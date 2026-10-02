"""open-jev (DeBERTa-v3-large) — register for RTX PRO 6000 Model Serving — AI Runtime CLI script.

Wraps the model as an MLflow pyfunc (`open_jev_pyfunc.py`) and registers it to Unity Catalog with
`env_pack="databricks_model_serving"` (GPU Express Deployment), so it can be served on
`GPU_LARGE_RTX` (1x RTX PRO 6000 Blackwell, 96 GB).

Why env_pack: RTX PRO 6000 is Blackwell (compute capability 12.0), and PyTorch only ships Blackwell
kernels in its CUDA 12.8+ builds. The YAML's `command` installs torch 2.10 (cu128) into this job's
environment; env_pack packs exactly that environment next to the model and restores it verbatim at
serving time, so the endpoint runs a Blackwell-capable torch.

Run it with (from this directory):
    COPYFILE_DISABLE=1 air run --file register.yaml --watch --profile <your-profile>
The notebook equivalent is ../notebook/01_register.py.

KNOWN ISSUE (2026-10): registration succeeds, but endpoints created from a version registered by
THIS AIR CLI job stay at "Creating resources for served entity." for 10+ hours (reproduced on
GPU_MEDIUM and GPU_LARGE_RTX, two workspaces). Versions registered by the notebook
../notebook/01_register.py deploy in ~3 min. Use the notebook path for serving.
"""

import os

# AI Runtime presets HF_HUB_ENABLE_HF_TRANSFER=1 but does not ship `hf_transfer`; force it off
# BEFORE huggingface_hub is imported.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import logging
import tempfile
import time
from importlib.metadata import version

for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)

UC_CATALOG = os.environ.get("UC_CATALOG", "main")
UC_SCHEMA = os.environ.get("UC_SCHEMA", "air_samples")
UC_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.{os.environ.get('MODEL_NAME', 'open_jev_deberta_v3_large')}"
HF_REPO = os.environ.get("HF_REPO", "com-kotobalabs/open-jev-deberta-v3-large")
HF_REVISION = os.environ.get("HF_REVISION", "188ee67a5c93122b916e5acd5bdb0cb3623e380a")
PYFUNC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "open_jev_pyfunc.py")

QUESTIONS = [
    {"type": "choice", "instructions": "Which product area is the message about?",
     "options": ["fees & charges", "pin & security", "refund & dispute", "top-up", "card", "transfer", "other"]},
    {"type": "score", "instructions": "How positive is the sentiment of this message?",
     "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
    {"type": "noul", "instructions": "The customer is asking for a refund."},
]


def ensure_schema() -> None:
    from databricks.sdk import WorkspaceClient

    try:
        WorkspaceClient().schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
        print(f"Created schema {UC_CATALOG}.{UC_SCHEMA}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise RuntimeError(f"Cannot create schema {UC_CATALOG}.{UC_SCHEMA}: {e}") from e


def main() -> None:
    import importlib.util

    import mlflow
    import pandas as pd
    import torch
    from huggingface_hub import snapshot_download
    from mlflow.models import infer_signature
    from mlflow.utils.env_pack import EnvPackConfig

    print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| arch list:", torch.cuda.get_arch_list())
    assert "sm_120" in torch.cuda.get_arch_list(), "this torch build has no Blackwell (sm_120) kernels"
    for p in ("transformers", "mlflow", "databricks-sdk", "sentencepiece", "protobuf"):
        print(f"  {p}: {version(p)}")
    print(f"UC model: {UC_MODEL}  HF: {HF_REPO}@{HF_REVISION[:10]}")
    ensure_schema()

    # Keep large files on local disk.
    work_dir = tempfile.mkdtemp()
    t0 = time.perf_counter()
    snapshot = snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=os.path.join(work_dir, "open_jev"))
    print(f"Downloaded snapshot in {time.perf_counter() - t0:.0f}s")

    input_example = pd.DataFrame([
        {"state": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
         "questions": json.dumps(QUESTIONS)},
        {"state": "My new card arrived today and activation took two seconds. Great app!",
         "questions": json.dumps(QUESTIONS)},
    ])
    model_config = {"max_batch_states": 32}

    # Smoke-test: call load_context + predict exactly like Model Serving will.
    spec = importlib.util.spec_from_file_location("open_jev_pyfunc", PYFUNC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    class _Ctx:  # minimal stand-in for mlflow's PythonModelContext
        artifacts = {"snapshot": snapshot}

    _Ctx.model_config = model_config
    smoke = mod.OpenJevModel()
    smoke.load_context(_Ctx())
    pred = smoke.predict(None, input_example)
    for _, r in pred.iterrows():
        print(r["latency_ms"], "ms", r["error"] or json.loads(r["answers"]))
    signature = infer_signature(input_example, pred)
    del smoke
    torch.cuda.empty_cache()

    mlflow.set_registry_uri("databricks-uc")
    # AI Runtime sets MLFLOW_EXPERIMENT_ID (from the YAML `experiment_name`) and MLFLOW_RUN_ID.
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")) as run:
        mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "torch": torch.__version__,
                           "serving_workload_type": "GPU_LARGE_RTX"})
        t0 = time.perf_counter()
        info = mlflow.pyfunc.log_model(
            name="open_jev",
            python_model=PYFUNC_PATH,
            artifacts={"snapshot": snapshot},  # weights + bundled inference code: no Hub egress at serving
            model_config=model_config,
            signature=signature,
            input_example=input_example,
            pip_requirements=[
                "--extra-index-url https://download.pytorch.org/whl/cu128",
                f"torch=={torch.__version__.split('+')[0]}",
                f"transformers=={version('transformers')}",
                f"sentencepiece=={version('sentencepiece')}",
                f"protobuf=={version('protobuf')}",
                f"mlflow=={mlflow.__version__}",
            ],
        )
        print(f"Logged {info.model_uri} in {time.perf_counter() - t0:.0f}s (run {run.info.run_id})")

    # Register separately with env_pack; install_dependencies=False restores the packed env as-is.
    t0 = time.perf_counter()
    mv = mlflow.register_model(
        model_uri=info.model_uri,
        name=UC_MODEL,
        env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
    )
    print(f"Registered {UC_MODEL} version {mv.version} in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
