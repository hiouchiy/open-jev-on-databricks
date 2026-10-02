# Databricks notebook source
# MAGIC %md
# MAGIC # open-jev (DeBERTa-v3-large) — Model Serving 1/2: register for RTX PRO 6000 serving
# MAGIC
# MAGIC Wraps [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
# MAGIC as an MLflow **pyfunc** and registers it to Unity Catalog with
# MAGIC **`env_pack="databricks_model_serving"`** (a.k.a. *GPU Express Deployment*), so it can be
# MAGIC served on `GPU_LARGE_RTX` (1x **NVIDIA RTX PRO 6000 Blackwell, 96 GB**) in `02_serve_rtx`.
# MAGIC
# MAGIC **Why env_pack?** The RTX PRO 6000 is a Blackwell GPU (compute capability 12.0). PyTorch only
# MAGIC ships Blackwell kernels in its **CUDA 12.8+** builds, so the serving container must run a
# MAGIC cu128 torch. `env_pack` *packs this notebook's Python environment* (torch cu128 + transformers
# MAGIC + this model's deps) next to the model and **restores it verbatim at serving time** — no
# MAGIC container rebuild that could swap torch for a build without Blackwell support.
# MAGIC
# MAGIC **What this notebook does**
# MAGIC 1. Installs a pinned, serving-ready stack (torch 2.10 / cu128, transformers 4.57, MLflow 3.12).
# MAGIC 2. Downloads the model snapshot (weights + bundled `typed_decisions` inference code) from the Hub.
# MAGIC 3. Writes the pyfunc (`state` + `questions` → calibrated answers, batched into one forward pass).
# MAGIC 4. Smoke-tests the pyfunc on this GPU, logs it with the snapshot as an **artifact**
# MAGIC    (so serving needs no Hugging Face egress and is pinned to one revision), and registers it
# MAGIC    with `env_pack`.
# MAGIC
# MAGIC **Compute:** attach **Serverless GPU** (A10 is enough — RTX is only needed for *serving*).
# MAGIC Compute drop-down → **Serverless GPU** → environment panel → Accelerator **A10** → Apply.
# MAGIC Or run it headless as a job (see the README, `databricks jobs submit`).
# MAGIC Runtime: ~6-10 min (package install, model logging, then env_pack packing + upload).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install the serving stack
# MAGIC Everything installed here is what Model Serving will run, because env_pack packs this env.
# MAGIC - `torch==2.10.0` from the **cu128** index → includes Blackwell (sm_120) kernels for RTX PRO 6000.
# MAGIC - A **matching `torchvision`** from the same index: the serverless GPU image preinstalls a
# MAGIC   torchvision built for its older base torch, which is ABI-broken against torch 2.10 and
# MAGIC   crashes `transformers` when it probes torchvision. Installing the matching build fixes it.
# MAGIC - `transformers` 4.57.x: the checkpoint was saved with 4.57.6 (`DebertaV2Model`).
# MAGIC - `sentencepiece` + `protobuf`: the DeBERTa-v3 tokenizer needs them.
# MAGIC - `mlflow==3.12.0` + `databricks-sdk>=0.102.0`: required by env_pack (older SDKs time out
# MAGIC   uploading the multi-GB packed environment).

# COMMAND ----------

# MAGIC %pip install --quiet --extra-index-url https://download.pytorch.org/whl/cu128 \
# MAGIC   "torch==2.10.0" "torchvision" \
# MAGIC   "transformers==4.57.6" "safetensors>=0.4.3" "huggingface_hub>=0.34,<1.0" \
# MAGIC   "sentencepiece>=0.2.0" "protobuf>=5.29.5,<6" \
# MAGIC   "mlflow==3.12.0" "databricks-sdk>=0.102.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration
# MAGIC Change these with the widgets at the top of the notebook (or `base_parameters` when run as a job).

# COMMAND ----------

import json
import logging
import os
import sys
import tempfile
import time

# Benign serverless warnings during MLflow logging (py4j whitelist / Spark Connect chatter).
for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)

dbutils.widgets.text("uc_catalog", "main", "UC catalog")
dbutils.widgets.text("uc_schema", "air_samples", "UC schema")
dbutils.widgets.text("model_name", "open_jev_deberta_v3_large", "UC model name")

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.{dbutils.widgets.get('model_name')}"

HF_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
# Pin the Hub revision so the registered model is reproducible.
HF_REVISION = "188ee67a5c93122b916e5acd5bdb0cb3623e380a"

# AI Runtime presets HF_HUB_ENABLE_HF_TRANSFER=1 without installing hf_transfer; disable it.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

print(f"UC model : {UC_MODEL}\nHF model : {HF_REPO}@{HF_REVISION[:10]}")

# COMMAND ----------

import torch
from importlib.metadata import version

print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| arch list:", torch.cuda.get_arch_list())
assert "sm_120" in torch.cuda.get_arch_list(), "this torch build has no Blackwell (sm_120) kernels"
print("GPU:", torch.cuda.get_device_name(0))
for p in ("transformers", "mlflow", "databricks-sdk", "sentencepiece", "protobuf"):
    print(f"  {p}: {version(p)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create the UC schema (idempotent)

# COMMAND ----------

from databricks.sdk import WorkspaceClient

try:
    WorkspaceClient().schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
    print(f"Created schema {UC_CATALOG}.{UC_SCHEMA}")
except Exception as e:
    if "already exists" not in str(e).lower():
        raise RuntimeError(
            f"Cannot create schema {UC_CATALOG}.{UC_SCHEMA}: {e}\n"
            "Ask for CREATE SCHEMA on the catalog, or point the widgets at an existing schema."
        ) from e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Download the model snapshot
# MAGIC Serverless notebooks should keep large files on local disk (`tempfile.mkdtemp()`), not `/Workspace`.
# MAGIC The snapshot contains the DeBERTa-v3-large backbone, the 3-layer scoring head
# MAGIC (`head.safetensors`), the tokenizer with the `[STATE]`/`[Q]`/`[OPT]` marker tokens,
# MAGIC `open_jev_config.json` (pooling mode, calibrated temperature) and the `typed_decisions/`
# MAGIC inference package — so no extra `pip install` is needed to run it.

# COMMAND ----------

from huggingface_hub import snapshot_download

WORK_DIR = tempfile.mkdtemp()
t0 = time.perf_counter()
SNAPSHOT_DIR = snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=os.path.join(WORK_DIR, "open_jev"))
print(f"Downloaded to {SNAPSHOT_DIR} in {time.perf_counter() - t0:.0f}s")
print(sorted(os.listdir(SNAPSHOT_DIR)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. The serving pyfunc
# MAGIC **Request** — one row per *state* (MLflow `dataframe_records`):
# MAGIC ```json
# MAGIC {"dataframe_records": [{
# MAGIC   "state": "I was charged twice for the same order. I want my money back.",
# MAGIC   "questions": "[{\"type\": \"choice\", \"instructions\": \"Which team?\", \"options\": [\"billing\", \"tech\"]},
# MAGIC                  {\"type\": \"score\",  \"instructions\": \"How angry?\", \"options\": [\"calm\", \"annoyed\", \"angry\"]},
# MAGIC                  {\"type\": \"noul\",   \"instructions\": \"Asks for a refund.\"}]"
# MAGIC }]}
# MAGIC ```
# MAGIC `questions` is a JSON string (a list of typed questions). A string column keeps the
# MAGIC MLflow schema simple while every row can carry a different number/kind of questions.
# MAGIC
# MAGIC **Response** — one row per state: `answers` (JSON list, one entry per question:
# MAGIC `choice`+`probabilities`+`confidence` / `score`+`probabilities`+`confidence` / `noul` = p(yes)),
# MAGIC `latency_ms`, `device` (GPU + dtype that served the row), and `error` (per-row validation errors are returned, not raised, so one bad row
# MAGIC never fails the whole request).
# MAGIC
# MAGIC All valid rows of a request are encoded into **one batched forward pass** (bf16 on Ampere+ GPUs, fp16 on T4).

# COMMAND ----------

PYFUNC_SRC = r'''
"""open-jev typed-decision model — MLflow pyfunc for Databricks Model Serving.

Input rows : {"state": str, "questions": JSON str (list of {type, instructions, options?})}
Output rows: {"answers": JSON str, "latency_ms": float, "device": str, "error": str}

Precision: bf16 autocast on Ampere and newer (A10G, H100, RTX PRO 6000); fp16 on older GPUs such as
T4 (Turing has no native bf16); fp32 on CPU. `device` reports the GPU and dtype that served the row.
"""
import json
import os
import sys
import time

import mlflow.pyfunc
import pandas as pd


class OpenJevModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch

        snapshot = context.artifacts["snapshot"]
        sys.path.insert(0, snapshot)  # the snapshot ships the `typed_decisions` package
        from typed_decisions.open_jev import OpenJev
        from typed_decisions.schema import readout

        self._jev = OpenJev.from_pretrained(snapshot)  # picks cuda when available
        self._readout = readout
        self._amp = self._jev.device.type == "cuda"
        if self._amp:
            major, _ = torch.cuda.get_device_capability(0)
            self._dtype = torch.bfloat16 if major >= 8 else torch.float16
            self._device = f"{torch.cuda.get_device_name(0)} ({str(self._dtype).split('.')[-1]})"
        else:
            self._dtype, self._device = torch.float32, "cpu (float32)"
        self._max_batch = int((context.model_config or {}).get("max_batch_states", 32))
        self._torch = torch

    def _forward(self, items):
        """items: list of (state, [Question]) -> list of per-state answer lists (one forward per chunk)."""
        torch, jev = self._torch, self._jev
        out = []
        for i in range(0, len(items), self._max_batch):
            chunk = items[i : i + self._max_batch]
            b = jev.collator(chunk, jev.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=self._dtype, enabled=self._amp):
                logits = jev.model(b["input_ids"], b["attention_mask"], b["opt_pos"],
                                   b["opt_mask"], b["q_pos"], b["seg"]).float()
            probs = (logits / jev.model.temperature).softmax(-1).cpu()
            for bi, (_, qs) in enumerate(chunk):
                answers = []
                for qi, q in enumerate(qs):
                    p = probs[bi, qi, : len(q.options)].tolist()
                    r = self._readout(q.kind, p)
                    if q.kind == "noul":
                        answers.append({"noul": r["noul"]})
                    else:
                        key = "choice" if q.kind == "choice" else "score"
                        val = q.options[r["choice"]] if q.kind == "choice" else r["score"]
                        answers.append({key: val, "probabilities": dict(zip(q.options, p)),
                                        "confidence": r["confidence"]})
                out.append(answers)
        return out

    def predict(self, context, model_input, params=None):
        rows = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        t0 = time.perf_counter()
        results = [{"answers": "[]", "latency_ms": 0.0, "device": self._device, "error": ""} for _ in rows]
        items, index = [], []
        for i, row in enumerate(rows):
            try:
                qs = row.get("questions")
                qs = json.loads(qs) if isinstance(qs, str) else list(qs)
                questions = [self._jev._question(j, q) for j, q in enumerate(qs)]
                if not questions:
                    raise ValueError("at least one question is required")
                # Validate length up front so one over-long row can't fail the whole batch.
                self._jev.collator.encode_one(str(row.get("state", "")), questions)
                items.append((str(row.get("state", "")), questions))
                index.append(i)
            except Exception as e:
                results[i]["error"] = f"{type(e).__name__}: {e}"
        if items:
            for i, answers in zip(index, self._forward(items)):
                results[i]["answers"] = json.dumps(answers, ensure_ascii=False)
        ms = (time.perf_counter() - t0) * 1000
        for r in results:
            r["latency_ms"] = round(ms, 2)
        return pd.DataFrame(results)


mlflow.models.set_model(OpenJevModel())
'''

PYFUNC_PATH = os.path.join(WORK_DIR, "open_jev_pyfunc.py")
with open(PYFUNC_PATH, "w") as f:
    f.write(PYFUNC_SRC)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Smoke-test the pyfunc on this GPU (same code path Model Serving will run)

# COMMAND ----------

import mlflow
import pandas as pd
from mlflow.models import infer_signature

QUESTIONS = [
    {"type": "choice", "instructions": "Which product area is the message about?",
     "options": ["fees & charges", "pin & security", "refund & dispute", "top-up", "card", "transfer", "other"]},
    {"type": "score", "instructions": "How positive is the sentiment of this message?",
     "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
    {"type": "noul", "instructions": "The customer is asking for a refund."},
]
input_example = pd.DataFrame([
    {"state": "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
     "questions": json.dumps(QUESTIONS)},
    {"state": "My new card arrived today and activation took two seconds. Great app!",
     "questions": json.dumps(QUESTIONS)},
])

model_config = {"max_batch_states": 32}
# Load the pyfunc file directly and call load_context + predict — the same calls Model Serving makes.
import importlib.util

spec = importlib.util.spec_from_file_location("open_jev_pyfunc", PYFUNC_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class _Ctx:  # minimal stand-in for mlflow's PythonModelContext
    artifacts = {"snapshot": SNAPSHOT_DIR}
    model_config = model_config


smoke = mod.OpenJevModel()
smoke.load_context(_Ctx())
pred = smoke.predict(None, input_example)
for _, r in pred.iterrows():
    print(r["latency_ms"], "ms", r["error"] or json.loads(r["answers"]))
signature = infer_signature(input_example, pred)
del smoke
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Log, then register with `env_pack`
# MAGIC - Log **without** `registered_model_name`, then call `mlflow.register_model(..., env_pack=...)`.
# MAGIC - `install_dependencies=False`: restore the packed env as-is (no re-resolution at serving).
# MAGIC - `pip_requirements` is listed explicitly for documentation/portability; env_pack is what
# MAGIC   actually provides the runtime environment.

# COMMAND ----------

from mlflow.utils.env_pack import EnvPackConfig

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/open-jev-serving")

with mlflow.start_run(run_name="open-jev-register-env-pack") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "torch": torch.__version__,
                       "serving_workload_type": "GPU_LARGE_RTX"})
    t0 = time.perf_counter()
    info = mlflow.pyfunc.log_model(
        name="open_jev",
        python_model=PYFUNC_PATH,
        artifacts={"snapshot": SNAPSHOT_DIR},
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
    print(f"Logged {info.model_uri} in {time.perf_counter() - t0:.0f}s")

t0 = time.perf_counter()
mv = mlflow.register_model(
    model_uri=info.model_uri,
    name=UC_MODEL,
    env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
)
print(f"Registered {UC_MODEL} version {mv.version} in {time.perf_counter() - t0:.0f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC Next: **`02_serve_rtx`** creates a `GPU_LARGE_RTX` serving endpoint for this version.

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "version": str(mv.version), "run_id": run.info.run_id}))
