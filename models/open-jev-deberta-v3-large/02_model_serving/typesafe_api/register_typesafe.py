# Databricks notebook source
# MAGIC %md
# MAGIC # open-jev — register a TypeSafe System One compatible model for Model Serving
# MAGIC
# MAGIC Registers [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
# MAGIC with a **TypeSafe AI API compatible** request/response (`typesafe_pyfunc.py` + `typesafe_adapter.py`
# MAGIC in this folder) and **env_pack**, so it can be served on any GPU including `GPU_LARGE_RTX`
# MAGIC (RTX PRO 6000, Blackwell → needs the torch cu128 build installed below).
# MAGIC
# MAGIC Put the whole `typesafe_api/` folder in the workspace first — clone the repo as a **Git folder**, or
# MAGIC `databricks workspace import-dir typesafe_api /Workspace/Users/<you>/typesafe_api` — because this
# MAGIC notebook loads the two `.py` files next to it.
# MAGIC
# MAGIC **Compute:** Serverless GPU (A10). ~10 min. Next: deploy the endpoint and the gateway app (see README).

# COMMAND ----------

# MAGIC %pip install --quiet --extra-index-url https://download.pytorch.org/whl/cu128 \
# MAGIC   "torch==2.10.0" "torchvision" \
# MAGIC   "transformers==4.57.6" "safetensors>=0.4.3" "huggingface_hub>=0.34,<1.0" \
# MAGIC   "sentencepiece>=0.2.0" "protobuf>=5.29.5,<6" \
# MAGIC   "mlflow==3.12.0" "databricks-sdk>=0.102.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json
import logging
import os
import sys
import tempfile
import time

for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)

dbutils.widgets.text("uc_catalog", "main", "UC catalog")
dbutils.widgets.text("uc_schema", "air_samples", "UC schema")
dbutils.widgets.text("model_name", "open_jev_deberta_v3_large_typesafe", "UC model name")
# How criteria are turned into option texts — defaults are the most accurate variants measured by
# eval_criteria_modes.py (see README).
dbutils.widgets.dropdown("choice_text_mode", "name", ["name", "name_description", "description"], "Choice text mode")
dbutils.widgets.dropdown("noul_criteria_mode", "fallback", ["fallback", "options"], "Noul criteria mode")

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_MODEL = f"{UC_CATALOG}.{UC_SCHEMA}.{dbutils.widgets.get('model_name')}"
CHOICE_TEXT_MODE = dbutils.widgets.get("choice_text_mode")
NOUL_CRITERIA_MODE = dbutils.widgets.get("noul_criteria_mode")

HF_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
HF_REVISION = "188ee67a5c93122b916e5acd5bdb0cb3623e380a"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # AI Runtime presets 1 without shipping hf_transfer

# This notebook's folder in the workspace holds typesafe_pyfunc.py and typesafe_adapter.py.
nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
HERE = "/Workspace" + os.path.dirname(nb_path)
PYFUNC_PATH, ADAPTER_PATH = f"{HERE}/typesafe_pyfunc.py", f"{HERE}/typesafe_adapter.py"
assert os.path.exists(PYFUNC_PATH) and os.path.exists(ADAPTER_PATH), f"put typesafe_api/ in {HERE} (see header)"
print(f"UC model: {UC_MODEL}\nchoice_text_mode: {CHOICE_TEXT_MODE}  noul_criteria_mode: {NOUL_CRITERIA_MODE}\ncode: {HERE}")

# COMMAND ----------

import torch

print("torch", torch.__version__, "| arch list:", torch.cuda.get_arch_list(), "|", torch.cuda.get_device_name(0))
assert "sm_120" in torch.cuda.get_arch_list(), "this torch build has no Blackwell (sm_120) kernels"

from databricks.sdk import WorkspaceClient

try:
    WorkspaceClient().schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG)
except Exception as e:
    if "already exists" not in str(e).lower():
        raise

from huggingface_hub import snapshot_download

WORK_DIR = tempfile.mkdtemp()  # large files on local disk, not /Workspace
SNAPSHOT_DIR = snapshot_download(HF_REPO, revision=HF_REVISION, local_dir=os.path.join(WORK_DIR, "open_jev"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test — the same `load_context` + `predict` calls Model Serving makes

# COMMAND ----------

import importlib.util

import pandas as pd

sys.path.insert(0, HERE)  # typesafe_adapter (shipped to serving via code_paths)
spec = importlib.util.spec_from_file_location("typesafe_pyfunc", PYFUNC_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

MODEL_CONFIG = {"choice_text_mode": CHOICE_TEXT_MODE, "noul_criteria_mode": NOUL_CRITERIA_MODE, "model_aliases": ["jev-latest", "open-jev-deberta-v3-large"],
                "model_id": "open-jev-deberta-v3-large", "max_batch_requests": 32}


class _Ctx:  # minimal stand-in for mlflow's PythonModelContext
    artifacts = {"snapshot": SNAPSHOT_DIR}
    model_config = MODEL_CONFIG


EXAMPLE = {
    "model": "jev-latest",
    "state": "I was charged twice. Please refund the duplicate charge.",
    "questions": {
        "requests_refund": {"type": "noul", "instructions": "Is the customer requesting a refund?"},
        "department": {"type": "choice", "instructions": "Which team should handle this?",
                       "criteria": {"billing": "Charges, payments, invoices, and refunds",
                                    "technical": "Bugs, outages, and integrations",
                                    "shipping": "Delivery, tracking, and lost packages"}},
        "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                        "criteria": ["Calm and neutral", "Frustrated but civil", "Very angry or threatening to leave"]},
    },
}
input_example = pd.DataFrame([{"request": json.dumps(EXAMPLE)},
                              {"request": json.dumps({"model": "jev-latest", "questions": {}})}])  # -> 422

smoke = mod.TypeSafeOpenJevModel()
smoke.load_context(_Ctx())
pred = smoke.predict(None, input_example)
for _, r in pred.iterrows():
    print(r["status"], r["device"], json.loads(r["body"]))
assert list(pred["status"]) == [200, 422]

from mlflow.models import infer_signature

signature = infer_signature(input_example, pred)
del smoke
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log, then register with env_pack

# COMMAND ----------

import mlflow
from importlib.metadata import version
from mlflow.utils.env_pack import EnvPackConfig

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/open-jev-typesafe-serving")

with mlflow.start_run(run_name="open-jev-typesafe-register") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "torch": torch.__version__,
                       "api": "typesafe-systemone", **{k: str(v) for k, v in MODEL_CONFIG.items()}})
    info = mlflow.pyfunc.log_model(
        name="open_jev_typesafe",
        python_model=PYFUNC_PATH,
        code_paths=[ADAPTER_PATH],
        artifacts={"snapshot": SNAPSHOT_DIR},  # weights + bundled inference code: no Hub egress at serving
        model_config=MODEL_CONFIG,
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

mv = mlflow.register_model(
    model_uri=info.model_uri,
    name=UC_MODEL,
    env_pack=EnvPackConfig(name="databricks_model_serving", install_dependencies=False),
)
print(f"Registered {UC_MODEL} version {mv.version}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "version": str(mv.version), "run_id": run.info.run_id}))
