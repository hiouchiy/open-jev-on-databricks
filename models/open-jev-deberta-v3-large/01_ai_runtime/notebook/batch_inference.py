# Databricks notebook source
# MAGIC %md
# MAGIC # open-jev (DeBERTa-v3-large) — AI Runtime: GPU batch inference + evaluation
# MAGIC
# MAGIC [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
# MAGIC is a **typed-decision model**: one text (the *state*) plus any number of typed *questions* go in, and a
# MAGIC calibrated probability distribution per question comes back from **one encoder forward pass**.
# MAGIC
# MAGIC | question type | what you give | what you get back |
# MAGIC |---|---|---|
# MAGIC | `choice` | instructions + 2-255 options | `choice` (the option), `probabilities`, `confidence` |
# MAGIC | `score` | instructions + 2-10 *ordered* levels | `score` (expected level index), `probabilities`, `confidence` |
# MAGIC | `noul` | a yes/no statement | `noul` = p(yes) |
# MAGIC
# MAGIC Nothing is generated, so the output is always well-formed — a fast, cheap alternative to an LLM
# MAGIC for routing / triage / extraction-style decisions. Base model: DeBERTa-v3-large (435M params), Apache-2.0.
# MAGIC
# MAGIC **This notebook**, on one serverless GPU:
# MAGIC 1. runs a few ad-hoc decisions (no labels needed — any questions you like),
# MAGIC 2. scores the model authors' published test sets (banking77 / SST-5 / BoolQ, gold labels) in batched bf16,
# MAGIC 3. logs accuracy / Brier / ECE + throughput to MLflow and writes predictions to a UC Volume.
# MAGIC
# MAGIC **Compute:** compute drop-down → **Serverless GPU** → environment panel → Accelerator **A10**,
# MAGIC Base environment **AI v5** (or newer) → **Apply**. Then *Run all*. (~5 min incl. downloads.)
# MAGIC
# MAGIC The AI Runtime CLI version of this notebook is `../cli/batch_inference.py` + `../cli/batch_inference.yaml`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install
# MAGIC The AI base environment already ships torch, transformers, MLflow and pandas. The DeBERTa-v3
# MAGIC tokenizer additionally needs `sentencepiece`. `transformers<5`: the checkpoint was saved with 4.57
# MAGIC (a no-op on AI v5, which ships 4.57); no floor is raised, so the base env is never downgraded.

# COMMAND ----------

# MAGIC %pip install --quiet "sentencepiece>=0.2.0" "transformers>=4.48,<5"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration (widgets at the top of the notebook)

# COMMAND ----------

import os

# AI Runtime presets HF_HUB_ENABLE_HF_TRANSFER=1 but does not ship `hf_transfer`, which makes every
# Hub download fail. Force the plain downloader BEFORE huggingface_hub is imported.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
import logging
import sys
import time
import urllib.request
from collections import defaultdict

# Benign serverless warnings during MLflow logging (py4j whitelist / Spark Connect chatter).
for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)

dbutils.widgets.text("uc_catalog", "main", "UC catalog")
dbutils.widgets.text("uc_schema", "air_samples", "UC schema")
dbutils.widgets.text("uc_volume", "predictions", "UC volume")
dbutils.widgets.text("max_states", "0", "Texts per split (0 = all)")
dbutils.widgets.text("batch_size", "64", "Texts per forward pass")

UC_CATALOG = dbutils.widgets.get("uc_catalog")
UC_SCHEMA = dbutils.widgets.get("uc_schema")
UC_VOLUME = dbutils.widgets.get("uc_volume")
MAX_STATES = int(dbutils.widgets.get("max_states"))
BATCH_SIZE = int(dbutils.widgets.get("batch_size"))
BF16 = True  # bf16 autocast: ~2x faster on A10 with the same accuracy
OUTPUT_NAME = "open_jev_predictions"

# Model on the Hugging Face Hub, pinned to a revision for reproducibility.
HF_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
HF_REVISION = "188ee67a5c93122b916e5acd5bdb0cb3623e380a"

# Evaluation data: the model authors' published splits (pinned commit of kotoba-lang/typed-decisions).
#   test     = question types seen in training (in-domain)
#   ood-test = never-seen instructions / option sets on the same texts (out-of-domain)
DATA_BASE_URL = ("https://raw.githubusercontent.com/kotoba-lang/typed-decisions/"
                 "e65215db7e554a31f557936f0bd1d8a0a78b6af5/data-fam")
SPLITS = ["test", "ood-test"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load the model
# MAGIC The model repo ships its own small inference package (`typed_decisions/`), so a snapshot download
# MAGIC is all that's needed — no extra `pip install`.

# COMMAND ----------

import torch
from huggingface_hub import snapshot_download

t0 = time.perf_counter()
snapshot = snapshot_download(HF_REPO, revision=HF_REVISION)
sys.path.insert(0, snapshot)
from typed_decisions.open_jev import OpenJev
from typed_decisions.schema import Example, readout

jev = OpenJev.from_pretrained(snapshot)
print(f"Loaded {HF_REPO} on {jev.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}) "
      f"in {time.perf_counter() - t0:.1f}s; calibrated temperature={jev.model.temperature}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ad-hoc decisions — no training, any questions
# MAGIC Several questions of different types on one text are answered in a single forward pass.
# MAGIC (You may see a tokenizer "incorrect regex pattern" warning from transformers — it is a false
# MAGIC positive for this DeBERTa tokenizer; the accuracy below matches the model card.)

# COMMAND ----------

answers = jev.decide(
    "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    [{"type": "choice", "instructions": "Which team should handle this message?",
      "options": ["billing", "fraud & security", "card delivery", "technical support", "other"]},
     {"type": "score", "instructions": "How positive is the sentiment of this message?",
      "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
     {"type": "noul", "instructions": "The customer is asking for a refund."}],
)
for a in answers:
    print(json.dumps(a, indent=None, default=lambda x: round(x, 3)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Batch inference + evaluation on the published test sets

# COMMAND ----------

def load_examples(split):
    url = f"{DATA_BASE_URL}/{split}.jsonl"
    lines = urllib.request.urlopen(url, timeout=120).read().decode().splitlines()
    examples = [Example.from_json(l) for l in lines if l.strip()]
    if MAX_STATES:
        examples = examples[:MAX_STATES]
    print(f"{split}: {len(examples)} texts, {sum(len(e.questions) for e in examples)} questions")
    return examples


def score(examples):
    """Batched inference. Returns one record per question with the full distribution."""
    use_amp = BF16 and jev.device.type == "cuda"
    records = []
    for i in range(0, len(examples), BATCH_SIZE):
        chunk = examples[i : i + BATCH_SIZE]
        b = jev.collator([(e.state, e.questions) for e in chunk], jev.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            logits = jev.model(b["input_ids"], b["attention_mask"], b["opt_pos"],
                               b["opt_mask"], b["q_pos"], b["seg"]).float()
        probs = (logits / jev.model.temperature).softmax(-1).cpu()
        for bi, e in enumerate(chunk):
            for qi, q in enumerate(e.questions):
                p = probs[bi, qi, : len(q.options)].tolist()
                pred = max(range(len(p)), key=p.__getitem__)
                r = readout(q.kind, p)
                records.append({
                    "qid": q.qid, "source": e.source, "kind": q.kind,
                    "instructions": q.instructions, "options": json.dumps(q.options),
                    "gold": q.gold, "pred": pred, "pred_option": q.options[pred],
                    "confidence": p[pred], "correct": pred == q.gold,
                    # kind-specific read-out: expected level index for `score`, p(yes) for `noul`
                    "readout": r.get("score", r.get("noul")),
                    "probabilities": json.dumps(p),
                })
    return records


def metrics(records, n_bins=15):
    """accuracy, multi-class Brier score and expected calibration error (ECE, max-prob binning)."""
    if not records:
        return {}
    acc = sum(r["correct"] for r in records) / len(records)
    brier = 0.0
    bins = defaultdict(lambda: [0, 0.0, 0])  # bin -> [count, sum(conf), n_correct]
    for r in records:
        p = json.loads(r["probabilities"])
        brier += sum((pj - (j == r["gold"])) ** 2 for j, pj in enumerate(p))
        k = min(int(r["confidence"] * n_bins), n_bins - 1)
        bins[k][0] += 1
        bins[k][1] += r["confidence"]
        bins[k][2] += r["correct"]
    ece = sum(abs(c / n - s / n) * n for n, s, c in bins.values()) / len(records)
    return {"n": len(records), "accuracy": acc, "brier": brier / len(records), "ece": ece}


def evaluate(records, split):
    """Flatten overall / per-kind / per-source metrics into MLflow-friendly keys."""
    out = {f"{split}/{k}": v for k, v in metrics(records).items()}
    for field in ("kind", "source"):
        for value in sorted({r[field] for r in records}):
            sub = [r for r in records if r[field] == value]
            out[f"{split}/{field}={value}/accuracy"] = metrics(sub)["accuracy"]
    return out

# COMMAND ----------

import mlflow
import pandas as pd

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/open-jev-batch-inference")

all_records = []
with mlflow.start_run(run_name="open-jev-batch-inference") as run:
    mlflow.log_params({"hf_repo": HF_REPO, "hf_revision": HF_REVISION, "batch_size": BATCH_SIZE,
                       "bf16": BF16, "max_states": MAX_STATES,
                       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"})
    for split in SPLITS:
        examples = load_examples(split)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        records = score(examples)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        m = evaluate(records, split)
        m[f"{split}/questions_per_sec"] = len(records) / dt
        m[f"{split}/texts_per_sec"] = len(examples) / dt
        mlflow.log_metrics({k.replace("=", "_"): v for k, v in m.items()})
        all_records += [dict(r, split=split) for r in records]
    mlflow.log_metric("peak_gpu_mem_gib", torch.cuda.max_memory_allocated() / 2**30)

pdf = pd.DataFrame(all_records)
summary = (pdf.groupby(["split", "source", "kind"])["correct"].agg(["mean", "count"])
              .rename(columns={"mean": "accuracy", "count": "questions"}).reset_index())
display(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC Expected (full splits): **in-domain accuracy ≈ 0.86, ECE ≈ 0.015**; **OOD ≈ 0.69** — in line with the
# MAGIC model card (0.854 / 0.690). `score` questions on unseen level sets are the weakest, as the card notes.
# MAGIC Calibration means the `confidence` is usable for routing: e.g. auto-handle above 0.9, send the rest to a human.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save predictions to a Unity Catalog Volume

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
for fn, what in [
    (lambda: w.schemas.create(name=UC_SCHEMA, catalog_name=UC_CATALOG), f"schema {UC_CATALOG}.{UC_SCHEMA}"),
    (lambda: w.volumes.create(catalog_name=UC_CATALOG, schema_name=UC_SCHEMA, name=UC_VOLUME,
                              volume_type=VolumeType.MANAGED), f"volume {UC_CATALOG}.{UC_SCHEMA}.{UC_VOLUME}"),
]:
    try:
        fn()
        print(f"Created {what}")
    except Exception as e:
        if "already exists" not in str(e).lower():
            raise RuntimeError(f"Cannot create {what}: {e}\nPre-create it or change the widgets.") from e

# Write locally, then copy: sequential writes are FUSE-safe on /Volumes.
local = f"/tmp/{OUTPUT_NAME}.parquet"
pdf.to_parquet(local, index=False)
dest = f"/Volumes/{UC_CATALOG}/{UC_SCHEMA}/{UC_VOLUME}/{OUTPUT_NAME}.parquet"
with open(local, "rb") as src, open(dest, "wb") as dst:
    dst.write(src.read())
print(f"Wrote {len(pdf)} predictions to {dest} (MLflow run {run.info.run_id})")
