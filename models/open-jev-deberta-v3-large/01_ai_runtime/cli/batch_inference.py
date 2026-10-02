"""open-jev (DeBERTa-v3-large) GPU batch inference + evaluation — AI Runtime CLI script.

open-jev is a *typed-decision* model: one text ("state") plus any number of typed questions go in,
and a calibrated probability distribution per question comes back from ONE encoder forward pass:
    choice  pick one of up to 255 options        -> choice, probabilities, confidence
    score   one of 2-10 ordered levels           -> score (expected level index), probabilities, confidence
    noul    yes / no                             -> noul = p(yes)
Nothing is generated, so the output is always well-formed.

This script, on one AI Runtime GPU:
  1. downloads the model snapshot (pinned revision) from the Hugging Face Hub,
  2. downloads the model authors' published test sets (banking77 / SST-5 / BoolQ, gold labels),
  3. scores every question in batched bf16 forward passes,
  4. logs accuracy / Brier / ECE (overall, per question type, per dataset) + throughput to MLflow,
  5. writes per-question predictions as Parquet to a Unity Catalog Volume.

Run it with (from this directory):
    COPYFILE_DISABLE=1 air run --file batch_inference.yaml --watch --profile <your-profile>
The notebook equivalent is ../notebook/batch_inference.py.
"""

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
from dataclasses import asdict, dataclass

# Benign serverless warnings during MLflow logging (py4j whitelist / Spark Connect chatter).
for _name in ("mlflow.tracking.context.registry", "pyspark.sql.connect", "py4j"):
    logging.getLogger(_name).setLevel(logging.ERROR)


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # Unity Catalog destination for the predictions file (override in the YAML `env_variables`).
    uc_catalog: str = _env("UC_CATALOG", "main")
    uc_schema: str = _env("UC_SCHEMA", "air_samples")
    uc_volume: str = _env("UC_VOLUME", "predictions")
    output_name: str = _env("OUTPUT_NAME", "open_jev_predictions")

    # Model on the Hugging Face Hub, pinned to a revision for reproducibility.
    hf_repo: str = _env("HF_REPO", "com-kotobalabs/open-jev-deberta-v3-large")
    hf_revision: str = _env("HF_REVISION", "188ee67a5c93122b916e5acd5bdb0cb3623e380a")

    # Evaluation data: the model authors' published splits (pinned commit of kotoba-lang/typed-decisions).
    #   test     = question types seen in training (in-domain)
    #   ood-test = never-seen instructions / option sets on the same texts (out-of-domain)
    data_base_url: str = _env(
        "DATA_BASE_URL",
        "https://raw.githubusercontent.com/kotoba-lang/typed-decisions/"
        "e65215db7e554a31f557936f0bd1d8a0a78b6af5/data-fam",
    )
    splits: str = _env("SPLITS", "test,ood-test")
    max_states: int = int(_env("MAX_STATES", "0"))  # 0 = all (3,000 texts per split); small = smoke test

    batch_size: int = int(_env("BATCH_SIZE", "64"))  # texts per forward pass (each carries 1-3 questions)
    bf16: bool = _env("BF16", "1") == "1"            # bf16 autocast: ~2x faster on A10, same accuracy


CFG = Config()
print(CFG)


def ensure_uc_volume(catalog: str, schema: str, volume: str) -> None:
    """Create the schema + MANAGED volume if missing (idempotent). GPU nodes have no Spark, so we use
    the Databricks SDK. Without CREATE rights, fail with an actionable message."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.catalog import VolumeType

    w = WorkspaceClient()
    steps = [
        (lambda: w.schemas.create(name=schema, catalog_name=catalog), f"schema {catalog}.{schema}"),
        (lambda: w.volumes.create(catalog_name=catalog, schema_name=schema, name=volume,
                                  volume_type=VolumeType.MANAGED), f"volume {catalog}.{schema}.{volume}"),
    ]
    for fn, what in steps:
        try:
            fn()
            print(f"Created {what}")
        except Exception as e:
            if "already exists" in str(e).lower():
                continue
            raise RuntimeError(
                f"Cannot create {what}: {e}\nPre-create it (see setup.sh) or point "
                "UC_CATALOG/UC_SCHEMA/UC_VOLUME at one you can write to."
            ) from e


def load_model(cfg: Config):
    """Download the snapshot and build the model with the inference code bundled in it."""
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    snapshot = snapshot_download(cfg.hf_repo, revision=cfg.hf_revision)
    # The model repo ships its own `typed_decisions` package (no pip install needed).
    sys.path.insert(0, snapshot)
    from typed_decisions.open_jev import OpenJev

    jev = OpenJev.from_pretrained(snapshot)
    print(f"Loaded {cfg.hf_repo} on {jev.device} in {time.perf_counter() - t0:.1f}s "
          f"(calibrated temperature={jev.model.temperature})")
    return jev


def load_examples(cfg: Config, split: str):
    from typed_decisions.schema import Example

    url = f"{cfg.data_base_url}/{split}.jsonl"
    lines = urllib.request.urlopen(url, timeout=120).read().decode().splitlines()
    examples = [Example.from_json(l) for l in lines if l.strip()]
    if cfg.max_states:
        examples = examples[: cfg.max_states]
    print(f"{split}: {len(examples)} texts, {sum(len(e.questions) for e in examples)} questions")
    return examples


def score(jev, examples, cfg: Config):
    """Batched inference. Returns one record per question with the full distribution."""
    import torch

    from typed_decisions.schema import readout

    use_amp = cfg.bf16 and jev.device.type == "cuda"
    records = []
    for i in range(0, len(examples), cfg.batch_size):
        chunk = examples[i : i + cfg.batch_size]
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


def metrics(records, n_bins: int = 15) -> dict:
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


def evaluate(records, split: str) -> dict:
    """Flatten overall / per-kind / per-source metrics into MLflow-friendly keys."""
    out = {f"{split}/{k}": v for k, v in metrics(records).items()}
    for field in ("kind", "source"):
        for value in sorted({r[field] for r in records}):
            sub = [r for r in records if r[field] == value]
            out[f"{split}/{field}={value}/accuracy"] = metrics(sub)["accuracy"]
    return out


def main() -> None:
    import mlflow
    import pandas as pd
    import torch

    jev = load_model(CFG)

    # AI Runtime sets MLFLOW_EXPERIMENT_ID (from the YAML `experiment_name`); log into its run.
    with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")) as run:
        mlflow.log_params({k: v for k, v in asdict(CFG).items() if k != "data_base_url"})
        mlflow.log_param("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

        all_records = []
        for split in [s.strip() for s in CFG.splits.split(",") if s.strip()]:
            examples = load_examples(CFG, split)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            records = score(jev, examples, CFG)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0

            m = evaluate(records, split)
            m[f"{split}/questions_per_sec"] = len(records) / dt
            m[f"{split}/texts_per_sec"] = len(examples) / dt
            mlflow.log_metrics({k.replace("=", "_"): v for k, v in m.items()})
            for k, v in m.items():
                print(f"  {k:45s} {v:.4f}" if isinstance(v, float) else f"  {k:45s} {v}")
            all_records += [dict(r, split=split) for r in records]

        if torch.cuda.is_available():
            mlflow.log_metric("peak_gpu_mem_gib", torch.cuda.max_memory_allocated() / 2**30)

        # Write predictions to a UC Volume. Write to local disk first, then copy (sequential write
        # is FUSE-safe), and never mkdir the Volume root itself (a Volume can't be created by mkdir).
        ensure_uc_volume(CFG.uc_catalog, CFG.uc_schema, CFG.uc_volume)
        local = f"/tmp/{CFG.output_name}.parquet"
        pd.DataFrame(all_records).to_parquet(local, index=False)
        dest = f"/Volumes/{CFG.uc_catalog}/{CFG.uc_schema}/{CFG.uc_volume}/{CFG.output_name}.parquet"
        with open(local, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        mlflow.log_param("predictions_path", dest)
        print(f"Wrote {len(all_records)} predictions to {dest} (MLflow run {run.info.run_id})")


if __name__ == "__main__":
    main()
