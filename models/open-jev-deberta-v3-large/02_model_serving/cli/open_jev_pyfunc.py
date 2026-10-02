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
