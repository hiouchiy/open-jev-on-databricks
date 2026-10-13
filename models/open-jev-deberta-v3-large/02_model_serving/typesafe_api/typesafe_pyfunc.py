"""open-jev with a TypeSafe System One compatible request/response — MLflow pyfunc for Model Serving.

Model Serving fixes the transport (`POST /serving-endpoints/<name>/invocations`, MLflow input
formats, responses wrapped in `predictions`), so the TypeSafe body travels as one JSON string per row:

  input  rows : {"request": "<TypeSafe POST /v1/systemone body as JSON>"}
  output rows : {"status": 200 | 422, "body": "<TypeSafe response body, or {'detail': [...]} on 422>",
                 "device": "<GPU (dtype)>"}

The companion gateway app (../typesafe_gateway) unwraps this so the official `typesafe-sdk` can talk to
it unchanged. All valid rows of one invocation are scored in a single batched forward pass.
Translation rules live in `typesafe_adapter.py` (shipped with the model via `code_paths`).
"""

import json
import sys

import mlflow.pyfunc
import pandas as pd


class TypeSafeOpenJevModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch

        import typesafe_adapter

        snapshot = context.artifacts["snapshot"]
        sys.path.insert(0, snapshot)  # the snapshot ships the `typed_decisions` package
        from typed_decisions.open_jev import OpenJev

        cfg = context.model_config or {}
        self._A = typesafe_adapter
        self._aliases = tuple(cfg.get("model_aliases", typesafe_adapter.DEFAULT_ALIASES))
        self._model_id = cfg.get("model_id", typesafe_adapter.MODEL_ID)
        self._mode = cfg.get("choice_text_mode", typesafe_adapter.DEFAULT_CHOICE_TEXT_MODE)
        self._noul_mode = cfg.get("noul_criteria_mode", typesafe_adapter.DEFAULT_NOUL_CRITERIA_MODE)
        self._max_batch = int(cfg.get("max_batch_requests", 32))

        self._jev = OpenJev.from_pretrained(snapshot)
        self._torch = torch
        self._amp = self._jev.device.type == "cuda"
        if self._amp:
            # bf16 on Ampere and newer (A10G, H100, RTX PRO 6000); fp16 on T4 (no native bf16).
            major, _ = torch.cuda.get_device_capability(0)
            self._dtype = torch.bfloat16 if major >= 8 else torch.float16
            self._device = f"{torch.cuda.get_device_name(0)} ({str(self._dtype).split('.')[-1]})"
        else:
            self._dtype, self._device = torch.float32, "cpu (float32)"

    def _forward(self, batch):
        """batch: list of (state_text, [Question]) -> list of per-request probability lists."""
        torch, jev = self._torch, self._jev
        out = []
        for i in range(0, len(batch), self._max_batch):
            chunk = batch[i : i + self._max_batch]
            b = jev.collator(chunk, jev.device)
            with torch.no_grad(), torch.autocast("cuda", dtype=self._dtype, enabled=self._amp):
                logits = jev.model(b["input_ids"], b["attention_mask"], b["opt_pos"],
                                   b["opt_mask"], b["q_pos"], b["seg"]).float()
            probs = (logits / jev.model.temperature).softmax(-1).cpu()
            for bi, (_, qs) in enumerate(chunk):
                out.append([probs[bi, qi, : len(q.options)].tolist() for qi, q in enumerate(qs)])
        return out

    def _error(self, detail):
        return {"status": 422, "body": json.dumps({"detail": detail}, ensure_ascii=False), "device": self._device}

    def predict(self, context, model_input, params=None):
        rows = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        results, pending = [None] * len(rows), []
        for i, row in enumerate(rows):
            raw = row.get("request")
            try:
                body = json.loads(raw) if isinstance(raw, str) else raw
            except json.JSONDecodeError as e:
                results[i] = self._error([{"loc": ["body", e.pos], "msg": "JSON decode error", "type": "json_invalid"}])
                continue
            try:
                state, items = self._A.parse_request(body, self._aliases, self._mode, self._noul_mode)
                questions = self._A.to_jev_questions(items)
                ids, *_ = self._jev.collator.encode_one(state, questions)  # raises if > 512 tokens
            except self._A.RequestValidationError as e:
                results[i] = self._error(e.detail)
                continue
            except ValueError as e:
                results[i] = self._error([{"loc": ["body"], "msg": str(e), "type": "value_error"}])
                continue
            pending.append((i, state, items, questions, len(ids)))

        if pending:
            probs = self._forward([(state, qs) for _, state, _, qs, _ in pending])
            for (i, _, items, _, n_tokens), p in zip(pending, probs):
                body = self._A.build_response(items, p, n_tokens, self._model_id)
                results[i] = {"status": 200, "body": json.dumps(body, ensure_ascii=False), "device": self._device}
        return pd.DataFrame(results)


mlflow.models.set_model(TypeSafeOpenJevModel())
