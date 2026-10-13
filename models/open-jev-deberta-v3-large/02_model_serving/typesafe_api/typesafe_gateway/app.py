"""TypeSafe AI API compatible gateway for an open-jev Model Serving endpoint — Databricks App (FastAPI).

Exposes the TypeSafe System One routes so the official `typesafe-sdk` (or any TypeSafe client) works by
changing only `base_url` and `api_key`:

  POST /v1/systemone  -> forwards the body to the serving endpoint and returns the TypeSafe response
  GET  /v1/models     -> lists the accepted model names (MODEL_ALIASES)
  GET  /              -> liveness / which endpoint this gateway fronts

Auth: callers send `Authorization: Bearer <Databricks OAuth token>` (what the SDK does with `api_key`);
Databricks Apps validates it before the request reaches this code. The app then calls the endpoint as
its own service principal (`serving-endpoint` resource with CAN_QUERY in app.yaml).

Errors follow the TypeSafe API: 422 {"detail": [...]} for invalid requests (produced by the model's
adapter), 429 when the endpoint rate-limits, 503 when it is unavailable (the SDK retries 429/5xx).
"""

import json
import logging
import os
import uuid

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError, TooManyRequests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

ENDPOINT = os.environ["SERVING_ENDPOINT"]
MODEL_ALIASES = [m.strip() for m in os.environ.get("MODEL_ALIASES", "jev-latest,open-jev-deberta-v3-large").split(",") if m.strip()]
MODEL_DESCRIPTION = os.environ.get(
    "MODEL_DESCRIPTION", "open-jev-deberta-v3-large (DeBERTa-v3-large typed-decision model) on Databricks Model Serving")
MODEL_RELEASE_DATE = os.environ.get("MODEL_RELEASE_DATE", "2026-09-24")

log = logging.getLogger("typesafe_gateway")
app = FastAPI(title="open-jev TypeSafe-compatible gateway", version="1.0")
w = WorkspaceClient()  # app service principal (auto-injected credentials)


def _reply(status: int, body, request_id: str) -> JSONResponse:
    return JSONResponse(body, status_code=status, headers={"x-typesafe-request-id": request_id})


@app.get("/")
def root():  # /healthz is answered by the Databricks Apps platform itself
    return {"status": "ok", "endpoint": ENDPOINT}


@app.get("/v1/models")
def models():
    return {"models": [{"name": m, "description": MODEL_DESCRIPTION, "release_date": MODEL_RELEASE_DATE}
                       for m in MODEL_ALIASES]}


@app.post("/v1/systemone")
async def systemone(request: Request):
    request_id = str(uuid.uuid4())
    raw = await request.body()
    try:
        body = json.loads(raw)
    except json.JSONDecodeError as e:
        return _reply(422, {"detail": [{"loc": ["body", e.pos], "msg": "JSON decode error", "type": "json_invalid"}]}, request_id)

    try:
        # One TypeSafe request = one row; the model returns {"status", "body"} per row.
        resp = w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT}/invocations",
                               body={"dataframe_records": [{"request": json.dumps(body, ensure_ascii=False)}]})
        row = resp["predictions"][0]
        return _reply(int(row["status"]), json.loads(row["body"]), request_id)
    except TooManyRequests as e:
        return _reply(429, {"detail": f"Serving endpoint is rate limiting: {e}"}, request_id)
    except DatabricksError as e:
        log.warning("serving call failed (%s): %s", request_id, e)
        return _reply(503, {"detail": f"Serving endpoint unavailable: {e}"}, request_id)
