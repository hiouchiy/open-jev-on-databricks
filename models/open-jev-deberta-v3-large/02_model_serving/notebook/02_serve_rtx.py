# Databricks notebook source
# MAGIC %md
# MAGIC # open-jev (DeBERTa-v3-large) — Model Serving 2/2: serve on RTX PRO 6000 (`GPU_LARGE_RTX`)
# MAGIC
# MAGIC Creates (or updates) a Model Serving endpoint for the version registered by
# MAGIC **`01_register`**, on **`GPU_LARGE_RTX`** = 1x NVIDIA **RTX PRO 6000 Blackwell (96 GB)**,
# MAGIC then sends typed-decision requests to it.
# MAGIC
# MAGIC | Model Serving `workload_type` | GPU | Memory |
# MAGIC |---|---|---|
# MAGIC | `GPU_SMALL` | 1x T4 | 16 GB |
# MAGIC | `GPU_MEDIUM` | 1x A10G | 24 GB |
# MAGIC | **`GPU_LARGE_RTX`** | **1x RTX PRO 6000** | **96 GB** |
# MAGIC | `GPU_XLARGE` | 1x H100 | 80 GB (enrollment, us-west-2) |
# MAGIC
# MAGIC `GPU_LARGE_RTX` is available in `us-west-2`, `us-east-1`, `us-east-2`, `ap-northeast-1` (Tokyo)
# MAGIC and `ap-northeast-2`; see the [docs](https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms).
# MAGIC
# MAGIC **Compute:** any compute works (serverless CPU is fine) — this notebook only makes control-plane
# MAGIC calls. First deployment takes ~15-30 min (the packed environment + 1.7 GB of weights are staged).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuration

# COMMAND ----------

import json
import time

dbutils.widgets.text("uc_catalog", "main", "UC catalog")
dbutils.widgets.text("uc_schema", "air_samples", "UC schema")
dbutils.widgets.text("model_name", "open_jev_deberta_v3_large", "UC model name")
dbutils.widgets.text("model_version", "", "Model version (empty = latest)")
dbutils.widgets.text("endpoint_name", "open-jev-rtx", "Serving endpoint name")

UC_MODEL = f"{dbutils.widgets.get('uc_catalog')}.{dbutils.widgets.get('uc_schema')}.{dbutils.widgets.get('model_name')}"
ENDPOINT = dbutils.widgets.get("endpoint_name")
WORKLOAD_TYPE = "GPU_LARGE_RTX"

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
VERSION = dbutils.widgets.get("model_version") or str(
    max(int(v.version) for v in client.search_model_versions(f"name='{UC_MODEL}'"))
)
print(f"Serving {UC_MODEL} v{VERSION} on {WORKLOAD_TYPE} as endpoint '{ENDPOINT}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create or update the endpoint
# MAGIC The MLflow Deployments client takes the REST-shaped config as a plain dict, which stays stable
# MAGIC across SDK versions. `scale_to_zero_enabled=True` keeps idle cost at zero (the first request
# MAGIC after idle waits for a cold start).

# COMMAND ----------

from mlflow.deployments import get_deploy_client

deploy = get_deploy_client("databricks")
config = {
    "served_entities": [{
        "name": f"{UC_MODEL.split('.')[-1]}-{VERSION}",
        "entity_name": UC_MODEL,
        "entity_version": VERSION,
        "workload_type": WORKLOAD_TYPE,
        "workload_size": "Small",        # 1 replica, up to 4 concurrent requests
        "scale_to_zero_enabled": True,
    }]
}

try:
    deploy.get_endpoint(ENDPOINT)
    print("Endpoint exists -> updating its config")
    deploy.update_endpoint(endpoint=ENDPOINT, config=config)
except Exception as e:
    if "does not exist" not in str(e).lower() and "not found" not in str(e).lower():
        raise
    print("Creating endpoint")
    deploy.create_endpoint(name=ENDPOINT, config=config)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Wait until the endpoint is READY

# COMMAND ----------

deadline = time.time() + 60 * 60
while True:
    ep = deploy.get_endpoint(ENDPOINT)
    state = ep["state"]
    print(time.strftime("%H:%M:%S"), state)
    if state.get("ready") == "READY" and state.get("config_update") == "NOT_UPDATING":
        break
    if state.get("config_update") == "UPDATE_FAILED":
        raise RuntimeError(f"Deployment failed — open Serving > {ENDPOINT} > Logs. State: {state}")
    if time.time() > deadline:
        raise TimeoutError("endpoint not ready after 60 min")
    time.sleep(60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Query: one text, several typed questions, one forward pass
# MAGIC Each row is a *state* (any text) plus a JSON list of typed questions. The same endpoint answers
# MAGIC any question set — there is no fixed label space, so new routing/triage/QA questions need no retraining.

# COMMAND ----------

QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this message?",
     "options": ["billing", "fraud & security", "card delivery", "technical support", "other"]},
    {"type": "score", "instructions": "How positive is the sentiment of this message?",
     "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
    {"type": "noul", "instructions": "The customer is asking for a refund."},
    {"type": "noul", "instructions": "The message needs an urgent response."},
]
texts = [
    "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    "Someone used my card in another country an hour ago. I did not make that payment!",
    "My new card arrived today and activation took two seconds. Great app!",
]
payload = {"dataframe_records": [{"state": t, "questions": json.dumps(QUESTIONS)} for t in texts]}

t0 = time.perf_counter()
resp = deploy.predict(endpoint=ENDPOINT, inputs=payload)
print(f"round trip {1000 * (time.perf_counter() - t0):.0f} ms")

for text, row in zip(texts, resp["predictions"]):
    print("\n" + text)
    if row["error"]:
        print("  ERROR:", row["error"])
        continue
    for q, a in zip(QUESTIONS, json.loads(row["answers"])):
        if q["type"] == "noul":
            print(f"  [noul  ] {q['instructions']:<45} p(yes)={a['noul']:.3f}")
        elif q["type"] == "score":
            print(f"  [score ] {q['instructions']:<45} level={a['score']:.2f} (conf {a['confidence']:.2f})")
        else:
            print(f"  [choice] {q['instructions']:<45} {a['choice']} (conf {a['confidence']:.2f})")
    print(f"  server latency {row['latency_ms']} ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Query from outside Databricks (curl)
# MAGIC ```bash
# MAGIC curl -s -X POST "https://<workspace-host>/serving-endpoints/open-jev-rtx/invocations" \
# MAGIC   -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \
# MAGIC   -d '{"dataframe_records":[{"state":"I lost my card yesterday.",
# MAGIC        "questions":"[{\"type\":\"noul\",\"instructions\":\"The customer lost a card.\"}]"}]}'
# MAGIC ```
# MAGIC The endpoint scales to zero when idle; delete it when you no longer need it:
# MAGIC `databricks serving-endpoints delete open-jev-rtx`.
