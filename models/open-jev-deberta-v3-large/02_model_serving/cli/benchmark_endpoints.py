"""Compare open-jev serving endpoints on different GPUs (T4 / A10G / RTX PRO 6000) — stdlib only.

For each endpoint it measures, against the same requests:
  1. quality    accuracy on the model authors' in-domain test split (gold labels) — catches precision
                regressions (e.g. fp16 on T4 vs bf16 on Ampere+)
  2. latency    sequential single-text requests (1 text x 4 questions): server-side and round-trip p50/p95
  3. throughput concurrent batched requests (BATCH texts per request, CONCURRENCY in flight)
and, given each GPU tier's DBU/hour and your $/DBU, the cost per hour and per 1M questions.

    python benchmark_endpoints.py --profile <profile> \
        --endpoint T4=open-jev-t4 --endpoint A10G=open-jev-a10 --endpoint RTX=open-jev-rtx \
        --dbu T4=10.48 --dbu A10G=20 --dbu RTX=80 --usd-per-dbu 0.09

DBU/hour: Model Serving pricing page (T4 10.48, A10G 20) or measured from system.billing.usage
(RTX PRO 6000 = 80, see the model README). $/DBU: system.billing.list_prices for your region's
SERVERLESS_REAL_TIME_INFERENCE SKU. Round-trip numbers include your network distance to the workspace.
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DATA_URL = ("https://raw.githubusercontent.com/kotoba-lang/typed-decisions/"
            "e65215db7e554a31f557936f0bd1d8a0a78b6af5/data-fam/test.jsonl")
LATENCY_QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this message?",
     "options": ["billing", "fraud & security", "card delivery", "technical support", "other"]},
    {"type": "score", "instructions": "How positive is the sentiment of this message?",
     "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
    {"type": "noul", "instructions": "The customer is asking for a refund."},
    {"type": "noul", "instructions": "The message needs an urgent response."},
]
LATENCY_TEXT = "I was charged twice for the same order and nobody answers my emails. I want my money back now."


def auth(profile):
    host = json.loads(subprocess.check_output(
        ["databricks", "auth", "describe", "--profile", profile, "-o", "json"]))["details"]["host"]
    token = json.loads(subprocess.check_output(["databricks", "auth", "token", "--profile", profile]))["access_token"]
    return host.rstrip("/"), token


def invoke(host, token, endpoint, records, timeout=900):
    req = urllib.request.Request(
        f"{host}/serving-endpoints/{endpoint}/invocations",
        data=json.dumps({"dataframe_records": records}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["predictions"]


def to_request(example):
    """Authors' test record -> serving row + gold indices."""
    qs = [{"type": q["kind"], "instructions": q["instructions"],
           **({} if q["kind"] == "noul" else {"options": q["options"]})} for q in example["questions"]]
    return {"state": example["state"], "questions": json.dumps(qs)}, example["questions"]


def predicted_index(q, a):
    if q["kind"] == "noul":
        return int(a["noul"] >= 0.5)  # options are ("no", "yes")
    p = a["probabilities"]
    return q["options"].index(max(p, key=p.get))


def pct(xs, p):
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]


def bench(name, endpoint, host, token, examples, args):
    out = {"name": name, "endpoint": endpoint}
    invoke(host, token, endpoint, [{"state": LATENCY_TEXT, "questions": json.dumps(LATENCY_QUESTIONS)}])  # warm up
    rows = [to_request(e) for e in examples]

    # 1. quality
    correct = n = 0
    for i in range(0, len(rows), args.batch):
        chunk = rows[i : i + args.batch]
        preds = invoke(host, token, endpoint, [r for r, _ in chunk])
        out["device"] = preds[0].get("device", "?")
        for (_, golds), p in zip(chunk, preds):
            for q, a in zip(golds, json.loads(p["answers"])):
                correct += predicted_index(q, a) == q["gold"]
                n += 1
    out["accuracy"], out["accuracy_n"] = correct / n, n

    # 2. latency: sequential single-text requests
    srv, rt = [], []
    one = [{"state": LATENCY_TEXT, "questions": json.dumps(LATENCY_QUESTIONS)}]
    for _ in range(args.latency_requests):
        t0 = time.perf_counter()
        p = invoke(host, token, endpoint, one)[0]
        rt.append((time.perf_counter() - t0) * 1000)
        srv.append(p["latency_ms"])
    out.update(server_p50_ms=statistics.median(srv), server_p95_ms=pct(srv, .95),
               rtt_p50_ms=statistics.median(rt), rtt_p95_ms=pct(rt, .95))

    # 3. throughput: CONCURRENCY threads, each sending BATCH-text requests
    batches = [[r for r, _ in rows[(i * args.batch) % len(rows):][: args.batch]] for i in range(args.throughput_requests)]
    nq = sum(len(json.loads(r["questions"])) for b in batches for r in b)
    srv_batch = []

    def send(b):
        p = invoke(host, token, endpoint, b)
        srv_batch.append(p[0]["latency_ms"])

    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        list(pool.map(send, batches))
    dt = time.perf_counter() - t0
    out.update(throughput_qps=nq / dt, throughput_texts_ps=sum(len(b) for b in batches) / dt,
               batch_server_p50_ms=statistics.median(srv_batch))

    dbu = args.dbu.get(name)
    if dbu:
        usd_h = dbu * args.usd_per_dbu
        out.update(dbu_per_hour=dbu, usd_per_hour=usd_h,
                   usd_per_1m_questions=usd_h / (out["throughput_qps"] * 3600) * 1e6)
    return out


def main():
    kv = lambda s: tuple(s.split("=", 1))
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--endpoint", type=kv, action="append", required=True, help="LABEL=endpoint-name")
    ap.add_argument("--dbu", type=kv, action="append", default=[], help="LABEL=DBU per hour")
    ap.add_argument("--usd-per-dbu", type=float, default=0.07)
    ap.add_argument("--quality-texts", type=int, default=300)
    ap.add_argument("--latency-requests", type=int, default=100)
    ap.add_argument("--throughput-requests", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32, help="texts per request (quality + throughput)")
    ap.add_argument("--concurrency", type=int, default=4, help="Small workload size = up to 4 concurrent")
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()
    args.dbu = {k: float(v) for k, v in args.dbu}

    host, token = auth(args.profile)
    lines = urllib.request.urlopen(DATA_URL, timeout=120).read().decode().splitlines()
    examples = [json.loads(l) for l in lines if l.strip()][: args.quality_texts]

    results = []
    for name, ep in args.endpoint:
        print(f"== {name} ({ep})", flush=True)
        r = bench(name, ep, host, token, examples, args)
        print(json.dumps(r, indent=1), flush=True)
        results.append(r)
    json.dump(results, open(args.out, "w"), indent=1)

    cols = [("GPU", "name", "{}"), ("device", "device", "{}"), ("acc", "accuracy", "{:.3f}"),
            ("srv p50 ms", "server_p50_ms", "{:.1f}"), ("srv p95 ms", "server_p95_ms", "{:.1f}"),
            ("rtt p50 ms", "rtt_p50_ms", "{:.0f}"), ("q/s", "throughput_qps", "{:.0f}"),
            ("$/h", "usd_per_hour", "{:.2f}"), ("$/1M q", "usd_per_1m_questions", "{:.2f}")]
    print("\n| " + " | ".join(c[0] for c in cols) + " |\n|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(f.format(r[k]) if k in r else "-" for _, k, f in cols) + " |")


if __name__ == "__main__":
    main()
