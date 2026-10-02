"""Query the open-jev RTX serving endpoint from your laptop — standard library only.

    python query_endpoint.py --profile <your-profile> [--endpoint open-jev-rtx] [--bench 50]

Auth comes from the Databricks CLI (`databricks auth token`), so no token handling is needed.
A long HTTP timeout survives a scale-from-zero cold start (the CLI's `serving-endpoints query`
gives up after 60 s).
"""

import argparse
import json
import statistics
import subprocess
import time
import urllib.request

QUESTIONS = [
    {"type": "choice", "instructions": "Which team should handle this message?",
     "options": ["billing", "fraud & security", "card delivery", "technical support", "other"]},
    {"type": "score", "instructions": "How positive is the sentiment of this message?",
     "options": ["very negative", "negative", "neutral", "positive", "very positive"]},
    {"type": "noul", "instructions": "The customer is asking for a refund."},
    {"type": "noul", "instructions": "The message needs an urgent response."},
]
TEXTS = [
    "I was charged twice for the same order and nobody answers my emails. I want my money back now.",
    "Someone used my card in another country an hour ago. I did not make that payment!",
    "My new card arrived today and activation took two seconds. Great app!",
]


def auth(profile: str):
    host = json.loads(subprocess.check_output(
        ["databricks", "auth", "describe", "--profile", profile, "-o", "json"]))["details"]["host"]
    token = json.loads(subprocess.check_output(
        ["databricks", "auth", "token", "--profile", profile]))["access_token"]
    return host.rstrip("/"), token


def invoke(host, token, endpoint, records, timeout=900):
    req = urllib.request.Request(
        f"{host}/serving-endpoints/{endpoint}/invocations",
        data=json.dumps({"dataframe_records": records}).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["predictions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--endpoint", default="open-jev-rtx")
    ap.add_argument("--bench", type=int, default=0, help="extra timed single-text requests")
    args = ap.parse_args()
    host, token = auth(args.profile)

    records = [{"state": t, "questions": json.dumps(QUESTIONS)} for t in TEXTS]
    t0 = time.perf_counter()
    preds = invoke(host, token, args.endpoint, records)
    print(f"first request: {time.perf_counter() - t0:.1f}s (includes cold start if scaled to zero)\n")

    for text, row in zip(TEXTS, preds):
        print(text)
        if row["error"]:
            print("  ERROR:", row["error"])
            continue
        for q, a in zip(QUESTIONS, json.loads(row["answers"])):
            if q["type"] == "noul":
                print(f"  [noul  ] {q['instructions']:<42} p(yes)={a['noul']:.3f}")
            elif q["type"] == "score":
                print(f"  [score ] {q['instructions']:<42} level={a['score']:.2f} (conf {a['confidence']:.2f})")
            else:
                print(f"  [choice] {q['instructions']:<42} {a['choice']} (conf {a['confidence']:.2f})")
        print(f"  server-side {row['latency_ms']} ms\n")

    if args.bench:
        rt, srv = [], []
        for i in range(args.bench):
            t0 = time.perf_counter()
            row = invoke(host, token, args.endpoint, [records[i % len(records)]])[0]
            rt.append((time.perf_counter() - t0) * 1000)
            srv.append(row["latency_ms"])
        q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))]
        print(f"{args.bench} requests x {len(QUESTIONS)} questions — "
              f"server p50 {statistics.median(srv):.1f} ms / p95 {q(srv, .95):.1f} ms; "
              f"round trip p50 {statistics.median(rt):.0f} ms / p95 {q(rt, .95):.0f} ms")


if __name__ == "__main__":
    main()
