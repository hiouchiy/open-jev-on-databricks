"""Call the open-jev gateway with the official TypeSafe Python SDK — only base_url and api_key change.

    pip install typesafe-sdk
    python sdk_smoke_test.py --profile <profile> --app-url https://<app-name>-<id>.aws.databricksapps.com

api_key is a Databricks OAuth token (`databricks auth token --profile <profile>`), which Databricks Apps
accepts as `Authorization: Bearer ...` — exactly the header the SDK sends.
"""

import argparse
import json
import statistics
import subprocess
import time

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient
from typesafe_sdk import TypeSafeUnprocessableEntityError


def databricks_token(profile: str) -> str:
    return json.loads(subprocess.check_output(["databricks", "auth", "token", "--profile", profile]))["access_token"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="DEFAULT")
    ap.add_argument("--app-url", required=True)
    ap.add_argument("--bench", type=int, default=20, help="timed requests after the first one")
    args = ap.parse_args()

    # Generous timeout: a scaled-to-zero endpoint needs a few minutes for the first request.
    with TypeSafeClient(api_key=databricks_token(args.profile), base_url=args.app_url.rstrip("/"), timeout=600) as client:
        print("models:", [m.name for m in client.models.list().models])

        questions = {
            "requests_refund": Noul(instructions="Is the customer requesting a refund?"),
            "department": Choice(instructions="Which team should handle this?",
                                 criteria={"billing": "Charges, payments, invoices, and refunds",
                                           "technical": "Bugs, outages, and integrations",
                                           "shipping": "Delivery, tracking, and lost packages"}),
            "frustration": Score(instructions="How frustrated is the customer?",
                                 criteria=["Calm and neutral", "Frustrated but civil", "Very angry or threatening to leave"]),
        }
        state = "I was charged twice. Please refund the duplicate charge."
        t0 = time.perf_counter()
        r = client.system_one(state=state, questions=questions)
        print(f"first call {time.perf_counter() - t0:.1f}s  model={r.model}  usage={r.usage}")
        print("  requests_refund.noul   =", round(r.nouls["requests_refund"].noul, 3))
        print("  department.choice      =", r.choices["department"].choice,
              {k: round(v, 3) for k, v in r.choices["department"].probabilities.items()})
        print("  frustration.score      =", round(r.scores["frustration"].score, 2), dict(r.scores["frustration"].legend))

        # Structured state (object) works too.
        r2 = client.system_one(state={"subject": "Order 1042", "body": "My parcel never arrived."},
                               questions={"department": questions["department"]})
        print("  object state ->", r2.choices["department"].choice)

        # Validation errors surface as the SDK's 422 exception.
        try:
            client.system_one(state=state, questions=questions, model="unknown-model")
        except TypeSafeUnprocessableEntityError as e:
            print("  422 as expected:", e)

        if args.bench:
            lat = []
            for _ in range(args.bench):
                t0 = time.perf_counter()
                client.system_one(state=state, questions=questions)
                lat.append((time.perf_counter() - t0) * 1000)
            print(f"{args.bench} SDK calls (3 questions): p50 {statistics.median(lat):.0f} ms, max {max(lat):.0f} ms")


if __name__ == "__main__":
    main()
