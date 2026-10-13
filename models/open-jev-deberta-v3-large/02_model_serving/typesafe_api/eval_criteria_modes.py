"""Which way of turning TypeSafe `criteria` into open-jev option texts is most accurate? — AI Runtime CLI.

open-jev was trained on short option labels ("fees & charges"), while TypeSafe-style requests carry a
choice *name* plus a natural-language *description* ({"fees_and_charges": "Fees or unexpected amounts
taken from the account"}). This script rewrites the model authors' in-domain test questions
(banking77 product area / SST-5 polarity, level and yes-no questions) into TypeSafe form with
hand-written descriptions, sends them through `typesafe_adapter` exactly as the serving pyfunc does,
and compares accuracy per variant:

  choice  label             original training labels (reference: what the model was trained on)
          name              snake_case names only                    (choice_text_mode="name")
          description       descriptions only                        (choice_text_mode="description")
          name_description  "name: description"                      (choice_text_mode="name_description")
  score   label             original level labels
          description       level descriptions
  noul    plain             no criteria (built-in "no"/"yes" options)
          criteria          criteria.true / criteria.false as the two option texts (noul_criteria_mode="options")

Measured (2026-10, in-domain test split, A10G): choice label 0.866 / name 0.855 / name_description
0.807 / description 0.745; noul plain 0.935 / criteria 0.914; score label 0.590 / description 0.525.
→ adapter defaults: choice_text_mode="name", noul_criteria_mode="fallback".

Run (from this directory):
    COPYFILE_DISABLE=1 air run --file eval_criteria_modes.yaml --watch --profile <your-profile>
"""

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # AI Runtime presets 1 without shipping hf_transfer

import json
import sys
import time
import urllib.request
from collections import defaultdict

import typesafe_adapter as A

HF_REPO = "com-kotobalabs/open-jev-deberta-v3-large"
HF_REVISION = "188ee67a5c93122b916e5acd5bdb0cb3623e380a"
DATA_URL = ("https://raw.githubusercontent.com/kotoba-lang/typed-decisions/"
            "e65215db7e554a31f557936f0bd1d8a0a78b6af5/data-fam/test.jsonl")
MAX_STATES = int(os.environ.get("MAX_STATES", "0"))
BATCH = int(os.environ.get("BATCH_SIZE", "64"))

# Hand-written TypeSafe-style criteria, aligned with the original option order of each test question.
PRODUCT_AREA = {  # banking77 "Which product area is the message about?"
    "fees_and_charges": "Fees, charges, or unexpected amounts taken from the account",
    "pin_and_security": "PIN codes, passcodes, blocked or compromised cards, and account security",
    "refund_and_dispute": "Refunds, disputed or unrecognised payments, and reverted transactions",
    "top_up": "Topping up or adding money to the account",
    "exchange_and_fiat": "Currency exchange, exchange rates, and foreign or crypto currencies",
    "atm_and_cash": "ATM withdrawals, cash, and cash machines",
    "transfer": "Sending, receiving, or tracking bank transfers",
    "card": "Ordering, receiving, activating, or using a physical or virtual card",
    "account_and_identity": "Account details, identity verification, personal data, or closing the account",
    "other": "Anything that fits none of the other areas",
}
POLARITY = {  # SST-5 "What is the overall polarity of the sentence?"
    "negative": "The sentence is critical or unfavourable",
    "neutral": "The sentence is neither clearly favourable nor unfavourable",
    "positive": "The sentence is favourable or praising",
}
LEVELS = [  # SST-5 "How positive is the sentiment of this review sentence?"
    "Very negative: harshly critical",
    "Negative: somewhat critical",
    "Neutral: mixed, or no clear opinion",
    "Positive: somewhat favourable",
    "Very positive: enthusiastic praise",
]
NOUL_CRITERIA = {
    "The customer is asking about a card (physical or virtual).":
        {"true": "The message is about a physical or virtual card", "false": "The message is not about a card"},
    "The sentence expresses a positive opinion.":
        {"true": "The sentence expresses a positive opinion", "false": "The sentence does not express a positive opinion"},
}
CHOICE_SETS = {"Which product area is the message about?": PRODUCT_AREA,
               "What is the overall polarity of the sentence?": POLARITY}


def variants(q):
    """Yield (family, variant, TypeSafe question, choice_text_mode) for one original test question."""
    instr, opts = q["instructions"], q["options"]
    if q["kind"] == "choice" and instr in CHOICE_SETS:
        crit = CHOICE_SETS[instr]
        yield "choice", "label", {"type": "choice", "instructions": instr, "criteria": {o: None for o in opts}}, "name"
        for mode in A.CHOICE_TEXT_MODES:
            yield "choice", mode, {"type": "choice", "instructions": instr, "criteria": crit}, mode
    elif q["kind"] == "score" and len(opts) == len(LEVELS):
        yield "score", "label", {"type": "score", "instructions": instr, "criteria": opts}, "name"
        yield "score", "description", {"type": "score", "instructions": instr, "criteria": LEVELS}, "name"
    elif q["kind"] == "noul" and instr in NOUL_CRITERIA:
        yield "noul", "plain", {"type": "noul", "instructions": instr}, "name"
        yield "noul", "criteria", {"type": "noul", "instructions": instr, "criteria": NOUL_CRITERIA[instr]}, "name"


def main():
    import torch
    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(HF_REPO, revision=HF_REVISION)
    sys.path.insert(0, snapshot)
    from typed_decisions.open_jev import OpenJev

    jev = OpenJev.from_pretrained(snapshot)
    amp = jev.device.type == "cuda"
    lines = urllib.request.urlopen(DATA_URL, timeout=120).read().decode().splitlines()
    examples = [json.loads(l) for l in lines if l.strip()]
    examples = [e for e in examples if e["source"] in ("banking77", "sst5")]
    if MAX_STATES:
        examples = examples[:MAX_STATES]

    # One TypeSafe request per (example, choice_text_mode); each carries the variants that use that mode.
    jobs = []  # (state, mode, [(family, variant, gold, question)])
    for e in examples:
        by_mode = defaultdict(list)
        for q in e["questions"]:
            for fam, var, tq, mode in variants(q):
                gold = q["gold"]
                by_mode[mode].append((fam, var, gold, tq))
        for mode, qs in by_mode.items():
            jobs.append((e["state"], mode, qs))

    correct, total = defaultdict(int), defaultdict(int)
    t0 = time.perf_counter()
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i : i + BATCH]
        parsed = []
        for state, mode, qs in chunk:
            body = {"model": "jev-latest", "state": state,
                    "questions": {f"{fam}|{var}|{j}": tq for j, (fam, var, _, tq) in enumerate(qs)}}
            # noul "criteria" variant = always use the descriptions (noul_criteria_mode="options")
            _, items = A.parse_request(body, choice_text_mode=mode, noul_criteria_mode="options")
            parsed.append((state, items, qs))
        batch = jev.collator([(s, A.to_jev_questions(items)) for s, items, _ in parsed], jev.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits = jev.model(batch["input_ids"], batch["attention_mask"], batch["opt_pos"],
                               batch["opt_mask"], batch["q_pos"], batch["seg"]).float()
        probs = (logits / jev.model.temperature).softmax(-1).cpu()
        for bi, (_, items, qs) in enumerate(parsed):
            for qi, (it, (fam, var, gold, _)) in enumerate(zip(items, qs)):
                p = probs[bi, qi, : len(it.options)].tolist()
                pred = max(range(len(p)), key=p.__getitem__)
                correct[(fam, var)] += pred == gold
                total[(fam, var)] += 1
    print(f"scored {sum(total.values())} questions in {time.perf_counter() - t0:.1f}s")

    results = {f"{f}/{v}": {"accuracy": correct[(f, v)] / total[(f, v)], "n": total[(f, v)]} for f, v in sorted(total)}
    print(json.dumps(results, indent=1))
    try:
        import mlflow

        with mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID")):
            mlflow.log_metrics({k.replace("/", "_"): v["accuracy"] for k, v in results.items()})
    except Exception as e:  # MLflow is optional for this experiment
        print("MLflow logging skipped:", e)


if __name__ == "__main__":
    main()
