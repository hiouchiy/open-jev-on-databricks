"""TypeSafe System One API  <->  open-jev adapter.

Translates a TypeSafe `POST /v1/systemone` request body into open-jev typed questions, and the model's
probability distributions back into a TypeSafe response body. Pure Python (no torch / MLflow import),
so the serving pyfunc, the evaluation script and the gateway tests all share exactly this code.

Spec: https://docs.typesafe.ai/api  (OpenAPI: https://api.typesafe.ai/openapi.json)

Request : {"model": str, "state": str|object|array,
           "questions": {name: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}}
Response: {"model": str, "answers": {name: answer}, "usage": {"input_tokens": int, "output_tokens": int}}
Errors  : HTTP 422 with {"detail": [{"loc": [...], "msg": str, "type": str}]}  (FastAPI style, as TypeSafe)

How TypeSafe concepts map onto open-jev (which scores a list of option *texts* per question):
  choice  criteria {name: description|None}  -> one option text per name (see CHOICE_TEXT_MODES);
                                                the answer reports the *name*
  score   criteria [level description, ...]  -> the level descriptions are the ordered options;
                                                legend/probabilities are keyed "0", "1", ...
  noul    criteria {"true": ..., "false": ...} (optional)
                                             -> open-jev's noul is a 2-option softmax over
                                                ("no", "yes"); the descriptions can replace those two
                                                option texts (see NOUL_CRITERIA_MODES), noul = p(true)
State / instructions / criteria given as objects or arrays are serialized to compact JSON text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

MODEL_ID = "open-jev-deberta-v3-large"
DEFAULT_ALIASES = ("jev-latest", MODEL_ID)

# How a choice criterion {name: description} becomes the option text the model reads.
#   name             "fees and charges"                      (description ignored)
#   description      "Fees or unexpected amounts taken"      (falls back to name when None)
#   name_description "fees and charges: Fees or unexpected amounts taken"
CHOICE_TEXT_MODES = ("name", "description", "name_description")
# Default = "name": measured best on the authors' in-domain test split (eval_criteria_modes.py) —
# open-jev was trained on short labels, and adding descriptions lowered choice accuracy
# (name 0.855 / name_description 0.807 / description 0.745; original labels 0.866).
DEFAULT_CHOICE_TEXT_MODE = "name"

# How noul criteria {"true": ..., "false": ...} are used.
#   fallback  only when the question has no instructions (default; plain noul measured 0.935 vs 0.914)
#   options   always use them as the two option texts
NOUL_CRITERIA_MODES = ("fallback", "options")
DEFAULT_NOUL_CRITERIA_MODE = "fallback"

MAX_CHOICE_OPTIONS = 255  # open-jev limit (collator option stride)
MAX_SCORE_LEVELS = 10     # open-jev trained range


class RequestValidationError(Exception):
    """Invalid request; `detail` is a FastAPI-style list of {loc, msg, type}."""

    def __init__(self, detail: list[dict]):
        super().__init__(detail[0]["msg"] if detail else "invalid request")
        self.detail = detail


@dataclass
class Item:
    """One TypeSafe question, lowered to what open-jev scores."""

    name: str
    type: str            # TypeSafe type: noul | choice | score
    kind: str            # open-jev kind actually used: noul | choice | score
    instructions: str
    options: list[str]   # option texts the model reads
    keys: list[str] = field(default_factory=list)  # choice: names, aligned with options
    legend: dict = field(default_factory=dict)      # score: {"0": original criterion, ...}


def _text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _humanize(name: str) -> str:
    return name.replace("_", " ").strip() or name


def _err(loc, msg, typ="value_error"):
    return {"loc": ["body", *loc], "msg": msg, "type": typ}


def parse_request(body, aliases=DEFAULT_ALIASES, choice_text_mode: str = DEFAULT_CHOICE_TEXT_MODE,
                  noul_criteria_mode: str = DEFAULT_NOUL_CRITERIA_MODE):
    """Validate a System One request body. Returns (state_text, [Item]); raises RequestValidationError."""
    if choice_text_mode not in CHOICE_TEXT_MODES:
        raise ValueError(f"choice_text_mode must be one of {CHOICE_TEXT_MODES}")
    if noul_criteria_mode not in NOUL_CRITERIA_MODES:
        raise ValueError(f"noul_criteria_mode must be one of {NOUL_CRITERIA_MODES}")
    errors: list[dict] = []
    if not isinstance(body, dict):
        raise RequestValidationError([{"loc": ["body"], "msg": "Input should be a valid object", "type": "dict_type"}])

    for key in ("state", "model", "questions"):
        if key not in body:
            errors.append(_err([key], "Field required", "missing"))
    if "model" in body and body["model"] not in aliases:
        errors.append(_err(["model"], f"Unknown model {body['model']!r}. Available: {', '.join(aliases)}"))
    state = body.get("state")
    if "state" in body and not isinstance(state, (str, dict, list)):
        errors.append(_err(["state"], "Input should be a string, object or array", "type_error"))
    questions = body.get("questions")
    if "questions" in body and (not isinstance(questions, dict) or not questions):
        errors.append(_err(["questions"], "Questions should be an object with at least 1 item", "too_short"))
        questions = {}

    items: list[Item] = []
    for name, q in (questions or {}).items():
        loc = ["questions", name]
        if not isinstance(q, dict) or q.get("type") not in ("noul", "choice", "score"):
            errors.append(_err([*loc, "type"], "Input tag should be one of 'noul', 'choice', 'score'", "union_tag_invalid"))
            continue
        qtype, instr, crit = q["type"], _text(q.get("instructions")), q.get("criteria")

        if qtype == "choice":
            if not isinstance(crit, dict) or not crit:
                errors.append(_err([*loc, "choice", "criteria"], "Choice criteria should be an object with at least 1 item", "too_short"))
                continue
            if len(crit) > MAX_CHOICE_OPTIONS:
                errors.append(_err([*loc, "choice", "criteria"], f"At most {MAX_CHOICE_OPTIONS} choices are supported", "too_long"))
                continue
            keys, options = [], []
            for k, desc in crit.items():
                d = _text(desc)
                if d is None or choice_text_mode == "name":
                    options.append(_humanize(k))
                elif choice_text_mode == "description":
                    options.append(d)
                else:
                    options.append(f"{_humanize(k)}: {d}")
                keys.append(k)
            items.append(Item(name, "choice", "choice", instr or "", options, keys=keys))

        elif qtype == "score":
            if not isinstance(crit, list) or not crit:
                errors.append(_err([*loc, "score", "criteria"], "Score criteria should be a list with at least 1 item", "too_short"))
                continue
            if len(crit) > MAX_SCORE_LEVELS:
                errors.append(_err([*loc, "score", "criteria"], f"At most {MAX_SCORE_LEVELS} score levels are supported", "too_long"))
                continue
            if any(c is None for c in crit):
                errors.append(_err([*loc, "score", "criteria"], "Score levels must not be null", "type_error"))
                continue
            options = [_text(c) for c in crit]
            # open-jev's score head needs >= 2 levels; a single level is a degenerate 1-option choice.
            kind = "score" if len(options) >= 2 else "choice"
            items.append(Item(name, "score", kind, instr or "", options, legend={str(i): c for i, c in enumerate(crit)}))

        else:  # noul
            if crit is not None and not isinstance(crit, dict):
                errors.append(_err([*loc, "noul", "criteria"], "Noul criteria should be an object", "dict_type"))
                continue
            t, f = (_text(crit.get("true")), _text(crit.get("false"))) if crit else (None, None)
            if instr is None and t is None and f is None:
                errors.append(_err([*loc, "noul"], "Noul question needs instructions or criteria", "value_error"))
                continue
            if (t is None and f is None) or (instr is not None and noul_criteria_mode == "fallback"):
                items.append(Item(name, "noul", "noul", instr, ["no", "yes"]))
            else:
                # Same 2-option softmax as a plain noul, with the descriptions as the option texts.
                items.append(Item(name, "noul", "choice", instr or t or "", [f or "no", t or "yes"]))

    if errors:
        raise RequestValidationError(errors)
    return (state if isinstance(state, str) else _text(state)), items


def to_jev_questions(items: list[Item]):
    """Build typed_decisions Question objects (import deferred: needs the model snapshot on sys.path)."""
    from typed_decisions.schema import Question

    return [Question(f"q{i}", it.kind, it.instructions, list(it.options), 0) for i, it in enumerate(items)]


def build_answers(items: list[Item], probs: list[list[float]]) -> dict:
    """probs[i] = probability distribution over items[i].options. Returns TypeSafe `answers`."""
    answers = {}
    for it, p in zip(items, probs):
        p = [float(x) for x in p]
        if it.type == "noul":
            answers[it.name] = {"type": "noul", "noul": p[1]}
        elif it.type == "choice":
            k = max(range(len(p)), key=p.__getitem__)
            answers[it.name] = {"type": "choice", "choice": it.keys[k], "confidence": p[k],
                                "probabilities": dict(zip(it.keys, p))}
        else:
            answers[it.name] = {"type": "score", "score": float(sum(i * x for i, x in enumerate(p))),
                                "confidence": max(p), "legend": it.legend,
                                "probabilities": {str(i): x for i, x in enumerate(p)}}
    return answers


def build_response(items: list[Item], probs: list[list[float]], input_tokens: int, model_id: str = MODEL_ID) -> dict:
    # Nothing is generated, so there are no output tokens.
    return {"model": model_id, "answers": build_answers(items, probs),
            "usage": {"input_tokens": int(input_tokens), "output_tokens": 0}}
