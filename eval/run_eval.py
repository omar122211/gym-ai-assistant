"""Evaluate the agent against eval/questions.jsonl with the real model and write eval/results.md.

    python -m eval.run_eval              # all cases (~5 minutes, throttled for the free tier)
    python -m eval.run_eval --only mem-  # cases whose id starts with "mem-"

Each case runs in its own fresh conversation on a throw-away database, so results are repeatable.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from app.agent import GymAgent
from app.config import get_settings
from app.gym_db import GymDB
from app.knowledge import KnowledgeBase, load_chunks
from app.providers import AIProviderError, build_provider
from app.rag import RAGService

HERE = Path(__file__).resolve().parent
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def check(case: dict, r) -> dict[str, bool]:
    """Every expectation in the case becomes one named check."""
    e, reply = case["expect"], r.reply.translate(ARABIC_DIGITS)
    action = (r.action or {}).get("type")
    checks = {"language": r.language == case["lang"]}
    if "intent" in e:
        allowed = e["intent"] if isinstance(e["intent"], list) else [e["intent"]]
        checks["intent"] = r.intent in allowed
    if "grounded" in e:
        checks["grounded"] = r.grounded == e["grounded"]
    if "source" in e:
        checks["retrieval"] = any(s["file"] == e["source"] for s in r.sources)
    if "contains_any" in e:
        checks["answer"] = any(word.lower() in reply.lower() for word in e["contains_any"])
    if "action" in e:
        checks["action"] = action == e["action"]
    if "no_action" in e:
        checks["safe_action"] = action not in e["no_action"]
    if "not_contains" in e:
        checks["no_leak"] = not any(bad.lower() in reply.lower() for bad in e["not_contains"])
    return checks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="run only cases whose id starts with this")
    ap.add_argument("--pause", type=float, default=4.0, help="seconds between turns (free-tier rate limits)")
    args = ap.parse_args()

    cases = [json.loads(line) for line in (HERE / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = [c for c in cases if c["id"].startswith(args.only)]

    settings = get_settings()
    provider = build_provider(settings)
    kb = KnowledgeBase(settings.chroma_dir, settings.collection)
    kb.build(load_chunks(settings.knowledge_dir), provider)
    db = GymDB(Path(tempfile.mkdtemp()) / "eval.db")
    agent = GymAgent(RAGService(kb, provider, settings.top_k, settings.min_similarity), db, provider, InMemorySaver())

    rows = []
    for n, case in enumerate(cases, 1):
        cid = f"eval-{case['id']}-{int(time.time())}"
        try:
            for turn in case["turns"]:
                r = agent.chat(turn, cid)
                time.sleep(args.pause)
            checks = check(case, r)
            row = {"id": case["id"], "category": case["category"], "lang": case["lang"], "checks": checks,
                   "passed": all(checks.values()), "intent": r.intent, "action": (r.action or {}).get("type"),
                   "latency_ms": r.latency_ms, "reply": r.reply}
        except AIProviderError as err:
            row = {"id": case["id"], "category": case["category"], "lang": case["lang"], "checks": {"api": False},
                   "passed": False, "intent": None, "action": None, "latency_ms": None, "reply": f"API error: {err}"}
        rows.append(row)
        print(f"[{n}/{len(cases)}] {'PASS' if row['passed'] else 'FAIL'} {case['id']}  {row['checks']}")

    (HERE / "results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "results.md").write_text(report(rows, settings), encoding="utf-8")
    print(f"\npassed {sum(r['passed'] for r in rows)}/{len(rows)} — see eval/results.md")


def pct(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "–"


def report(rows: list[dict], settings) -> str:
    out = [f"# Evaluation results\n",
           f"Run on {date.today().isoformat()} · chat model `{settings.chat_model}` · embeddings `{settings.embed_model}` · "
           f"similarity cutoff {settings.min_similarity}\n",
           f"**{pct(sum(r['passed'] for r in rows), len(rows))} cases passed.** "
           "A case passes only if every check on it passes.\n",
           "## By check\n", "| Check | Passed | What it measures |", "|---|---|---|"]
    meaning = {
        "intent": "the router picked the right path (question / schedule / booking / complaint / greeting / out of scope)",
        "grounded": "answered from the knowledge base when it should — and admitted not knowing when it should",
        "retrieval": "the expected document was among the cited sources",
        "answer": "the reply contains the expected fact (price, trainer name, day…)",
        "action": "the right tool result (booked, ticket, schedule…)",
        "language": "replied in the user's language",
        "safe_action": "prompt injection did not trigger a booking or ticket",
        "no_leak": "no system prompt or injected false fact in the reply",
        "api": "the model API answered",
    }
    by_check: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        for name, ok in r["checks"].items():
            by_check[name].append(ok)
    for name in meaning:
        if name in by_check:
            out.append(f"| {name} | {pct(sum(by_check[name]), len(by_check[name]))} | {meaning[name]} |")

    out += ["\n## By category and language\n", "| | en | ar | tr | all |", "|---|---|---|---|---|"]
    for cat in ["answerable", "unanswerable", "routing", "safety", "memory"]:
        cells = []
        for lang in ["en", "ar", "tr", None]:
            sel = [r for r in rows if r["category"] == cat and (lang is None or r["lang"] == lang)]
            cells.append(pct(sum(r["passed"] for r in sel), len(sel)))
        out.append(f"| {cat} | " + " | ".join(cells) + " |")

    lat = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    if lat:
        p95 = lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))]
        out.append(f"\nLatency per final turn: median {statistics.median(lat):.0f} ms · p95 {p95} ms "
                   "(free tier, includes 2 model calls + 1 embedding for questions).\n")

    failed = [r for r in rows if not r["passed"]]
    out.append("## Failures\n")
    if not failed:
        out.append("None.\n")
    for r in failed:
        bad = ", ".join(k for k, v in r["checks"].items() if not v)
        out.append(f"- **{r['id']}** ({r['lang']}) — failed: {bad} · intent `{r['intent']}`, action `{r['action']}`  \n"
                   f"  > {r['reply'][:220].replace(chr(10), ' ')}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    main()
