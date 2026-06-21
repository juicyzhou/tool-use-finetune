#!/usr/bin/env python3
"""批量跑 eval_cases.json 中的交互评测案例。

用法：
  python3 scripts/eval/run_cases.py
  python3 scripts/eval/run_cases.py --case multi_turn_followup_city
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.join(SCRIPT_DIR, "..", "infer")
sys.path.insert(0, INFER_DIR)

from chat_agent import DEMO_TOOLS, LocalEngine, run_turn  # noqa: E402


def _check_mentions(text: str, keywords: list[str]) -> list[str]:
    missing = []
    lower = text.lower()
    for kw in keywords:
        if kw.lower() not in lower:
            missing.append(kw)
    return missing


def run_case(engine, case: dict) -> dict:
    history: list[dict[str, str]] = []
    turn_results = []

    for i, turn in enumerate(case["turns"], start=1):
        if turn["role"] != "user":
            continue
        history.append({"role": "user", "content": turn["content"]})
        answer = run_turn(engine, list(history), DEMO_TOOLS, max_turns=8)
        history.append({"role": "assistant", "content": answer})

        exp_key = f"turn_{i}"
        exp = case["expected"].get(exp_key, case["expected"] if i == len(case["turns"]) else {})
        missing = _check_mentions(answer, exp.get("final_answer_should_mention", []))
        has_tool_call = "<tool_call>" in answer or bool(re.search(r'"name"\s*:', answer))

        turn_results.append(
            {
                "turn": i,
                "query": turn["content"],
                "answer": answer,
                "missing_keywords": missing,
                "has_tool_call_in_final": has_tool_call,
            }
        )

    passed = all(not tr["missing_keywords"] for tr in turn_results)
    return {"id": case["id"], "category": case["category"], "passed": passed, "turns": turn_results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=os.path.join(SCRIPT_DIR, "cases.json"))
    parser.add_argument("--case", default=None, help="只跑指定 case id")
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args()

    with open(args.cases, encoding="utf-8") as f:
        data = json.load(f)
    cases = data["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            raise SystemExit(f"case not found: {args.case}")

    from chat_agent import _default_paths, _resolve_latest_adapter

    default_model, train_out = _default_paths()
    adapter = args.adapter or _resolve_latest_adapter(train_out)
    engine = LocalEngine(default_model, adapter, "hermes")

    print(f"Running {len(cases)} case(s), adapter={adapter}\n")
    summary = []
    for case in cases:
        result = run_case(engine, case)
        summary.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']} ({result['category']})")
        for tr in result["turns"]:
            print(f"  Turn {tr['turn']}: {tr['query'][:60]}...")
            if tr["missing_keywords"]:
                print(f"    missing: {tr['missing_keywords']}")
            print(f"    answer: {tr['answer'][:200]}{'...' if len(tr['answer'])>200 else ''}")
        print()

    passed = sum(1 for r in summary if r["passed"])
    print(f"Summary: {passed}/{len(summary)} passed")


if __name__ == "__main__":
    main()
