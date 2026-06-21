#!/usr/bin/env python3
"""验证集工具调用评测：只评估该轮 tool name + arguments，忽略口语和最终回复。

Gold 提取：每个 user turn 到下一个 user 之间，跳过 assistant 口头铺垫，
收集该轮所有 tool_call 作为标准答案。

用法：
  bash scripts/run.sh eval
  bash scripts/run.sh eval --max-samples 100
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INFER_DIR = os.path.join(SCRIPT_DIR, "..", "infer")
sys.path.insert(0, INFER_DIR)

from chat_agent import LocalEngine, _resolve_latest_adapter  # noqa: E402


@dataclass
class ToolCall:
    name: str
    arguments: Any


@dataclass
class EvalPoint:
    sample_idx: int
    turn_idx: int
    prefix: list[dict[str, str]]
    tools: list[dict[str, Any]]
    gold_calls: list[ToolCall]


@dataclass
class Metrics:
    name_tp: int = 0
    name_fp: int = 0
    name_fn: int = 0
    name_tn: int = 0
    args_tp: int = 0
    args_fp: int = 0
    args_fn: int = 0
    args_tn: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)

    def _prf1(self, tp: int, fp: int, fn: int) -> dict[str, float]:
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}

    def to_dict(self) -> dict[str, Any]:
        should_call = self.name_tp + self.name_fn
        should_not_call = self.name_fp + self.name_tn
        return {
            "tool_name": {
                **self._prf1(self.name_tp, self.name_fp, self.name_fn),
                "tp": self.name_tp,
                "fp": self.name_fp,
                "fn": self.name_fn,
                "tn": self.name_tn,
                "false_call_rate": round(self.name_fp / should_not_call, 4) if should_not_call else 0.0,
            },
            "tool_name_and_args": {
                **self._prf1(self.args_tp, self.args_fp, self.args_fn),
                "tp": self.args_tp,
                "fp": self.args_fp,
                "fn": self.args_fn,
                "tn": self.args_tn,
                "false_call_rate": round(self.args_fp / should_not_call, 4) if should_not_call else 0.0,
            },
            "should_call_turns": should_call,
            "should_not_call_turns": should_not_call,
            "total_turns": should_call + should_not_call,
        }


KEY_ALIASES: dict[str, str] = {
    "original_price": "price",
    "amount": "price",
    "discounted_price": "price",
    "discount": "discount_percent",
    "discount_percentage": "discount_percent",
    "birthdate": "birth_date",
    "date_of_birth": "birth_date",
    "currentdate": "current_date",
    "source_currency": "from_currency",
    "base_currency": "from_currency",
    "target_currency": "to_currency",
    "destination_currency": "to_currency",
    "client_name": "customer_name",
    "name": "customer_name",
    "contact_name": "customer_name",
    "event_title": "title",
    "event_name": "title",
    "search_query": "query",
    "keyword": "query",
    "keywords": "query",
    "location1": "origin",
    "location2": "destination",
    "point1": "origin",
    "point2": "destination",
    "start_point": "origin",
    "end_point": "destination",
    "start_location": "origin",
    "end_location": "destination",
    "source": "origin",
    "event_date": "date",
    "start_date": "date",
    "end_date": "date",
}


def _parse_tools(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _parse_tool_call_content(content: str) -> ToolCall:
    payload = json.loads(content)
    args = payload.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    return ToolCall(name=payload["name"], arguments=args)


def _extract_turn_block(messages: list[dict[str, str]], user_idx: int) -> list[dict[str, str]]:
    block: list[dict[str, str]] = []
    i = user_idx + 1
    while i < len(messages) and messages[i]["role"] != "user":
        block.append(messages[i])
        i += 1
    return block


def _gold_calls_from_block(block: list[dict[str, str]]) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for msg in block:
        if msg["role"] == "tool_call":
            calls.append(_parse_tool_call_content(msg["content"]))
    return calls


def extract_eval_points(sample: dict[str, Any], sample_idx: int) -> list[EvalPoint]:
    messages = sample["messages"]
    tools = _parse_tools(sample.get("tools"))
    points: list[EvalPoint] = []

    for i, msg in enumerate(messages):
        if msg["role"] != "user":
            continue
        block = _extract_turn_block(messages, i)
        points.append(
            EvalPoint(
                sample_idx=sample_idx,
                turn_idx=i,
                prefix=[dict(m) for m in messages[: i + 1]],
                tools=tools,
                gold_calls=_gold_calls_from_block(block),
            )
        )
    return points


def _canonical_key(key: str) -> str:
    k = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower().replace("-", "_")
    return KEY_ALIASES.get(k, k)


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, str):
        s = value.strip()
        if re.fullmatch(r"-?\d+\.?\d*", s):
            return float(s) if "." in s else float(int(s))
        return s.lower()
    return value


def _normalize_args(args: Any) -> Any:
    if isinstance(args, list):
        return {"_list": [_normalize_scalar(v) for v in args]}
    if not isinstance(args, dict):
        return {"_value": _normalize_scalar(args)}
    out: dict[str, Any] = {}
    for key, value in args.items():
        ckey = _canonical_key(str(key))
        if isinstance(value, dict):
            out[ckey] = _normalize_args(value)
        elif isinstance(value, list):
            out[ckey] = [_normalize_scalar(v) for v in value]
        else:
            out[ckey] = _normalize_scalar(value)
    return out


def _values_equal(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return False
        return all(_values_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_values_equal(x, y) for x, y in zip(a, b))
    return a == b


def args_match(gold_args: Any, pred_args: Any) -> bool:
    gold = _normalize_args(gold_args)
    pred = _normalize_args(pred_args)

    if "_list" in gold:
        pred_list = pred.get("_list")
        if pred_list is None and len(pred) == 1:
            only = next(iter(pred.values()))
            if isinstance(only, list):
                pred_list = only
        return _values_equal(gold["_list"], pred_list)

    for key, gval in gold.items():
        if key in pred:
            if not _values_equal(gval, pred[key]):
                return False
            continue
        pred_values = list(pred.values())
        if not any(_values_equal(gval, pv) for pv in pred_values):
            return False
    return True


def calls_match_name(gold: list[ToolCall], pred: list[ToolCall]) -> bool:
    if len(gold) != len(pred):
        return False
    return all(g.name == p.name for g, p in zip(gold, pred))


def calls_match_name_and_args(gold: list[ToolCall], pred: list[ToolCall]) -> bool:
    if len(gold) != len(pred):
        return False
    return all(g.name == p.name and args_match(g.arguments, p.arguments) for g, p in zip(gold, pred))


def _predict_calls(engine: LocalEngine, point: EvalPoint) -> list[ToolCall]:
    response = engine.generate(point.prefix, point.tools)
    parsed = engine.parse_tool_calls(response)
    calls: list[ToolCall] = []
    for call in parsed:
        args = call.arguments
        if isinstance(args, str):
            args = json.loads(args)
        calls.append(ToolCall(name=call.name, arguments=args))
    return calls


def _serialize_calls(calls: list[ToolCall]) -> list[dict[str, Any]]:
    return [{"name": c.name, "arguments": c.arguments} for c in calls]


def evaluate(engine: LocalEngine, points: list[EvalPoint], error_limit: int = 50) -> Metrics:
    from tqdm import tqdm

    metrics = Metrics()
    for point in tqdm(points, desc="eval tool call"):
        user_text = point.prefix[-1]["content"][:160]
        gold = point.gold_calls
        should_call = bool(gold)

        try:
            pred = _predict_calls(engine, point)
        except Exception as exc:  # noqa: BLE001
            if len(metrics.errors) < error_limit:
                metrics.errors.append(
                    {
                        "sample_idx": point.sample_idx,
                        "turn_idx": point.turn_idx,
                        "error": str(exc),
                        "user": user_text,
                        "gold": _serialize_calls(gold),
                    }
                )
            if should_call:
                metrics.name_fn += 1
                metrics.args_fn += 1
            else:
                metrics.name_tn += 1
                metrics.args_tn += 1
            continue

        name_ok = calls_match_name(gold, pred)
        args_ok = calls_match_name_and_args(gold, pred)

        if should_call:
            if name_ok:
                metrics.name_tp += 1
            else:
                metrics.name_fn += 1
            if args_ok:
                metrics.args_tp += 1
            else:
                metrics.args_fn += 1
        else:
            if pred:
                metrics.name_fp += 1
                metrics.args_fp += 1
            else:
                metrics.name_tn += 1
                metrics.args_tn += 1

        if len(metrics.errors) < error_limit and not args_ok:
            kind = "wrong_args" if (should_call and name_ok) else (
                "missed_call" if (should_call and not pred) else (
                    "wrong_tool" if (should_call and pred) else "false_call"
                )
            )
            metrics.errors.append(
                {
                    "sample_idx": point.sample_idx,
                    "turn_idx": point.turn_idx,
                    "kind": kind,
                    "gold": _serialize_calls(gold),
                    "pred": _serialize_calls(pred),
                    "user": user_text,
                }
            )

    return metrics


def _repo_paths() -> tuple[str, str, str]:
    repo_root = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
    model = os.environ.get("MODEL_PATH", os.path.join(repo_root, "models", "Qwen3-4B"))
    train_out = os.environ.get("TRAIN_OUT_DIR", os.path.join(repo_root, "outputs", "tool-use-lora"))
    return repo_root, model, train_out


def _resolve_latest_val_dataset(train_out: str) -> str:
    import glob

    paths = sorted(
        glob.glob(os.path.join(train_out, "v*/val_dataset.jsonl")),
        key=os.path.getmtime,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"no val_dataset.jsonl under {train_out}/v*/")
    return paths[0]


def main() -> None:
    repo_root, default_model, train_out = _repo_paths()
    default_val = os.environ.get("VAL_FILE")
    if not default_val:
        try:
            default_val = _resolve_latest_val_dataset(train_out)
        except FileNotFoundError:
            default_val = os.path.join(train_out, "v0-*/val_dataset.jsonl")

    parser = argparse.ArgumentParser(description="Evaluate tool name + args on validation set")
    parser.add_argument("--val-file", default=default_val)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output",
        default=os.path.join(repo_root, "outputs", "eval", "tool_selection_metrics.json"),
    )
    parser.add_argument("--error-limit", type=int, default=50)
    args = parser.parse_args()

    adapter = args.adapter or _resolve_latest_adapter(train_out)
    print(f"[INFO] model={args.model}")
    print(f"[INFO] adapter={adapter}")
    print(f"[INFO] val_file={args.val_file}")

    points: list[EvalPoint] = []
    with open(args.val_file, encoding="utf-8") as f:
        for sample_idx, line in enumerate(f):
            if args.max_samples is not None and sample_idx >= args.max_samples:
                break
            points.extend(extract_eval_points(json.loads(line), sample_idx))

    should_call = sum(1 for p in points if p.gold_calls)
    should_not_call = len(points) - should_call
    print(f"[INFO] eval points: {len(points)} (should_call={should_call}, should_not_call={should_not_call})")

    engine = LocalEngine(args.model, adapter, "hermes")
    metrics = evaluate(engine, points, error_limit=args.error_limit)

    result = {
        "adapter": adapter,
        "val_file": args.val_file,
        "max_samples": args.max_samples,
        "metrics": metrics.to_dict(),
        "errors_sample": metrics.errors,
    }

    print("\n=== Tool Name Metrics ===")
    for k, v in metrics.to_dict()["tool_name"].items():
        print(f"{k}: {v}")
    print("\n=== Tool Name + Args Metrics ===")
    for k, v in metrics.to_dict()["tool_name_and_args"].items():
        print(f"{k}: {v}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] saved to {args.output}")


if __name__ == "__main__":
    main()
