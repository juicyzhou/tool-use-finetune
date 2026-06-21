#!/usr/bin/env python3
"""统计 ms-swift 训练数据的 Token 长度分布（Hermes 物理 Prompt）。

用法：
  python3 scripts/stats_token_length.py
  python3 scripts/stats_token_length.py --max-samples 5000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


DEFAULT_DATASET = os.path.join(_repo_root(), "data/glaive-function-calling-v2-ms-swift/glaive-ms-swift.jsonl")
DEFAULT_MODEL = os.path.join(_repo_root(), "models/Qwen3-4B")


@dataclass
class RunningStats:
    values: list[int] = field(default_factory=list)

    def add(self, v: int) -> None:
        self.values.append(v)

    def summary(self) -> dict[str, float | int]:
        if not self.values:
            return {}
        xs = sorted(self.values)
        n = len(xs)

        def pct(p: float) -> float:
            if n == 1:
                return float(xs[0])
            idx = min(n - 1, int(round((n - 1) * p)))
            return float(xs[idx])

        return {
            "count": n,
            "min": xs[0],
            "max": xs[-1],
            "mean": round(statistics.mean(xs), 1),
            "p50": pct(0.50),
            "p90": pct(0.90),
            "p95": pct(0.95),
            "p99": pct(0.99),
        }

    def over_rate(self, threshold: int) -> float:
        if not self.values:
            return 0.0
        return round(sum(v > threshold for v in self.values) / len(self.values), 4)


def _parse_tools(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _percentile(xs: list[int], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    if n == 1:
        return float(xs[0])
    idx = min(n - 1, int(round((n - 1) * p)))
    return float(xs[idx])


def _build_template(model: str, agent_template: str, max_length: int):
    from swift.arguments import SftArguments
    from swift.pipelines.utils import prepare_model_template

    args = SftArguments(
        model=model,
        agent_template=agent_template,
        dataset=[DEFAULT_DATASET],
        max_length=max_length,
    )
    _, template = prepare_model_template(args)
    template.set_mode("train")
    tokenizer = template.tokenizer
    return template, tokenizer


def _encode_sample(template, sample: dict[str, Any]) -> dict[str, Any]:
    from swift.infer_engine import InferRequest

    tools = _parse_tools(sample.get("tools"))
    req = InferRequest(messages=sample["messages"], tools=tools)
    try:
        enc = template.encode(req)
    except Exception:
        # 超长样本：临时放宽 max_length 再编码，用于统计真实长度
        old = template.max_length
        template.max_length = 131072
        try:
            enc = template.encode(req)
        finally:
            template.max_length = old
    labels = enc.get("labels") or []
    total = len(enc["input_ids"])
    label = sum(1 for x in labels if x != -100)
    prompt = total - label
    return {"total": total, "label": label, "prompt": prompt}


def _segment_token_lens(tokenizer, messages: list[dict[str, str]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"assistant": [], "tool_call": [], "user": [], "tool_response": []}
    for msg in messages:
        role = msg["role"]
        if role not in out:
            continue
        out[role].append(len(tokenizer.encode(msg["content"], add_special_tokens=False)))
    return out


def analyze(
    dataset: str,
    model: str,
    agent_template: str,
    max_length: int,
    max_samples: int | None,
) -> dict[str, Any]:
    template, tokenizer = _build_template(model, agent_template, max_length)

    seq_total = RunningStats()
    seq_prompt = RunningStats()
    seq_label = RunningStats()
    turns = RunningStats()
    tool_count = RunningStats()
    role_stats = {k: RunningStats() for k in ("assistant", "tool_call", "user", "tool_response")}
    single_output = RunningStats()  # max per-sample single assistant/tool_call segment

    thresholds = [2048, 4096, 8192, 16384, 32768]
    trunc = {t: 0 for t in thresholds}
    n = 0

    from tqdm import tqdm

    with open(dataset, encoding="utf-8") as f:
        lines = f if max_samples is None else None
        if max_samples is not None:
            iterator = (json.loads(line) for i, line in enumerate(f) if i < max_samples)
        else:
            iterator = (json.loads(line) for line in f)

        for sample in tqdm(iterator, desc="token stats", total=max_samples):

            enc = _encode_sample(template, sample)
            seq_total.add(enc["total"])
            seq_prompt.add(enc["prompt"])
            seq_label.add(enc["label"])
            for t in thresholds:
                if enc["total"] > t:
                    trunc[t] += 1

            msgs = sample["messages"]
            turns.add(sum(1 for m in msgs if m["role"] == "user"))
            tools = _parse_tools(sample.get("tools")) or []
            tool_count.add(len(tools))

            seg = _segment_token_lens(tokenizer, msgs)
            sample_single_max = 0
            for role, lens in seg.items():
                for v in lens:
                    role_stats[role].add(v)
                    if role in ("assistant", "tool_call"):
                        sample_single_max = max(sample_single_max, v)
            # Hermes XML wrapper overhead for tool_call (~15 tok); assistant prefix (~5 tok)
            if sample_single_max:
                single_output.add(sample_single_max + (20 if any(m["role"] == "tool_call" for m in msgs) else 5))
            n += 1

    def pack(rs: RunningStats) -> dict[str, Any]:
        s = rs.summary()
        if not s:
            return {}
        return {
            **{k: (int(v) if k != "mean" else v) for k, v in s.items()},
            "over_rate": {str(t): rs.over_rate(t) for t in thresholds},
        }

    # Recommended settings
    total_p99 = int(seq_total.summary().get("p99", 0))
    total_max = int(seq_total.summary().get("max", 0))
    single_p99 = int(single_output.summary().get("p99", 0))
    single_max = int(single_output.summary().get("max", 0))

    rec_ctx = 4096 if total_p99 <= 3800 else (8192 if total_p99 <= 7800 else 16384)
    rec_out = 512 if single_p99 <= 480 else (1024 if single_p99 <= 980 else 2048)

    return {
        "dataset": dataset,
        "model": model,
        "agent_template": agent_template,
        "train_max_length": max_length,
        "samples": n,
        "sequence": {
            "total_tokens": pack(seq_total),
            "prompt_tokens": pack(seq_prompt),
            "label_tokens": pack(seq_label),
        },
        "single_turn_output": {
            "assistant_or_tool_call_content_plus_overhead": pack(single_output),
            "assistant_content": pack(role_stats["assistant"]),
            "tool_call_content": pack(role_stats["tool_call"]),
        },
        "role_content_tokens": {k: pack(v) for k, v in role_stats.items()},
        "dialogue": {
            "user_turns_per_sample": pack(turns),
            "tools_per_sample": pack(tool_count),
        },
        "truncation_rate_if_max_length": {
            str(t): round(trunc[t] / n, 4) if n else 0.0 for t in thresholds
        },
        "recommendation": {
            "max_length_context": rec_ctx,
            "max_new_tokens_inference": rec_out,
            "rationale": {
                "max_length_context": f"p99 全序列={total_p99}, max={total_max}；建议覆盖 p99 并留 ~5% 余量",
                "max_new_tokens_inference": f"p99 单轮输出={single_p99}, max={single_max}；含 Hermes XML 开销估算",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Token length stats for ms-swift training data")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--agent-template", default="hermes")
    parser.add_argument("--max-length", type=int, default=131072, help="Template max_length for stats (use large value to avoid truncation)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"[INFO] dataset={args.dataset}")
    print(f"[INFO] model={args.model}, agent_template={args.agent_template}")
    result = analyze(args.dataset, args.model, args.agent_template, args.max_length, args.max_samples)

    print("\n=== Sequence Token Length (Hermes physical prompt) ===")
    for section, key in [("Total", "total_tokens"), ("Prompt (context)", "prompt_tokens"), ("Label (supervised)", "label_tokens")]:
        s = result["sequence"][key]
        print(f"\n{section}: count={s['count']} min={s['min']} p50={s['p50']} p90={s['p90']} p95={s['p95']} p99={s['p99']} max={s['max']} mean={s['mean']}")

    print("\n=== Single-turn Output (assistant / tool_call) ===")
    s = result["single_turn_output"]["assistant_or_tool_call_content_plus_overhead"]
    print(f"content+overhead: p50={s['p50']} p95={s['p95']} p99={s['p99']} max={s['max']}")

    print("\n=== Truncation rate if max_length = ===")
    for k, v in result["truncation_rate_if_max_length"].items():
        print(f"  {k}: {v*100:.2f}%")

    print("\n=== Recommendation ===")
    rec = result["recommendation"]
    print(f"  max_length (context): {rec['max_length_context']}")
    print(f"  max_new_tokens (infer): {rec['max_new_tokens_inference']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] saved to {args.output}")


if __name__ == "__main__":
    main()
