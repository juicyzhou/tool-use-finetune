#!/usr/bin/env python3
"""Convert Glaive Function Calling v2 dataset to ms-swift Agent format.

ms-swift Agent format (JSONL):
{
  "tools": "[{\"type\": \"function\", \"function\": {...}}]",  # optional
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "tool_call", "content": "{\"name\": \"...\", \"arguments\": {...}}"},
    {"role": "tool_response", "content": "{...}"},
    ...
  ]
}

Reference: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Agent-support.md
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROLE_PATTERN = re.compile(r"(USER:|ASSISTANT:|FUNCTION RESPONSE:)")
def _extract_functioncall_payload(region: str) -> str:
    """Extract the JSON-like payload after <functioncall>."""
    region = region.strip()
    brace_start = region.find("{")
    if brace_start < 0:
        raise ValueError(f"No payload found in: {region[:80]}")

    args_key = re.search(r'"arguments"\s*:\s*', region[brace_start:])
    if not args_key:
        end = region.rfind("}")
        return region[brace_start : end + 1]

    args_value_start = brace_start + args_key.end()
    value = region[args_value_start:].lstrip()
    if value.startswith("\\'"):
        close = value.find("\\'", 2)
        if close < 0:
            raise ValueError("Unterminated functioncall payload")
        payload_end = args_value_start + close + 2
        if region[payload_end:].strip().startswith("}"):
            payload_end = region.find("}", payload_end) + 1
        return region[brace_start:payload_end]
    if value.startswith("'"):
        i = 1
        while i < len(value):
            if value[i] == "\\":
                i += 2
                continue
            if value[i] == "'":
                payload_end = args_value_start + i + 1
                if region[payload_end:].strip().startswith("}"):
                    payload_end = region.find("}", payload_end) + 1
                return region[brace_start:payload_end]
            i += 1
        raise ValueError("Unterminated functioncall payload")
    if value.startswith("{"):
        depth = 0
        for i, ch in enumerate(value):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    payload_end = args_value_start + i + 1
                    return region[brace_start:payload_end]
    end = region.rfind("}")
    return region[brace_start : end + 1]


def _extract_functioncall_blocks(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, payload) for each <functioncall> block."""
    blocks: list[tuple[int, int, str]] = []
    search_from = 0
    while True:
        start = text.find("<functioncall>", search_from)
        if start < 0:
            break
        payload_start = start + len("<functioncall>")
        end_marker = text.find("<|endoftext|>", payload_start)
        region_end = end_marker if end_marker >= 0 else len(text)
        region = text[payload_start:region_end]
        payload = _extract_functioncall_payload(region)
        payload_abs_end = payload_start + region.find(payload) + len(payload)
        block_end = end_marker if end_marker >= 0 else payload_abs_end
        blocks.append((start, block_end, payload))
        search_from = block_end
    return blocks
ENDOFTEXT_PATTERN = re.compile(r"<\|endoftext\|>")
TOOL_JSON_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _strip_endoftext(text: str) -> str:
    return ENDOFTEXT_PATTERN.sub("", text).strip()


def _parse_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise ValueError(f"Unsupported arguments type: {type(raw)}")


def _parse_quoted_argument_string(value: str) -> str:
    """Extract JSON object string from quoted argument value."""
    value = value.strip()
    if value.startswith("\\'"):
        close = value.find("\\'", 2)
        if close < 0:
            raise ValueError("Unterminated escaped single-quoted string")
        return value[2:close]
    if value.startswith("'"):
        i = 1
        while i < len(value):
            if value[i] == "\\":
                i += 2
                continue
            if value[i] == "'":
                return value[1:i]
            i += 1
        raise ValueError("Unterminated single-quoted string")
    raise ValueError(f"Unsupported quoted argument prefix: {value[:20]}")


def _parse_functioncall_payload(payload: str) -> dict[str, Any]:
    """Parse Glaive functioncall JSON, tolerating single-quoted arguments."""
    payload = payload.strip()

    # Glaive often uses: {"name": "fn", "arguments": '{...}'} or \' {...} \'
    name_match = re.search(r'"name"\s*:\s*"([^"]+)"', payload)
    if not name_match:
        raise ValueError(f"Missing function name in payload: {payload[:120]}")
    name = name_match.group(1)

    args_match = re.search(r'"arguments"\s*:\s*', payload)
    if not args_match:
        return {"name": name, "arguments": {}}

    rest = payload[args_match.end() :].strip()
    if rest == "{}":
        arguments = {}
    elif rest.startswith("\\'") or rest.startswith("'"):
        args_str = _parse_quoted_argument_string(rest)
        arguments = json.loads(args_str)
    elif rest.startswith("{"):
        depth = 0
        end = 0
        for i, ch in enumerate(rest):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        arguments = json.loads(rest[:end])
    else:
        data = json.loads(payload)
        arguments = _parse_arguments(data.get("arguments", {}))

    return {"name": name, "arguments": arguments}


def extract_tools_from_system(system: str) -> list[dict[str, Any]]:
    """Extract tool definitions from Glaive system prompt."""
    if "no access to external functions" in system:
        return []

    marker = "Use them if required -"
    if marker in system:
        tools_text = system.split(marker, 1)[1]
    else:
        tools_text = system

    tools: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for idx, ch in enumerate(tools_text):
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                chunk = tools_text[start : idx + 1]
                try:
                    tool_def = json.loads(chunk)
                except json.JSONDecodeError:
                    start = -1
                    continue
                if "name" in tool_def:
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool_def["name"],
                                "description": tool_def.get("description", ""),
                                "parameters": tool_def.get("parameters", {"type": "object", "properties": {}}),
                            },
                        }
                    )
                start = -1
    return tools


def parse_chat(chat: str) -> list[dict[str, str]]:
    """Parse Glaive chat string into ms-swift messages."""
    chat = chat.strip()
    if not chat:
        return []

    parts = ROLE_PATTERN.split(chat)
    messages: list[dict[str, str]] = []
    idx = 1
    while idx < len(parts):
        role_tag = parts[idx].strip()
        content = parts[idx + 1] if idx + 1 < len(parts) else ""
        content = _strip_endoftext(content)

        if role_tag == "USER:":
            if content:
                messages.append({"role": "user", "content": content})
        elif role_tag == "ASSISTANT:":
            remaining = content
            cursor = 0
            blocks = _extract_functioncall_blocks(remaining)
            if not blocks:
                text = _strip_endoftext(remaining)
                if text:
                    messages.append({"role": "assistant", "content": text})
            else:
                for block_start, block_end, payload in blocks:
                    prefix = remaining[cursor:block_start].strip()
                    if prefix:
                        messages.append({"role": "assistant", "content": prefix})
                    call = _parse_functioncall_payload(payload)
                    messages.append(
                        {
                            "role": "tool_call",
                            "content": json.dumps(call, ensure_ascii=False),
                        }
                    )
                    cursor = block_end
                tail = remaining[cursor:].strip()
                if tail:
                    messages.append({"role": "assistant", "content": _strip_endoftext(tail)})
        elif role_tag == "FUNCTION RESPONSE:":
            response_text = content.strip()
            if response_text:
                # ms-swift expects JSON string content; keep raw JSON if possible.
                try:
                    json.loads(response_text)
                    messages.append({"role": "tool_response", "content": response_text})
                except json.JSONDecodeError:
                    messages.append(
                        {
                            "role": "tool_response",
                            "content": json.dumps({"result": response_text}, ensure_ascii=False),
                        }
                    )
        idx += 2

    return messages


def convert_sample(sample: dict[str, Any], skip_truncated: bool = True) -> dict[str, Any] | None:
    system = sample.get("system", "")
    chat = sample.get("chat", "")

    if skip_truncated and chat.strip() and not chat.rstrip().endswith("<|endoftext|>"):
        return None

    messages = parse_chat(chat)
    if not messages:
        return None

    result: dict[str, Any] = {"messages": messages}
    tools = extract_tools_from_system(system)
    if tools:
        result["tools"] = json.dumps(tools, ensure_ascii=False)

    return result


def convert_file(
    input_path: Path,
    output_path: Path,
    skip_truncated: bool = True,
    max_samples: int | None = None,
) -> dict[str, int]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    stats = {
        "total": len(data),
        "converted": 0,
        "skipped_truncated": 0,
        "skipped_empty": 0,
        "errors": 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        limit = max_samples if max_samples is not None else len(data)
        for sample in data[:limit]:
            try:
                converted = convert_sample(sample, skip_truncated=skip_truncated)
            except Exception:
                stats["errors"] += 1
                continue

            if converted is None:
                if skip_truncated and sample.get("chat", "").strip() and not sample["chat"].rstrip().endswith(
                    "<|endoftext|>"
                ):
                    stats["skipped_truncated"] += 1
                else:
                    stats["skipped_empty"] += 1
                continue

            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["converted"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Glaive FC v2 to ms-swift Agent JSONL")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/glaive-function-calling-v2/glaive-function-calling-v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/glaive-function-calling-v2-ms-swift/glaive-ms-swift.jsonl"),
    )
    parser.add_argument("--keep-truncated", action="store_true", help="Keep truncated samples")
    parser.add_argument("--max-samples", type=int, default=None, help="Only convert first N samples")
    args = parser.parse_args()

    stats = convert_file(
        input_path=args.input,
        output_path=args.output,
        skip_truncated=not args.keep_truncated,
        max_samples=args.max_samples,
    )

    print(json.dumps(stats, indent=2))
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
