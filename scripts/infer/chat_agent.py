#!/usr/bin/env python3
"""交互式工具调用评测（直连 ms-swift 推理引擎，含完整 tool 循环）。

用法：
  python3 scripts/infer/chat_agent.py
  python3 scripts/infer/chat_agent.py --adapter outputs/tool-use-lora/v0-*/checkpoint-800
  python3 scripts/infer/chat_agent.py --api http://127.0.0.1:8000/v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Any

# ── 演示用工具（可按业务替换为真实 API）────────────────────────────────────────
DEMO_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Get exchange rate between two currencies",
            "parameters": {
                "type": "object",
                "properties": {
                    "base_currency": {"type": "string", "description": "Source currency code, e.g. USD"},
                    "target_currency": {"type": "string", "description": "Target currency code, e.g. CNY"},
                },
                "required": ["base_currency", "target_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a simple math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. (12 + 8) * 3"},
                },
                "required": ["expression"],
            },
        },
    },
]


def _mock_execute(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name == "get_weather":
        city = arguments.get("city", "Unknown")
        unit = arguments.get("unit", "celsius")
        temp = 25 if unit == "celsius" else 77
        return {"city": city, "temperature": temp, "unit": unit, "condition": "Sunny", "source": "mock"}
    if name == "get_exchange_rate":
        base = arguments.get("base_currency", "USD").upper()
        target = arguments.get("target_currency", "CNY").upper()
        rates = {("USD", "CNY"): 7.24, ("EUR", "CNY"): 7.85, ("USD", "EUR"): 0.92}
        rate = rates.get((base, target), 1.0)
        return {"base_currency": base, "target_currency": target, "rate": rate, "source": "mock"}
    if name == "calculate":
        expr = str(arguments.get("expression", "0"))
        allowed = set("0123456789+-*/(). ")
        if not set(expr) <= allowed:
            return {"error": "unsupported characters in expression"}
        try:
            return {"expression": expr, "result": eval(expr, {"__builtins__": {}}, {})}  # noqa: S307
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)}
    return {"error": f"unknown tool: {name}"}


def _resolve_latest_adapter(train_out: str) -> str:
    import glob

    paths = sorted(glob.glob(os.path.join(train_out, "v*/checkpoint-*")), key=os.path.getmtime, reverse=True)
    if not paths:
        raise FileNotFoundError(f"no checkpoint under {train_out}/v*/checkpoint-*")
    return paths[0]


def _default_paths() -> tuple[str, str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model = os.environ.get("MODEL_PATH", os.path.join(repo_root, "models", "Qwen3-4B"))
    train_out = os.environ.get("TRAIN_OUT_DIR", os.path.join(repo_root, "outputs", "tool-use-lora"))
    return model, train_out


def _parse_args() -> argparse.Namespace:
    default_model, train_out = _default_paths()
    parser = argparse.ArgumentParser(description="Interactive tool-use chat for fine-tuned Qwen3-4B LoRA")
    parser.add_argument("--model", default=default_model)
    adapter_default = os.environ.get("ADAPTER_PATH")
    if not adapter_default:
        try:
            adapter_default = _resolve_latest_adapter(train_out)
        except FileNotFoundError:
            adapter_default = None
    parser.add_argument("--adapter", default=adapter_default)
    parser.add_argument("--agent-template", default="hermes")
    parser.add_argument("--api", default=None, help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1")
    parser.add_argument("--model-name", default="qwen3-4b-glaive-tool-use")
    parser.add_argument("--max-turns", type=int, default=8, help="Max tool-call rounds per user query")
    parser.add_argument("--tools-file", default=None, help="JSON file with tools list; default uses built-in demo tools")
    return parser.parse_args()


def _load_tools(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return DEMO_TOOLS
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "tools" in data:
        tools = data["tools"]
        if isinstance(tools, str):
            return json.loads(tools)
        return tools
    return data


class LocalEngine:
    def __init__(self, model: str, adapter: str, agent_template: str) -> None:
        from swift.arguments import InferArguments
        from swift.infer_engine import RequestConfig, TransformersEngine
        from swift.pipelines.utils import prepare_model_template

        self.request_config = RequestConfig(max_tokens=2048, temperature=0.1)
        infer_args = InferArguments(
            model=model,
            adapters=[adapter],
            agent_template=agent_template,
            infer_backend="transformers",
            torch_dtype="bfloat16",
        )
        model_obj, template = prepare_model_template(infer_args)
        self.engine = TransformersEngine(model_obj, template=template)
        self.template = template

    def generate(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        from swift.infer_engine import InferRequest

        req = InferRequest(messages=messages, tools=tools)
        resp = self.engine.infer([req], self.request_config, use_tqdm=False)[0]
        return resp.choices[0].message.content

    def parse_tool_calls(self, text: str):
        return self.template.agent_template.get_toolcall(text)

    def get_processed_history(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> list[dict[str, str]]:
        from swift.template import StdTemplateInputs
        import copy
        # 模拟 ms-swift 推理前的预处理过程
        messages_copy = copy.deepcopy(messages)
        inputs = StdTemplateInputs(messages=messages_copy, tools=tools)
        self.template._preprocess_inputs(inputs)
        return inputs.messages

    def get_prompt(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        from swift.infer_engine import InferRequest
        req = InferRequest(messages=messages, tools=tools)
        # 编码并解码回字符串，展示物理 Prompt
        example = self.template.encode(req)
        return self.template.tokenizer.decode(example['input_ids'])


class ApiEngine:
    def __init__(self, base_url: str, model_name: str, agent_template: str) -> None:
        from openai import OpenAI
        from swift.agent_template import agent_template_map

        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        self.model_name = model_name
        self.agent_template = agent_template_map[agent_template]()

    def generate(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            temperature=0.1,
            max_tokens=2048,
        )
        return resp.choices[0].message.content or ""

    def parse_tool_calls(self, text: str):
        return self.agent_template.get_toolcall(text)


def run_turn(engine, messages: list[dict[str, str]], tools: list[dict[str, Any]], max_turns: int) -> str:
    for _ in range(max_turns):
        response = engine.generate(messages, tools)
        tool_calls = engine.parse_tool_calls(response)
        if not tool_calls:
            messages.append({"role": "assistant", "content": response})
            return response

        print(f"\n[assistant/tool_call]\n{response}\n")
        messages.append({"role": "assistant", "content": response})

        for call in tool_calls:
            args = call.arguments if isinstance(call.arguments, dict) else json.loads(call.arguments)
            result = _mock_execute(call.name, args)
            print(f"[tool] {call.name}({json.dumps(args, ensure_ascii=False)})")
            print(f"[tool_response] {json.dumps(result, ensure_ascii=False)}")
            messages.append({"role": "tool_response", "content": json.dumps(result, ensure_ascii=False)})

    return "[ERROR] exceeded max tool-call turns"


def main() -> None:
    args = _parse_args()
    tools = _load_tools(args.tools_file)

    if args.api:
        engine = ApiEngine(args.api, args.model_name, args.agent_template)
        backend = f"API {args.api}"
    else:
        engine = LocalEngine(args.model, args.adapter, args.agent_template)
        backend = f"local LoRA {args.adapter}"

    print("=" * 60)
    print("Qwen3-4B Glaive Tool-Use 交互评测")
    print(f"Backend : {backend}")
    print(f"Tools   : {[t['function']['name'] for t in tools]}")
    print("Commands: exit | quit | clear | tools | history | encode")
    print("=" * 60)

    history: list[dict[str, str]] = []
    while True:
        try:
            query = input("\n[user] ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        cmd = query.lower()
        if cmd in {"exit", "quit"}:
            break
        if cmd == "clear":
            history = []
            print("[INFO] history cleared")
            continue
        if cmd == "tools":
            print(json.dumps(tools, ensure_ascii=False, indent=2))
            continue
        if cmd == "history":
            print(json.dumps(history, ensure_ascii=False, indent=2))
            continue
        if cmd == "encode":
            if hasattr(engine, "get_processed_history"):
                processed = engine.get_processed_history(history, tools)
                print("\n[Processed History (Pre-processed by Template)]")
                print(json.dumps(processed, ensure_ascii=False, indent=2))
                
                print("\n" + "="*40)
                print("[Final Physical Prompt]")
                print("="*40)
                print(engine.get_prompt(history, tools))
            else:
                print("[ERROR] encode 命令仅支持 local LoRA 引擎（需访问 swift 模板对象）")
            continue
        if not query:
            continue

        turn_messages = history + [{"role": "user", "content": query}]
        answer = run_turn(engine, turn_messages, tools, args.max_turns)
        print(f"\n[assistant]\n{answer}")
        history = turn_messages


if __name__ == "__main__":
    main()
