---
license: apache-2.0
language:
  - en
tags:
  - tool-use
  - function-calling
  - agent
  - ms-swift
  - hermes
  - glaive
  - fine-tuning
size_categories:
  - 10K<n<100K
---

# Glaive Function Calling v2 — ms-swift Agent Format

由 [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) 转换而来，适用于 **[ms-swift](https://github.com/modelscope/ms-swift)** + **Hermes Agent 模板** 的工具调用微调（SFT / LoRA）。

> 转换脚本：[data_process/glaive_to_ms_swift.py](https://github.com/juicyzhou/tool-use-finetune/blob/main/data_process/glaive_to_ms_swift.py)

## 文件说明

| 文件 | 条数 | 说明 |
|------|------|------|
| `glaive-ms-swift.jsonl` | ~109k | 全量训练集 |
| `glaive-ms-swift-10000.jsonl` | 10k | 快速复现子集（默认训练数据） |
| `glaive-ms-swift-sample-1000.jsonl` | 1k | 评测 / smoke test 样例 |

## 数据格式

每行一条 JSON，ms-swift Agent 格式：

```json
{
  "tools": "[{\"type\": \"function\", \"function\": {...}}]",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "tool_call", "content": "{\"name\": \"...\", \"arguments\": {...}}"},
    {"role": "tool_response", "content": "{...}"},
    {"role": "assistant", "content": "..."}
  ]
}
```

- 原始 Glaive `system` 人设已丢弃；工具定义提取到顶层 `tools` 字段
- 角色：`user` / `assistant` / `tool_call` / `tool_response`
- 训练时需设置 `agent_template=hermes`

## 下载

```bash
pip install huggingface_hub

huggingface-cli download hhzhou/glaive-function-calling-v2-ms-swift \
  --repo-type dataset \
  --local-dir data/glaive-function-calling-v2-ms-swift
```

数据集页面：<https://huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift>

## 训练示例

配合 [tool-use-finetune](https://github.com/juicyzhou/tool-use-finetune) 代码仓库：

```bash
git clone https://github.com/juicyzhou/tool-use-finetune.git && cd tool-use-finetune
pip install -r requirements.txt

bash scripts/run.sh train
# 或指定全量
DATASET=data/glaive-function-calling-v2-ms-swift/glaive-ms-swift.jsonl bash scripts/run.sh train
```

## 转换说明

相对原始 Glaive FC v2 的变更：

1. 解析 `<functioncall>` 为独立 `tool_call` / `tool_response` 消息
2. 丢弃 assistant 人设类 system 文本
3. 工具 JSON 提取到样本级 `tools` 字段

本地从原始数据转换：

```bash
huggingface-cli download glaiveai/glaive-function-calling-v2 \
  glaive-function-calling-v2.json --repo-type dataset \
  --local-dir data/glaive-function-calling-v2

python3 data_process/glaive_to_ms_swift.py \
  --input  data/glaive-function-calling-v2/glaive-function-calling-v2.json \
  --output data/glaive-function-calling-v2-ms-swift/glaive-ms-swift.jsonl
```

## 许可

本数据集为 [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) 的转换衍生版本，请遵循原数据集许可证（Apache 2.0）。

## 引用

```bibtex
@misc{glaive-function-calling-v2-ms-swift,
  title={Glaive Function Calling v2 — ms-swift Agent Format},
  author={juicyzhou},
  howpublished={\url{https://huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift}},
  note={Converted from glaiveai/glaive-function-calling-v2. Code: https://github.com/juicyzhou/tool-use-finetune}
}
```
