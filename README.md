# Tool Use — Glaive Function Calling 微调复现

[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) + LoRA + [ms-swift](https://github.com/modelscope/ms-swift) + Hermes，在 [Glaive FC v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) 上微调工具调用。

---

## 快速开始

```bash
git clone https://github.com/juicyzhou/tool-use-finetune.git && cd tool-use-finetune
```

### 1. 环境安装

**方式 A：pip（本地环境）**

需自行安装与 CUDA 匹配的 PyTorch，再安装项目依赖：

```bash
# 示例：CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

**方式 B：Docker 镜像（推荐，免配环境）**

使用 [ms-swift 官方镜像](https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html)（已内置 PyTorch / CUDA / ms-swift / vLLM）：

```bash
IMAGE=modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.10.0-vllm0.17.1-modelscope1.34.0-swift4.0.3

docker run --gpus all -it --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  "${IMAGE}" bash
```

容器内无需再 `pip install`；若需 SwanLab 等可选组件，可 `pip install swanlab`。

> 更多镜像版本与安装说明见 [SWIFT 安装文档](https://swift.readthedocs.io/zh-cn/latest/GetStarted/SWIFT-installation.html)。

### 2. 准备数据与模型

```bash
# 训练数据（ms-swift 格式）
huggingface-cli download hhzhou/glaive-function-calling-v2-ms-swift \
  --repo-type dataset --local-dir data/glaive-function-calling-v2-ms-swift

# 基座模型 → models/Qwen3-4B/（已 gitignore，可下载或软链接）
huggingface-cli download Qwen/Qwen3-4B --local-dir models/Qwen3-4B
```

| 准备项 | 说明 |
|--------|------|
| 基座模型 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) → `models/Qwen3-4B/` |
| 训练数据 | [hhzhou/glaive-function-calling-v2-ms-swift](https://huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift) · [本地说明](data/glaive-function-calling-v2-ms-swift/README.md) |
| 训练输出 | `outputs/tool-use-lora/`（已 gitignore） |

### 3. 训练与评测

```bash
make train ARGS="--gpus 0,1,2,3"
make eval ARGS="--max-samples 500"
make chat
```

---

## 统一入口

所有命令通过 **`scripts/run.sh`**（或 **`make`**）调用，配置集中在 **`scripts/config.sh`**。

```bash
bash scripts/run.sh help          # 命令列表

bash scripts/run.sh train         # 训练
bash scripts/run.sh eval          # 主评测
bash scripts/run.sh eval-sample   # 批量推理样例
bash scripts/run.sh eval-cases    # 手工 case
bash scripts/run.sh chat          # 交互 Demo
bash scripts/run.sh deploy        # API 部署
bash scripts/run.sh convert       # 数据转换
bash scripts/run.sh stats         # Token 统计
```

---

## 项目结构

```
tool-use-finetune/
├── scripts/
│   ├── run.sh              # ★ 统一入口
│   ├── config.sh           # ★ 全局配置
│   ├── train.sh            # 训练实现
│   ├── train_full.sh       # 全量 / 后台 / SwanLab
│   ├── eval/               # 评测（三层，见下）
│   └── infer/              # chat_agent.py + deploy.sh
├── data_process/           # Glaive 转换脚本
├── data/glaive-function-calling-v2-ms-swift/   # HF: hhzhou/glaive-function-calling-v2-ms-swift
├── docs/
├── Makefile
└── requirements.txt
```

---

## 训练

```bash
# 8 卡生产
PER_DEVICE_BATCH_SIZE=8 GRAD_ACCUM_STEPS=2 make train

# 单卡调试
CUDA_DEVICES=0 PER_DEVICE_BATCH_SIZE=1 GRAD_ACCUM_STEPS=4 make train ARGS="--gpus 0"
```

| 场景 | PER_DEVICE | GRAD_ACCUM | global batch |
|------|------------|------------|--------------|
| 单卡 | 1 | 4–8 | 4–8 |
| 4 卡 | 4 | 8 | 128 |
| 8 卡 | 8 | 2 | 128 |

自动计算：`CUDA_VISIBLE_DEVICES`、`NPROC_PER_NODE`、`dataset_num_proc`

高级：`make train-full`（全量数据、后台、本地 SwanLab）

---

## 评测（三层）

按用途由主到辅：

```
L1  eval/tool_selection.py   验证集工具名 + 参数 F1（主指标，报告用）
L2  eval/sample.sh           批量推理 JSONL，快速看生成效果
L3  eval/run_cases.py        手工 case 交互验证（cases.json）
```

```bash
make eval                              # L1：自动找最新 checkpoint + val 集
make eval ARGS="--max-samples 500"

make eval-sample                       # L2：默认 20 条 sample 数据
VAL_SAMPLES=50 make eval-sample

make eval-cases                        # L3：跑全部 case
make eval-cases ARGS="--case single_weather"
```

---

## 推理与部署

```bash
make chat
make deploy                            # 后台: make deploy ARGS="--background"
python3 scripts/infer/chat_agent.py --api http://127.0.0.1:8000/v1
```


**API 调用示例**（需先 `make deploy`）：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
  "model": "qwen3-4b-glaive-tool-use",
  "messages": [
    {"role": "user", "content": "What is the weather in Shanghai? Use celsius."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
          },
          "required": ["city"]
        }
      }
    }
  ],
  "temperature": 0.1
}'
```

---

## 配置参考（`scripts/config.sh`）

| 变量 | 默认值 |
|------|--------|
| `MODEL_PATH` | `models/Qwen3-4B` |
| `DATASET` | `data/glaive-function-calling-v2-ms-swift/glaive-ms-swift-10000.jsonl` |
| `TRAIN_OUT_DIR` | `outputs/tool-use-lora` |
| `CUDA_DEVICES` | `0,1,2,3,4,5,6,7` |
| `PER_DEVICE_BATCH_SIZE` | `1` |
| `GRAD_ACCUM_STEPS` | `4` |
| `DEEPSPEED` | `zero3` |

---

## 参考结果

Qwen3-4B + LoRA，验证集 checkpoint-1600：工具名 F1 **99.6%**，工具名+参数 F1 **98.7%**

---

## FAQ

- **模型目录** → `models/Qwen3-4B/`，仓库已 ignore，需自行下载或软链接
- **数据缺失** → 从 [HuggingFace](https://huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift) 下载，或见 [data/glaive-function-calling-v2-ms-swift/README.md](data/glaive-function-calling-v2-ms-swift/README.md)

## 链接

- **代码仓库**：[github.com/juicyzhou/tool-use-finetune](https://github.com/juicyzhou/tool-use-finetune)
- **数据集**：[huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift](https://huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift)
- **无 system 角色** → Hermes 模板训练时自动注入 `# Tools` 前缀

## 致谢

[Glaive FC v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) · [ms-swift](https://github.com/modelscope/ms-swift) · [Qwen3](https://huggingface.co/Qwen/Qwen3-4B) · [模板说明](docs/ms_swift_template_report.md)
