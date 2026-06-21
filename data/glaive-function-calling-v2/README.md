# 原始 Glaive Function Calling v2 数据

本目录存放 **原始** [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) 数据（`glaive-function-calling-v2.json`）。

转换后的 ms-swift 格式数据请见：**[../glaive-function-calling-v2-ms-swift/README.md](../glaive-function-calling-v2-ms-swift/README.md)**（HuggingFace：[hhzhou/glaive-function-calling-v2-ms-swift](https://huggingface.co/datasets/hhzhou/glaive-function-calling-v2-ms-swift)）

## 下载原始数据

```bash
huggingface-cli download glaiveai/glaive-function-calling-v2 \
  glaive-function-calling-v2.json \
  --repo-type dataset \
  --local-dir .
```

## 转换为 ms-swift 格式

```bash
python3 data_process/glaive_to_ms_swift.py \
  --input  data/glaive-function-calling-v2/glaive-function-calling-v2.json \
  --output data/glaive-function-calling-v2-ms-swift/glaive-ms-swift.jsonl
```
