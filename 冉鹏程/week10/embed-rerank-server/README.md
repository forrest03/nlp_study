# Embedding Server

本服务提供两个本地模型能力：

- `bge-large-zh-v1.5`：文本向量化
- `bge-reranker-v2-m3`：query-document 重排

默认模型目录：

- embedding：`/data/modelscope_cache/models/BAAI/bge-large-zh-v1.5`
- reranker：`/data/modelscope_cache/models/BAAI/bge-reranker-v2-m3`

## 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8354
```

## 后台启动

先激活服务器上的 conda 环境，再执行：

```bash
bash scripts/start_server.sh
```

后台启动脚本会：

- 默认使用当前激活环境里的 `python`
- 以 `nohup` 方式后台启动服务
- 默认固定使用第 2 块 GPU，对应 `CUDA_VISIBLE_DEVICES=1`
- 将日志写入 `/data/logs/embedding-server.log`
- 将进程号写入 `/data/logs/embedding-server.pid`

如需改用其他 GPU，可在启动前覆盖：

```bash
export EMBEDDING_GPU_INDEX=3
bash scripts/start_server.sh
```

如需显式指定 Python，可在启动前覆盖：

```bash
export EMBEDDING_PYTHON=/data/miniconda3/envs/embedding/bin/python
bash scripts/start_server.sh
```

## 停止服务

```bash
bash scripts/stop_server.sh
```

## 查看状态

```bash
bash scripts/status_server.sh
```

## 打包上传

```bash
bash scripts/package_release.sh
```

打包脚本会：

- 输出 zip 到项目下 `dist/`
- 文件名带时间戳，例如 `embedding-server-20260707-120000.zip`
- zip 内带一个外层目录，例如 `embedding-server-20260707/`
- 自动排除本地缓存、IDE 文件和系统垃圾文件，例如 `.idea`、`.DS_Store`、`Thumbs.db`

## 远程接口测试

```bash
python scripts/test_remote_api.py
```

也可以单独测试某一个接口：

```bash
python scripts/test_remote_api.py health
python scripts/test_remote_api.py embeddings
python scripts/test_remote_api.py rerank
```

测试脚本固定调用以下 method：

- `embed_rerank_server.health`
- `embed_rerank_server.v1.embeddings`
- `embed_rerank_server.v1.rerank`

可选环境变量：

- `EMBEDDING_MODEL_PATH`：模型目录，默认 `/data/modelscope_cache/models/BAAI/bge-large-zh-v1.5`
- `RERANKER_MODEL_PATH`：模型目录，默认 `/data/modelscope_cache/models/BAAI/bge-reranker-v2-m3`
- `EMBEDDING_DEVICE`：推理设备，默认 `auto`
- `EMBEDDING_MAX_BATCH_SIZE`：单次最大文本数，默认 `32`
- `RERANKER_MAX_BATCH_SIZE`：单次最大文档数，默认 `64`
- `EMBEDDING_NORMALIZE`：默认是否归一化，默认 `true`

## 请求示例

```bash
curl -X POST 'http://127.0.0.1:8000/v1/embeddings' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": ["第一段文本", "第二段文本"]
  }'
```

返回示例：

```json
{
  "data": [
    {
      "index": 0,
      "embedding": [0.123, -0.456, 0.789]
    },
    {
      "index": 1,
      "embedding": [0.234, -0.567, 0.891]
    }
  ],
  "model": "bge-large-zh-v1.5",
  "usage": {
    "prompt_tokens": 24,
    "total_tokens": 24
  }
}
```

## 健康检查

```bash
curl 'http://127.0.0.1:8000/health'
```

## 重排请求示例

```bash
curl -X POST 'http://127.0.0.1:8000/v1/rerank' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "投标文件是否满足资格要求",
    "documents": ["文档A", "文档B", "文档C"]
  }'
```

返回示例：

```json
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {
      "index": 1,
      "relevance_score": 0.9821,
      "document": "文档B"
    },
    {
      "index": 2,
      "relevance_score": 0.9134,
      "document": "文档C"
    },
    {
      "index": 0,
      "relevance_score": 0.6742,
      "document": "文档A"
    }
  ],
  "usage": {
    "prompt_tokens": 38,
    "total_tokens": 38
  }
}
```
