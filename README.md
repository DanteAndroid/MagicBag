# MagicRAG

![image-20260414140123497](structure.png)

一个给魔术资料做知识库检索的 RAG 服务，技术栈是 FastAPI + Qdrant + DeepSeek 兼容接口。

目前这个项目已经能做的事：

- 启动 Web 服务
- 扫描本地文档目录
- 读取 `md`、`txt`、`html`、`pdf`、`docx`
- 按配置切成文本块
- 提供健康检查、入库、搜索、问答接口骨架

目前还没做完的事：

- 向量生成
- 写入 Qdrant
- 真正的语义搜索
- 基于检索结果生成回答

如果你是第一次接触这个项目，先看“快速开始”和“现在项目做到哪一步”。

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 创建环境变量文件

```bash
cp .env.example .env
```

把 `.env` 里这几项先填好：

```env
RAG_DOCUMENTS_DIR=/Users/l/RAG
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
DEEPSEEK_CHAT_MODEL=deepseek-ai/DeepSeek-V3.2-Exp
```

说明：

- `RAG_DOCUMENTS_DIR` 是你的知识库文档目录
- `DEEPSEEK_*` 是模型接口配置
- 如果你还没有接好 Qdrant，也可以先不填 `QDRANT_*`

### 3. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

启动后打开：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### 4. 扫描本地文档

接口：

```text
POST /api/v1/rag/ingest
```

请求体：

```json
{
  "force_reindex": false
}
```

当前效果：

- 读取 `RAG_DOCUMENTS_DIR` 下的文档
- 按 `CHUNK_SIZE` 和 `CHUNK_OVERLAP` 切块
- 返回发现了多少文档、切出了多少 chunk
- 还不会写入 Qdrant

## 现在项目做到哪一步

### 已完成

- Web 服务入口可运行
- `GET /health`
- `GET /stats`
- `POST /api/v1/rag/ingest`
- 本地文档扫描和切块
- Railway Docker 部署

### 未完成

- embedding 生成
- Qdrant collection 初始化
- 向量 upsert
- 语义检索
- 回答生成

## 接口说明

### `GET /health`

健康检查。

### `GET /stats`

返回当前配置下的基础统计信息。

### `POST /api/v1/rag/ingest`

扫描并切分本地文档。

### `POST /api/v1/rag/search`

接口已预留，暂时返回占位结果。

### `POST /api/v1/rag/query`

接口已预留，暂时返回占位结果。

## 本地文档目录说明

你当前的文档目录建议直接用：

```env
RAG_DOCUMENTS_DIR=/Users/l/RAG
```

这个目录里已经有大量资料，包括：

- Markdown 文档
- TXT 文档
- PDF 文档

项目现在会读取这些文件并做切块。

## Railway 部署

Railway 检测到 `Dockerfile` 后会直接构建镜像。

至少需要配置这些环境变量：

```env
DEEPSEEK_API_KEY=你的线上密钥
DEEPSEEK_BASE_URL=https://api.siliconflow.cn/v1
DEEPSEEK_CHAT_MODEL=deepseek-ai/DeepSeek-V3.2-Exp
RAG_DOCUMENTS_DIR=/data/rag
```

项目也兼容 OpenAI 风格变量名：

- `OPENAI_API_KEY` 等价于 `DEEPSEEK_API_KEY`
- `OPENAI_BASE_URL` 等价于 `DEEPSEEK_BASE_URL`
- `OPENAI_MODEL` 等价于 `DEEPSEEK_CHAT_MODEL`
- `OPENAI_EMBEDDING_MODEL` 等价于 `DEEPSEEK_EMBEDDING_MODEL`

注意：

- Railway 上的 `RAG_DOCUMENTS_DIR` 必须是容器内真实存在的目录
- 如果文档没有打进镜像或挂载 volume，线上服务虽然能启动，但不会扫到资料

## 常用命令

安装依赖：

```bash
pip install -r requirements.txt
```

运行服务：

```bash
uvicorn app.main:app --reload
```

运行测试：

```bash
pytest -q
```

## 项目结构

```text
.
├── app
│   ├── api/routes
│   ├── core
│   ├── db
│   ├── llm
│   ├── models
│   ├── schemas
│   ├── services
│   └── main.py
├── tests
├── .env.example
├── Dockerfile
├── requirements.txt
└── README.md
```
