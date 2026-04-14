# MagicRAG

一个可直接部署到 Railway 的魔术知识 RAG 项目骨架，技术栈为 FastAPI + Qdrant Cloud + DeepSeek。

当前版本目标：

- 提供完整可运行的项目结构
- 预留文档入库、语义搜索、问答、健康检查、统计接口
- 提供 Dockerfile、环境变量示例、基础测试和部署说明

当前版本不包含具体业务逻辑实现，所有关键位置都已留出注释，方便后续补全。

## 项目结构

```text
.
├── app
│   ├── api/routes          # 路由层
│   ├── core                # 配置与日志
│   ├── db                  # Qdrant 客户端
│   ├── llm                 # DeepSeek 客户端
│   ├── models              # 内部领域模型
│   ├── schemas             # 请求/响应模型
│   ├── services            # RAG 核心编排逻辑
│   └── main.py             # FastAPI 入口
├── tests                   # 基础测试
├── .env.example            # 环境变量模板
├── Dockerfile              # Railway 部署镜像
├── requirements.txt        # Python 依赖
└── README.md
```

## 接口列表

### `GET /`

基础根接口。

### `GET /health`

健康检查接口。

### `GET /stats`

统计接口。

当前返回占位统计信息，后续应补 Qdrant collection stats、token 使用量、入库进度等。

### `POST /api/v1/rag/ingest`

文档入库接口。

请求体：

```json
{
  "force_reindex": false
}
```

当前行为：

- 扫描 `RAG_DOCUMENTS_DIR` 目录下支持的文档
- 返回发现的文件数量
- 不执行真实切片、向量化和入库

### `POST /api/v1/rag/search`

语义搜索接口。

请求体：

```json
{
  "query": "纸牌迫选原理",
  "top_k": 5
}
```

当前返回占位结果，后续需要在 `app/services/rag_service.py` 中接入：

- DeepSeek embedding
- Qdrant 向量检索
- 可选 rerank

### `POST /api/v1/rag/query`

LLM 回答接口。

请求体：

```json
{
  "question": "双翻转控制的经典思路是什么？",
  "top_k": 5
}
```

目标行为：

- 先走 RAG 检索
- 检索不到或分数过低时 fallback 到 DeepSeek 通用知识

当前仅返回可运行的占位响应。

## 本地运行

### 1. 创建虚拟环境并安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
```

至少需要检查这些变量：

- `RAG_DOCUMENTS_DIR`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `DEEPSEEK_API_KEY`

如果你的知识库文档放在本机 `l/RAG` 目录，按实际绝对路径填写，例如：

```env
RAG_DOCUMENTS_DIR=/Users/你的用户名/l/RAG
```

### 3. 启动服务

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打开：

- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### 4. 运行测试

```bash
pytest
```

## Railway 部署

### 方式一：直接从仓库部署

Railway 检测到 `Dockerfile` 后会直接按镜像构建。

需要在 Railway 配置以下环境变量：

- `APP_NAME`
- `APP_VERSION`
- `ENVIRONMENT`
- `HOST`
- `PORT`
- `LOG_LEVEL`
- `RAG_DOCUMENTS_DIR`
- `QDRANT_URL`
- `QDRANT_API_KEY`
- `QDRANT_COLLECTION_NAME`
- `QDRANT_VECTOR_SIZE`
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL`
- `DEEPSEEK_CHAT_MODEL`
- `DEEPSEEK_EMBEDDING_MODEL`
- `TOP_K`
- `SCORE_THRESHOLD`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`

### 文档目录说明

如果要让 Railway 上的入库接口扫描文档，需要保证 `RAG_DOCUMENTS_DIR` 在容器内存在。常见做法有两种：

1. 把知识库文件一起提交进仓库，再在环境变量里指向容器内路径。
2. 给 Railway 服务挂载 volume，把文档同步到挂载目录。

当前骨架不会自动上传文件到 Railway。

## 后续应该补什么

优先完善这些文件：

- `app/services/document_loader.py`
  - 实现 markdown、txt、pdf、docx 的读取
  - 实现 chunk 切片逻辑

- `app/services/rag_service.py`
  - 文档切片后生成 embedding
  - 创建或检查 Qdrant collection
  - upsert 向量与 payload
  - 搜索命中阈值判断
  - 构建回答 prompt
  - DeepSeek fallback

- `app/db/qdrant.py`
  - collection 初始化
  - 索引检查
  - stats 查询

- `app/llm/deepseek.py`
  - embedding 调用
  - chat completion 调用
  - 统一错误处理和超时控制

## 设计原则

- 路由层只做参数校验和错误转换
- 业务逻辑集中在 service 层
- 第三方客户端集中在 `db` 与 `llm` 层
- 所有环境变量统一从 `app/core/config.py` 管理

## 备注

这个骨架已经可以直接：

- 本地启动 FastAPI
- 运行基础测试
- 构建 Docker 镜像
- 部署到 Railway

但它现在是“工程骨架可运行”，不是“RAG 业务已完成”。
