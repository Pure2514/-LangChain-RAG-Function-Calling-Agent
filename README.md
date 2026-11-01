# DeviceMate · 设备维修智能助手

基于 Streamlit + LangChain + Chroma 的工业设备维修问答系统，
支持 PDF 手册知识库检索（RAG）与设备实时状态查询（Function Calling）。

## 快速开始

### 本地运行

1. 安装依赖：`pip install -r requirements.txt`
2. 复制 `.env.example` 为 `.env`，填入 `SILICONFLOW_KEY`
3. 将设备手册放入根目录，命名为 `manual.pdf`
4. 运行：`streamlit run app_agent.py`

### Docker 部署

1. 确保 `.env` 文件已配置好 `SILICONFLOW_KEY`
2. 构建镜像：`docker build -t device-mate:2.0 .`
3. 运行容器：`docker compose up -d --build`
4. 浏览器打开 http://localhost:8501`

## 目录结构

```
device-mate/
├── app_agent.py           # 主程序
├── requirements.txt       # Python 依赖
├── Dockerfile             # Docker 构建文件
├── .dockerignore          # Docker 构建排除文件
├── .env                   # 环境变量（不进 Git）
├── .env.example           # 环境变量模板（可提交到 Git）
├── docker-compose.yml     # Docker Compose 配置
├── manual.pdf             # 设备维修手册
├── .streamlit/
│   └── config.toml        # Streamlit 配置
└── chroma/                # Chroma 向量库持久化目录（自动创建）
```

## 技术栈

- **Web 框架**：Streamlit
- **LLM 编排**：LangChain
- **向量数据库**：Chroma
- **嵌入模型**：BAAI/bge-m3（通过 SiliconFlow API）
- **大语言模型**：Qwen/Qwen3-8B（通过 SiliconFlow API）
- **容器化**：Docker + Docker Compose

## 功能特性

- **RAG 知识库检索**：上传 PDF 手册，自动构建向量索引，支持语义搜索
- **Function Calling 工具调用**：大模型自动判断意图，调用设备状态查询或知识库检索
- **多轮对话**：支持上下文记忆，对话历史自动管理
- **Agent 执行过程可视化**：折叠面板展示工具调用详情，方便调试

## 注意事项

- `.env` 文件包含敏感密钥，已加入 `.gitignore`，不要提交到 Git
- Chroma 向量库数据持久化在 `chroma/` 目录，删除该目录会丢失索引
- 首次启动需要下载模型文件，请耐心等待
