# app_agent.py  —  Agent版（Function Calling + RAG 双工具）
import os
import json
import uuid
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import sqlite3

# ========== 基础配置 ==========
load_dotenv()
My_Key = os.getenv("SILICONFLOW_KEY")

if not My_Key:
    raise RuntimeError(
        "SILICONFLOW_KEY 未设置！请检查：\n"
        "1. 项目根目录是否有 .env 文件\n"
        "2. .env 文件中是否写入了 SILICONFLOW_KEY=your-key-here\n"
        "3. 如果使用 Docker，请确保通过 --env-file .env 注入变量"
    )

# 统一使用这个目录，避免多处硬编码
PERSIST_DIR = "./chroma_db"

# 嵌入模型
embeddings = OpenAIEmbeddings(
    model="BAAI/bge-m3",
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1"
)

# 大模型（LangChain 封装，保留用于其他场景）
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1"
)

# 原始 OpenAI 客户端（用于工具调用 Loop）
client = OpenAI(
    api_key=My_Key,
    base_url="https://api.siliconflow.cn/v1",
)

# ========== 全局切分器（函数内外共用，解决 NameError） ==========
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, chunk_overlap=20
)

# ========== 工具函数定义 ==========

def get_device_status(device_name: str) -> str:
    """查询指定设备的当前运行状态。"""
    try:
        conn = sqlite3.connect('factory.db')
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM devices WHERE name = ?", (device_name,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return result[0]
        return f"未找到设备：{device_name}"
    except Exception as e:
        return f"数据库查询异常：{str(e)}"


retriever = None


def search_knowledge_base(query: str) -> str:
    """在设备维修手册知识库中搜索相关内容"""
    if retriever is None:
        return "知识库尚未初始化，请稍后再试。"
    retrieved_docs = retriever.invoke(query)
    context_text = "\n".join([doc.page_content for doc in retrieved_docs])
    return context_text if context_text else "未在知识库中找到相关内容。"


# ========== 工具 Schema ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_device_status",
            "description": (
                "查询指定设备的当前运行状态。"
                "当用户询问某台设备是否正常运行、是否停机、状态如何时，必须调用此函数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "设备名称，例如 '设备A'、'设备B'"
                    }
                },
                "required": ["device_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "在设备维修手册知识库中搜索相关内容。"
                "当用户询问设备维修方法、故障排除步骤、设备参数、保养说明、"
                "操作规范等手册相关内容时，必须调用此函数。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题描述"
                    }
                },
                "required": ["query"],
            },
        },
    }
]

FUNCTION_MAP = {
    "get_device_status": get_device_status,
    "search_knowledge_base": search_knowledge_base,
}


# ========== RAG 知识库初始化 ==========
@st.cache_resource
def init_rag_system():
    """加载PDF、构建向量库，返回 (vectorstore, retriever)"""
    if os.path.exists(PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        loader = PyPDFLoader("manual.pdf")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
        )

    retriever_instance = vectorstore.as_retriever(search_kwargs={"k": 5})
    return vectorstore, retriever_instance


# ========== Agent 对话引擎 ==========
def agent_chat(user_input: str, history_messages: list) -> tuple:
    system_prompt = (
        "你是一个工业设备维修专家助手。你拥有以下工具：\n"
        "1. get_device_status(device_name) - 查询设备实时运行状态\n"
        "2. search_knowledge_base(query) - 搜索设备维修手册知识库\n\n"
        "你的工作原则：\n"
        "- 用户问设备状态 -> 调用 get_device_status\n"
        "- 用户问维修方法/手册内容 -> 调用 search_knowledge_base\n"
        "- 用户问题同时涉及两方面 -> 连续调用多个工具，综合回答\n"
        "- 如果问题既不需要工具也不需要检索 -> 直接回答\n"
        "- 回答要简洁、专业、针对用户问题"
    )

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history_messages[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})

    max_rounds = 5
    rounds = 0
    tool_log = []

    while rounds < max_rounds:
        rounds += 1
        resp = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
            })
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                tool_log.append(f"[工具调用] {name}")
                tool_log.append(f"  参数: {args}")
                fn = FUNCTION_MAP.get(name)
                if fn:
                    try:
                        result = fn(**args)
                    except Exception as e:
                        result = f"工具执行异常: {e}"
                else:
                    result = f"未知工具: {name}"
                tool_log.append(f"  结果: {result}")
                tool_log.append("")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
            continue

        return msg.content, tool_log

    return "处理超时，请换个问法", tool_log


# ========== Streamlit UI ==========
with st.spinner("正在加载知识库..."):
    vectorstore, retriever = init_rag_system()

with st.sidebar:
    st.header("关于本系统")
    st.write("基于 LangChain + Chroma 搭建的工业设备维修 Agent 助手")
    st.write("支持 Function Calling 工具调用")
    st.write("开发者：所以然")
    st.write("当前版本：v2.0（Agent版 · 支持工具调用 + 实时状态查询）")

uploaded_file = st.sidebar.file_uploader("动态更新知识库", type=["pdf", "txt"])
st.title("设备维修智能助手 DeviceMate")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if uploaded_file is not None:
    if st.sidebar.button("一键写入知识库"):
        # 1. 使用 UUID 生成一个绝对不重复的临时文件名，防止覆盖！
        file_extension = uploaded_file.name.split('.')[-1]
        temp_file_path = f"./temp_{uuid.uuid4().hex}.{file_extension}"

        try:
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if file_extension == 'txt':
                loader = TextLoader(temp_file_path, encoding="utf-8")
            else:
                loader = PyPDFLoader(temp_file_path)

            new_docs = loader.load_and_split(text_splitter)

            # 如果切分出来的内容为空，或者异常少，给出警告
            if len(new_docs) == 0:
                st.sidebar.error("解析失败：文件可能是图片扫描件，或者提取不到文本。")
            else:
                vectorstore.add_documents(new_docs)
                st.sidebar.success(f"成功写入 {len(new_docs)} 个文本块！")

        except Exception as e:
            st.sidebar.error(f"处理失败: {e}")
        finally:
            # 2. 核心动作：无论成功失败，最后把临时文件删掉！
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

user_query = st.chat_input("请输入故障设备代码或问题：")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Agent 正在思考并查询..."):
        answer, tool_log = agent_chat(user_query, st.session_state.messages)

    if tool_log:
        with st.expander("查看 Agent 执行过程"):
            for log_line in tool_log:
                st.write(log_line)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)