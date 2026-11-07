import streamlit as st
import os
from pathlib import Path
from fastembed import TextEmbedding
from dotenv import load_dotenv
from datetime import datetime
from langchain_ollama.llms import OllamaLLM
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, MatchValue, Filter
from openai import OpenAI
import logging
import sys
import os
from mcp.server.mcpserver import MCPServer
st.set_page_config(
    page_title="Habel's AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

name="ai-mcp-server"
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s_%(timestamp)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(name)
port = int(os.environ.get('PORT', 8080))

mcp = MCPServer(name)

# @mcp.tool()
# def add(a: int, b: int) -> int:
#     logger.info(f"Tool: add({a}, {b})")
#     return a + b

# @mcp.tool()
# def get_secret_word() -> str:
#     logger.info("Tool: get_secret_word()")
#     return random.choice(["apple", "banana", "cherry"])

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import (IDENTITY, KNOWLEDGE, VLLM_MODEL, build_system_prompt, client, embed, expand_query, llm)

from remember import recall
from store import append_turn, load_session, new_session_id

collection_name = KNOWLEDGE
llm_model = VLLM_MODEL

def expand_query(q: str) -> str:
    ql = q.lower()
    extra = []
    if "old" in ql or "age" in ql or "birthday" in ql:
        extra.append("born birthdate age years old")
    if "brother" in ql or "sibling" in ql or "sister" in ql or "family" in ql or "parent" in ql or "mother" in ql or "father" in ql:
        extra.append("family brother mother father sibling born")
    return (q + " " + " ".join(extra)).strip() if extra else q


@st.cache_resource
def connect_to_memory():
    try:
        return client.get_collection(collection_name)
    except Exception:
        return None

collection = connect_to_memory()
# # print(collection)
# print(collection.points_count)

# st.title("🤖 Habel's AI")
# st.markdown("*Habel's representation when he's not available*")

# with st.sidebar:
#     st.header("🧠 Memory")
#     if collection:
#         st.success(f"Memory loaded successfully!")
#     else:
#         st.error("❌ **No memory found**")
#     st.markdown("---")
#     st.header("🔌 LLM status")
#     try:
#         served = openai_client.models.list()
#         st.success(f"LLM connected @ {vllm_base_url}")
#         st.caption(f"Using model: {llm_model}")
#         st.caption(f"Served: {[m.id for m in served.data]}")
#     except Exception as e:
#         st.error(f"LLM NOT reachable: {e}")
#         st.caption(f"Base URL in use: {vllm_base_url}")
#     st.markdown("---")
#     st.subheader("💡 Tips")
#     st.write("Ask about your preferences")
#     st.write("Try asking 'What do you want to know'")

# st.header("💬 Chat with Your Assistant")

# if "messages" not in st.session_state:
#     st.session_state.messages = []
#     welcome_message = "Welcome. This is Habel's AI Agents. A representation of Habel Siregar in human. What would you like to know about?"
#     st.session_state.messages.append({"role": "assistant", "content": welcome_message})

# # Render the full conversation on every rerun (welcome + all prior turns).
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# def calculate_similarity(embedding1, embedding2):
#     return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))



# if prompt := st.chat_input("Ask me anything!"):
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         with st.spinner("Thinking... "):

#             retrieved_memory = ""
#             response = ""

#             if collection:
#                 try:
#                     query_vector = list(embedder.embed([expand_query(prompt)]))[0]
#                     search_results = client.query_points(
#                         collection_name=collection_name,
#                         query=query_vector,
#                         limit=6,
#                     )
#                     retrieved_memory = "\n---\n".join(r.payload["text"] for r in search_results.points)

#                     # Fix #2: fully in-character prompt. IDENTITY is always present;
#                     # retrieved facts are added as extra memory. No "AI/assistant" framing.
#                     system_prompt = IDENTITY
#                     if retrieved_memory:
#                         system_prompt += f"""

# More details recalled from your own memories (use them where relevant):
# {retrieved_memory}"""
#                     system_prompt += """

# How to answer:
# - Reply as Habel himself, in first person, warm and casual like a real person texting.
# - Use the facts above naturally; do not force them in.
# - If something personal isn't covered, answer honestly and briefly in character (e.g. "hmm I'm not sure about that one") — never break character or mention being an AI.
# - Keep it conversational, not robotic. No bullet-point lists unless asked."""

#                     # session_state.messages = [welcome, ...prior turns..., current user]
#                     # Drop the welcome and the current user turn to get real history.
#                     history = [
#                         {"role": m["role"], "content": m["content"]}
#                         for m in st.session_state.messages[1:-1]
#                     ][-MAX_CONTEXT_MESSAGES:]
#                     messages = (
#                         [{"role": "system", "content": system_prompt}]
#                         + history
#                         + [{"role": "user", "content": prompt}]
#                     )

#                     completion = openai_client.chat.completions.create(
#                         model=llm_model,
#                         messages=messages,
#                         temperature=0.2,
#                         max_tokens=8192,
#                         stream=True,
#                     )
#                     response = st.write_stream(completion)
#                 except Exception as e:
#                     st.error(f"Error generating response: {e}")
#                     response = "Sorry, I encountered an error while processing your question. Please try again."
#         st.session_state.messages.append({"role": "assistant", "content": response})
#         if len(st.session_state.messages) > 1: 
#             if st.sidebar.button("Clear Chat History", type="secondary"):
#                 st.session_state.messages = []
#                 st.rerun()


if __name__ =="__main__":
    logger.info(f"Starting MCP server on port {port}")    
    try:
        mcp.run(transport="sse")
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Server terminated")



