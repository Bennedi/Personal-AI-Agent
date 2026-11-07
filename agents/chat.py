"""Persona chat loop with RAG, backed by LMDeploy's OpenAI-compatible server.

Configure in .env:
  VLLM_BASE_URL=http://localhost:23333/v1  # lmdeploy serve api_server endpoint
  VLLM_MODEL=qwen2.5-7b-instruct          # must match the served model name

Run from repo root:  python agents/chat.py
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastembed import TextEmbedding
from openai import OpenAI
from qdrant_client import QdrantClient
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from memory import VLLM_MODEL, build_system_prompt, expand_query, llm
from remember import facts_block
from store import append_turn, load_session, new_session_id

MAX_CONTEXT_MESSAGES = int(os.getenv("MAX_CONTEXT_MESSAGES", "8"))

def ask_llm(messages: list[dict]) -> str:
    resp = llm.chat.completions.create(model=VLLM_MODEL, messages=messages, temperature=0.6)
    return resp.choices[0].message.content


def main():
    session_id = new_session_id()
    print(f"Session {session_id}. LMDeploy model={VLLM_MODEL}. Type 'exit' or 'quit' to end.\n")
    print("Type /remember <fact> to teach Habel something durable. \n")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue
    
        facts = facts_block(question)
        messages = (
            [{"role": "system", "content": build_system_prompt(facts)}]+ load_session(session_id, MAX_CONTEXT_MESSAGES) + [{"role": "user", "content": question}]
        )
        try:
            answer = ask_llm(messages)
        except Exception as e:
            print("LLM error:", e)
            continue
        print(f"\nHabel: {answer}")
        append_turn(session_id, "user", question)
        append_turn(session_id, "assistant", answer, meta={"facts_retrieved": facts.count("---") + 1})

if __name__ == "__main__":
    main()
