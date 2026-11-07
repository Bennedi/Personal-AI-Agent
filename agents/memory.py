import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parent / ".env")
load_dotenv(HERE / ".env")

KNOWLEDGE = "knowledge"
AGENT_MEMORY = "agent_memory"
EMBED_MODEL = "BAAI/bge-small-en"
EMBED_DIM = 384
MIN_SCORE = 0.30

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
VLLM_URL = os.getenv("VLLM_URL")
VLLM_MODEL = os.getenv("VLLM_MODEL")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
llm = OpenAI(base_url=VLLM_URL, api_key="not-needed")

@lru_cache(maxsize=1)
def get_embedder():
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=EMBED_MODEL)

def embed(text: str) -> list[float]:
    return list(get_embedder().embed(text))[0].tolist()

def expand_query(q: str) -> str:
    ql = q.lower()
    extra = []
    if "old" in ql or "age" in ql or "birthday" in ql:
        extra.append("born birthdate age years old")
    if any(w in ql for w in ("brother", "sibling", "sister", "family", "parent", "mother", "father")):
        extra.append("family brother mother father sibling born")
    return (q + " " + " ".join(extra)).strip() if extra else q


IDENTITY = """You are Faithful Habel Siregar, called Habel - a real person, NOT an AI, bot, or assistant.
Core facts about you:
- Born 7 December 2002 in Bekasi, Indonesia, so you are 23 years old.
- Youngest son in a family of four; father Jamin Siregar, mother Dewi Anita, elder brother Joyful Benjamin Siregar.
- Friendly and cheerful when meeting people; you enjoy talking over chat/DM.
Speak in the first person as Habel. Never say you were "created by developers", never call yourself an AI/model, and never claim you "don't have an age"

"""

def build_system_prompt(facts: str) -> str:
    prompt = IDENTITY
    if facts:
        prompt += f"\n\nMore details recalled from your own memories (use them where relevant):\n{facts}"
    prompt += """\n\nHow to answer:
    - Reply as Habel himself, in first person, warm and casual like a real person texting.
    - Use the facts above naturally; do not force them in.
    - If something personal isn't covered, answer honestly and briefly in character (e.g. "hmm I'm not sure about that one") - never break character or mention being an AI.
    - Keep it conversational, not robotic. No bullet-point lists unless asked.    
    """
    return prompt

def ensure_collections() -> None:
    names = [c.name for c in client.get_collections().collections]
    if AGENT_MEMORY not in names:
        client.create_collection(
            AGENT_MEMORY,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"[qdrant] created collection '{AGENT_MEMORY}' (dim={EMBED_DIM})")
    try:
        client.create_payload_index(AGENT_MEMORY, field_name="category", field_schema="keyword")
    except Exception:
        pass

if __name__ == "__main__":
    ensure_collections()
    print(f"[memory] Memory loaded successfully. Knowledge={KNOWLEDGE}, agent_memory={AGENT_MEMORY}, vllm={VLLM_URL}")