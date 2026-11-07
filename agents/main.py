from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, FieldCondition, MatchValue, Filter
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
load_dotenv()

qdrant_api_key=os.getenv('QDRANT_API_KEY')
#Knowledge
url = "http://localhost:23333/v1"
client = QdrantClient(url="https://675cb084-207c-468e-9877-5a485523e479.us-west-1-0.aws.cloud.qdrant.io", api_key=qdrant_api_key)
collection_name="knowledge"
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=4, distance=Distance.COSINE),
# )
# collections = client.get_collections()
# print("Existing collections:", collections)
# points = [
#     PointStruct(
#         id=1,
#         vector=[0.1, 0.2, 0.3, 0.4],
#         payload={"category": "favorite_food"}
#     ),
#     PointStruct(
#         id=2,
#         vector=[0.2, 0.3, 0.4, 0.5],
#         payload={"category": "future_goals"}
#     ),
# ]

# client.upsert(
#     collection_name=collection_name,
#     points=points
# )

# print("Inserted successfully")
def get_openai_client():
    """
    Initializes the OpenAI client on demand.
    """
    try:
        import openai
        return openai.Client(base_url=url, api_key="not-needed")    
    except Exception as e:
        print(f"CRITICAL: Failed to initialize OpenAI client. Error: {e}", level="error")
        return None

#Vector Search
from fastembed import TextEmbedding
embedder = TextEmbedding(model_name="BAAI/bge-small-en")
# query = "Where did I was born?"

# for r in search_results.points:
#     print(r.score, r.payload["text"])

# #FilteredSearch
# search_filter = Filter(
#     must=[
#         FieldCondition(
#             key="category",
#             match=MatchValue(value="favorite_food")
#         )
#     ]
# )

# filtered_results = client.query_points(
#     collection_name=collection_name,
#     query=query_vector,
#     query_filter=search_filter,
#     limit=1
# )

# print("Filtered search results:", filtered_results)
# try:
#     file_path = os.path.join('knowledge', 'about_me.txt')
#     with open(file_path, 'r') as f:
#         knowledge_text = f.read()
# except FileNotFoundError:
#     print("Error: File was not found")
#     exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# docs = text_splitter.split_text(knowledge_text)
# print(f"Total chunks created: {len(docs)}\n")
# for i, chunk in enumerate(docs):
#     print(f"--Chunk {i}--\n{chunk}\n")

# print("Res:", res)

# query = ef.encode_queries(["What is my mother's birthdate?"])

# results = client.search(
#     collection_name="knowledge_habel",
#     data=query,
#     limit=1,
#     output_fields=["text", "subject"]
# )

# print(results)

print("""
Hello! I'm your personal AI assistant. Please ask me anything
Type 'exit' or 'quit' to end the chat
""")

while True:
    openai_client = get_openai_client()
    user_question = input("\nQuestion: ")
    if user_question.lower() in ["quit", "exit"]:
        print("Goodbye (: !")
        break

 
    retrieved_memory = ""
    query_vector = list(embedder.embed([user_question]))[0]
    search_results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
        with_payload=True,
    )
    for r in search_results.points:
        retrieved_memory+=(r.payload["text"])
    if retrieved_memory:
        prompt = f"""
        You are a helpful personal assistant.
        Based on this fact that I'm providing you:
        {retrieved_memory}
        Please answer the following question:
        {user_question}
        Constraint:
        1. DO NOT ADD Any addition text 
        2. Act as a human interaction
        """
    else:
        prompt = user_question
    
    try:
        response = openai_client.chat.completions.create(
            model="OpenGVLab/InternVL2_5-4B-AWQ",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=4096
        )
        result = response.choices[0].message.content
        print(result)
    except Exception as e:
        print(e)