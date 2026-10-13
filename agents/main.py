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
url = "http://localhost:11434/api/chat"
client = QdrantClient(url="https://675cb084-207c-468e-9877-5a485523e479.us-west-1-0.aws.cloud.qdrant.io", api_key=qdrant_api_key)
collection_name="knowledge"
# client.create_collection(
#     collection_name=collection_name,
#     vectors_config=VectorParams(size=4, distance=Distance.COSINE),
# )
collections = client.get_collections()
print("Existing collections:", collections)
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

#Vector Search
query_vector = [0.08, 0.14, 0.33, 0.28]
search_results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=1
)
print("Search results:", search_results)

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

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=400,
#     chunk_overlap=50,
#     separators=["\n\n", "\n", ". ", " ", ""]
# )

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

# print("""
# Hello! I'm your personal AI assistant. Please ask me anything
# Type 'exit' or 'quit' to end the chat
# """)

# while True:
#     user_question = input("\nQuestion: ")
#     if user_question.lower() in ["quit", "exit"]:
#         print("Goodbye (: !")
#         break

 
#     retrieved_memory = ""
  
#     if retrieved_memory:
#         prompt = f"""
#         You are a helpful personal assistant.
#         Based on this fact that I'm providing you:
#         {retrieved_memory}
#         Please answer the following question:
#         {user_question}
#         """
#     else:
#         prompt = user_question
#     payload = {
#         "model": "llama3.1:8b",
#         "messages": [
#             {
#                 "role": "user",
#                 "content": prompt,
                
#             }
#         ],
#         "stream": False,
#         "options":{
#             "temperature": 0,
#             "top_p": 0.9,
#             "top_k": 40,
#             "num_ctx": 8192,
#         }
#     }
#     try:
#         response = requests.post(url, json=payload, timeout=50)
#         response.raise_for_status()
#         result = response.json()
#         ai_response = result.get('message', {}).get('content', '')
#         print(ai_response)
#     except Exception as e:
#         print(e)