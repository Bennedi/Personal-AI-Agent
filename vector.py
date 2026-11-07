from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
URI = "./character_vector.db"
from dotenv import load_dotenv

openai_key = os.getenv("OPEN")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key="")

vector_store = Milvus(
    collection_name="./character_vector.db", 
    embedding_function=embeddings,
    connection_args={"uri": URI},
    index_params={
        "index_type": "FLAT",
        "metric_type": "L2"
    },
    consistency_level="Strong"
)