from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
URI = "./character_vector.db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key="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")

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