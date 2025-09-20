from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from src.config import Config
import os

def build_vector_store(chunks=None):
    embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)
    
    if os.path.exists(Config.VECTOR_DB_PATH):
        print("✅ Loading existing vector store...")
        vectorstore = FAISS.load_local(Config.VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        if chunks is None:
            raise ValueError("Chunks required to build new vector store")
        print("🛠️ Building new vector store...")
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local(Config.VECTOR_DB_PATH)
        print(f"✅ Vector store saved at '{Config.VECTOR_DB_PATH}'")
    
    return vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})