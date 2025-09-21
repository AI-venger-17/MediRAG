import os
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import Config

def build_vector_store(chunks=None):
    embedding_model = HuggingFaceEmbeddings(model_name=Config.EMBEDDING_MODEL_NAME)

    if os.path.exists(Config.VECTOR_DB_PATH):
        try:
            print("✅ Loading existing vector store...")
            vectorstore = FAISS.load_local(
                Config.VECTOR_DB_PATH,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        except KeyError as e:
            print(f"⚠️ Vector DB schema mismatch ({e}), rebuilding from scratch...")
            if chunks is None:
                raise ValueError("Chunks required to rebuild vector store after schema mismatch")
            vectorstore = _build_and_save_vectorstore(chunks, embedding_model)
    else:
        if chunks is None:
            raise ValueError("Chunks required to build new vector store")
        vectorstore = _build_and_save_vectorstore(chunks, embedding_model)

    return vectorstore.as_retriever(search_kwargs={"k": Config.RETRIEVER_K})


def _build_and_save_vectorstore(chunks, embedding_model):
    print("🛠️ Building new vector store...")
    # Wrap chunks with tqdm for progress display
    vectorstore = FAISS.from_documents(
        tqdm(chunks, desc="🔧 Embedding Chunks", unit="chunk"),
        embedding_model
    )
    vectorstore.save_local(Config.VECTOR_DB_PATH)
    print(f"✅ Vector store saved at '{Config.VECTOR_DB_PATH}'")
    return vectorstore
