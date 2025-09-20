from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.config import Config

def load_and_split_pdf():
    loader = PyPDFLoader(Config.PDF_PATH)
    pages = loader.load()
    print(f"📄 Total pages loaded: {len(pages)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE,
        chunk_overlap=Config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_splitter.split_documents(pages)
    print(f"📑 Total chunks after splitting: {len(chunks)}")
    return chunks