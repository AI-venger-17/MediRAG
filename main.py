
import argparse
from src.data_loader import load_and_split_pdf
from src.vector_store import build_vector_store
from src.inference import generate_rag_answer

def main():
    parser = argparse.ArgumentParser(description="MediRAG CLI: Medical Query Answering")
    parser.add_argument("--query", type=str, help="Single query to process")
    parser.add_argument("--build-db", action="store_true", help="Force rebuild vector DB")
    args = parser.parse_args()

    # Build/load retriever
    if args.build_db:
        chunks = load_and_split_pdf()
        retriever = build_vector_store(chunks)
    else:
        retriever = build_vector_store()

    if args.query:
        # Single query mode
        answer = generate_rag_answer(args.query, retriever)
        print(f"\nQ: {args.query}\nA: {answer}\n")
    else:
        # Interactive mode
        print("Enter queries (type 'exit' to quit):")
        while True:
            query = input("> ")
            if query.lower() == "exit":
                break
            answer = generate_rag_answer(query, retriever)
            print(f"A: {answer}\n")

if __name__ == "__main__":
    main()