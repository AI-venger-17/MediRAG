from groq import Groq
from src.config import Config

groq_client = Groq(api_key=Config.GROQ_API_KEY)

def generate_rag_answer(query, retriever):
    docs = retriever.invoke(query)
    context = "\n\n".join([f"[Source: {doc.metadata.get('source', 'Unknown')}] {doc.page_content}" for doc in docs])

    prompt = f"""
Use the following extracted context to answer the question.

If the answer is not available in the context, say:
"The answer is not available in the provided context."

Context:
{context}

Question:
{query}

Answer:
"""
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a highly knowledgeable medical assistant trained using The Merck Manual."},
            {"role": "user", "content": prompt}
        ],
        model=Config.GROQ_MODEL,
        max_tokens=Config.MAX_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P
    )
    return chat_completion.choices[0].message.content.strip()