import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n".join(chunks)

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1:novita",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer questions using only the provided context. If the answer is not in the context, say 'I don't have enough information to answer that.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ],
        max_tokens=300,
        temperature=0.3,
    )

    return completion.choices[0].message.content.strip()