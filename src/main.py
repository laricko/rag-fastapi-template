from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, Field

from config import settings
from text_splitter import split_text
from vectorstore import get_vectorstore

app = FastAPI(title="RAG API Template")
client = OpenAI(api_key=settings.openai_api_key)


class UploadRequest(BaseModel):
    text: str
    doc_id: str = Field(..., description="Уникальный идентификатор документа", examples=["kafka_guide", "политика_конфиденциальности"])


@app.post("/upload")
async def upload_doc(body: UploadRequest):
    vs = get_vectorstore()
    chunks = split_text(body.text)
    vs.add_texts(chunks, metadatas=[{"doc_id": body.doc_id}] * len(chunks))
    vs.persist()


@app.post("/ask", description="k - количество чанков которое будет использоваться для контекста. Чем меньше тем более точен будет ответ.")
async def ask(
    query: str,
    k: int = 2
):
    vs = get_vectorstore()
    docs = vs.similarity_search(query, k=k)
    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""
    Используй только приведённый контекст, чтобы ответить на вопрос.
    Если ответа нет, скажи, что не знаешь.

    Контекст:
    {context}

    Вопрос:
    {query}
    """

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return {"answer": resp.choices[0].message.content}
