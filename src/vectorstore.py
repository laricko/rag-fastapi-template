from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name="docs",
        embedding_function=embedder,
        persist_directory=settings.persist_dir,
    )
