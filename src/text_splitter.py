from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=40,
    separators=["\n\n", "\n", ".", " "]
)


def split_text(text: str):
    return splitter.split_text(text)
