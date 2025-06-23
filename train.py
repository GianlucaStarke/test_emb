import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv(find_dotenv())
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

documents = []
with open("dataset/sql_schema.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        table = data["table"]
        schema_lines = [f"{col['column']}: {col['dtype']}" for col in data["schema"]]
        schema_text = "\n".join(schema_lines)
        examples = "\n".join([str(row) for row in data["examples"]])
        content = f"Tabela: {table}\nColunas:\n{schema_text}\nExemplos:\n{examples}"
        documents.append(Document(page_content=content, metadata={"table": table}))

print(f" {len(documents)} documentos carregados")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     
    chunk_overlap=100    
)
chunks = text_splitter.split_documents(documents)
print(f"🔹 {len(chunks)} chunks gerados para embeddings")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002") 
db = FAISS.from_documents(chunks, embeddings)

os.makedirs("outputs", exist_ok=True)
db.save_local("outputs/faiss_index")
print("Índice salvo com sucesso!")
