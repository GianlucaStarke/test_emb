import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore.document import Document

load_dotenv(find_dotenv())

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

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    openai_api_version="2024-05-01-preview"
)

db = FAISS.from_documents(documents, embeddings)

os.makedirs("outputs", exist_ok=True)
db.save_local("outputs/faiss_index")
print("Índice FAISS salvo em outputs/faiss_index")
