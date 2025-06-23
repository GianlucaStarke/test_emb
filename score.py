from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Embeddings e LLM
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    openai_api_version="2024-05-01-preview"
)
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    openai_api_version="2024-05-01-preview"
)

# Carregar índice FAISS
db = FAISS.load_local("outputs/faiss_index", embeddings, allow_dangerous_deserialization=True)

# Criar cadeia de QA
chain = load_qa_chain(llm, chain_type="stuff")

# Loop de perguntas
while True:
    pergunta = input(" Faça sua pergunta sobre o banco: ")
    if pergunta.strip().lower() in ["sair", "exit", "quit"]:
        break
    docs = db.similarity_search(pergunta, k=5)
    resposta = chain.run(input_documents=docs, question=pergunta)
    print("💬", resposta)
