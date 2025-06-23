from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("outputs/faiss_index", embeddings, allow_dangerous_deserialization=True)

retriever = db.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo")  
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    pergunta = input(" Pergunta sobre o schema: ")
    if pergunta.strip().lower() in ["sair", "exit", "quit"]:
        break
    resposta = qa.invoke({"query": pergunta})
    print(f"💡 Resposta:\n{resposta['result']}")
