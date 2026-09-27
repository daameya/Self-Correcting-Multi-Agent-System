import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0,
)

response = model.invoke(
    "Explain in one sentence what LangGraph is."
)

print(response.content)