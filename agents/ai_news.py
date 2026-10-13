import os
import requests
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

NAME = "ai_news"

def search_news():
    r = requests.post(
        "https://ollama.com/api/web_search",
        headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"},
        json={"query": "Current traffic conditions in routes between Bekasi Barat and Central Jakarta", "max_results": 5},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["results"]

def run():
    results = search_news()
    agent = create_agent(
        model=ChatOllama(model="llama3.1:8b", temperature=0),
        tools=[],
        system_prompt="""
You are a traffic assistant that provides information about the latest traffic news and conditions.
Given the search results, write a concise summary of the latest traffic news and conditions in Markdown format
Note: Verify if the news are not outdated or irrelevant. If the news is outdated, please mention it in the summary.
        """
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": str(results)}]
    })

    return f"# Latest Traffic News Summary\n\n{result['messages'][-1].content}"