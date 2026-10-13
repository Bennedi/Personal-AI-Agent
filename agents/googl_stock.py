import requests
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

NAME = "googl_stock"

def fetch_googl():
    url = "https://query1.finance.yahoo.com/v8/finance/chart/GOOGL?interval=1d&range=1d"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()

    meta = r.json()["chart"]["result"][0]["meta"]
    price = meta["regularMarketPrice"]
    prev = meta["chartPreviousClose"]
    change = price - prev
    percent_change = (change / prev) * 100 if prev else 0

    return {
        "symbol": "GOOGL",
        "price": round(price, 2),
        "previous_close": round(prev, 2),
        "change": round(change, 2),
        "percent_change": round(percent_change, 2)
    }

def run():
    data = fetch_googl()

    agent = create_agent(
        model = ChatOllama(model="llama3.1:8b", temperature=0),
        tools=[],
        system_prompt=
        """
You are a financial assistant that provides information about stock prices.
Given stock data, write 2 concise Markdown bullet points explaining the price
move and whether it was an up or down day
        """
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": str(data)}]
    })

    return (
        "# GOOGL Stock Price Summary\n\n"
        f"{result['messages'][-1].content}"
        f"**Raw data:** {data}`\n"
    )

