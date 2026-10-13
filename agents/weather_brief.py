import requests
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

NAME = "weather_brief"

def fetch_weather():
    r = requests.get("https://wttr.in/Indonesia?format=j1", timeout=15)
    r.raise_for_status()

    current = r.json()["current_condition"][0]
    return{
        "temp_f": current["temp_F"],
        "feels_like_f": current["FeelsLikeF"],
        "humidity": current["humidity"],
        "wind_mph": current["windspeedMiles"],
        "description": current["weatherDesc"][0]["value"],
    }

def run():
    weather = fetch_weather()

    agent = create_agent(
        model = ChatOllama(model="llama3.1:8b", temperature=0),
        tools=[],
        system_prompt=(
            """
You are a weather assistant that provides information about the current weather conditions.
Add also percentage if today is a sunny day or a rainy day.
Summarizing the conditions in Bahasa Indonesia, including temperature, humidity, wind speed, and a brief description of the weather.
            """
        )
    )

    result = agent.invoke({
        "messages": [{"role": "user", "content": str(weather)}]
    })

    return f"# Weather Brief for Indonesia\n\n{result['messages'][-1].content}"