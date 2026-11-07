# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage

# llm = ChatOpenAI(
#     base_url = "http://localhost:23333/v1",
#     model_name="OpenGVLab/InternVL2_5-4B-AWQ",
#     openai_api_key="EMPTY",
#     temperature=0.7
# )

# response = llm.invoke([
#     HumanMessage(content="Tell me all countries in Asia")
# ])

# print(response.content)

from guardrails import Guard, OnFailAction
from guardrails_ai.regex_match import RegexMatch
guard = Guard().use(
    RegexMatch(regex=r"\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}", on_fail=OnFailAction.EXCEPTION)
)

guard.validate("123-456-7890")
print("Success!")
try:
    guard.validate("1234-789-0000")
except Exception as e:
    print(e)
