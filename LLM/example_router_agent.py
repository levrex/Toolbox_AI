"""A token-efficient example with two LangChain specialist agents."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
import os


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=300
)

router_agent = create_agent(
    model=llm,
    system_prompt=(
        "Route the request. Reply with exactly one word: RESEARCH for factual "
        "questions, or WRITE for rewriting and drafting requests."
    ),
)

research_agent = create_agent(
    model=llm,
    system_prompt="Research the topic. Return at most 5 concise bullet points.",
)

writer_agent = create_agent(
    model=llm,
    system_prompt="Write a clear answer in at most 100 words using the input.",
)


def invoke_agent(agent, prompt: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    return str(result["messages"][-1].content)


def route_request(request: str) -> str:
    result = invoke_agent(router_agent, request).strip().upper()
    print('RESULTS:', result)
    return "WRITE" if result.startswith("WRITE") else "RESEARCH"


def main() -> None:
    request = "Explain why reusable skills are useful when building AI agents."
    route = route_request(request)
    agent = writer_agent if route == "WRITE" else research_agent
    print(invoke_agent(agent, request))


if __name__ == "__main__":
    main()
