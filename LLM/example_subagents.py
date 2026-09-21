"""Route a request to one specialist subagent."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_groq import ChatGroq


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.2,
    max_tokens=300
)

research_subagent = create_agent(
    model=llm,
    system_prompt="Research factual questions. Return at most 5 concise bullet points.",
)

writing_subagent = create_agent(
    model=llm,
    system_prompt="Rewrite or draft text clearly in at most 100 words.",
)


@tool
def ask_research_subagent(request: str) -> str:
    """Use for factual questions that need research or explanation."""
    result = research_subagent.invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return str(result["messages"][-1].content)


@tool
def ask_writing_subagent(request: str) -> str:
    """Use for rewriting, drafting, or improving text."""
    result = writing_subagent.invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return str(result["messages"][-1].content)


router_agent = create_agent(
    model=llm,
    tools=[ask_research_subagent, ask_writing_subagent],
    system_prompt=(
        "You are a router. Call exactly one specialist tool. "
        "Use ask_research_subagent for factual questions. "
        "Use ask_writing_subagent for rewriting or drafting. "
        "Return the selected specialist's answer without adding commentary."
    ),
)


def main() -> None:
    request = "Explain why reusable skills are useful when building AI agents."
    result = router_agent.invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
