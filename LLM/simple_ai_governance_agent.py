"""Evaluate the responsible use of AI for a funding application."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq


from langchain_groq import ChatGroq

def load_skill() -> str:
    skill_path = Path(os.getenv("REPOSITORY_ROOT")) / "skills" / "governance" / "SKILL.md"
    return skill_path.read_text(encoding="utf-8")


def create_governance_agent():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Set GROQ_API_KEY in the repository .env file.")

    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.1,
        max_tokens=1500,
        #api_key=groq_api_key,
    )

    skill = load_skill()
    return create_agent(
        model=model,
        system_prompt=f"""
You evaluate whether and how AI should be used for a user request.
Follow this skill exactly:

{skill}

For a funding application, assess the use of AI, not the application's chance
of receiving funding. Return a concise report with exactly these sections:
1. Scenario: RAG, top-of-shelf LLM, small generic LLM, or refuse/escalate.
2. Advice: what AI may do and what it must not do (limit do what is directly relevant to request).
3. Governance report: privacy, data protection, bias, human oversight,
   traceability, sources, security, and expected costs.
4. Missing information: concrete questions if your assessment is uncertain.
5. Decision: approve the proposed AI use, approve with conditions, or escalate.

Do not invent facts about the organization, the applicant, or the funding rules.
If personal, medical, financial, or confidential information is present, mention
that it should be minimized or redacted before using an external model.
""",
    )


def evaluate_request(agent, request: str) -> str:
    result = agent.invoke(
        {"messages": [{"role": "user", "content": request}]}
    )
    return str(result["messages"][-1].content)


def main() -> None:
    print("Beschrijf waarvoor je AI wilt gebruiken.")
    print("Voorbeeld: laat een fondsaanvraag samenvatten en controleren op volledigheid.")
    request = input("\nAI-verzoek: ").strip()

    if not request:
        request = (
            "Beoordeel of AI een fondsaanvraag mag samenvatten, controleren op "
            "volledigheid en aandachtspunten kan markeren voor een medewerker."
        )
        print(f"Voorbeeldverzoek gebruikt: {request}")

    agent = create_governance_agent()
    report = evaluate_request(agent, request)

    print("\n--- AI-gebruiksadvies en governanceverslag ---\n")
    print(report)


if __name__ == "__main__":
    main()
