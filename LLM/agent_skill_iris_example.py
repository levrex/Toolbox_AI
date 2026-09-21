import os
from langchain.agents import create_agent
import pandas as pd
import asyncio
from langchain_core.tools import tool

from langchain_groq import ChatGroq


# Initialize the model using an active Groq free-tier model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)



# 2. Define the execution tool to read and process the local Iris CSV
@tool
def inspect_iris_data(file_path: str = "iris_dataset.csv") -> str:
    """Reads the Iris dataset CSV file and returns summary statistics grouped by species.
    
    Args:
        file_path: The local path to the Iris dataset CSV file.
    """
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    df = pd.read_csv(file_path)
    
    # Generate basic metrics
    summary = df.groupby(df.columns[-1]).describe().to_string()
    headers = list(df.columns)
    
    return f"Columns: {headers}\n\nGrouped Summary Statistics:\n{summary}"


@tool
def list_files(path: str = ".") -> str:
    """Lists files and directories in a local directory."""
    directory = os.path.abspath(path)
    if not os.path.isdir(directory):
        return f"Error: Directory '{path}' not found."

    entries = sorted(os.listdir(directory))
    if not entries:
        return f"Directory '{path}' is empty."

    return "\n".join(entries)

# bind_tools tells the model "these tools are available to you."
#llm_w_tools = llm.bind_tools([inspect_iris_data])

# 3. Load the SKILL.md file into the agent instructions
def load_skill_instructions(skill_path: str) -> str:
    skill_file = os.path.join(skill_path, "SKILL.md")
    with open(skill_file, "r") as f:
        return f.read()

iris_skill = load_skill_instructions("C:\\Users\\tjardo_maarseveen\\OneDrive - Transfer Solutions\\Documents\\Project\\Agentic_AI_course\\Oracle-Agentic-AI-Foundations-main\\skills\\data_analysis")

# 4. Instantiate the Agent with both the Skill instructions and the Tool


agent = create_agent(
    name="Data Assistant",
    model=llm,
    system_prompt =f"You are a helpful assistant. Follow these skill rules when relevant:\n\n{iris_skill}",
    # tools=tools,
    tools=[inspect_iris_data, list_files]
)


def run_agent(question: str):
    """Run the agent and print a clean, beginner-friendly execution trace."""

    print(f"\n🧑 User: {question}")
    print("-" * 60)

    result = agent.invoke({
        "messages": [("user", question)]
    })

    print("🔎 Clean Agent Execution Trace")
    print("-" * 60)

    step = 1

    for msg in result["messages"]:

        # 1. Human message = original user question
        if msg.type == "human":
            print(f"{step}. User asked:")
            print(f"   {msg.content}")
            step += 1

        # 2. AI message with tool_calls = agent decided to use a tool
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tool_call in msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                print(f"{step}. Agent decision:")
                print(f"   I need to use the tool: {tool_name}")
                print(f"   Tool input: {tool_args}")
                step += 1

        # 3. Tool message = result returned by the tool
        elif msg.type == "tool":
            print(f"{step}. Tool observation:")
            print(f"   Tool returned: {msg.content}")
            step += 1

        # 4. Final AI message = final response to user
        elif msg.type == "ai" and msg.content:
            print(f"{step}. Final answer:")
            print(f"   {msg.content}")
            step += 1

    print("=" * 60)

# 5. Run the agent against the local file
async def main():
    
    
    user_prompt = "Run the data-analysis skill on the local file 'iris_dataset.csv'."
    run_agent(user_prompt)
    #result = agent.invoke({"input": user_prompt})
    #print("\n--- Agent Response ---\n")
    #print(result["messages"][-1].content)  # Print the last message from the agent

if __name__ == "__main__":
    print(os.getcwd())
    #exit()
    asyncio.run(main())