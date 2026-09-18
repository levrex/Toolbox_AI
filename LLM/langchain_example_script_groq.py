import os

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Initialize the model using an active Groq free-tier model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)

messages = [
    SystemMessage(content="You are an expert software engineer."),
    HumanMessage(content="Explain prompt caching in two bullet points.")
]

response = llm.invoke(messages)
print(response.content)

# ─────────────────────────────────────────────
# 2. PROMPT TEMPLATES — Steering the Model
# ─────────────────────────────────────────────
# A prompt template is a reusable sentence with blanks ({placeholders})
# you fill in later — write the wording once, reuse it many times.
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# 2. Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise programming tutor."),
    ("user", "Explain {topic} in simple terms for a beginner.")
])

# 3. Build LCEL Chain (Prompt -> LLM -> String Output)
chain = prompt | llm | StrOutputParser()

# 4. Invoke Chain
result = chain.invoke({"topic": "recursion"})
print(result)

# ─────────────────────────────────────────────
# 4. MEMORY — Giving the Model Context
# ─────────────────────────────────────────────
# Models are stateless — they forget everything between calls.
# "Memory" is simply us storing past messages and feeding them back in.
# (Modern LangChain uses ChatMessageHistory; the old
#  ConversationBufferMemory is deprecated and out of the core package.)

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# 3. Store in-memory sessions
store = {}

# 2. Define prompt with a placeholder for history
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain = prompt | llm

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 4. Wrap chain with history handler
conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# 5. Usage with a Session ID
config = {"configurable": {"session_id": "user_session_1"}}

# Turn 1
res1 = conversational_chain.invoke({"input": "Hi! My name is Alice."}, config=config)
print("Bot:", res1.content)

# Turn 2 (Remembers previous message)
res2 = conversational_chain.invoke({"input": "What is my name?"}, config=config)
print("Bot:", res2.content)

# ─────────────────────────────────────────────
# 5. TOOLS — Giving the Model Abilities   (Lesson 2)
# ─────────────────────────────────────────────
# A tool is just a normal Python function the model is ALLOWED to call.
# The @tool decorator exposes it; the model reads the function's name,
# docstring, and type hints to decide WHEN and HOW to use it.

from langchain_core.tools import tool

@tool
def calculate_bmi(weight_kg: float, height_m: float) -> str:
    """Calculate Body Mass Index (BMI) given weight in kg and height in meters."""
    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal weight"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"
    return f"BMI: {bmi:.1f} ({category})"

@tool
def get_word_count(text: str) -> int:
    """Count the number of words in a given text string."""
    return len(text.split())

# This is exactly what the model "sees" about a tool — the same info it
# uses to decide whether the tool fits the question.
#print("=== Tool Info ===")
#print(f"Name: {calculate_bmi.name}")
#print(f"Description: {calculate_bmi.description}")
#print(f"Args: {calculate_bmi.args}")
#print()

# bind_tools tells the model "these tools are available to you."
model_with_tools = llm.bind_tools([calculate_bmi, get_word_count])

# IMPORTANT (say this out loud): the model only *requests* a tool call —
# it does NOT run the tool. It hands back which tool to call and with what
# arguments. Actually executing the tool and looping the result back is the
# AGENT's job.
response = model_with_tools.invoke("What's the BMI for someone who is 70kg and 1.75m tall?")
print("=== Tool Call Response ===")
print(f"Tool calls: {response.tool_calls}")
#print()

print("✅ All core concepts demonstrated! Next: first_agent.py")
