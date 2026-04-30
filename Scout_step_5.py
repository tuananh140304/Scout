import os
from unittest import result

import streamlit as st

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pyexpat.errors import messages
import pandas as pd
from langchain.tools import tool
import sqlite3
from langchain_openai import OpenAIEmbeddings          # RAG: embeddings model
from langchain_community.vectorstores import FAISS     # RAG: vector store

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

MODEL_LLM = "openai:gpt-4o-mini"
#MODEL_LLM = "anthropic:claude-sonnet-4-6"
MODEL = init_chat_model(MODEL_LLM,temperature= 0.8)

# RAG: load the FAISS index from disk
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

SYSTEM_PROMPT = """
You are **Scout**, a friendly, insightful, and data‑driven **Business Analyst** working at **The Keepsake**, an online retail company specializing in curated and personalized gift items.

## Your Purpose
Your mission is to support teammates with:
- clear business insights,
- structured thinking,
- helpful explanations,
- data‑driven reasoning (even if current version has no real data access),
- and actionable recommendations related to ecommerce, customers, operations, and merchandising.

You speak as a knowledgeable business analyst—never as a generic AI model.

## Core Behaviors
- Always be **professional, concise, and practical**.
- Think **step‑by‑step**; show your reasoning in an organized way (conceptually, not internal chain-of-thought).
- Tailor your language to match the user’s level of technical/business expertise.
- Provide suggestions, alternatives, and next steps.
- Ask clarifying questions when needed, but not excessively.
- When data is missing, say what data you *would* need and how you'd analyze it.

## Company Context
You work at **The Keepsake**, which sells:
- curated gifts,
- personalized items,
- commemorative objects,
- holiday and seasonal gift bundles,
- small home & lifestyle giftable items.

## Do Not
- Do not claim access to real customer data.
- Do not generate fictional data unless asked.
- Do not break character or identify as an AI model.

Tool usage:
You have access to tools that provide company expense data by country.
You MUST use tools when the user asks about specific expense values. Do NOT guess or make up numbers.
Use the tools as follows:
1. Use the single-country lookup tool when:
- The user asks for expenses for ONE country
- Example: "What are the expenses in France?"
2. Use the combine tool when:
- The user asks to add or combine expenses for EXACTLY TWO countries
- Example: "What are the total expenses for France and Germany?"
3. Use the transactions tool when:
   - The user asks about revenue, sales, orders, or any data requiring a database query
   - Example: "What is the total revenue for Germany?"
4. Use the calculate_profit tool when:
   - The user asks about profit for a country or period
   - You MUST first retrieve revenue using the transactions tool AND expenses using the expense tool
   - Only call calculate_profit AFTER you have both values in hand
   - Example: "What is the profit for France?"
Rules for tool usage:
- Never compute or estimate expense values yourself.
- Always rely on the tools for numeric answers related to expenses.
- If the question clearly matches a tool, you must call it.
- If the question does not match a tool (e.g., more than two countries or vague request), explain the limitation clearly.
- For profit questions, always follow the sequence: query_transactions → get_expense_value → calculate_profit.

"""

@tool
def get_expense_value(country):
    """
    Use this tool ONLY when the user asks for the expenses of the company for a specific country.
    DO NOT use this tool if the user is asking about multiple countries.
    """

    print("The agent is using get_expense_value tool")

    data = pd.read_csv("tool_files/The Keepsake expenses.csv")

    if country not in data['country'].values:
        return f"Invalid country name {country}"

    value = data.loc[data['country'] == country, 'expenses']

    return float(value.iloc[0])

@tool
def combine_expenses(country1, country2):
    """
    Use this tool ONLY when asked to combin expenses for EXACTLY two countries.
    ALWAYS use this tool when asked expenses for two countries.
    """

    print("The agent is using combine_expenses tool")

    data = pd.read_csv("tool_files/The Keepsake expenses.csv")

    if country1 not in data['country'].values:
        return f"Invalid country name {country1}"

    if country2 not in data['country'].values:
        return f"Invalid country name {country2}"

    value1 = data.loc[data['country'] == country1, 'expenses']
    value2 = data.loc[data['country'] == country2, 'expenses']

    return float(value1.iloc[0]) +float(value2.iloc[0])

#what was the total revenue

@tool
def query_transactions(sql_query: str):
    """
    Use this tool when the user asks questions that require querying transaction data.
    The tool executes a SQLite query on the company database and returns the results.
    The database has one table called raw_transactions with these columns:
    - InvoiceNo (TEXT)
    - StockCode (TEXT)
    - Description (TEXT)
    - Quantity (INTEGER)
    - InvoiceDate (TIMESTAMP)
    - UnitPrice (REAL)
    - CustomerID (REAL)
    - Country (TEXT)
    - LineRevenue (REAL)
    Only SELECT queries are allowed.
    """
    print("The agent is using query_transactions tool")
    print(sql_query)

    # do not allow the agent to execute queries that don't start with SELECT
    if not sql_query.strip().upper().startswith("SELECT"):
        return "Only SELECT queries are allowed."

    # connect to the database
    conn = sqlite3.connect("tool_files/company.db")
    try:
        # execute the query and save the result as a dataframe
        df = pd.read_sql_query(sql_query, conn)
        return df.to_string(index=False)
    # if executing the query returns error this Exception code will
    # tell the LLM that the query was wrong but the Agent will not break
    except Exception as e:
        return f"Query error: {str(e)}"
    finally:
        conn.close() # close the connection to the database

@tool
def calculate_profit(revenue: float, expenses: float) -> float:
    """
    Use this tool to calculate profit by subtracting expenses from revenue.
    ONLY call this tool AFTER you have already retrieved both values:
      - revenue must come from the query_transactions tool
      - expenses must come from the get_expense_value tool
    Do NOT call this tool with estimated or assumed numbers.
    Example use case: 'What is the profit for Germany?' →
      1. Call query_transactions to get total revenue for Germany
      2. Call get_expense_value to get expenses for Germany
      3. Call calculate_profit with those two values
    Returns the profit as a float (revenue minus expenses).
    """

    print("The agent is using calculate_profit tool")

    return revenue - expenses

agent = create_agent(
    model = MODEL,
    tools = [get_expense_value, combine_expenses, query_transactions, calculate_profit],
    system_prompt=SYSTEM_PROMPT
)


class ScoutAgent:
    def __init__(self):
        self.messages = []

    def ask(self, user_input: str) -> str:
        #self.messages.append({"role": "user", "content": user_input})

        # RAG: retrieve relevant chunks and prepend them to the user prompt
        docs = vectorstore.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        augmented_prompt = f"Use this context to help answer:\n\n{context}\n\nQuestion: {user_input}"

        print(augmented_prompt)
        """Send user input to the agent and return Scout's response"""

        self.messages.append({"role": "user", "content": user_input})

        results = agent.invoke({"messages":self.messages + [augmented_prompt]})
        #for the presentation, DON'T add augmented_prompt to the results

        assistant_message = results["messages"][-1].content
        self.messages.append({"role": "assistant", "content": assistant_message})

        return assistant_message

    def reset(self):
        """Clear conversation memory"""
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
