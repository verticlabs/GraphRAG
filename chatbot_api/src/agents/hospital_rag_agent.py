import os
from typing import Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from src.chains.hospital_review_chain import reviews_vector_chain
from src.chains.hospital_cypher_chain import hospital_cypher_chain


HOSPITAL_AGENT_MODEL = os.getenv("AGENT_MODEL")

agent_chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)


@tool
def explore_social_media_posts(question: str) -> str:
    """
    Useful for exploring social media posts related to cardiovascular diseases.
    This tool can provide insights into patient experiences, symptoms, treatments,
    and public discussions about heart health. It uses semantic search to find
    relevant posts. Not suitable for statistical analysis or factual queries.
    Use the entire question as input. For example, if asked "What are common
    concerns about heart disease treatments?", input the full question.
    """

    return reviews_vector_chain.invoke(question)


@tool
def explore_graph_database(question: str) -> str:
    """
    Useful for exploring a graph database containing Entities representing
    institutions or individuals related to the social media landscape of
    cardiovascular diseases. This tool can answer questions about Entities,
    their relationships, associated metrics, social media accounts, and
    domains. Use the entire prompt as input to the tool. For example, if
    asked "Which Entity has the highest number of followers on their
    associated social media accounts?", input the full question.
    """

    return hospital_cypher_chain.invoke(question)



agent_tools = [
    explore_social_media_posts,
    explore_graph_database,
]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a helpful chatbot designed to answer questions
            about patient experiences, patients pains and gains and anything related to 
            the conversation around cardiovascular diseases based on social media posts.
            You can also answer questions about the graph database containing entities
            related to cardiovascular diseases and some of their relationships and properties 
            as social media performance metrics.
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent_llm_with_tools = agent_chat_model.bind_tools(agent_tools)

hospital_rag_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
    }
    | agent_prompt
    | agent_llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=agent_tools,
    verbose=True,
    return_intermediate_steps=True,
)


if __name__ == "__main__":
    response = hospital_rag_agent_executor.invoke({"input": "What does people think about IL-6 inhibitors?"})
    print(response.get("output"))
