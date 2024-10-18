from fastapi import FastAPI
from src.agents.cvd_rag_agent import cvd_rag_agent_executor
from src.models.cvd_rag_query import CVDQueryInput, CVDQueryOutput
from src.utils.async_utils import async_retry

app = FastAPI(
    title="CVD Chatbot",
    description="Endpoints for a cvd system graph RAG chatbot",
)


@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """
    Retry the agent if a tool fails to run. This can help when there
    are intermittent connection issues to external APIs.
    """

    return await cvd_rag_agent_executor.ainvoke({"input": query})


@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/cvd-rag-agent")
async def ask_cvd_agent(query: CVDQueryInput) -> CVDQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
