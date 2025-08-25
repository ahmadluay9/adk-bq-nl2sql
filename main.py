# main.py
import uuid
import base64
import io
import os
import logging
import sys
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, List, Dict, Any

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
log = logging.getLogger(__name__)

from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for matplotlib

# Used for LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- FastAPI App Initialization ---
app = FastAPI(
    title="BigQuery Q&A with Human Approval API",
    description="An API to generate and execute SQL queries on BigQuery with a human approval step.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Pydantic Models for API validation ---
class QueryRequest(BaseModel):
    project_id: str
    dataset_id: str
    question: str
    chat_history: List[Dict[str, str]] = []

class ExecutionRequest(BaseModel):
    thread_id: str
    approved: bool
    query: str # Allow user to modify the query before approval
    project_id: str
    dataset_id: str

# --- LangGraph State Definition ---
class State(TypedDict):
    question: str
    query: str
    result: str
    structured_result: List[Dict[str, Any]]
    insight: str
    answer: str
    chart: str # To store the base64 encoded chart
    project_id: str
    dataset_id: str
    error: str
    is_corrected: bool
    chat_history: List[Dict[str, str]]

# --- Global Variables & In-memory Checkpointer ---
# In a production environment, you would use a more persistent checkpointer like Redis or a database.
memory = MemorySaver()

# --- LangGraph Nodes ---

def write_query(state: State):
    """Generates a SQL query from the user's question."""
    log.info("Entering 'write_query' node.")
    question = state["question"]
    project_id = state.get("project_id")
    dataset_id = state.get("dataset_id")
    chat_history = state.get("chat_history", [])
    llm = ChatVertexAI(model="gemini-2.5-pro")

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history]) if chat_history else "No previous conversation."

    template = f"""You are an expert BigQuery data analyst. Your task is to generate a SQL query based on a conversation history and a new user question.
    If the new question is a follow-up to the previous one (e.g., "sort it differently", "show me more"), modify the last SQL query from the history.
    If the new question is on a completely new topic, generate a brand new query.

    Conversation History:
    ---
    {history_str}
    ---

    Based on the full conversation history above and the user's **new question**, write a single, executable Google BigQuery SQL query.

    **Table Schema:**
    {{schema}}

    **New Question:**
    {{question}}

    **BigQuery SQL Dialect Rules:**
    - When filtering for a period like "last year" on a TIMESTAMP column, always use DATE functions (e.g., `WHERE DATE(timestamp_column) >= DATE_SUB(CURRENT_DATE(), INTERVAL 1 YEAR)`).
    - Do **NOT** use `TIMESTAMP_SUB` with `YEAR` or `MONTH` intervals.

    SQL Query:
    """
    prompt = PromptTemplate.from_template(template)
    db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")

    def get_schema(_):
        return db.get_table_info()

    sql_query_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    generated_query = sql_query_chain.invoke({"question": question})
    clean_query = generated_query.replace("```sql", "").replace("```", "").strip()
    log.info("Successfully generated SQL query.")
    return {"query": clean_query, "is_corrected": False}

def check_query(state: State):
    """Validates the SQL query for common mistakes."""
    log.info("Entering 'check_query' node.")
    original_query = state["query"]
    project_id = state.get("project_id")
    dataset_id = state.get("dataset_id")

    llm = ChatVertexAI(model="gemini-2.5-pro")
    db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")

    system = """You are a SQL expert. Double check the user's {dialect} query for common mistakes.
If you find any mistakes, rewrite the query to fix them.
If the query is already correct, just return the original query exactly as it is.

Common mistakes include:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query.
If there are no mistakes, just reproduce the original query with no further commentary.

Output the final SQL query only."""

    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{query}")]
    ).partial(dialect=db.dialect)

    validation_chain = prompt | llm | StrOutputParser()
    validated_query_text = validation_chain.invoke({"query": original_query})
    clean_validated_query = validated_query_text.replace("```sql", "").replace("```", "").strip()
    
    # Check if the query was changed
    if original_query.strip() != clean_validated_query.strip():
        log.warning(f"Query was corrected. Original: '{original_query}' | Corrected: '{clean_validated_query}'")
        return {"query": clean_validated_query, "is_corrected": True}
    else:
        log.info("SQL query is valid, no changes made.")
        return {"is_corrected": False}

def execute_query(state: State):
    """Executes the SQL query and gets the result."""
    log.info(f"Entering 'execute_query' node.")
    query = state["query"]
    project_id = state.get("project_id")
    dataset_id = state.get("dataset_id")

    db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")
    try:
        query_result_dicts = db._execute(query, fetch="all")
        log.info(f"Query executed successfully, returned {len(query_result_dicts)} rows.")
        query_result_str = str(query_result_dicts)
        return {"result": query_result_str, "structured_result": query_result_dicts, "error": ""}
    except Exception as e:
        log.error(f"Error executing query: {e}", exc_info=True)
        # NEW: Capture the error to potentially loop back
        return {"error": str(e)}

def generate_chart(state: State):
    """Generates a bar chart from the query result and saves it as a base64 string."""
    log.info("Entering 'generate_chart' node.")
    data = state.get("structured_result", [])

    if not data or len(data) < 1 or len(data[0].keys()) < 2:
        print("Query result is not suitable for a chart. Skipping chart generation.")
        return {"chart": None}

    try:
        headers = list(data[0].keys())
        # Assume the first column is the label (x-axis) and the second is the value (y-axis)
        labels = [str(row[headers[0]]) for row in data]
        values = [row[headers[1]] for row in data]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.bar(labels, values)
        ax.set_xlabel(headers[0])
        ax.set_ylabel(headers[1])
        ax.set_title(f"Chart of {headers[1]} by {headers[0]}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save chart to a byte buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        # Encode as base64
        chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        print("Chart successfully generated.")
        return {"chart": chart_base64}

    except Exception as e:
        print(f"Could not generate chart: {e}")
        return {"chart": None}

def generate_insight(state: State):
    """Generates insight from the SQL query and its result."""
    log.info("Entering 'generate_insight' node.")
    query = state["query"]
    result = state["result"]
    llm = ChatVertexAI(model="gemini-2.5-pro")
    insight_prompt = PromptTemplate.from_template(
        """
        Anda adalah seorang analis data. Berdasarkan query SQL dan hasilnya berikut ini,
        berikan insight singkat dalam satu kalimat dalam Bahasa Indonesia mengenai apa yang diungkapkan oleh data tersebut.

        SQL Query: {query}
        SQL Result: {result}
        Insight (dalam Bahasa Indonesia):
        """
    )
    insight_chain = insight_prompt | llm
    insight_text = insight_chain.invoke({"query": query, "result": result})
    log.info("Insight generated successfully in Indonesian.")
    return {"insight": insight_text.content}

def generate_answer(state: State):
    """Generates a natural language answer."""
    log.info("Entering 'generate_answer' node.")
    question = state["question"]
    insight = state["insight"]
    result = state["result"]
    llm = ChatVertexAI(model="gemini-2.5-pro")
    answer_prompt = PromptTemplate.from_template(
        """
        Berdasarkan pertanyaan pengguna dan insight data, berikan jawaban akhir yang ringkas **dalam Bahasa Indonesia**.

        Pertanyaan: {question}
        Insight: {insight}
        Hasil Data (untuk konteks): {result}
        Jawaban Akhir:
        """
    )
    answer_chain = answer_prompt | llm
    final_answer = answer_chain.invoke({"question": question, "insight": insight, "result": result})
    log.info("Final answer generated successfully in Indonesian.")
    return {"answer": final_answer.content}

def handle_no_results(state: State):
    log.info("No results found. Generating direct answer.")
    return {"answer": "Maaf, query tidak menemukan data apapun yang cocok.", "insight": "Tidak ada data yang ditemukan."}

def decide_to_continue_or_rewrite(state: State):
    """
    Decides whether to continue to execution or to rewrite the query.
    """
    log.info("Entering decider.")
    if state.get("is_corrected"):
        log.info("Decision: Query was corrected, looping back to re-check.")
        # If the query was corrected, we loop back to check it again
        return "check_query"
    else:
        log.info("Decision: Query is valid, proceeding to execution.")
        # If the query is valid, we proceed
        return "execute_query"

def route_after_execution(state: State):
    log.info("Entering execution result router.")
    structured_result = state.get("structured_result", [])
    if not structured_result:
        log.warning("Decision: No results found, routing to handle_no_results.")
        return "handle_no_results"
    else:
        log.info(f"Decision: {len(structured_result)} results found, proceeding to generate_chart.")
        return "generate_chart"
    
# --- Graph Definition ---
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("check_query", check_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_chart", generate_chart)
graph_builder.add_node("generate_insight", generate_insight)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("handle_no_results", handle_no_results)

graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "check_query")
graph_builder.add_conditional_edges(
    "check_query",
    decide_to_continue_or_rewrite,
    {
        "check_query": "check_query", # Loop back to itself if corrected
        "execute_query": "execute_query" # Proceed if valid
    }
)

graph_builder.add_conditional_edges(
    "execute_query",
    route_after_execution,
    {"generate_chart": "generate_chart", "handle_no_results": "handle_no_results"}
)

# Path for results
graph_builder.add_edge("generate_chart", "generate_insight")
graph_builder.add_edge("generate_insight", "generate_answer")
graph_builder.add_edge("generate_answer", END)

# Path for no results
graph_builder.add_edge("handle_no_results", END)

# The graph will pause *before* executing the 'execute_query' node.
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["execute_query"]
)
log.info("LangGraph with validation loop compiled successfully.")

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    """Serves the frontend HTML file."""
    try:
        return FileResponse("index.html")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="index.html not found. Make sure the frontend file is in the same directory as the backend script.")

@app.post("/generate-query")
async def generate_query(request: QueryRequest):
    """
    Takes a user's question and generates a SQL query for approval.
    """
    log.info(f"Received request for /generate-query for project '{request.project_id}'.")
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    # PERUBAHAN DI SINI: Menyertakan riwayat chat di state awal
    initial_state = {
        "question": request.question,
        "project_id": request.project_id,
        "dataset_id": request.dataset_id,
        "chat_history": request.chat_history
    }
    try:
        for _ in graph.stream(initial_state, config, stream_mode="values"): pass
        current_state = graph.get_state(config)
        generated_query = current_state.values.get("query")
        if not generated_query:
            log.error("Graph execution finished but no query was generated.")
            raise HTTPException(status_code=500, detail="Failed to generate SQL query.")
        log.info(f"Successfully generated and validated query for thread_id: {thread_id}")
        return {
                "message": "Query generated successfully.", 
                "thread_id": thread_id, 
                "query": generated_query
                }
    except Exception as e:
        log.error(f"Error during /generate-query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.post("/execute-query")
async def execute_query_endpoint(request: ExecutionRequest):
    """
    Executes the query after human approval and returns the final result.
    """
    log.info(f"Received request for /execute-query for thread_id: {request.thread_id}")
    if not request.approved:
        log.warning(f"Query denied by user for thread_id: {request.thread_id}")
        return {"message": "Query execution denied by user.", "result": None}
    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        current_state = graph.get_state(config)
        if not current_state:
             log.error(f"Session not found for thread_id: {request.thread_id}")
             raise HTTPException(status_code=404, detail="Session not found.")
        
        current_state.values["query"] = request.query
        current_state.values["project_id"] = request.project_id
        current_state.values["dataset_id"] = request.dataset_id

        final_result = None
        for step in graph.stream(None, config, stream_mode="values"):
            final_result = step
        if not final_result:
            log.error(f"Graph did not produce a final result for thread_id: {request.thread_id}")
            raise HTTPException(status_code=500, detail="Failed to get a final result from the graph.")
        log.info(f"Successfully executed query for thread_id: {request.thread_id}")
        return {
            "message": "Query executed successfully.",
            "answer": final_result.get("answer"),
            "insight": final_result.get("insight"),
            "raw_result": final_result.get("structured_result"),
            "chart_png_base64": final_result.get("chart")
        }
    except HTTPException as e:
        raise e
    except Exception as e:
        log.error(f"Error during /execute-query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

# To run this app, save it as main.py and run: uvicorn main:app --reload
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
