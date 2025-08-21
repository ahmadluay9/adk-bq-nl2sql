# main.py
import uuid
import base64
import io
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import TypedDict, List, Dict, Any

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

class ExecutionRequest(BaseModel):
    thread_id: str
    approved: bool
    query: str # Allow user to modify the query before approval

# --- LangGraph State Definition ---
class State(TypedDict):
    question: str
    query: str
    result: str
    structured_result: List[Dict[str, Any]]
    insight: str
    answer: str
    chart: str # To store the base64 encoded chart
    project_id: str # Added to state
    dataset_id: str # Added to state


# --- Global Variables & In-memory Checkpointer ---
# In a production environment, you would use a more persistent checkpointer like Redis or a database.
memory = MemorySaver()

# --- LangGraph Nodes ---
# Note: These are mostly the same as your original code, with minor adjustments for the API.

def write_query(state: State):
    """Generates a SQL query from the user's question."""
    print("---GENERATING SQL QUERY---")
    question = state["question"]
    project_id = state.get("project_id")
    dataset_id = state.get("dataset_id")

    if not project_id or not dataset_id:
        raise ValueError("project_id and dataset_id must be in the initial state")

    llm = ChatVertexAI(model="gemini-2.5-pro")

    template = """
    Based on the table schema below, write a SQL query that would answer the user's question.
    Pay attention to use only the column names that you can see in the schema description.
    Be careful to not query for columns that do not exist.
    Pay attention to which column is in which table.
    Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    Table Schema: {schema}
    Question: {question}
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
    return {"query": clean_query}

def check_query(state: State):
    """Validates the SQL query for common mistakes."""
    print("---CHECKING SQL QUERY---")
    query = state["query"]
    project_id = state.get("project_id")
    dataset_id = state.get("dataset_id")

    llm = ChatVertexAI(model="gemini-2.5-pro")
    db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")

    system = """Double check the user's {dialect} query for common mistakes, including:
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
    validated_query_text = validation_chain.invoke({"query": query})
    clean_validated_query = validated_query_text.replace("```sql", "").replace("```", "").strip()
    return {"query": clean_validated_query}

def execute_query(state: State):
    """Executes the SQL query and gets the result."""
    print("---EXECUTING SQL QUERY---")
    query = state["query"]
    project_id = state.get("project_id")
    dataset_id = state.get("dataset_id")

    db = SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")
    try:
        query_result_dicts = db._execute(query, fetch="all")
        if query_result_dicts:
            headers = query_result_dicts[0].keys()
            data = [list(row.values()) for row in query_result_dicts]
            print("\n--- QUERY RESULT ---")
            print(tabulate(data, headers=headers, tablefmt="grid"))
            print("--------------------\n")
        else:
            print("\n--- QUERY RESULT ---")
            print("Query returned no results.")
            print("--------------------\n")

        query_result_str = str(query_result_dicts)
        return {"result": query_result_str, "structured_result": query_result_dicts}
    except Exception as e:
        print(f"Error executing query: {e}")
        raise HTTPException(status_code=400, detail=f"Error executing query: {str(e)}")


def generate_chart(state: State):
    """Generates a bar chart from the query result and saves it as a base64 string."""
    print("---GENERATING CHART---")
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
    print("---GENERATING INSIGHT---")
    query = state["query"]
    result = state["result"]
    llm = ChatVertexAI(model="gemini-2.5-pro")
    insight_prompt = PromptTemplate.from_template(
        """
        You are a data analyst. Given the following SQL query and its result,
        provide a brief, one-sentence insight into what the data reveals.

        SQL Query: {query}
        SQL Result: {result}
        Insight:
        """
    )
    insight_chain = insight_prompt | llm
    insight_text = insight_chain.invoke({"query": query, "result": result})
    return {"insight": insight_text.content}

def generate_answer(state: State):
    """Generates a natural language answer."""
    print("---GENERATING FINAL ANSWER---")
    question = state["question"]
    insight = state["insight"]
    result = state["result"]
    llm = ChatVertexAI(model="gemini-2.5-pro")
    answer_prompt = PromptTemplate.from_template(
        """
        Given the user's question and the data insight, provide a final answer.

        Question: {question}
        Insight: {insight}
        Data Result (for context): {result}
        Final Answer:
        """
    )
    answer_chain = answer_prompt | llm
    final_answer = answer_chain.invoke({"question": question, "insight": insight, "result": result})
    return {"answer": final_answer.content}


# --- Graph Definition ---
graph_builder = StateGraph(State)
graph_builder.add_node("write_query", write_query)
graph_builder.add_node("check_query", check_query)
graph_builder.add_node("execute_query", execute_query)
graph_builder.add_node("generate_chart", generate_chart)
graph_builder.add_node("generate_insight", generate_insight)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.add_edge(START, "write_query")
graph_builder.add_edge("write_query", "check_query")
graph_builder.add_edge("check_query", "execute_query")
graph_builder.add_edge("execute_query", "generate_chart")
graph_builder.add_edge("generate_chart", "generate_insight")
graph_builder.add_edge("generate_insight", "generate_answer")
graph_builder.add_edge("generate_answer", END)

# The graph will pause *before* executing the 'execute_query' node.
graph = graph_builder.compile(
    checkpointer=memory,
    interrupt_before=["execute_query"]
)

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
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "question": request.question,
        "project_id": request.project_id,
        "dataset_id": request.dataset_id,
    }
    
    # Stream the graph to generate the query
    try:
        for _ in graph.stream(initial_state, config, stream_mode="values"):
            pass # We just need to run it until the interruption point
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in graph execution: {str(e)}")

    # Get the current state to retrieve the generated query
    current_state = graph.get_state(config)
    generated_query = current_state.values.get("query")

    if not generated_query:
        raise HTTPException(status_code=500, detail="Failed to generate SQL query.")

    return {
        "message": "Query generated successfully. Please review and approve.",
        "thread_id": thread_id,
        "query": generated_query,
    }

@app.post("/execute-query")
async def execute_query_endpoint(request: ExecutionRequest):
    """
    Executes the query after human approval and returns the final result.
    """
    if not request.approved:
        return {
            "message": "Query execution denied by user.",
            "result": None
        }

    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Get the state before resuming to update the query if it was modified
    current_state = graph.get_state(config)
    if not current_state:
         raise HTTPException(status_code=404, detail="Session not found. Please generate a query first.")
         
    # Update the state with the (potentially modified) query from the user
    current_state.values["query"] = request.query

    # Resume graph execution from the interruption point
    final_result = None
    try:
        for step in graph.stream(None, config, stream_mode="values"):
            final_result = step
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in graph execution: {str(e)}")
    
    if not final_result:
        raise HTTPException(status_code=500, detail="Failed to get a final result from the graph.")

    return {
        "message": "Query executed successfully.",
        "answer": final_result.get("answer"),
        "insight": final_result.get("insight"),
        "raw_result": final_result.get("structured_result"),
        "chart_png_base64": final_result.get("chart"),
    }

# To run this app, save it as main.py and run: uvicorn main:app --reload
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
