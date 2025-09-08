import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_experimental.utilities import PythonREPL
import pandas as pd
import uuid
import os
import io
import matplotlib.pyplot as plt
# Used for LangGraph
from typing import TypedDict, List, Dict, Any, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# --- 1. Define the State for the Graph ---
# This dictionary will hold the data that moves between the nodes of our graph.
class State(TypedDict):
    question: str
    query: str
    result: str
    structured_result: List[Dict[str, Any]] # To hold structured data for charts/tables
    insight: str
    answer: str
    chart_image: bytes | None # To hold the chart image in memory
    chart_code: str | None # To hold the Python code for the chart
    chart_type: str | None # To hold the chosen chart type
    x_col: str | None # To hold the chosen x-axis column
    visualization_reason: str | None # To hold the reason for the chart choice
    y_col: str | None # To hold the chosen y-axis column
    log: List[str] # To hold the processing steps
    error: str | None # To hold error messages for the conditional edge
    retry_count: int # To prevent infinite loops

# --- 2. Define the Nodes for the Graph ---

# st.cache_resource to initialize the LLM and DB connection once
@st.cache_resource
def get_llm():
    return ChatVertexAI(model="gemini-2.5-flash")

def get_db(project_id, dataset_id):
    return SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")

def write_query(state: State):
    """Generates a SQL query from the user's question."""
    log = state.get("log", [])
    log.append("✍️ Generating SQL query...")
    question = state["question"]
    project_id = st.session_state.project_id
    dataset_id = st.session_state.dataset_id
    
    llm = get_llm()
    db = get_db(project_id, dataset_id)

    template = """
    You are a Google BigQuery SQL expert.
    Based on the table schema below, write a SQL query that would answer the user's question.
    Use only Google BigQuery standard SQL syntax.
    
    **CRITICAL BIGQUERY RULES FOR DATES AND TIMESTAMPS:**
    1.  `TIMESTAMP_SUB` **CANNOT** be used with `YEAR`, `QUARTER`, or `MONTH`. It will cause an error.
    2.  To subtract years or months from a `TIMESTAMP` column, you **MUST** cast the column to a `DATE` first and use `DATE_SUB`.
        - **Correct Example:** `WHERE DATE(t.TGL_TRANSAKSI) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR)`
        - **Incorrect Example:** `WHERE t.TGL_TRANSAKSI >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 YEAR)`
    3.  For date/time formatting, use `FORMAT_DATE` or `FORMAT_TIMESTAMP`.
    4.  To extract parts of a date, use `EXTRACT`.

    Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    Table Schema: {schema}
    Question: {question}
    SQL Query:
    """
    prompt = PromptTemplate.from_template(template)
    
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
    
    return {"query": clean_query, "log": log}

def check_query(state: State):
    """Validates the SQL query for common mistakes."""
    log = state.get("log", [])
    log.append("🧐 Validating SQL query...")
    query = state["query"]
    project_id = st.session_state.project_id
    dataset_id = st.session_state.dataset_id

    llm = get_llm()
    db = get_db(project_id, dataset_id)

    system = """You are a Google BigQuery SQL expert. Double check the user's {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query to be compatible with Google BigQuery standard SQL.
If there are no mistakes, just reproduce the original query with no further commentary.

Output the final SQL query only."""
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "{query}")]
    ).partial(dialect=db.dialect)
    
    validation_chain = prompt | llm | StrOutputParser()
    validated_query_text = validation_chain.invoke({"query": query})
    clean_validated_query = validated_query_text.replace("```sql", "").replace("```", "").strip()

    return {"query": clean_validated_query, "log": log}

def execute_query(state: State):
    """Executes the SQL query and gets the result."""
    log = state.get("log", [])
    log.append("🚀 Executing SQL query...")
    query = state["query"]
    project_id = st.session_state.project_id
    dataset_id = st.session_state.dataset_id

    db = get_db(project_id, dataset_id)
    
    try:
        query_result_dicts = db._execute(query, fetch="all")
        return {"structured_result": query_result_dicts, "log": log, "error": None}
    except Exception as e:
        error_message = f"Query execution failed: {e}"
        return {"structured_result": [], "log": log + [error_message], "error": error_message}

def rewrite_query_on_error(state: State):
    """Takes the failed query and error, and asks the LLM to rewrite it."""
    log = state.get("log", [])
    log.append("🔧 Query failed. Attempting to fix...")
    
    query = state["query"]
    error = state["error"]
    retry_count = state.get("retry_count", 0)

    if retry_count >= 2: # Limit retries
        log.append("❌ Reached max retries. Stopping.")
        return {"log": log}

    llm = get_llm()
    
    rewrite_prompt = PromptTemplate.from_template(
        """
        You are a Google BigQuery SQL expert. The following SQL query failed with an error.
        Your task is to analyze the query and the error message, and rewrite the query to fix the issue.
        Pay close attention to the error message as it contains crucial clues.
        
        **CRITICAL BIGQUERY RULES FOR DATES AND TIMESTAMPS:**
        1.  `TIMESTAMP_SUB` **CANNOT** be used with `YEAR`, `QUARTER`, or `MONTH`. It will cause an error.
        2.  To subtract years or months from a `TIMESTAMP` column, you **MUST** cast the column to a `DATE` first and use `DATE_SUB`.
            - **Correct Example:** `WHERE DATE(t.TGL_TRANSAKSI) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR)`
            - **Incorrect Example:** `WHERE t.TGL_TRANSAKSI >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 YEAR)`

        Original Failed Query:
        {query}

        Error Message:
        {error}

        Corrected Google BigQuery SQL Query:
        """
    )
    rewriter_chain = rewrite_prompt | llm | StrOutputParser()
    corrected_query = rewriter_chain.invoke({"query": query, "error": error})
    clean_corrected_query = corrected_query.replace("```sql", "").replace("```", "").strip()

    return {
        "query": clean_corrected_query, 
        "log": log, 
        "error": None, # Clear the error for the next attempt
        "retry_count": retry_count + 1
    }

def choose_chart(state: State):
    """Chooses an appropriate chart type and columns based on the query result."""
    log = state.get("log", [])
    log.append("📊 Choosing visualization...")
    
    structured_result = state["structured_result"]
    
    if not structured_result or len(structured_result) < 2:
        log.append("No chart generated (not enough data).")
        return {"log": log, "chart_type": "table"}

    try:
        df = pd.DataFrame(structured_result)
        column_info = f"Columns: {', '.join(df.columns)}\nData Types:\n{df.dtypes.to_string()}"
        first_few_rows = df.head(3).to_string()
    except Exception:
         log.append("Could not process data for chart selection.")
         return {"log": log, "chart_type": "table"}

    llm = get_llm()

    chart_prompt = PromptTemplate.from_template(
        """
        You are a data visualization expert. Your task is to choose the best chart type 
        to visualize the data from a user's question and a SQL query result.

        1.  **Analyze the data**: Review the column names, data types, and a few sample rows.
        2.  **Determine the best chart type**: Choose from 'bar', 'line', 'pie', 'scatter', or 'table'.
        3.  **Select columns**: Identify the most appropriate column for the x-axis and y-axis.
            - For pie charts, the x-axis (`x_col`) should be the labels and the y-axis (`y_col`) should be the values.
            - If no chart is suitable, choose 'table'.
        4.  **Provide a reason**: Briefly explain why this visualization is a good choice.

        User's Question: {question}
        
        Data Information:
        {column_info}
        
        Sample Data:
        {first_few_rows}
        
        Return a JSON object with your choices. Example:
        {{
          "chart_type": "bar",
          "x_col": "category_column",
          "y_col": "value_column",
          "reason": "A bar chart is best to compare values across different categories."
        }}
        """
    )
    
    parser = JsonOutputParser()
    chart_chain = chart_prompt | llm | parser
    
    try:
        chart_details = chart_chain.invoke({
            "question": state["question"],
            "column_info": column_info,
            "first_few_rows": first_few_rows
        })
        log.append(f"Chart choice: {chart_details.get('chart_type')} - {chart_details.get('reason')}")
        return {
            "log": log,
            "chart_type": chart_details.get("chart_type"),
            "x_col": chart_details.get("x_col"),
            "y_col": chart_details.get("y_col"),
            "visualization_reason": chart_details.get("reason")
        }
    except Exception as e:
        log.append(f"Error choosing chart type: {e}")
        return {"log": log, "chart_type": "table"}
    
# --- Define Python REPL tool for chart execution ---
repl = PythonREPL()

@tool
def python_repl_tool(code: Annotated[str, "The Python code to execute to generate your chart."]):
    """Execute Python code using a Python REPL (Read-Eval-Print Loop).
    
    Args:
        code (str): The Python code to execute.
    Returns:
        str: The result of the executed code or an error message if execution fails.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    
    result_str = f"Successfully executed:\n```python\n{code}\n```\nStdout: {result}"
    return result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."


def generate_chart(state: State):
    """Generate chart code and image using LLM + Python REPL based on state."""
    log = state.get("log", [])
    log.append("📈 Generating chart code and visualization...")

    chart_type = state.get("chart_type", "table")
    x_col = state.get("x_col")
    y_col = state.get("y_col")
    structured_result = state.get("structured_result")

    if chart_type == "table" or not structured_result:
        log.append("No chart generated. Returning table view.")
        return {"chart_image": None, "chart_code": None, "log": log}

    df = pd.DataFrame(structured_result)

    llm = get_llm()
    chart_prompt = PromptTemplate.from_template(
    """
    You are a Python data visualization expert.
    Generate Python code that uses matplotlib (and optionally seaborn) 
    to plot a clear and visually appealing {chart_type} chart from the given DataFrame `df`.

    IMPORTANT RULES:
    1. DO NOT create or simulate data. Always use the provided `df` directly.
    2. Assume `df` is already defined and contains the real query result.
    3. x-axis column: {x_col}
       y-axis column: {y_col}
    4. Always import matplotlib.pyplot as plt (and seaborn if needed).
    5. Add a descriptive title, axis labels, and legend if applicable.
    6. Use `plt.tight_layout()` for spacing.
    7. End the code with:
       buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)

    DataFrame columns: {columns}

    Return only valid Python code. Do NOT include explanations or markdown.
    """
)

    chart_chain = chart_prompt | llm | StrOutputParser()
    chart_code = chart_chain.invoke({
        "chart_type": chart_type,
        "columns": list(df.columns),
        "x_col": x_col,
        "y_col": y_col
    })

    # Clean chart code
    chart_code = chart_code.replace("```python", "").replace("```", "").strip()

    # Try executing the code in a local sandbox
    try:
        buf = io.BytesIO()
        # Inject df into locals so code can use it
        local_vars = {"df": df, "io": io, "plt": plt, "pd": pd}
        exec(chart_code, {}, local_vars)
        buf = local_vars.get("buf", None)
        chart_image = buf.getvalue() if buf else None
    except Exception as e:
        log.append(f"⚠️ Chart execution failed: {e}")
        return {"chart_image": None, "chart_code": chart_code, "log": log}

    log.append("✅ Chart generated successfully.")
    return {"chart_image": chart_image, "chart_code": chart_code, "log": log}

def generate_insight(state: State):
    """Generates insight from the SQL query and its result."""
    log = state.get("log", [])
    log.append("💡 Generating insight...")
    query = state["query"]
    
    if not state.get("structured_result"):
        return {"insight": "No data was returned from the query.", "log": log}
    
    structured_result = state["structured_result"]
    result_str = str(structured_result)

    llm = get_llm()
    insight_prompt = PromptTemplate.from_template(
        """
        You are a data analyst. Given the following SQL query and its result,
        provide a brief, one or two-sentence insight into what the data reveals.
        Focus on the most important finding.

        SQL Query: {query}
        SQL Result: {result}
        Insight:
        """
    )
    insight_chain = insight_prompt | llm | StrOutputParser()
    insight_text = insight_chain.invoke({"query": query, "result": result_str})
    return {"insight": insight_text, "log": log}

def generate_answer(state: State):
    """Generates a natural language answer from the insight."""
    log = state.get("log", [])
    log.append("💬 Formulating final answer...")
    question = state["question"]
    insight = state["insight"]
    
    llm = get_llm()
    answer_prompt = PromptTemplate.from_template(
        """
        Given the user's question and the data insight, provide a concise, natural language final answer.
        Directly answer the question based on the insight provided.

        Question: {question}
        Insight: {insight}
        Final Answer:
        """
    )
    answer_chain = answer_prompt | llm | StrOutputParser()
    final_answer = answer_chain.invoke({"question": question, "insight": insight})
    return {"answer": final_answer, "log": log}

def should_continue(state: State):
    """Determines the next step after query execution based on whether an error occurred."""
    if state.get("error"):
        if state.get("retry_count", 0) >= 2:
            return "end"
        return "rewrite_query_on_error"
    else:
        return "choose_visualization"

# --- 3. Build and Compile the Graph ---

@st.cache_resource
def get_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("write_query", write_query)
    graph_builder.add_node("check_query", check_query)
    graph_builder.add_node("execute_query", execute_query)
    graph_builder.add_node("rewrite_query_on_error", rewrite_query_on_error)
    graph_builder.add_node("choose_chart", choose_chart)
    graph_builder.add_node("generate_chart", generate_chart)
    graph_builder.add_node("generate_insight", generate_insight)
    graph_builder.add_node("generate_answer", generate_answer)

    graph_builder.add_edge(START, "write_query")
    graph_builder.add_edge("write_query", "check_query")
    graph_builder.add_edge("check_query", "execute_query")
    
    graph_builder.add_conditional_edges(
        "execute_query",
        should_continue,
        {
            "rewrite_query_on_error": "rewrite_query_on_error",
            "choose_visualization": "choose_chart",
            "end": END
        }
    )
    graph_builder.add_edge("rewrite_query_on_error", "execute_query")
    graph_builder.add_edge("choose_chart", "generate_chart")
    graph_builder.add_edge("generate_chart", "generate_insight")
    graph_builder.add_edge("generate_insight", "generate_answer")
    graph_builder.add_edge("generate_answer", END)
    
    return graph_builder.compile()

# --- 4. Streamlit Frontend ---

st.set_page_config(page_title="BigQuery Q&A Assistant 📊", layout="wide")
st.title("BigQuery Q&A Assistant 📊")

with st.sidebar:
    st.header("Configuration")
    st.info("Enter your Google Cloud Project ID and the BigQuery Dataset ID you want to query.")
    
    if 'PROJECT_ID' in os.environ:
        st.session_state.project_id = os.environ['PROJECT_ID']
    if 'DATASET_ID' in os.environ:
        st.session_state.dataset_id = os.environ['DATASET_ID']

    project_id = st.text_input("GCP Project ID", value=st.session_state.get("project_id", "eikon-dev-data-team"), key="project_id")
    dataset_id = st.text_input("BigQuery Dataset ID", value=st.session_state.get("dataset_id", "ancoldbdufanlive"), key="dataset_id")

    if not project_id or not dataset_id:
        st.warning("Please provide both Project ID and Dataset ID.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "results" in message and message["results"]["query"] != "N/A":
            with st.expander("🔍 Lihat Detail"):
                st.code(message["results"]["query"], language="sql")
                if message["results"].get("visualization_reason"):
                    st.info(f"**Visualization Reason:** {message['results']['visualization_reason']}")
                if message["results"]["chart"] is not None:
                    st.image(message["results"]["chart"], caption="Generated Chart")
                if message["results"].get("chart_code"):
                    with st.expander("🐍 Show Python Code for Chart"):
                        st.code(message["results"]["chart_code"], language="python")
                if message["results"]["data"]:
                    st.dataframe(pd.DataFrame(message["results"]["data"]))

if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        graph = get_graph()
        initial_state = {"question": prompt, "log": [], "retry_count": 0}
        
        status_placeholder = st.empty()
        final_state = {}

        # This expander will hold the step-by-step results
        with st.expander("🔍 Lihat Detail", expanded=True) as details_expander:
            query_placeholder = st.empty()
            reason_placeholder = st.empty()
            chart_placeholder = st.empty()
            code_placeholder = st.empty()
            data_placeholder = st.empty()

        # The final answer will appear here, outside the expander
        answer_placeholder = st.empty()
        
        for chunk in graph.stream(initial_state):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            final_state.update(node_output)

            # Update status text
            if "log" in final_state and final_state["log"]:
                last_log = final_state["log"][-1]
                if "failed" in last_log.lower() or "stopping" in last_log.lower():
                    status_placeholder.error(last_log)
                    break
                else:
                    status_placeholder.markdown(last_log)
            
            # Check the output of the current node and update the UI placeholders
            if "query" in node_output and node_output["query"]:
                query_placeholder.code(node_output["query"], language="sql")

            if "structured_result" in node_output and node_output["structured_result"]:
                data_placeholder.dataframe(pd.DataFrame(node_output["structured_result"]))
            
            if "visualization_reason" in node_output and node_output["visualization_reason"]:
                reason_placeholder.info(f"**Visualization Reason:** {node_output['visualization_reason']}")
            
            if "chart_image" in node_output and node_output["chart_image"]:
                chart_placeholder.image(node_output["chart_image"], caption="Generated Chart")

            if "chart_code" in node_output and node_output["chart_code"]:
                with code_placeholder.expander("🐍 Show Python Code for Chart"):
                    st.code(node_output["chart_code"], language="python")
        
        if "failed" in final_state.get("log", [""])[-1].lower() or "stopping" in final_state.get("log", [""])[-1].lower():
             st.stop()
        
        status_placeholder.markdown("💬 Formulating final answer...")

        # --- Final Answer Streaming ---
        llm = get_llm()
        answer_prompt = PromptTemplate.from_template(
            """
            Given the user's question and the data insight, provide a concise, natural language final answer.
            Directly answer the question based on the insight provided.

            Question: {question}
            Insight: {insight}
            Final Answer:
            """
        )
        answer_chain = answer_prompt | llm | StrOutputParser()

        insight = final_state.get("insight", "I couldn't generate an insight from the data.")
        
        # Use the placeholder to write the stream
        answer_stream = answer_chain.stream({
            "question": prompt,
            "insight": insight
        })
        answer = answer_placeholder.write_stream(answer_stream)
        
        # All results are now available, so clear the final status message
        status_placeholder.empty()

        # Save the complete results to session state for history
        results_to_store = {
            "query": final_state.get("query", "N/A"),
            "data": final_state.get("structured_result", []),
            "chart": final_state.get("chart_image", None),
            "chart_code": final_state.get("chart_code", None),
            "visualization_reason": final_state.get("visualization_reason", None)
        }

        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, 
            "results": results_to_store
        })