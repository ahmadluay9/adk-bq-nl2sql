import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain.schema.runnable import RunnablePassthrough
from langchain_postgres import PGVector
from langchain_core.documents import Document
import pandas as pd
import uuid
import os
import io
import matplotlib.pyplot as plt
from dotenv import load_dotenv
# Used for LangGraph
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv() # Load environment variables from .env file

# --- 1. Define the State for the Graph ---
# This dictionary will hold the data that moves between the nodes of our graph.
class State(TypedDict):
    question: str
    query: str
    structured_result: List[Dict[str, Any]] # To hold structured data for charts/tables
    insight: str
    answer: str
    chart_image: bytes | None # To hold the chart image in memory
    chart_code: str | None # To hold the Python code for the chart
    chart_type: str | None # To hold the chosen chart type
    visualization_reason: str | None # To hold the reason for the chart choice
    log: List[str] # To hold the processing steps
    error: str | None # To hold error messages for the conditional edge
    retry_count: int # To prevent infinite loops

# --- 2. Define the Nodes for the Graph ---
# st.cache_resource to initialize the LLM and DB connection once
@st.cache_resource
def get_llm():
    return ChatVertexAI(model=os.environ.get("LLM_MODEL_NAME"))

def get_db(project_id, dataset_id):
    return SQLDatabase.from_uri(f"bigquery://{project_id}/{dataset_id}")

# --- Dynamic Few-Shot Examples Setup ---
examples = [
    {
        "question": "berapa jumlah tiket yang terjual tiap tahunnya dari tahun 2012-2015?",
        "query": "SELECT EXTRACT(YEAR FROM TGL_TRANSAKSI) AS transaction_year, SUM(JML_TIKET_BYR + JML_TIKET_GRATIS) AS total_tickets_sold FROM `tt_tiketing` WHERE EXTRACT(YEAR FROM TGL_TRANSAKSI) BETWEEN 2012 AND 2015 GROUP BY transaction_year ORDER BY transaction_year;"
    },
    {
        "question": "berapa penjualan tiket per hari di bulan maret 2013?",
        "query": "SELECT FORMAT_DATE('%Y-%m-%d', DATE(TGL_TRANSAKSI)) AS sales_date, SUM(JML_TIKET_BYR) AS total_paid_tickets_sold FROM `tt_tiketing` WHERE EXTRACT(YEAR FROM TGL_TRANSAKSI) = 2013 AND EXTRACT(MONTH FROM TGL_TRANSAKSI) = 3 GROUP BY sales_date ORDER BY sales_date;"
    },
    {
        "question": "buatkan analisa pendapatan tiket dunia fantasi dari tahun 2012-2014",
        "query": "SELECT EXTRACT(YEAR FROM TGL_TRANSAKSI) AS tahun_transaksi,SUM(TTL_BAYAR) AS total_pendapatan FROM tt_tiketing WHERE EXTRACT(YEAR FROM TGL_TRANSAKSI) BETWEEN 2012 AND 2014 GROUP BY tahun_transaksi ORDER BY tahun_transaksi;"
    },
]

@st.cache_resource
def get_example_selector():
    # --- PGVector Connection ---
    db_user = os.environ.get("POSTGRES_USER")
    db_password = os.environ.get("POSTGRES_PASSWORD")
    db_host = os.environ.get("POSTGRES_HOST")
    # db_host = os.environ.get("POSTGRES_LOCALHOST")
    db_port = os.environ.get("POSTGRES_PORT")
    db_name = os.environ.get("POSTGRES_DB")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        st.error(
            "Database connection variables are not fully set. "
            "Please create a `.env` file with `POSTGRES_USER`, `POSTGRES_PASSWORD`, "
            "`POSTGRES_HOST`, `POSTGRES_PORT`, and `POSTGRES_DB`."
        )
        st.stop()

    CONNECTION_STRING = f"postgresql+psycopg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    COLLECTION_NAME = "sql_examples_v3" # Changed version to avoid conflicts

    embeddings = VertexAIEmbeddings(model_name=os.environ.get("EMBEDDING_MODEL_NAME"))

    # Create Document objects for PGVector. The question is the content to be searched against.
    documents = [
        Document(
            page_content=ex["question"],
            metadata=ex  # Store the full example dict in metadata
        ) for ex in examples
    ]
    
    try:
        # Initialize PGVector. This will create the collection and embed the documents
        # if it's the first time, or connect to the existing collection.
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            pre_delete_collection=True # Start fresh during development
        )
    except Exception as e:
        st.error(f"Failed to connect to or initialize PGVector. Please check your connection string and database setup. Error: {e}")
        st.stop()

    # The selector will use the PGVector store for similarity searches
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=3,
    )

    # This prompt formats each selected example before it's added to the main prompt
    example_prompt = PromptTemplate.from_template("User input: {question}\nSQL query: {query}")
    
    return example_selector, example_prompt

# def write_query(state: State):
#     """Generates a SQL query from the user's question."""
#     log = state.get("log", [])
#     log.append("✍️ Generating SQL query...")
#     question = state["question"]
#     project_id = st.session_state.project_id
#     dataset_id = st.session_state.dataset_id
    
#     llm = get_llm()
#     db = get_db(project_id, dataset_id)

#     template = """
#     You are a Google BigQuery SQL expert.
#     Based on the table schema below, write a SQL query that would answer the user's question.
#     Use only Google BigQuery standard SQL syntax.
    
#     **CRITICAL BIGQUERY RULES FOR DATES AND TIMESTAMPS:**
#     1.  `TIMESTAMP_SUB` **CANNOT** be used with `YEAR`, `QUARTER`, or `MONTH`. It will cause an error.
#     2.  To subtract years or months from a `TIMESTAMP` column, you **MUST** cast the column to a `DATE` first and use `DATE_SUB`.
#         - **Correct Example:** `WHERE DATE(t.TGL_TRANSAKSI) >= DATE_SUB(CURRENT_DATE(), INTERVAL 3 YEAR)`
#         - **Incorrect Example:** `WHERE t.TGL_TRANSAKSI >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 3 YEAR)`
#     3.  For date/time formatting, use `FORMAT_DATE` or `FORMAT_TIMESTAMP`.
#     4.  To extract parts of a date, use `EXTRACT`.

#     Do not make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

#     Table Schema: {schema}
#     Question: {question}
#     SQL Query:
#     """
#     prompt = PromptTemplate.from_template(template)
    
#     def get_schema(_):
#         return db.get_table_info()

#     sql_query_chain = (
#         RunnablePassthrough.assign(schema=get_schema)
#         | prompt
#         | llm.bind(stop=["\nSQLResult:"])
#         | StrOutputParser()
#     )

#     generated_query = sql_query_chain.invoke({"question": question})
#     clean_query = generated_query.replace("```sql", "").replace("```", "").strip()
    
#     return {"query": clean_query, "log": log}

def write_query(state: State):
    """Generates a SQL query from the user's question using dynamic few-shot examples."""
    log = state.get("log", [])
    log.append("✍️ Generating SQL query...")
    question = state["question"]
    project_id = st.session_state.project_id
    dataset_id = st.session_state.dataset_id
    
    llm = get_llm()
    db = get_db(project_id, dataset_id)
    
    example_selector, example_prompt = get_example_selector()

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix="You are a Google BigQuery SQL expert. Given an input question, create a syntactically correct Google BigQuery query to run.\n\nHere is the relevant table info: {schema}\n\nBelow are a number of examples of questions and their corresponding SQL queries.",
        suffix="User input: {question}\nSQL query: ",
        input_variables=["question", "schema"],
    )

    def get_schema(_):
        return db.get_table_info()

    sql_query_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | few_shot_prompt
        | llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )

    generated_query = sql_query_chain.invoke({"question": question})
    clean_query = generated_query.replace("```sql", "").replace("```", "").strip()
    
    return {"query": clean_query, "log": log}

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

    if retry_count >= 5: # Limit retries
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

def generate_chart_with_choice(state: State):
    """Single LLM call: choose chart type, columns, and generate chart code & image."""
    log = state.get("log", [])
    log.append("🤖 Choosing chart type & generating visualization in one step...")

    structured_result = state.get("structured_result")
    if not structured_result or len(structured_result) < 2:
        log.append("No chart generated (not enough data).")
        return {"chart_type": "table", "chart_image": None, "chart_code": None, "log": log}

    df = pd.DataFrame(structured_result)
    column_info = f"Columns: {', '.join(df.columns)}\nData Types:\n{df.dtypes.to_string()}"
    first_few_rows = df.head(3).to_string()

    llm = get_llm()

    chart_prompt = PromptTemplate.from_template(
        """
        You are a Python data visualization expert. 
        Based on the data and the user's question, do the following in one response:

        1. **Choose chart type**: Decide between 'bar', 'line', 'pie', 'scatter', or 'table'.
        2. **Pick columns**: Select the most appropriate `x_col` and `y_col`.  
           - For pie: x_col = labels, y_col = values.
           - If no chart is suitable, use 'table'.
        3. **Explain reasoning**: Brief one-sentence reason.
        4. **Generate Python code**:  
           - Use `df` directly (already defined). DO NOT simulate or create data.  
           - Use matplotlib (and seaborn if useful).  
           - Apply chosen chart type, with x_col on x-axis, y_col on y-axis.  
           - Add title, labels, legend (if needed).  
           - Use `plt.tight_layout()`.  
           - Save image to buffer with:
             ```python
             buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
             ```

        User's Question: {question}

        Data Information:
        {column_info}

        Sample Data:
        {first_few_rows}

        Return JSON ONLY in the following format:
        {{
          "chart_type": "<bar|line|pie|scatter|table>",
          "x_col": "<column_name or null>",
          "y_col": "<column_name or null>",
          "reason": "<short reason>",
          "code": "<valid Python code string>"
        }}
        """
    )

    parser = JsonOutputParser()
    chart_chain = chart_prompt | llm | parser

    try:
        result = chart_chain.invoke({
            "question": state["question"],
            "column_info": column_info,
            "first_few_rows": first_few_rows
        })

        chart_type = result.get("chart_type", "table")
        chart_code = result.get("code", "")

        if chart_type == "table":
            log.append("No suitable chart. Returning table view.")
            return {"chart_type": "table", "chart_image": None, "chart_code": None, "log": log}

        # Clean and execute code
        chart_code = chart_code.replace("```python", "").replace("```", "").strip()

        buf = io.BytesIO()
        local_vars = {"df": df, "io": io, "plt": plt, "pd": pd}
        exec(chart_code, {}, local_vars)
        buf = local_vars.get("buf", None)
        chart_image = buf.getvalue() if buf else None

        log.append(f"✅ Chart generated successfully: {chart_type} - {result.get('reason')}")
        return {
            "chart_type": chart_type,
            "x_col": result.get("x_col"),
            "y_col": result.get("y_col"),
            "visualization_reason": result.get("reason"),
            "chart_image": chart_image,
            "chart_code": chart_code,
            "log": log
        }

    except Exception as e:
        log.append(f"⚠️ Failed: {e}")
        return {"chart_type": "table", "chart_image": None, "chart_code": None, "log": log}
    
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

def should_continue(state: State):
    """Determines the next step after query execution based on whether an error occurred."""
    if state.get("error"):
        if state.get("retry_count", 0) >= 2:
            return "end"
        return "rewrite_query_on_error"
    else:
        return "generate_chart_with_choice"
# --- 3. Build and Compile the Graph ---

@st.cache_resource
def get_graph():
    graph_builder = StateGraph(State)
    graph_builder.add_node("write_query", write_query)
    graph_builder.add_node("execute_query", execute_query)
    graph_builder.add_node("rewrite_query_on_error", rewrite_query_on_error)
    graph_builder.add_node("generate_chart_with_choice", generate_chart_with_choice)
    graph_builder.add_node("generate_insight", generate_insight)

    graph_builder.add_edge(START, "write_query")
    graph_builder.add_edge("write_query", "execute_query")
    
    graph_builder.add_conditional_edges(
        "execute_query",
        should_continue,
        {
            "rewrite_query_on_error": "rewrite_query_on_error",
            "generate_chart_with_choice": "generate_chart_with_choice",
            "end": END
        }
    )

    graph_builder.add_edge("rewrite_query_on_error", "execute_query")
    graph_builder.add_edge("rewrite_query_on_error", "execute_query")
    graph_builder.add_edge("generate_chart_with_choice", "generate_insight")
    graph_builder.add_edge("generate_insight", END)
    
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
                    with st.expander("Show Python Code for Chart"):
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
                with code_placeholder.expander("Show Python Code for Chart"):
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