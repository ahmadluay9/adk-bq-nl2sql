import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
import pandas as pd
import uuid
import os

# Used for LangGraph
from typing import TypedDict, List, Dict, Any
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
    log: List[str] # To hold the processing steps

# --- 2. Define the Nodes for the Graph ---

# We'll use st.cache_resource to initialize the LLM and DB connection once
@st.cache_resource
def get_llm():
    return ChatVertexAI(model="gemini-2.5-flash") # Using a more recent model

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
        query_result_str = str(query_result_dicts)
        return {"result": query_result_str, "structured_result": query_result_dicts, "log": log}
    except Exception as e:
        st.error(f"Query execution failed: {e}")
        # Return empty results to prevent downstream errors
        return {"result": "[]", "structured_result": [], "log": log}

def generate_chart(state: State):
    """Generates an appropriate chart from the query result if possible."""
    log = state.get("log", [])
    log.append("📊 Generating chart...")
    data = state.get("structured_result", [])

    if not data or len(data[0].keys()) < 2:
        return {"chart_image": None, "log": log}

    try:
        import matplotlib.pyplot as plt
        import io
        
        df = pd.DataFrame(data)
        # Attempt to convert date/time like columns to datetime objects
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # errors='coerce' will turn unparseable dates into NaT (Not a Time)
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except Exception:
                    pass # Keep as object if conversion fails
        
        headers = list(df.columns)
        
        # --- Axis Selection Logic ---
        datetime_cols = [h for h in headers if pd.api.types.is_datetime64_any_dtype(df[h])]
        categorical_cols = [h for h in headers if df[h].dtype == 'object']
        numeric_cols = [h for h in headers if pd.api.types.is_numeric_dtype(df[h]) and df[h].nunique() > 1]
        
        # Combine categorical and datetime for x-axis candidates
        x_candidates = datetime_cols + categorical_cols
        y_candidates = numeric_cols

        x_col, y_col = None, None

        if len(x_candidates) >= 1 and len(y_candidates) >= 1:
            x_col = x_candidates[0]
            y_col = y_candidates[0]
        elif len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            nunique1, nunique2 = df[col1].nunique(), df[col2].nunique()
            x_col, y_col = (col1, col2) if nunique1 < nunique2 else (col2, col1)
        else:
            x_col, y_col = headers[0], headers[1]
        
        if not x_col or not y_col:
            return {"chart_image": None, "log": log}
        
        # --- Chart Type Selection Logic ---
        chart_type = 'bar' # Default
        
        if x_col in datetime_cols:
            chart_type = 'line'
            df = df.sort_values(by=x_col)
        elif 2 <= df[x_col].nunique() <= 7:
            chart_type = 'pie'
            
        plt.figure(figsize=(10, 6))
        
        # --- Plotting based on selected chart type ---
        if chart_type == 'line':
            plt.plot(df[x_col], df[y_col], marker='o', linestyle='-')
            plt.ylabel(y_col.replace('_', ' ').title())
        
        elif chart_type == 'pie':
            plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%', startangle=90)
            plt.ylabel('') # Pie charts don't need a y-label
        
        else: # Default to bar chart
            df_chart = df.nlargest(20, y_col) if len(df) > 20 and y_col in numeric_cols else df
            plt.bar(df_chart[x_col].astype(str), df_chart[y_col])
            plt.ylabel(y_col.replace('_', ' ').title())

        # Common plot settings
        plt.xlabel(x_col.replace('_', ' ').title())
        plt.title(f"{y_col.replace('_', ' ').title()} by {x_col.replace('_', ' ').title()}")
        if chart_type != 'pie':
             plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return {"chart_image": buf.getvalue(), "log": log}
    except Exception as e:
        st.warning(f"Could not generate chart: {e}")
        return {"chart_image": None, "log": log}


def generate_insight(state: State):
    """Generates insight from the SQL query and its result."""
    log = state.get("log", [])
    log.append("💡 Generating insight...")
    query = state["query"]
    result = state["result"]
    
    if not state.get("structured_result"):
        return {"insight": "No data was returned from the query.", "log": log}
    
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
    insight_text = insight_chain.invoke({"query": query, "result": result})
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

# --- 3. Build and Compile the Graph ---

# Use st.cache_resource to ensure the graph is built only once
@st.cache_resource
def get_graph():
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
    
    # We don't need memory/checkpoints for this Streamlit app's flow
    return graph_builder.compile()


# --- 4. Streamlit Frontend ---

st.set_page_config(page_title="BigQuery Q&A Assistant 📊", layout="wide")
st.title("BigQuery Q&A Assistant 📊")

# --- Sidebar for Configuration ---
with st.sidebar:
    st.header("Configuration")
    st.info("Enter your Google Cloud Project ID and the BigQuery Dataset ID you want to query.")
    
    if 'PROJECT_ID' in os.environ:
        st.session_state.project_id = os.environ['PROJECT_ID']
    if 'DATASET_ID' in os.environ:
        st.session_state.dataset_id = os.environ['DATASET_ID']

    # Use the provided values as default placeholders
    project_id = st.text_input("GCP Project ID", value=st.session_state.get("project_id", "eikon-dev-data-team"), key="project_id")
    dataset_id = st.text_input("BigQuery Dataset ID", value=st.session_state.get("dataset_id", "ancoldbdufanlive"), key="dataset_id")

    if not project_id or not dataset_id:
        st.warning("Please provide both Project ID and Dataset ID.")
        st.stop()

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None

# --- Display existing chat messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "results" in message:
            with st.expander("🔍 Lihat Detail"):
                st.code(message["results"]["query"], language="sql")
                if message["results"]["chart"] is not None:
                    st.image(message["results"]["chart"], caption="Generated Chart")
                st.dataframe(pd.DataFrame(message["results"]["data"]))


# --- Main Logic ---
if prompt := st.chat_input("Ask a question about your data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        graph = get_graph()
        initial_state = {"question": prompt, "log": []}
        
        status_placeholder = st.empty()
        final_state = {}

        # Stream the graph execution to show live progress
        for chunk in graph.stream(initial_state):
            # The chunk is a dictionary with a single key, the node name
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]
            final_state.update(node_output) # Accumulate state from each node's output

            # Update the status message dynamically
            if "log" in final_state and final_state["log"]:
                status_placeholder.markdown(final_state["log"][-1])

        # Now that the graph has finished, display the final results
        answer = final_state.get("answer", "I couldn't generate an answer.")
        st.markdown(answer)
        
        # Prepare results for display and storing in history
        results_to_store = {
            "query": final_state.get("query", "N/A"),
            "data": final_state.get("structured_result", []),
            "chart": final_state.get("chart_image", None)
        }
        
        with st.expander("🔍 Lihat Detail"):
            st.code(results_to_store["query"], language="sql")
            if results_to_store["chart"]:
                st.image(results_to_store["chart"], caption="Generated Chart")
            st.dataframe(pd.DataFrame(results_to_store["data"]))

        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer, # Storing only the final answer for a cleaner history
            "results": results_to_store
        })