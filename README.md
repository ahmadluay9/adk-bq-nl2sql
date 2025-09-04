# BigQuery Natural Language Querying Assistant
This project provides an interactive web application built with Streamlit and LangChain that allows users to ask questions about their BigQuery data in natural language. The application translates the user's question into a SQL query, executes it against the specified BigQuery dataset, and presents the answer in natural language, along with a data table and an automatically generated chart.

## Key Features
- **Natural Language to SQL**: Translates plain English questions into valid BigQuery SQL queries.
- **Interactive Chat Interface**: A user-friendly, chat-based UI for asking questions and viewing results.
- **Automatic Chart Generation**: Intelligently selects the best chart type (Line, Bar, or Pie) to visualize the query results.
- **Dynamic Status Updates**: Provides real-time feedback as it processes a request through various stages (generating query, executing, charting, etc.).
- **Data Insights**: Generates a concise, one-sentence summary of the key finding from the data.
- **Configuration via UI**: Allows users to specify their GCP Project ID and BigQuery Dataset ID directly in the application.

## 📂 Project Structure
```
adk-bq-nl2sql/
├── README.md
├── .gitignore
├── cloudbuild.yaml
├── Dockerfile
├── app.py
├── notebook.ipynb
└── requirements.txt
```

- `app.py`: The main Streamlit application file containing all the frontend and backend logic.
- `requirements.txt`: A list of all the Python packages required to run the project.
- `notebook.ipynb`: A Jupyter notebook for development, testing, and exploration.
- `Dockerfile`: Instructions to build a Docker container for the application.
- `cloudbuild.yaml`: Configuration for deploying the application using Google Cloud Build.
- `.gitignore`: Specifies which files and directories to ignore for version control.
- `README.md`: This documentation file.

## Setup and Installation
1. Prerequisites
Python 3.8+
A Google Cloud Platform (GCP) project with the BigQuery API enabled.
A BigQuery dataset you want to query.
gcloud CLI installed and configured.
2. Clone the Repository
git clone <your-repository-url>
cd adk-bq-nl2sql


3. Install Dependencies
It's recommended to use a virtual environment.
```
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

4. Authenticate with Google Cloud
You need to authenticate your local environment so the application has permission to access your BigQuery data.
```
gcloud auth application-default login
```

This command will open a browser window for you to log in to your Google account.
## Running the Application
Once the setup is complete, you can run the Streamlit application with the following command:
```
streamlit run app.py
```

Your web browser should automatically open a new tab with the application running. If not, the terminal will provide a local URL (usually `http://localhost:8501`) that you can visit.
## Usage
1. Open the application in your browser.
2. Use the sidebar to enter your GCP Project ID and the BigQuery Dataset ID. The fields are pre-filled with example values.
3. Once configured, use the chat input at the bottom of the page to ask a question about your data (e.g., "What were the total sales per product last month?").
4. The assistant will show its progress and then display the final answer, along with an expandable "Lihat Detail" section containing the generated SQL, a chart, and the raw data table.
