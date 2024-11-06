import streamlit as st
from streamlit_option_menu import option_menu
import psycopg2
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from crewai import Crew, Process
from agents import data_analyst_agent, visualisation_agent, data_extraction_agent
from tasks import search_task, visualize_task
import os

load_dotenv()

st.set_page_config(page_title="IntelliHealthAI", page_icon=":material/smart_toy:")


def init_database() -> SQLDatabase:
    POSTGRES_USER='intellihealth'              
    POSTGRES_PASSWORD='intellihealth'                         
    POSTGRES_DB='intellihealth'
    # postgresql+psycopg2://user:password@host:port/dbname
    
    DATABASE_URL= f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@216.48.179.123:5432/{POSTGRES_DB}' 
    
    return SQLDatabase.from_uri(DATABASE_URL)

def get_sql_chain(db):
    template = """
    You are a Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the comapany's data.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Tell me about Patient data
    PostgresSQL Query: SELECT * FROM "Patient";
    
    Question: Tell me about Sarah Brown who enquired for medication refill
    PostgresSQL Query: SELECT * FROM "Patient" WHERE "Name" = 'Sarah Brown' AND "Enquiry" = 'Medication refill'
    
    Question: How many John Doe in the data?
    PostgresSQL Query: SELECT COUNT(*) FROM "Patient" WHERE "Name" = 'John Doe';
    
    Question: How many leaves were taken on 4th September 2024?
    PostgresSQL Query: SELECT COUNT(*) FROM "Employee" WHERE "Date of Leave" = '4th September 2024';
    
    Question: Who took leave for a family emergency?
    PostgresSQL Query: SELECT "Name" FROM "Employee" WHERE "Reason for Leave" = 'Family emergency';
    
    Question: What are the work hours scheduled for employees who took leave on 17th September 2024?
    PostgresSQL Query: SELECT "Name", "Schedule of Work Hours" FROM "Employee" WHERE "Date of Leave" = '17th September 2024';
    
    Question: How many patients were admitted on 9th September 2024?
    PostgresSQL Query: SELECT COUNT(*) FROM "Management" WHERE "Admission Date" = '9/9/2024';
    
    Question: What is the total treatment cost for patients diagnosed with COVID-19?
    PostgresSQL Query: SELECT SUM("Treatment Cost") FROM "Management" WHERE "Diagnosis" = 'COVID-19';
    
    Question: List the names of patients who have no insurance.
    PostgresSQL Query: SELECT DISTINCT "Name" FROM "Management" WHERE "Insurance" = 'None';
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # llm = ChatOpenAI(model="gpt-4")
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the company's data.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    print("prompt ",prompt)
    
    # llm = ChatOpenAI(model="gpt-4")
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })


with st.sidebar:
        selected = option_menu(
            menu_title=None,  # No menu title
            options=["Home", "Setup"],  # Menu options
            icons=["house", "gear"],  # Optional icons
            menu_icon="cast",  # Optional menu icon
            default_index=0,  # Default selected index
        )

database_list = os.environ['POSTGRES_DB'].split(",")

st.sidebar.success("Connected to PostgresDB successfully ")

crew = Crew(
    agents=[data_extraction_agent],
    tasks=[search_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

st.sidebar.success("Crew Agents Initialized. ")

if selected == "Home":
    st.title("IntelliHealth-AI")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm Pro Assist! Ask me anything"),
        ]
    
    # Initialize the database connection once and store it in session state
    if "db" not in st.session_state:
        with st.spinner("Connecting to database..."):
            db = init_database()
            st.session_state.db = db
            st.success("Connected to database!")
            
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
                
    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        
        with st.chat_message("Human"):
            st.markdown(user_query)
            
        with st.chat_message("AI"):
            
            result = crew.kickoff(inputs={'user_query': user_query, 'database_list': database_list, 'database_count': len(database_list)})

            print(result)
            
        st.session_state.chat_history.append(AIMessage(content=result))


if selected == "Setup":
    
    # Connect to the PostgreSQL database
    conn = psycopg2.connect(
        dbname="intellihealth",
        user="intellihealth",
        password="intellihealth",
        host="216.48.179.123",
        port="5432"
    )
    
    # Create a DataFrame from the SQL query
    df = pd.read_sql_query('SELECT * FROM "Patient"', conn)
    st.subheader("This is a sample setup data from Postgres")
    st.write(df)

    # Close the connection
    conn.close()