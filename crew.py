from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from crewai import Crew, Process
from agents import data_analyst_agent, visualisation_agent, data_extraction_agent
from tasks import search_task, visualize_task
import os

load_dotenv()

# llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.1-70b-versatile", temperature=0)

st.set_page_config(page_title="ProAssist", page_icon=":material/smart_toy:")


def init_database() -> SQLDatabase:
    POSTGRES_USER=os.environ['POSTGRES_USER']
    POSTGRES_PASSWORD=os.environ['POSTGRES_PASSWORD']                         
    POSTGRES_DB=os.environ['POSTGRES_DB']
    POSTGRES_HOST=os.environ['POSTGRES_HOST']
    # postgresql+psycopg2://user:password@host:port/dbname
    
    DATABASE_URL= f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}' 
    
    return SQLDatabase.from_uri(DATABASE_URL)

def init_lis_database() -> SQLDatabase:
    POSTGRES_USER=os.environ['POSTGRES_USER']
    POSTGRES_PASSWORD=os.environ['POSTGRES_PASSWORD']                         
    POSTGRES_DB=os.environ['POSTGRES_LIS_DB']
    POSTGRES_HOST=os.environ['POSTGRES_HOST']
    # postgresql+psycopg2://user:password@host:port/dbname
    
    DATABASE_URL= f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}' 
    
    return SQLDatabase.from_uri(DATABASE_URL)

def get_sql_chain(db):
    template = """
    You are a Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the comapany's data.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
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

def get_response(user_query: str, db: SQLDatabase):
    
    template = """
    You are Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the company's data.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
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
    })


# Get the SQL Query first
user_query = st.text_input("Ask Anything") # Give me the details about Harper'

if user_query:
    db = init_database()
    lis_db = init_lis_database()
    # # st.info(lis_db.get_table_info())
    # sql_chain = get_sql_chain(db)
    # sql_query = sql_chain.invoke({
    #                 "question": user_query
    #             })

    # st.write(sql_query)
    
    # ai_response = get_response(user_query, db)
    # st.write(ai_response)


    # Run the query in postgres and get the relavant data and visualize
    crew = Crew(
        agents=[data_extraction_agent],
        tasks=[search_task],
        process=Process.sequential,
        memory=True,
        cache=True,
        max_rpm=100,
        share_crew=True
    )

    st.info("Starting the crew! ...")
    # Start the execution of the crew with user input
    result = crew.kickoff(inputs={'user_query': user_query, 'schema': db.get_table_info()})

    st.success("Finished execution of the Crew!")
    
    # Display results in Streamlit app
    st.write(result)
    st.write(result.raw)




# from crewai import Crew, Process
# from tools import db_search_tool
# from agents import data_analyst_agent, visualisation_agent
# from tasks import search_task, visualize_task
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import streamlit as st
# import os

# load_dotenv()

# llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),model="llama-3.1-70b-versatile", temperature=0)

# st.set_page_config(page_title="ProAssist", page_icon=":material/smart_toy:")


# # # Building the Crew
# # crew = Crew(
# #     agents=[data_analyst_agent, visualisation_agent],
# #     tasks=[search_task, visualize_task],
# #     process=Process.sequential,
# #     memory=True,
# #     cache=True,
# #     max_rpm=100,
# #     share_crew=True
# # )

# # crew = Crew(
# #     agents=[visualisation_agent],
# #     tasks=[visualize_task],
# #     process=Process.sequential,
# #     memory=True,
# #     cache=True,
# #     max_rpm=100,
# #     share_crew=True
# # )

# crew = Crew(
#     agents=[data_analyst_agent],
#     tasks=[search_task],
#     process=Process.sequential,
#     memory=True,
#     cache=True,
#     max_rpm=100,
#     share_crew=True
# )

# query = st.text_input("Ask Anything") # Give me the details about Harper'

# if query:
#     # Start the execution of the crew
#     result = crew.kickoff(inputs={'query': query})

#     st.write(result)
#     st.write(result.raw)