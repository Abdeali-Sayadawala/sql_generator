from langchain_groq import ChatGroq
from dotenv import load_dotenv
from crewai import Crew, Process
from agents import data_analyst_agent, visualisation_agent, data_extraction_agent
from tasks import search_task, visualize_task
import os

load_dotenv()

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

database_list = os.environ['POSTGRES_DB'].split(",")
print("database_list: ", database_list)

user_query = input("Ask my anything: ")

crew = Crew(
    agents=[data_extraction_agent],
    tasks=[search_task],
    process=Process.sequential,
    memory=True,
    cache=True,
    max_rpm=100,
    share_crew=True
)

# Start the execution of the crew with user input
print("starting Crew...")
result = crew.kickoff(inputs={'user_query': user_query, 'database_list': database_list})

# Display results in Streamlit app
print(result)
print(result.raw)