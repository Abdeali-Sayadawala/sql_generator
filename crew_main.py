from dotenv import load_dotenv
from crewai import Crew, Process
from langchain_community.utilities.sql_database import SQLDatabase
from agents import data_extraction_agent
from tasks import search_task
import os

load_dotenv()

def get_db_schema():
    MYSQL_USER=os.environ['MYSQL_USER']
    MYSQL_PASSWORD=os.environ['MYSQL_PASSWORD']                         
    MYSQL_DB=os.environ['MYSQL_DB']
    MYSQL_HOST=os.environ['MYSQL_HOST']    
    DATABASE_URL= f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/{MYSQL_DB}'
    try:
        print("Connecting to Database")
        db = SQLDatabase.from_uri(DATABASE_URL)
        return db.get_table_info()
    except Exception as e:
        raise e
    finally:
        print("connection completed")

user_query = open('input.txt', 'r').read()
print("user query: ", user_query)

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
# result = crew.kickoff(inputs={'user_query': user_query, 'database_list': database_list, 'database_count': len(database_list)})
result = crew.kickoff(inputs={'user_query': user_query, 'schema': get_db_schema()})

# Display results in Streamlit app
print(result)
# print(result.raw)