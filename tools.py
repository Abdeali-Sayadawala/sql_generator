from crewai_tools import CSVSearchTool
from crewai_tools import NL2SQLTool
from crewai_tools import PGSearchTool
from crewai_tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    InfoSQLDatabaseTool,
    ListSQLDatabaseTool,
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_groq import ChatGroq
import os

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

# Establish a database connection
# db = SQLDatabase.from_uri(os.getenv("DATABASE_URL"))

def init_database(database_nm) -> SQLDatabase:
    POSTGRES_USER=os.environ['POSTGRES_USER']
    POSTGRES_PASSWORD=os.environ['POSTGRES_PASSWORD']                         
    POSTGRES_DB=database_nm
    POSTGRES_HOST=os.environ['POSTGRES_HOST']
    # postgresql+psycopg2://user:password@host:port/dbname
    
    DATABASE_URL= f'postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:5432/{POSTGRES_DB}' 
    
    return SQLDatabase.from_uri(DATABASE_URL)

# db = init_database()
# print(ListSQLDatabaseTool(db=db).invoke(""))

# # Tool 1: List all the tables in the database
# @tool
# def list_tables(database_nm: str) -> str:
#     """
#     Name: list_tables
#     Description: lists all the tables in the database passed in the argument
#     """
#     return ListSQLDatabaseTool(db=init_database(database_nm)).invoke("")

# list_tables.run()

# Tool 2: Return the schema and sample rows for given tables
@tool
def tables_schema(database_nm: str) -> str:
    """
    Name: tables_schema
    Description: get the schema for the tables from the database passed in the argument
    """
    # tool = InfoSQLDatabaseTool(db=init_database(database_nm))
    # return tool.invoke(tables)
    db = init_database(database_nm)
    return db.get_table_info()

# print(tables_schema.run("salaries"))

# Tool 3: Executes a given SQL query
@tool
def execute_sql(database_nm: str, sql_query: str) -> str:
    """
    Name: execute_sql
    Description: Executes the provided SQL query for the database passed in the argument and returns the result
    """
    return QuerySQLDataBaseTool(db=init_database(database_nm)).invoke(sql_query)

# execute_sql.run("SELECT * FROM salaries WHERE salary > 10000 LIMIT 5")

# Tool 4: Checks the SQL query before executing it
@tool("check_sql")
def check_sql(database_nm: str, sql_query: str) -> str:
    """
    Name: check_sql
    Description: Use this tool to double check if your query is correct for the database passed in the argument before executing it. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=init_database(database_nm), llm=llm).invoke({"query": sql_query})

# file_search_tool = CSVSearchTool(csv=['../staff_data.csv', '../management_data.csv', '../patient_data.csv'])

# psycopg2 was installed to run this example with PostgreSQL
db_search_tool = NL2SQLTool(db_uri=os.getenv("DATABASE_URL"))
# db_search_tool = PGSearchTool(db_uri=os.getenv("DATABASE_URL"), table_name="Patient")