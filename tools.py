# from crewai_tools import NL2SQLTool
from crewai_tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import (
    QuerySQLCheckerTool,
    QuerySQLDataBaseTool,
)
from langchain_groq import ChatGroq
import os

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

def init_database():
    MYSQL_USER=os.environ['MYSQL_USER']
    MYSQL_PASSWORD=os.environ['MYSQL_PASSWORD']                         
    MYSQL_DB=os.environ['MYSQL_DB']
    MYSQL_HOST=os.environ['MYSQL_HOST']    
    DATABASE_URL= f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/{MYSQL_DB}'
    try:
        return SQLDatabase.from_uri(DATABASE_URL)
    except Exception as e:
        raise e

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
# @tool
# def tables_schema() -> str:
#     """
#     Name: tables_schema
#     Description: get the schema for the tables from the database passed in the argument
#     """
#     # tool = InfoSQLDatabaseTool(db=init_database(database_nm))
#     # return tool.invoke(tables)
#     db = init_database()
#     return db.get_table_info()

# print(tables_schema.run("salaries"))

# Tool 3: Executes a given SQL query
@tool
def execute_sql(sql_query: str) -> str:
    """
    Name: execute_sql
    Description: Executes the provided SQL query passed in the argument using the database connection and returns the result
    """
    return QuerySQLDataBaseTool(db=init_database()).invoke(sql_query)

# execute_sql.run("SELECT * FROM salaries WHERE salary > 10000 LIMIT 5")

# Tool 4: Checks the SQL query before executing it
@tool("check_sql")
def check_sql(sql_query: str) -> str:
    """
    Name: check_sql
    Description: Use this tool to double check if your query is correct before executing it using the database connection. Always use this
    tool before executing a query with `execute_sql`.
    """
    return QuerySQLCheckerTool(db=init_database(), llm=llm).invoke({"query": sql_query})

# Tool 5: Merges two pandas dataframes and gives the output.
# @tool("merge_df")
# def merge_df(df_1: pd.DataFrame, df_2: pd.DataFrame, merge_columns: list, merge_type=None) -> pd.DataFrame:
#     """
#     Name: merge_df
#     Description: Use this tool to merge two pandas dataframes.
#     merge_columns argument will  be a list of columns on which the two dataframe is supposed to be merged on.
#     merge_type argument will be optional, it will denote how df_2 will be merged into df_1.
#     """
#     if merge_type:
#         pd.merge(df_1, df_2, how=merge_type, on=merge_columns)
#     else:
#         pd.merge(df_1, df_2, on=merge_columns)

# file_search_tool = CSVSearchTool(csv=['../staff_data.csv', '../management_data.csv', '../patient_data.csv'])

# psycopg2 was installed to run this example with PostgreSQL
# db_search_tool = NL2SQLTool(db_uri=os.getenv("DATABASE_URL"))
# db_search_tool = PGSearchTool(db_uri=os.getenv("DATABASE_URL"), table_name="Patient")