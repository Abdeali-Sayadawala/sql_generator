from crewai import Agent
from tools import db_search_tool, execute_sql, tables_schema, check_sql
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

data_analyst_agent = Agent(
    role='Question Answering Agent From Data',
    goal="""
    This is the user query {user_query} intended to query the database seeking natural language response.
    This is the SQL Query you need to run in the database using the PGSearchTool: {sql_query}
    Extract the relavant information according for the {user_query} and always provide complete answer in natural language
    
    Run the below SQL Query on the postgresDB and return all the resultant information in the form of a pandas dataframe.
    SQL Query: {sql_query}
    resultant dataframe:
    
    Once done with generating the the dataframe, share the data with the next agent which is visualization agent
    
    """,
    verbose=True,
    memory=True,
    backstory=(
        "Expert in analysing the data in the form postgresSQL database by running the provided SQL query, and retreiving information based on the {sql_query}. Analyses and extracts the relavant information according to the user query and provides the resultant data retreived in the form of dataframe."
    ),
    tools=[db_search_tool],
    llm='groq/llama-3.1-70b-versatile',
    allow_delegation = True  #To transfer info to another agent after task is finished
)


visualisation_agent = Agent(
    role='Data Visualization Agent',
    goal="""
    This is the user query {user_query} intended to query the database seeking natural language response.
    This is the SQL Query you need to run in the database using the PGSearchTool: {sql_query}
    
    Steps to follow: 
    1. Extract data according to the {sql_query} and store in a dataframe.
    2. Create a python script to generate a graph/plots based on the dataframe using libraries like matplotlib or seaborn. 
    3. Run the python script and return the plot as .jpg image type
    
    """,
    verbose=True,
    memory=True,
    backstory=(
        """Expert in understanding and user_query and using sql query to extract information and store in a dataframe, then creating and running a python script to generate a plot/graph to explain user_query using python libraries like matplotlib/seaborn. """
    ),
    tools=[db_search_tool],
    llm='groq/llama-3.1-70b-versatile',
    allow_code_execution=True, #functionality to run code to generate visualizations
    allow_delegation=False
)


data_extraction_agent = Agent(
    role='Senior Database Developer',
    goal='Construct and execute SQL queries based on a request',
    verbose=True,
    memory=True,
    backstory=(
        """You are a Pro Assist, an AI analyst at a company built to answer queries of users. You are interacting with a user who is asking you questions about the comapany's data.
        You are an experienced database engineer who is master at creating efficient and complex SQL queries.
        You have a deep understanding of how different databases work and how to optimize queries.

        Use the `tables_schema` to understand the metadata for the tables for all the databases in the list {database_list} one by one. 
        You can only pass one database at once from the list to `tables_schema`.
        If your user query requires you to get data from tables residing in different databases, 
        then you need to query each database table save the data in a pandas dataframe and join/merge the dataframes on appropriate column.

        Always follow a step by step procedure to generate and execute SQL query:
        1.) Generate the SQL query, do not wrap the SQL query in any other text, not even backticks.
            Use the `check_sql` to check your queries for correctness.
        2.) Using the query recieved from `check_sql` tool output execute the query using `execute_sql`

        do not use the data received from `tables_schema` to pass to the output.
        """
    ),
    tools=[tables_schema, execute_sql, check_sql],
    llm='groq/llama-3.1-70b-versatile',
    allow_delegation=True  # Enables passing data to the next agent
)

# python_script_generation_agent = Agent(
#     role='Python Script Generation Agent',
#     goal='Generate a Python script using libraries like matplotlib, seaborn, or plotly to create visualizations from the provided data.',
#     verbose=True,
#     memory=True,
#     backstory=(
#         "Skilled in writing Python scripts for data visualization, utilizing popular libraries to transform raw data into meaningful graphs or plots."
#     ),
#     tools=[],  # Add any specific tools if needed for script generation
#     llm=llm,
#     allow_delegation=True  # Enables passing the script to the next agent
# )

# visualization_display_agent = Agent(
#     role='Visualization Display Agent',
#     goal='Execute the provided Python script and display the resulting visualization.',
#     verbose=True,
#     memory=True,
#     backstory=(
#         "Specializes in executing Python scripts and rendering visual outputs, ensuring that users can see and interpret data insights effectively."
#     ),
#     tools=[],  # Add any specific tools if needed for executing scripts
#     llm=llm,
#     allow_code_execution=True,  # Allows execution of code to generate visualizations
# )