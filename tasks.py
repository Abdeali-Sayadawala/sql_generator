from crewai import Task
from tools import db_search_tool
from agents import data_analyst_agent, visualisation_agent, data_extraction_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)

# Information Retrieval Task
search_task = Task(
    description=(
        "Analyze and understand {user_query}. "
        "Generate SQL query and Python script if required based on the user input."
        "Run the SQL query in PostgresSQL database and extract all relevant information."
        "If more than one database need to be involved, use Python script to merge/join the data using Pandas."
    ),
    expected_output='A pandas dataframe containing all the data from running the generated SQL.',
    agent=data_extraction_agent,
)

# # Python Script Generation Task
# script_task = Task(
#     description=(
#         "Generate a Python script to create visualizations based on the extracted data. "
#         "Use libraries like matplotlib, seaborn, or plotly."
#     ),
#     expected_output='A Python script that generates a graph or plot.',
#     agent=python_script_generation_agent,
# )

# # Visualization Display Task
# display_task = Task(
#     description=(
#         "Execute the provided Python script to generate a visualization. "
#         "Display the resulting graph or plot."
#     ),
#     expected_output='A displayed graph or plot from the executed script.',
#     agent=visualization_display_agent,
#     async_execution=False,  # Ensure tasks run sequentially if needed
# )



## Information Retreival Task
# search_task = Task(
#     description=(
#         """Analyze and understand the user question: {user_query}
#         Here is the SQL Query to run in postgres to get relavant data: {sql_query}
#         store all the extracted information in a pandas dataframe and pass it to next agent in next tasks."""
#     ),
#     expected_output='a pandas dataframe with extracted information.',
#     tools=[db_search_tool],
#     agent=data_analyst_agent,
    
# )

visualize_task = Task(
    description=(
        """Analyze and understand the user question: {user_query}
        Here is the SQL Query to run in postgres to get relavant data: {sql_query}
        Extract information from postgres using the {sql_query} and store it in a pandas dataframe.
        Create a python script to generate a graph/plots based on the dataframe using libraries like matplotlib or seaborn. 
        Run the python script and return the plot as .jpg image type
        """
    ),
    expected_output='a clear plot/graph of .jpg image type',
    tools=[db_search_tool],
    agent=visualisation_agent,
    async_execution=False, # To enable both the agents working in parallel
    output_file='generated_image.jpg'
    
)
