from langchain_groq import ChatGroq
from dotenv import load_dotenv
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