
import os
import streamlit as st

from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool


# --------------------------------------------------
# 1. LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")
MODEL = os.getenv("MODEL")


# --------------------------------------------------
# 2. STREAMLIT PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Multi-Agent Customer Support",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent Customer Support")
st.write("Powered by CrewAI and Streamlit")


# --------------------------------------------------
# 3. VALIDATE ENVIRONMENT VARIABLES
# --------------------------------------------------

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing in your .env file.")
    st.stop()

if not SERPER_API_KEY:
    st.error("SERPER_API_KEY is missing in your .env file.")
    st.stop()

if not MODEL:
    st.error("MODEL is missing in your .env file.")
    st.stop()


# --------------------------------------------------
# 4. AGENT 1: CUSTOMER SUPPORT ASSISTANT
# --------------------------------------------------

assistant = Agent(
    role="Customer Support Assistant",

    goal=(
        "Provide clear, accurate and helpful answers "
        "to customer queries using your own knowledge."
    ),

    backstory=(
        "You are an experienced customer support professional. "
        "You understand customer problems and provide simple, "
        "friendly and useful solutions."
    ),

    llm=MODEL,
    verbose=True,
    allow_delegation=False
)


# --------------------------------------------------
# 5. WEB SEARCH TOOL
# --------------------------------------------------

search_tool = SerperDevTool()


# --------------------------------------------------
# 6. AGENT 2: WEB SEARCH ASSISTANT
# --------------------------------------------------

web_search_assistant = Agent(
    role="Web Search Customer Support Assistant",

    goal=(
        "Search the web for reliable and current information "
        "and provide an accurate answer to the customer's query."
    ),

    backstory=(
        "You are an expert online research assistant. "
        "You search the internet, analyse relevant information "
        "and provide helpful answers based on search results."
    ),

    tools=[search_tool],
    llm=MODEL,
    verbose=True,
    allow_delegation=False
)


# --------------------------------------------------
# 7. AGENT 3: CUSTOMER SUPPORT RECORD MANAGER
# --------------------------------------------------

entry_agent = Agent(
    role="Customer Support Record Manager",

    goal=(
        "Collect the customer's query and both answers from "
        "the previous agents and prepare a complete record "
        "for saving into a text file."
    ),

    backstory=(
        "You are a professional data management assistant. "
        "You carefully collect customer queries and responses "
        "from other agents and organise them into a structured "
        "record without losing any information."
    ),

    llm=MODEL,
    verbose=True,
    allow_delegation=False
)


# --------------------------------------------------
# 8. TASK 1: ASSISTANT TASK
# --------------------------------------------------

assistant_task = Task(
    description=(
        "Answer the following customer query using your own knowledge. "
        "Do not perform any web searches.\n\n"
        "Customer Query: {customer_query}"
    ),

    expected_output=(
        "A clear, accurate and helpful answer to the customer's query."
    ),

    agent=assistant
)


# --------------------------------------------------
# 9. TASK 2: WEB SEARCH TASK
# --------------------------------------------------

web_search_task = Task(
    description=(
        "Search the web for the following customer query. "
        "Use the available search tool to find relevant and current "
        "information. Analyse the results and generate a helpful answer. "
        "Include source URLs where available.\n\n"
        "Customer Query: {customer_query}"
    ),

    expected_output=(
        "A detailed answer based on web search results, "
        "including relevant source information."
    ),

    agent=web_search_assistant
)


# --------------------------------------------------
# 10. TASK 3: ENTRY AGENT TASK
# --------------------------------------------------

entry_task = Task(
    description=(
        "Create a customer support record.\n\n"

        "Customer Query: {customer_query}\n\n"

        "Include the complete answer from the Assistant agent "
        "and the complete answer from the Web Search Assistant.\n\n"

        "Organise the information into three sections:\n"
        "1. Customer Query\n"
        "2. Assistant Answer\n"
        "3. Web Search Assistant Answer\n\n"

        "Preserve both answers accurately without omitting information. "
        "Do not invent or change the answers."
    ),

    expected_output=(
        "A structured customer support record containing "
        "the original query and both agent answers."
    ),

    agent=entry_agent,

    context=[assistant_task, web_search_task],

    output_file="answers.txt"
)


# --------------------------------------------------
# 11. CREATE THE CREW
# --------------------------------------------------

crew = Crew(

    agents=[
        assistant,
        web_search_assistant,
        entry_agent
    ],

    tasks=[
        assistant_task,
        web_search_task,
        entry_task
    ],

    process=Process.sequential,

    verbose=True
)


# --------------------------------------------------
# 12. STREAMLIT USER INPUT
# --------------------------------------------------

st.subheader("Ask Your Customer Support Question")

customer_query = st.text_area(
    "Enter your query:",
    placeholder="Example: How can I reset my account password?",
    height=120
)


# --------------------------------------------------
# 13. RUN THE CREW
# --------------------------------------------------

if st.button("Submit Query", type="primary"):

    if not customer_query.strip():

        st.warning("Please enter a customer query.")

    else:

        try:

            with st.spinner(
                "AI agents are working on your query. Please wait..."
            ):

                result = crew.kickoff(
                    inputs={
                        "customer_query": customer_query
                    }
                )

            st.success("Customer support response generated successfully!")

            # Display Assistant Answer
            st.subheader("🤖 Assistant Answer")

            if assistant_task.output:
                st.write(assistant_task.output.raw)

            # Display Web Search Answer
            st.subheader("🌐 Web Search Assistant Answer")

            if web_search_task.output:
                st.write(web_search_task.output.raw)

            # Display Final Entry Agent Output
            st.subheader("📁 Final Customer Support Record")

            st.write(result.raw)

            st.info(
                "The final customer support record has been saved "
                "to answers.txt."
            )

        except Exception as e:

            st.error(f"An error occurred: {e}")
            st.exception(e)
