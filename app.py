import os
import streamlit as st
from openai import OpenAI
from crewai import Crew, Agent, Task, Process
from crewai_tools import OpenAITool

# Streamlit UI
st.title("Mixture of Agents Chatbot")

# Input OpenAI API key from user
api_key = st.text_input("Enter your OpenAI API key:", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key.")
    st.stop()

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key
client = OpenAI(api_key=api_key)

# Define the OpenAI tool with GPT-4o model
openai_tool = OpenAITool(model="gpt-4o")

# Setting up the agents
generator_agent = Agent(
    role='Generator Agent',
    goal='Generate diverse initial completions for the given prompt.',
    backstory='Skilled at creating varied and creative responses.',
    tools=[openai_tool]
)

critique_agent = Agent(
    role='Critique Agent',
    goal='Critique and evaluate the initial completions.',
    backstory='Provides constructive feedback to improve quality.',
    tools=[openai_tool]
)

synthesizer_agent = Agent(
    role='Synthesizer Agent',
    goal='Synthesize a final high-quality completion based on critiques.',
    backstory='Integrates feedback to produce refined output.',
    tools=[openai_tool]
)

# Setting up the tasks
generate_completions_task = Task(
    description='Generate 3 diverse initial completions for the given prompt.',
    expected_output='Three distinct completions for the input prompt.',
    agent=generator_agent
)

critique_completions_task = Task(
    description='Critique the 3 initial completions generated.',
    expected_output='Detailed critiques for each of the 3 completions.',
    agent=critique_agent
)

synthesize_completion_task = Task(
    description='Synthesize a final completion based on the original prompt, initial completions, and their critiques.',
    expected_output='A high-quality final completion that integrates the best aspects of the initial completions and addresses the critiques.',
    agent=synthesizer_agent
)

# Forming the crew
crew = Crew(
    agents=[generator_agent, critique_agent, synthesizer_agent],
    tasks=[generate_completions_task, critique_completions_task, synthesize_completion_task],
    process=Process.sequential
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your message here:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Use CrewAI to generate a response
    inputs = {'prompt': prompt}
    result = crew.kickoff(inputs=inputs)
    response = result.get("result", "")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
