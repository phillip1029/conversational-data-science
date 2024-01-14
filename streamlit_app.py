
import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import pandas as pd
import os

# load the .env file
from dotenv import load_dotenv
load_dotenv()

API_KEY =os.getenv("OPENAI_API_KEY")

llm = OpenAI(api_token=API_KEY, model="gpt-4-1106-preview")

st.title("Conversational Data Science")

st.sidebar.write("Select a dataset to load:")


if "sdf" not in st.session_state:
    st.session_state.sdf = None

if "prompt_history" not in st.session_state:
    st.session_state.prompt_history = []

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file. This should be in long format (one datapoint per row).",
    type="csv",
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    sdf = SmartDataframe(df, config={"llm": llm})
    st.session_state.sdf = sdf

def get_text():
    input_text = st.text_input("You: ","Would you please give me a summary of the data?", key="input")
    return input_text 

def generate_answer(question):
    return sdf.chat(question)

if "conversation" not in st.session_state:
    st.session_state.conversation = []

with st.form(key='my_form'):
    user_input = st.text_input("You: ")
    submit_button = st.form_submit_button(label='Enter')

import matplotlib.pyplot as plt

if submit_button and user_input:
    response = generate_answer(user_input)
    st.session_state.conversation.append(("You: ", user_input))

    # If the response is a matplotlib figure, display the chart
    if isinstance(response, plt.Figure):
        st.pyplot(response)
        st.session_state.conversation.append(("AI-Analyst: ", "Here is the chart you requested:"))
    else:
        st.session_state.conversation.append(("AI-Analyst: ", response))

for i in range(len(st.session_state.conversation)-1, -1, -2):
    if i-1 >= 0:
        st.markdown(f'<div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px;">{st.session_state.conversation[i-1][0]} {st.session_state.conversation[i-1][1]}</div>', unsafe_allow_html=True)
    if i < len(st.session_state.conversation):
        st.text(f"{st.session_state.conversation[i][0]} {st.session_state.conversation[i][1]}")