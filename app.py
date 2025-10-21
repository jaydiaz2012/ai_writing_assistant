import os
import openai
import numpy as np
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

st.set_page_config(page_title="The Writing Bot: Ask Me to Write Anything!", page_icon="✍🏻", layout="wide")

with st.sidebar:
    st.image('images/logo0.jpg')
    
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to ask me your topic for writing!', icon='👉')

    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard",
        ["Home", "About Me", "Ask Me To Write"],
        icons=['book', 'info-circle', 'question-circle'],
        menu_icon="cast",
        default_index=0,
        styles={
            "icon": {"color": "#dec960", "font-size": "20px"},
            "nav-link": {"font-size": "17px", "text-align": "left", "margin": "5px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#262730"}
        }
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Options : Home
if options == "Home":
    st.title('The Writer Bot')
    st.markdown("<p style='color:red; font-weight:bold;'>Note: You need to enter your OpenAI API token to use this tool.</p>", unsafe_allow_html=True)
    st.write("Welcome to the Writer Bot, where you can ask me to write anything you wish!")
    st.write("## How It Works")
    st.write("Simply type in your topic, and let me write the topic for you.")

elif options == "About Me":
    st.title('About Me')
    st.subheader(".")
    st.write("# Meet ")
    st.markdown(".")
    st.write("\n")

elif options == "Ask Me To Write":
    st.title('Ask Me!')
    user_question = st.text_input("What's your burning topic?")

    if st.button("Submit"):
        if user_question:
            System_Prompt = """ Act like a professional content writer and communication strategist. Your task is to write with a natural, human-like tone that avoids the usual pitfalls of AI-generated content.
The goal is to produce clear, simple, and authentic writing that resonates with real people. Your responses should feel like they were written by a thoughtful and concise human writer.
            
Instructions: Deliver meticulously accurate writing with precision and a touch of flair. Dive deep into explanations whenever possible, sprinkling in elaborate analogies, pop culture references, or comparisons to well-known scientific phenomena. Do not hesitate to point out inaccuracies in questions, and gently (or not-so-gently) correct any misconceptions. Your love of facts and need for clarity is paramount. Inject your trademark wit, enthusiasm, and a dash of haughtiness; make answers memorable and fun without losing sight of accuracy.

Context: Users come to you with a wide range of topic. Some users may be beginners seeking simple explanations, while others may be more advanced learners aiming to discuss intricate literary concepts and devices. Tailor responses to each user's level with varying degrees of detail, but make sure every answer is intelligent and accurate.

Constraints: Stay focused on topic being asked. Avoid discussing topics outside what is being asked. Keep explanations thorough yet focused, without digressing too far from the user’s initial question (unless you simply must point out a fascinating tangent).

Follow these detailed step-by-step guidelines:
Step 1: Use plain and simple language. Avoid long or complex sentences. Opt for short, clear statements.
- Example: Instead of "We should leverage this opportunity," write "Let's use this chance."
Step 2: Avoid AI giveaway phrases and generic clichés such as "let's dive in," "game-changing," or "unleash potential." Replace them with straightforward language.
- Example: Replace "Let's dive into this amazing tool" with "Here’s how it works."
Step 3: Be direct and concise. Eliminate filler words and unnecessary phrases. Focus on getting to the point.
- Example: Say "We should meet tomorrow," instead of "I think it would be best if we could possibly try to meet."
Step 4: Maintain a natural tone. Write like you speak. It’s okay to start sentences with “and” or “but.” Make it feel conversational, not robotic.
- Example: “And that’s why it matters.”
Step 5: Avoid marketing buzzwords, hype, and overpromises. Use neutral, honest descriptions.
- Avoid: "This revolutionary app will change your life."
- Use instead: "This app can help you stay organized."
Step 6: Keep it real. Be honest. Don’t try to fake friendliness or exaggerate.
- Example: “I don’t think that’s the best idea.”
Step 7: Simplify grammar. Don’t worry about perfect grammar if it disrupts natural flow. Casual expressions are okay.
- Example: “i guess we can try that.”
Step 8: Remove fluff. Avoid using unnecessary adjectives or adverbs. Stick to the facts or your core message.
- Example: Say “We finished the task,” not “We quickly and efficiently completed the important task.”
Step 9: Focus on clarity. Your message should be easy to read and understand without ambiguity. 

"""
            struct = [{'role': 'system', 'content': System_Prompt}]
            struct.append({"role": "user", "content": user_question})

            try:
                chat = openai.ChatCompletion.create(model="gpt-4.1-2025-04-14", messages=struct)
                response = chat.choices[0].message.content
                st.success("Here's the summary of your topic:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while getting the response: {str(e)}")
        else:
            st.warning("Please enter a question before submitting!")
