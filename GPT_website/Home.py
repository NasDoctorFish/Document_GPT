import streamlit as st
from langchain.prompts import PromptTemplate
import nltk
nltk.download('punkt')

st.set_page_config(
    page_title="GPT Master JUWON LEE",
    page_icon="🐶",
    #in a dictionary format
    menu_items={
        'About': 'This is a Streamlit app showcasing something cool!',
        'Get help': 'https://www.example.com/help',
        'Report a bug': 'https://www.example.com/report',
    },
)
st.title("GPT Master JUWON LEE")

'''
st.sidebar.title("Sidebar title")

st.sidebar.text_input("xxx")
'''


#using 'with' to simplify defining sidebar
with st.sidebar:
    st.title("Sidebar title")
    st.text_input("xxx")

#How to define tabs
tab_one, tab_two, tab_three = st.tabs(["A","B","C"])

with tab_one:
    st.write("a")

with tab_two:
    st.write("b")

with tab_three:
    st.write("c")

st.markdown(
    """
# Hello!
            
Welcome to my FullstackGPT Portfolio!
            
Here are the apps I made:
            
- [ ] [DocumentGPT](/DocumentGPT)
- [ ] [PrivateGPT](/PrivateGPT)
- [ ] [QuizGPT](/QuizGPT)
- [ ] [SiteGPT](/SiteGPT)
- [ ] [MeetingGPT](/MeetingGPT)
- [ ] [InvestorGPT](/InvestorGPT)
"""
)







