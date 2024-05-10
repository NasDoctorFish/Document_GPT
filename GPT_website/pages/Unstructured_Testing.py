import nltk
nltk.download('punkt')
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS



from langchain.chat_models import ChatOpenAI
import streamlit as st

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

with st.sidebar:
    nltk_token=nltk.data.find('tokenizers/punkt')
    st.write(nltk_token)

class ChatCallbackHandler(BaseCallbackHandler):
    #*arg: arguments, **kwargs: keyword arguments(a=1, hello= 1972)
    def on_llm_start(self, *args, **kwargs):
         with st.sidebar:
             st.write("llm started!")
    def on_llm_end(self, *args, **kwargs):
        with st.sidebar:
            st.write("llm ended!")
    #every new token generated
    def on_llm_new_token(self, token: str, *arg, **kwargs):
        print(token) 



llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[
        ChatCallbackHandler(),
    ]
)


#Streamlit will see the file, 
#and if the file is the same, will not run
#The same file will be saved as a Cache, used later if the file is identical
@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    #return as a dictionary instead of retriever object
    return retriever


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)
with st.sidebar:
    file = st.file_uploader(
        "Testing",
        type=["pdf", "txt", "docx"],
    )

if(file):
    retriever = embed_file(file)
    st.write(retriever.content)



