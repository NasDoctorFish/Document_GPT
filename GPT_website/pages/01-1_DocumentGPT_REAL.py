from typing import Union
from langchain.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.document_loaders import UnstructuredFileLoader
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

#return the type of file in a form of pdf, docx, txt
def get_file_extension(file_name):
    #file_name.r(ight)split(text_dividing, 1)
    #The second argument 1 tells the method to perform the split operation at most once
    return file_name.rsplit('.', 1)[-1].lower()




class ChatCallbackHandler(BaseCallbackHandler):

    message = ""

    #*arg: arguments, **kwargs: keyword arguments(a=1, hello= 1972)
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        with st.sidebar:
            st.write("llm started!")
    def on_llm_end(self, *args, **kwargs):
        with st.sidebar:
            st.write("llm ended!")
        save_messages(self.message,"ai")

    #every new token generated
    def on_llm_new_token(self, token: str, *arg, **kwargs):
        self.message += token
        #what does markdown mean?
        #Write down markdown HTML formatted text
        self.message_box.markdown(self.message)



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
    file_extension = get_file_extension(file.name)
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = flexible_file_loaders(file_path, file_extension)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    
    #return as a dictionary instead of retriever object
    return retriever

def flexible_file_loaders(file_path :str, file_extension :str ) -> Union[Docx2txtLoader,PyPDFLoader,TextLoader]:
    if(file_extension == 'docx'):
        loader = Docx2txtLoader(file_path)

    if(file_extension == 'pdf'):
        loader = PyPDFLoader(file_path)

    if(file_extension == 'txt'):
        loader = TextLoader(file_path)
    
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    return loader

def save_messages(message :str, role :str):
    st.session_state["messages"].append({"message": message, "role": role})

#ex) send_message("Hello", "ai")
#role should be "ai", "human", "system"
def send_message(message,role,save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_messages(message, role)
        

def paint_history():
    for message in st.session_state["messages"]:
        #rewrite all the session_state saved conversations whenever the Streamlit is running over again.
        #no need to cache conversation we have session_state in Streamlit
        send_message(
            message["message"],
            message["role"],
            save=False,
        )
    

def format_docs(docs):
    result = "\n\n".join(document.page_content for document in docs)
    return result
    

template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)

st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!
"""
)
with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )


if file:
    #retriever object, makes more easier to represent vectorstore object
    retriever = embed_file(file)
    #save=False doesn't let the current message saved to session_state
    send_message("I'm ready: Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file.....")
    if message:
        #Prior version
        #showing up the human message into human design form of Streamlit
        send_message(message, "human")
        chain = {
            #retriever in chain automatically does retriever.invoke(message)
            #seems that RunnablePassthrough() is used at which the query pass through chain like below
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        } | template | llm
        with st.chat_message("ai"):
            #When invoke, Chat model will automatically call ChatCallbackHandler
            #And also theorhetically update the message within the st.empty in ChatCallbackHandler
            response = chain.invoke(message)
        


        
        
else:
    #This part is initializing the session_state when there's no file uploaded
    st.session_state["messages"] = []