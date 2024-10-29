#Install the required packages
##Pip install langchain
##Pip install torch
##Pip install accelerate
##Pip install sentence_transformers
##Pip install streamlit
##Pip install streamlit_chat
##Pip install faiss-cpu
##Pip install tiktoken
##Pip install huggingface-hub
##Pip install pypdf
##Pip install ctransformers

#Import required libraries
import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory

#Create a function to load docuemnts

def load_documents():
    loader=DirectoryLoader('data/',glob="*.pdf",loader_cls=PyPDFLoader)
    documents= loader.load()
    return documents

#Function to split text into chunks
def split_text_into_chunks(documents):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks=text_splitter.split_documents(documents)
    return text_chunks

#Function to create embdeddings
def create_embdeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={'device':"cuda"})
    return embeddings

#Function to create vector store
def create_vector_store(text_chunks,embeddings):
    vector_store=FAISS.from_documents(text_chunks,embeddings)
    return vector_store

#Function to create LLMS Model
def create_llms_model():
    llm=CTransformers(model="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                      config={'max_new_tokens':2048,
                              'temperature':0.01},
                              streaming=True)
    return llm

#Initialize Streamlit app
st.title("RICS APC submission Prep Chatbot")
st.title("Personalised RICS APC Sbumission Friend")
st.markdown('<style>h1{color:orange;text-align:center;}</style>',unsafe_allow_html=True)
st.subheader("Get your comptency and level !")
st.markdown('<style>h3{color:pink;text-align:center;}</style>',unsafe_allow_html=True)

#loading of documents
documents=load_documents()

#Split text into chunks
text_chunks=split_text_into_chunks(documents)

#Create embeddings
embeddings=create_embdeddings()

#Create Vector Store
vector_store=create_vector_store(text_chunks,embeddings)

#Create LLMS model
llm=create_llms_model()

#Initialize the conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! Ask me about any area of competency!"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey! Let's have a fruitful chat"]

#Create memory
memory= ConversationBufferMemory(memory_key="chat_history",return_messages=True)

#Create chain
chain = ConversationalRetrievalChain.from_llm(llm=llm,chain_type='stuff',
                                              retriever=vector_store.as_retriever(search_kwargs={"k":2}),
                                              memory=memory)

#Define chat function
def conversation_chat(query):
    result=chain({"question":query,"chat_history":st.session_state['history']})
    st.session_state['history'].append((query,result["answer"]))
    return result["answer"]

#Display chat history
reply_container=st.container()
container=st.container()

with container:
    with st.form(key="my form",clear_on_submit=True):
        user_input=st.text_input("Question:",placeholder="Ask about your desired comptency and its level", key='input')
        submit_button=st.form_submit_button(label='send')
    
    if submit_button and user_input:
        output= conversation_chat(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

if st.session_state['generated']:
    with reply_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i],is_user=True,key=str(i)+'_user',avatar_style="thumbs")
            message(st.session_state["generated"][i],key=str(i),avatar_style="fun-emoji")
