import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import Groq
from langchain_groq import ChatGroq as Groq

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate
import os

os.environ["GROQ_API_KEY"] = "enter ur groq key here"

#  embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large"
)

# existing FAISS index
vector_store = FAISS.load_local('vectorstore_faiss/db', embeddings,allow_dangerous_deserialization=True)

# Groq LLM
llm = Groq(
    model_name="mixtral-8x7b-32768",  
    temperature=0.5, #medium to strike balance, more wouldve deviated our model
)

#prompt template
prompt_template = """
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say you don't know. Don't try to make up an answer. Sound proffesional and poilte.

Context: {context}

Question: {question}

Answer: """

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# creating chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

# front end using streamlit
st.title("🐶 WELCOME TO ANICARE!")
if 'messages' not in st.session_state:
    st.session_state.messages=[]

#displaying history
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])
# input by user
user_question = st.chat_input("Hello! How may i assist you today?")

if user_question:
    st.chat_message('user').markdown(user_question)
    st.session_state.messages.append({'role':'user',"content":user_question})
    with st.spinner("Thinking..."):
        #response by chain
        response = qa_chain({"query": user_question})
        
        # Display answer
        st.write("### Answer")
        st.session_state.messages.append({'role':'bot',"content":response["result"]})
        st.chat_message('bot').markdown(response["result"])
        # st.write(response["result"])
        # st.write(response)
        # display source documents- it wont save their history
        st.write("### Source Documents")
        for i, doc in enumerate(response["source_documents"]):
            with st.expander(f"Document {i + 1}"):
                st.write(doc.page_content)
                st.write("---")
                st.write(f"Source: {doc.metadata}")

