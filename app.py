import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

## load the GROQ And OpenAI API KEY 
groq_api_key=os.getenv('GROQ_API_KEY')
#os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY  
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

st.title("Clinisio🤖🤖")

llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="Deepseek-R1-Distill-Llama-70b")
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)
def vector_embedding():
    if "vectors" not in st.session_state:
        # Load embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load PDFs
        st.session_state.loader = PyPDFDirectoryLoader("./pdf")  # Ensure correct path
        st.session_state.docs = st.session_state.loader.load()  # Document Loading

        # Debugging: Print loaded documents
        st.write(f"Loaded {len(st.session_state.docs)} documents")

        # Check if documents were loaded
        if not st.session_state.docs:
            st.error("❌ No documents found! Please upload PDFs.")
            return

        # Text splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

        # Debugging: Print split document count
        st.write(f"Split into {len(st.session_state.final_documents)} chunks")

        # Check if text splitting worked
        if not st.session_state.final_documents:
            st.error("❌ Text splitting failed. Check document format.")
            return

        # Generate embeddings for each document
        st.session_state.embeddings = [embedding_model.embed_query(doc.page_content) for doc in st.session_state.final_documents]

        # Debugging: Print embedding size
        st.write(f"Generated {len(st.session_state.embeddings)} embeddings")

        # Check if embeddings were generated
        if not st.session_state.embeddings or len(st.session_state.embeddings) == 0:
            st.error("❌ Embeddings are empty. Check embedding model.")
            return

        # Create FAISS vector store
        st.session_state.vectors = FAISS.from_texts(
            [doc.page_content for doc in st.session_state.final_documents], embedding_model
        )

        st.write("✅ Vector Store DB is Ready!")






prompt1=st.text_input("Enter Your Question From Doduments")


if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")
import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
