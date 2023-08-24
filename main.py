import streamlit as st
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

OPENAI_API_KEY = os.environ.get('OPEN_API_KEY') 

# # Load OPENAI key ENV 
# def load_llm():  
#     llm = ChatOpenAI(temperature=.25, openai_api_key=OPENAI_API_KEY )  
#     return llm 

# check openAI key 
def get_openai_api_key():
    input_text = st.text_input(label="OpenAI API Key (or set it as .env variable)",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text

# Start streamlit 
st.set_page_config(page_title="PDF ChatBot", page_icon="💬")
    
 
st.header("Chat with PDF 💬")

st.markdown("Upload PDF files and chat with Bots to get infromations from your PDFS.")
st.markdown('''
    This app is a LLM PDF chatbot built using:
    [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), [OpenAI](https://platform.openai.com/docs/models), [FAISS](https://faiss.ai/index.html) 
    ''')
 
# upload a PDF file
pdf = st.file_uploader("", type='pdf')


 # st.write(pdf)
if pdf is not None:
    pdf_reader = PdfReader(pdf)

    text = ""
    for page in pdf_reader.pages:
        text+= page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
        )
    chunks = text_splitter.split_text(text)

    #embeddings
    store_name = pdf.name[:-4]
    # get Filename
    # st.write(f'PDF Filename: {store_name}')
    # st.write(chunks)

    # load it into Chroma
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            Vectordb = pickle.load(f)
            print('Embedding load from disk.')

    else:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        Vectordb = FAISS.from_texts(chunks, embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(Vectordb, f)

    # Accept user questions/query
    query = st.text_input("Ask questions about your PDF file:")

    if query:
        docs = Vectordb.similarity_search(query=query, k=3)

        llm = OpenAI(openai_api_key=OPENAI_API_KEY)
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write(response)
