import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
import google.generativeai as genai

# Streamlit Page Config
st.set_page_config(page_title="Ask PDF Bot - IPH", layout="wide")

# ---- Header UI ----
col1, col2 = st.columns([0.2, 0.8])
with col1:
    st.image("IPH.jpg", width=100)

with col2:
    st.title("Institute of Public Health – Ask PDF Bot")
    st.markdown("🔍 Powered By **Gemini AI**")

# ---- API Key Input ----
with st.expander("🔐 Enter your Google API Key", expanded=True):
    api_key = st.text_input("Paste your Google API Key here:", type="password", key="api_key_input")
    if api_key:
        genai.configure(api_key=api_key)
        st.session_state["GOOGLE_API_KEY"] = api_key
        st.success("Google API Key configured!")

# ---- PDF Processing Functions ----
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.session_state["GOOGLE_API_KEY"]
    )
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    vectorstore.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question using the context below. Be concise and accurate.
    If the answer is not in the context, say "Answer not found in the document."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",
        temperature=0.3,
        google_api_key=st.session_state["GOOGLE_API_KEY"]
    )
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=st.session_state["GOOGLE_API_KEY"]
    )
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vectorstore.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.markdown("### ✅ Answer:")
    st.write(response["output_text"])

# ---- Main App UI ----
st.markdown("---")
st.subheader("Once the Upload is Processed you can ask your questions related to the uploaded PDF file")

with st.sidebar:
    st.header("📤 Upload & Process PDF")
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])
    if st.button("🚀 Process PDFs"):
        if "GOOGLE_API_KEY" in st.session_state:
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processed and indexed! You can now ask questions.")
        else:
            st.error("Please enter your Google API Key first!")

st.markdown("---")
st.subheader("Ask a Question about the PDF uploaded")

user_question = st.text_input("💬 Type your question here:")
if user_question:
    if "GOOGLE_API_KEY" in st.session_state:
        user_input(user_question)
    else:
        st.warning("Please enter your Google API Key above before asking questions.")
