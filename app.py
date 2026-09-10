import pdfplumber
import streamlit as st

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF AI Chatbot",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# TITLE
# ============================================================

st.title("📄 PDF AI Chatbot")
st.caption("Ask questions about your PDF using Gemini AI")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ AI Configuration")

    provider = st.selectbox(
        "AI Provider",
        ["Google Gemini"]
    )

    api_key = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        placeholder="Paste your Gemini API key"
    )

    if api_key:
        st.session_state.api_key = api_key

    st.info(
        "🔐 Your API key is used for the current session. "
        "Do not share your API key."
    )

    st.divider()

    st.header("📁 Document")

    uploaded_file = st.file_uploader(
        "Upload a PDF file",
        type=["pdf"]
    )

    st.divider()

    if st.button("🗑️ Clear Session"):

        st.session_state.api_key = ""
        st.session_state.vector_store = None
        st.session_state.file_name = None

        st.rerun()


# ============================================================
# CHECK API KEY
# ============================================================

if not st.session_state.api_key:

    st.warning(
        "👈 Please enter your Gemini API key in the sidebar."
    )

    st.stop()


# ============================================================
# CHECK PDF
# ============================================================

if uploaded_file is None:

    st.info(
        "👈 Upload a PDF file from the sidebar."
    )

    st.stop()


# ============================================================
# PROCESS PDF
# ============================================================

if (
    st.session_state.vector_store is None
    or st.session_state.file_name != uploaded_file.name
):

    with st.spinner("📖 Processing PDF..."):

        try:

            # ------------------------------------------------
            # Extract text
            # ------------------------------------------------

            text = ""

            with pdfplumber.open(uploaded_file) as pdf:

                for page in pdf.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"


            # ------------------------------------------------
            # Check text
            # ------------------------------------------------

            if not text.strip():

                st.error(
                    "❌ No readable text was found in this PDF."
                )

                st.stop()


            # ------------------------------------------------
            # Split text
            # ------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    ""
                ],
                chunk_size=1000,
                chunk_overlap=200
            )

            chunks = text_splitter.split_text(text)


            # ------------------------------------------------
            # Gemini embeddings
            # ------------------------------------------------

            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=st.session_state.api_key
            )


            # ------------------------------------------------
            # FAISS vector database
            # ------------------------------------------------

            vector_store = FAISS.from_texts(
                chunks,
                embeddings
            )


            # ------------------------------------------------
            # Save in session
            # ------------------------------------------------

            st.session_state.vector_store = vector_store
            st.session_state.file_name = uploaded_file.name

            st.success(
                f"✅ PDF processed successfully! "
                f"{len(chunks)} chunks created."
            )

        except Exception as e:

            st.error(
                "❌ Error while processing the PDF."
            )

            st.code(str(e))

            st.stop()


# ============================================================
# RETRIEVER
# ============================================================

retriever = st.session_state.vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4
    }
)


# ============================================================
# FORMAT DOCUMENTS
# ============================================================

def format_docs(docs):

    return "\n\n".join(
        doc.page_content
        for doc in docs
    )


# ============================================================
# GEMINI LLM
# ============================================================

try:

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        max_output_tokens=100000,
        google_api_key=st.session_state.api_key
    )

except Exception as e:

    st.error("❌ Could not initialize Gemini.")

    st.code(str(e))

    st.stop()


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
You are a helpful AI assistant that answers questions
about a PDF document.

Follow these rules strictly:

1. Answer ONLY using information from the provided PDF
   context.

2. Do not use outside knowledge.

3. Do not invent or assume information.

4. If the answer is not available in the context,
   say:
   "The answer is not available in the provided PDF."

5. Provide clear and complete answers.

6. Include important numbers, dates, names and facts
   when available.

7. Use bullet points when useful.

PDF CONTEXT:
{context}
"""
    ),

    (
        "human",
        "{question}"
    )

])


# ============================================================
# RAG CHAIN
# ============================================================

chain = (

    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }

    | prompt
    | llm
    | StrOutputParser()

)


# ============================================================
# DOCUMENT STATUS
# ============================================================

st.success(
    f"📄 Currently chatting with: **{uploaded_file.name}**"
)

st.divider()


# ============================================================
# QUESTION
# ============================================================

user_question = st.text_input(
    "💬 Ask a question about your PDF",
    placeholder="Example: What is the main objective of this document?"
)


# ============================================================
# ANSWER
# ============================================================

if user_question:

    with st.spinner("🤖 Gemini is thinking..."):

        try:

            response = chain.invoke(
                user_question
            )

            st.subheader("Answer")

            st.write(response)

        except Exception as e:

            st.error(
                "❌ Error while generating the answer."
            )

            st.code(str(e))