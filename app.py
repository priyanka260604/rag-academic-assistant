
import warnings
import logging

# Suppress AFC warning from Google GenAI SDK
warnings.filterwarnings(
    "ignore",
    message=".*Direct use of automatic function calling.*"
)

# Suppress Gemini sampling warning
warnings.filterwarnings(
    "ignore",
    message=".*uses fixed sampling defaults.*"
)
logging.getLogger("google").setLevel(logging.ERROR)
logging.getLogger("google.genai").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai").setLevel(logging.ERROR)

import streamlit as st

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RAG Academic Assistant",
    page_icon="🎓",
    layout="wide"
)

# =========================================================
# CUSTOM UI
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.feature-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

.source-box {
    padding: 12px;
    border-radius: 10px;
    border: 1px solid #ddd;
    margin-top: 8px;
}

</style>
""", unsafe_allow_html=True)

from pypdf import PdfReader
from dotenv import load_dotenv
from pathlib import Path
import hashlib
import json
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI





# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error(
        "❌ GOOGLE_API_KEY not found.\n\n"
        "Please check your .env file."
    )
    st.stop()


# =========================================================
# CONSTANTS
# =========================================================

VECTOR_DB_PATH = "./vectorstore"

COLLECTION_NAME = "academic_documents"

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GEMINI_MODEL = "gemini-3.6-flash"


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()


if "quiz_started" not in st.session_state:
    st.session_state.quiz_started = False


if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False


if "quiz" not in st.session_state:
    st.session_state.quiz = []


if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0


# =========================================================
# EMBEDDING MODEL
# =========================================================

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


# =========================================================
# VECTOR DATABASE
# =========================================================

@st.cache_resource
def get_chroma_client():
    """
    Create one persistent ChromaDB client.

    This does not load the Hugging Face embedding model, so the
    homepage can open without waiting for model initialization.
    """
    return chromadb.PersistentClient(path=VECTOR_DB_PATH)


@st.cache_resource
def get_vectorstore():
    """
    Load the embedding model only when vector search or PDF
    indexing is actually needed.
    """
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        client=get_chroma_client()
    )


# =========================================================
# GEMINI MODEL
# =========================================================

@st.cache_resource
def get_llm():

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=GOOGLE_API_KEY
    )



# =========================================================
# HELPER: CLEAN GEMINI RESPONSE
# =========================================================

def clean_response(response):

    content = response.content

    if isinstance(content, list):

        text_parts = []

        for item in content:

            if isinstance(item, dict):

                text_parts.append(
                    item.get("text", "")
                )

            elif isinstance(item, str):

                text_parts.append(item)

        return "\n".join(
            text_parts
        ).strip()

    return str(content).strip()


# =========================================================
# HELPER: GET DOCUMENT COUNT
# =========================================================

def get_document_count():
    """
    Return the number of stored chunks without loading embeddings
    or fetching all document IDs.
    """
    try:
        collection = get_chroma_client().get_collection(
            name=COLLECTION_NAME
        )
        return collection.count()

    except Exception:
        # The collection may not exist before the first PDF upload.
        return 0


# =========================================================
# HELPER: BUILD CONTEXT
# =========================================================

def build_context(results):

    context_parts = []

    for result in results:

        source = result.metadata.get(
            "source",
            "Unknown"
        )

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        content = result.page_content

        context_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

CONTENT:
{content}
"""
        )

    return "\n\n".join(
        context_parts
    )

# =========================================================
# HELPER: BUILD CHAT HISTORY
# =========================================================

def build_chat_history():

    history = []

    for message in st.session_state.messages[-6:]:

        role = message["role"].upper()
        content = message["content"]

        history.append(
            f"{role}: {content}"
        )

    return "\n\n".join(history)

# =========================================================
# HELPER: EXTRACT JSON
# =========================================================

def extract_json(text):

    text = text.strip()

    # Remove markdown code blocks

    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = text.replace(
        "```",
        ""
    ).strip()

    # Try direct JSON

    try:

        return json.loads(text)

    except Exception:

        pass

    # Try finding JSON array

    start = text.find("[")

    end = text.rfind("]")

    if start != -1 and end != -1:

        json_text = text[
            start:end + 1
        ]

        try:

            return json.loads(
                json_text
            )

        except Exception:

            pass

    return None


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🎓 RAG Academic Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload your academic PDFs, ask questions, '
    'generate notes, create MCQs, and test yourself.'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "💡 Your personal AI study assistant powered by "
    "Retrieval-Augmented Generation (RAG)."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
        ### 📚
        **PDF Learning**

        Ask questions directly from your study material.
        """
    )

with col2:
    st.markdown(
        """
        ### 📝
        **Smart Notes**

        Generate revision-friendly academic notes.
        """
    )

with col3:
    st.markdown(
        """
        ### ❓
        **MCQ Generator**

        Automatically create exam questions.
        """
    )

with col4:
    st.markdown(
        """
        ### 🎯
        **Quiz Mode**

        Test yourself and track your score.
        """
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header(
        "📚 Academic Assistant"
    )

    st.markdown(
        "Upload your academic PDFs "
        "and ask questions."
    )

    uploaded_files = st.file_uploader(
        "📄 Upload Academic PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.divider()

    document_count = get_document_count()

    st.info(
        f"🗄️ Stored chunks: {document_count}"
    )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.quiz_started = False

        st.session_state.quiz_submitted = False

        st.rerun()


# =========================================================
# PROCESS MULTIPLE PDFs
# =========================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        # -------------------------------------------------
        # READ FILE
        # -------------------------------------------------

        file_bytes = uploaded_file.getvalue()


        # -------------------------------------------------
        # CREATE FILE HASH
        # -------------------------------------------------

        file_hash = hashlib.md5(
            file_bytes
        ).hexdigest()


        # -------------------------------------------------
        # SKIP ALREADY PROCESSED FILE
        # -------------------------------------------------

        if file_hash in st.session_state.processed_files:

            continue


        with st.spinner(
            f"📖 Processing {uploaded_file.name}..."
        ):

            try:

                # -----------------------------------------
                # READ PDF
                # -----------------------------------------

                reader = PdfReader(
                    uploaded_file
                )


                # -----------------------------------------
                # TEXT SPLITTER
                # -----------------------------------------

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )


                texts = []

                metadatas = []


                # -----------------------------------------
                # PROCESS EACH PAGE
                # -----------------------------------------

                for page_number, page in enumerate(
                    reader.pages,
                    start=1
                ):

                    page_text = page.extract_text()


                    if not page_text:

                        continue


                    page_chunks = splitter.split_text(
                        page_text
                    )


                    for chunk in page_chunks:

                        if not chunk.strip():

                            continue


                        texts.append(
                            chunk
                        )


                        metadatas.append(
                            {
                                "source":
                                    uploaded_file.name,

                                "page":
                                    page_number,

                                "file_hash":
                                    file_hash
                            }
                        )


                # -----------------------------------------
                # ADD TO VECTOR DATABASE
                # -----------------------------------------

                if texts:

                    get_vectorstore().add_texts(
                        texts=texts,
                        metadatas=metadatas
                    )


                    st.session_state.processed_files.add(
                        file_hash
                    )


                    st.sidebar.success(
                        f"✅ {uploaded_file.name}"
                    )

                else:

                    st.sidebar.warning(
                        f"⚠️ No readable text found in "
                        f"{uploaded_file.name}"
                    )


            except Exception as e:

                st.sidebar.error(
                    f"❌ Error processing "
                    f"{uploaded_file.name}: {e}"
                )


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# ACADEMIC AI TOOLS
# =========================================================

st.divider()

st.header(
    "🎓 Academic AI Tools"
)


tool = st.selectbox(
    "Choose an academic tool",
    [
        "Select a tool",
        "📝 Generate Notes",
        "📌 Summarize Documents",
        "❓ Generate MCQs",
        "🧠 Explain Simply",
        "🎯 Quiz Mode"
    ]
)


# =========================================================
# ACADEMIC TOOL PROCESSING
# =========================================================

if tool != "Select a tool":

    current_document_count = get_document_count()


    if current_document_count == 0:

        st.warning(
            "📄 Please upload a PDF first."
        )

    # =====================================================
    # QUIZ MODE
    # =====================================================

    elif tool == "🎯 Quiz Mode":

        st.subheader(
            "🎯 Academic Quiz"
        )


        number_of_questions = st.slider(
            "Number of questions",
            min_value=5,
            max_value=15,
            value=5
        )


        if st.button(
            "🚀 Start New Quiz"
        ):

            # ---------------------------------------------
            # RESET PREVIOUS QUIZ
            # ---------------------------------------------

            st.session_state.quiz_started = False

            st.session_state.quiz_submitted = False

            st.session_state.quiz = []

            st.session_state.quiz_score = 0


            with st.spinner(
                "🔎 Finding important concepts..."
            ):

                results = get_vectorstore().similarity_search(
                    "important concepts definitions "
                    "formulas theories examples",
                    k=10
                )


            if not results:

                st.error(
                    "No relevant content found."
                )

            else:

                context = build_context(
                    results
                )


                # -----------------------------------------
                # QUIZ PROMPT
                # -----------------------------------------

                quiz_prompt = f"""
You are an academic examination question generator.

Create exactly {number_of_questions}
multiple-choice questions using ONLY
the academic content provided below.

Return ONLY valid JSON.

Required format:

[
  {{
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": 0,
    "explanation": "Short explanation"
  }}
]

Rules:

- answer must be an integer.
- 0 means Option A.
- 1 means Option B.
- 2 means Option C.
- 3 means Option D.
- Exactly four options for every question.
- Questions must be based ONLY on the document.
- Do not include markdown.
- Do not include text outside JSON.

ACADEMIC CONTENT:

{context}
"""


                with st.spinner(
                    "🤖 Generating quiz..."
                ):

                    try:

                        response = get_llm().invoke(
                            quiz_prompt
                        )

                        quiz_text = clean_response(
                            response
                        )

                        quiz = extract_json(
                            quiz_text
                        )


                        if not quiz:

                            st.error(
                                "❌ Could not understand "
                                "the generated quiz."
                            )

                        elif not isinstance(
                            quiz,
                            list
                        ):

                            st.error(
                                "❌ Invalid quiz format."
                            )

                        else:

                            # Validate questions

                            valid_quiz = []


                            for question in quiz:

                                if (
                                    isinstance(
                                        question,
                                        dict
                                    )
                                    and
                                    "question" in question
                                    and
                                    "options" in question
                                    and
                                    "answer" in question
                                    and
                                    "explanation" in question
                                ):

                                    if (
                                        len(
                                            question[
                                                "options"
                                            ]
                                        ) == 4
                                    ):

                                        valid_quiz.append(
                                            question
                                        )


                            if not valid_quiz:

                                st.error(
                                    "❌ No valid questions "
                                    "were generated."
                                )

                            else:

                                st.session_state.quiz = (
                                    valid_quiz
                                )

                                st.session_state.quiz_started = (
                                    True
                                )

                                st.session_state.quiz_submitted = (
                                    False
                                )

                                st.rerun()


                    except Exception as e:

                        st.error(
                            f"❌ Quiz generation error: {e}"
                        )


        # =================================================
        # DISPLAY QUIZ
        # =================================================

        if st.session_state.quiz_started:

            quiz = st.session_state.quiz


            st.divider()

            st.subheader(
                "📝 Answer the Questions"
            )


            answers = []


            for i, question in enumerate(
                quiz
            ):

                st.markdown(
                    f"### Q{i + 1}. "
                    f"{question['question']}"
                )


                selected = st.radio(
                    "Choose your answer:",
                    question["options"],
                    key=f"quiz_question_{i}"
                )


                answers.append(
                    selected
                )


            if st.button(
                "✅ Submit Quiz",
                use_container_width=True
            ):

                score = 0


                for i, question in enumerate(
                    quiz
                ):

                    correct_index = int(
                        question["answer"]
                    )


                    if (
                        0 <= correct_index < 4
                    ):

                        correct_answer = (
                            question[
                                "options"
                            ][correct_index]
                        )


                        if answers[i] == correct_answer:

                            score += 1


                st.session_state.quiz_score = (
                    score
                )

                st.session_state.quiz_submitted = (
                    True
                )


                st.rerun()


        # =================================================
        # QUIZ RESULT
        # =================================================

        if st.session_state.quiz_submitted:

            quiz = st.session_state.quiz

            score = st.session_state.quiz_score

            total = len(quiz)


            percentage = (
                score / total
            ) * 100 if total else 0


            st.divider()

            st.subheader(
                "📊 Quiz Result"
            )


            col1, col2 = st.columns(2)


            with col1:

                st.metric(
                    "Your Score",
                    f"{score}/{total}"
                )


            with col2:

                st.metric(
                    "Percentage",
                    f"{percentage:.0f}%"
                )


            if percentage >= 80:

                st.success(
                    "🎉 Excellent! "
                    "You have a strong understanding."
                )

            elif percentage >= 60:

                st.info(
                    "👍 Good job! "
                    "A little more revision will help."
                )

            else:

                st.warning(
                    "📚 Keep practicing "
                    "and revise the important concepts."
                )


            st.subheader(
                "📖 Answer Review"
            )


            for i, question in enumerate(
                quiz
            ):

                correct_index = int(
                    question["answer"]
                )


                correct_answer = (
                    question[
                        "options"
                    ][correct_index]
                )


                st.markdown(
                    f"""
### Q{i + 1}

**Correct Answer:**  
{correct_answer}

**Explanation:**  
{question["explanation"]}
"""
                )


    # =====================================================
    # OTHER ACADEMIC TOOLS
    # =====================================================

    else:

        with st.spinner(
            "🔎 Searching your academic documents..."
        ):

            results = get_vectorstore().similarity_search(
                tool,
                k=8
            )


        if not results:

            st.warning(
                "No relevant information found."
            )

        else:

            context = build_context(
                results
            )


            # =============================================
            # NOTES
            # =============================================

            if "Generate Notes" in tool:

                prompt = f"""
You are an academic note-making assistant.

Create clear and structured study notes from
the provided academic content.

Include:

- Important definitions
- Key concepts
- Important points
- Formulas
- Examples
- Exam-important points

Use headings and bullet points.

Do NOT add information that is not present
in the provided content.

ACADEMIC CONTENT:

{context}

Generate detailed but easy-to-revise notes.
"""


            # =============================================
            # SUMMARY
            # =============================================

            elif "Summarize" in tool:

                prompt = f"""
You are an academic summarization assistant.

Summarize the provided academic content.

The summary should be:

- Clear
- Concise
- Easy to revise
- Academically accurate

Include important concepts and definitions.

Do not add outside information.

ACADEMIC CONTENT:

{context}

SUMMARY:
"""


            # =============================================
            # MCQS
            # =============================================

            elif "Generate MCQs" in tool:

                prompt = f"""
You are an academic exam-question generator.

Create 10 multiple-choice questions based ONLY
on the provided academic content.

For each question provide:

1. Question
2. Four options
3. Correct answer
4. Short explanation

Do not use outside information.

ACADEMIC CONTENT:

{context}

Generate 10 MCQs.
"""


            # =============================================
            # SIMPLE EXPLANATION
            # =============================================

            else:

                prompt = f"""
You are a friendly academic teacher.

Explain the academic material below in very
simple language.

Use:

- Simple explanations
- Analogies
- Small examples
- Step-by-step reasoning

Do not introduce unsupported information.

ACADEMIC CONTENT:

{context}

Explain the concepts simply.
"""


            # =============================================
            # GENERATE RESPONSE
            # =============================================

            with st.spinner(
                "🤖 Generating..."
            ):

                try:

                    response = get_llm().invoke(
                        prompt
                    )

                    output = clean_response(
                        response
                    )

                    st.markdown(
                        output
                    )


                    # =====================================
                    # SOURCES
                    # =====================================

                    st.markdown(
                        "### 📚 Sources"
                    )


                    shown_sources = set()


                    for result in results:

                        source = result.metadata.get(
                            "source",
                            "Unknown"
                        )

                        page = result.metadata.get(
                            "page",
                            "Unknown"
                        )


                        source_key = (
                            source,
                            page
                        )


                        if source_key not in shown_sources:

                            shown_sources.add(
                                source_key
                            )


                            st.markdown(
                                f"📄 **{source} — "
                                f"Page {page}**"
                            )


                except Exception as e:

                    st.error(
                        f"❌ Error generating response: {e}"
                    )


# =========================================================
# NORMAL RAG CHAT
# =========================================================

st.divider()

st.header(
    "💬 Ask Your Academic Assistant"
)


question = st.chat_input(
    "Ask a question about your PDFs..."
)


if question:

    # =====================================================
    # CHECK DOCUMENTS
    # =====================================================

    if get_document_count() == 0:

        st.warning(
            "📄 Please upload an academic PDF first."
        )

        st.stop()


    # =====================================================
    # USER MESSAGE
    # =====================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # =====================================================
    # ASSISTANT
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        try:

            # ---------------------------------------------
            # RETRIEVE
            # ---------------------------------------------

            with st.spinner(
                "🔎 Searching your documents..."
            ):

                results = get_vectorstore().similarity_search(
                    question,
                    k=5
                )


            if not results:

                answer = (
                    "I could not find relevant "
                    "information in your uploaded "
                    "academic documents."
                )

                st.markdown(
                    answer
                )


            else:

                # -----------------------------------------
                # CONTEXT
                # -----------------------------------------

                context = build_context(
                    results
                )


                # -----------------------------------------
                # RAG PROMPT
                # -----------------------------------------

                prompt = f"""
You are a helpful academic RAG assistant.

Answer the student's question using ONLY
the provided academic document context.

IMPORTANT RULES:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the answer is not available in the
   context, say:

"I could not find this information
in the uploaded academic documents."

4. Explain concepts clearly.
5. Use simple academic language.
6. Use bullet points where appropriate.
7. Mention the relevant source PDF and
   page number at the end.

ACADEMIC DOCUMENT CONTEXT:

{context}

STUDENT QUESTION:

{question}

ANSWER:
"""


                # -----------------------------------------
                # GEMINI
                # -----------------------------------------

                with st.spinner(
                    "🤖 Generating answer..."
                ):

                    response = get_llm().invoke(
                        prompt
                    )


                # -----------------------------------------
                # CLEAN ANSWER
                # -----------------------------------------

                answer = clean_response(
                    response
                )


                # -----------------------------------------
                # DISPLAY
                # -----------------------------------------

                st.markdown(
                    answer
                )


                # -----------------------------------------
                # SOURCES
                # -----------------------------------------

                st.markdown(
                    "### 📚 Retrieved Sources"
                )


                shown_sources = set()


                for result in results:

                    source = result.metadata.get(
                        "source",
                        "Unknown"
                    )

                    page = result.metadata.get(
                        "page",
                        "Unknown"
                    )


                    source_key = (
                        source,
                        page
                    )


                    if source_key not in shown_sources:

                        shown_sources.add(
                            source_key
                        )


                        st.markdown(
                            f"📄 **{source} — "
                            f"Page {page}**"
                        )


        except Exception as e:

            answer = (
                "❌ Something went wrong while "
                "generating the answer."
            )

            st.error(
                f"Error: {e}"
            )


    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )