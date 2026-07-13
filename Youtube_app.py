import os
from urllib.parse import urlparse, parse_qs

import streamlit as st
#from dotenv import load_dotenv

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)

#load_dotenv()

try:
    HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

st.set_page_config(
    page_title="YouTube AI Chat",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 YouTube AI Chat")
st.caption("Ask questions about any YouTube video using Retrieval-Augmented Generation (RAG).")
#--------------------------------------------------------------------------------------------------------------------------------
#Cache
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="ibm-granite/granite-embedding-107m-multilingual"
    )

@st.cache_resource
def load_llm():

    endpoint = HuggingFaceEndpoint(
        repo_id="MiniMaxAI/MiniMax-M2.5",
        task="text-generation",
        huggingfacehub_api_token=HF_TOKEN,
        max_new_tokens=512,
        temperature=0.3,
    )

    return ChatHuggingFace(llm=endpoint)

with st.spinner("Loading Embedding Model..."):
    embedding = load_embedding_model()

with st.spinner("Loading LLM..."):
    model = load_llm()

#--------------------------------------------------------------------------------------------------------------------------------
#prompt

prompt = PromptTemplate(
    template="""
You are a helpful AI assistant.

Answer ONLY using the supplied transcript.

If the transcript does not contain the answer, reply:

"I don't have enough information from this video."

Keep answers:

- concise
- factual
- easy to understand

Transcript Context:

{context}

Question:

{question}

Answer:
""",
    input_variables=["context", "question"],

)

#--------------------------------------------------------------------------------------------------------------------------------
#utilitites
def extract_video_id(url_or_id: str):

    url_or_id = url_or_id.strip()

    if "youtube.com" in url_or_id:

        parsed = urlparse(url_or_id)

        return parse_qs(parsed.query).get("v", [None])[0]

    elif "youtu.be" in url_or_id:

        return urlparse(url_or_id).path.strip("/")

    return url_or_id

def get_transcript(video_id):

    api = YouTubeTranscriptApi()

    try:
        transcript = api.fetch(video_id, languages=["en"])

    except Exception:

        transcript = api.fetch(video_id)

    return " ".join(chunk.text for chunk in transcript)

def format_context(docs):

    return "\n\n".join(doc.page_content for doc in docs)

# ============================================================
# Build Vector Store (Cached)
# ============================================================

@st.cache_resource(show_spinner=False)
def build_vector_store(video_id: str):

    transcript = get_transcript(video_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=lambda x: len(x.split()),
    )

    docs = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(
        docs,
        embedding,
    )

    return (
        vector_store,
        transcript,
        len(transcript.split()),
        len(docs),
    )

# ============================================================
# Build LangChain Pipeline
# ============================================================

def build_chain(retriever):

    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_context),
            "question": RunnablePassthrough(),
        }
    )

    chain = (
        parallel_chain
        | prompt
        | model
        | StrOutputParser()
    )

    return chain


# ============================================================
# Session State
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

if "video_id" not in st.session_state:
    st.session_state.video_id = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "word_count" not in st.session_state:
    st.session_state.word_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.header("📹 Load a YouTube Video")

    video_input = st.text_input(
        "Paste YouTube URL or Video ID",
        placeholder="https://www.youtube.com/watch?v=...",
    )

    load_btn = st.button(
        "Load Video",
        use_container_width=True,
    )

    if load_btn:

        if not video_input.strip():

            st.warning("Please enter a YouTube URL.")

        else:

            video_id = extract_video_id(video_input)

            if not video_id:

                st.error("Invalid YouTube URL.")

            elif video_id == st.session_state.video_id:

                st.info("Video already loaded.")

            else:

                try:

                    with st.spinner("Fetching transcript..."):

                        vector_store, transcript, words, chunks = build_vector_store(video_id)
                        st.session_state.transcript = transcript
                    st.session_state.video_loaded = True
                    st.session_state.video_id = video_id
                    st.session_state.vector_store = vector_store
                    st.session_state.word_count = words
                    st.session_state.chunk_count = chunks
                    st.session_state.messages = []

                    st.success("Video indexed successfully!")

                except (
                    TranscriptsDisabled,
                    NoTranscriptFound,
                ):

                    st.error(
                        "No transcript available for this video."
                    )

                except Exception as e:

                    st.error(str(e))

    if st.session_state.video_loaded:

        st.divider()

        st.subheader("Loaded Video")

        st.video(
            f"https://www.youtube.com/watch?v={st.session_state.video_id}"
        )

        st.metric(
            "Transcript Words",
            st.session_state.word_count,
        )

        st.metric(
            "Transcript Chunks",
            st.session_state.chunk_count,
        )

        st.caption(
            f"Video ID: {st.session_state.video_id}"
        )

        if st.button(
            "🗑 Clear Chat",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.rerun()

# ============================================================
# Main Interface
# ============================================================

if not st.session_state.video_loaded:

    st.info(
        "👈 Paste a YouTube URL or Video ID in the sidebar and click **Load Video**."
    )

    st.markdown(
        """
### Example Questions

- Summarize this video.
- What are the main topics discussed?
- Explain the key concepts.
- What conclusions does the speaker make?
- List the important points.
"""
    )

else:

    st.subheader("💬 Chat with the Video")

    # --------------------------------------------------------
    # Display Previous Messages
    # --------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])

            if (
                message["role"] == "assistant"
                and "sources" in message
                and message["sources"]
            ):

                with st.expander("📚 Retrieved Transcript Chunks"):

                    for i, doc in enumerate(message["sources"], start=1):

                        st.markdown(f"**Chunk {i}**")

                        st.info(doc.page_content)

    # --------------------------------------------------------
    # User Question
    # --------------------------------------------------------

    question = st.chat_input(
        "Ask anything about this YouTube video..."
    )

    if question:

        # ----------------------------------------------
        # Show User Message
        # ----------------------------------------------

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        # ----------------------------------------------
        # Build Retriever
        # ----------------------------------------------

        retriever = st.session_state.vector_store.as_retriever(

            search_type="mmr",

            search_kwargs={
                "k": 6,
                "fetch_k": 20,
                "lambda_mult": 0.7,
            },
        )

        retrieved_docs = retriever.invoke(question)

        # ----------------------------------------------
        # Build Chain
        # ----------------------------------------------

        chain = build_chain(retriever)

        # ----------------------------------------------
        # Assistant Message
        # ----------------------------------------------

        with st.chat_message("assistant"):

            placeholder = st.empty()

            answer = ""

            with st.spinner("Thinking..."):

                try:

                    # Streaming Response
                    for chunk in chain.stream(question):

                        answer += chunk

                        placeholder.markdown(answer + "▌")

                    placeholder.markdown(answer)

                except Exception as e:

                    answer = f"Error: {e}"

                    placeholder.error(answer)

            # ------------------------------------------
            # Show Sources
            # ------------------------------------------

            with st.expander("📚 Retrieved Transcript Chunks"):

                for i, doc in enumerate(retrieved_docs, start=1):

                    st.markdown(f"### Chunk {i}")

                    st.write(doc.page_content)

        # ----------------------------------------------
        # Save Assistant Response
        # ----------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": retrieved_docs,
            }
        )

# ============================================================
# Footer
# ============================================================

# st.divider()

# col1, col2, col3 = st.columns(3)

# with col1:
#     st.caption("🤖 Embedding")
#     st.write("IBM Granite 107M")

# with col2:
#     st.caption("🧠 LLM")
#     st.write("MiniMax M2.5")

# with col3:
#     st.caption("🔎 Retrieval")
#     st.write("FAISS + MMR")


# st.markdown(
#     """
# ---
# Made with ❤️ using

# - Streamlit
# - LangChain
# - HuggingFace
# - FAISS
# - YouTube Transcript API
# """
# )
