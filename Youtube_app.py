import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
HF_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.set_page_config(page_title="YouTube AI Chat", page_icon="🎬", layout="wide")
st.title("🎬 YouTube AI Chat")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1: Cache models with @st.cache_resource
# They load ONCE and are shared across all reruns.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="ibm-granite/granite-embedding-107m-multilingual"
    )

@st.cache_resource
def load_llm():
    llm = HuggingFaceEndpoint( repo_id="MiniMaxAI/MiniMax-M2.5", task="text-generation", huggingfacehub_api_token = HF_TOKEN )
    return ChatHuggingFace(llm=llm)

# Load models (shows spinner only on first load)
with st.spinner("Loading embedding model..."):
    embedding = load_embedding_model()

with st.spinner("Loading LLM..."):
    model = load_llm()

# ── Prompt ────────────────────────────────────────────────────────────────────
prompt = PromptTemplate(
    template="""
You are a helpful assistant that answers questions based on a YouTube video transcript.
Answer ONLY from the provided transcript context.
If the context is insufficient, say "I don't have enough information from this video to answer that."

Context:
{context}

Question: {question}

Answer:
""",
    input_variables=["context", "question"]
)

# ── Helper functions ──────────────────────────────────────────────────────────
def get_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    #transcript_list = api.fetch(video_id=video_id, languages=["en"])
    transcript = api.fetch(video_id)
    #return " ".join(chunk.text for chunk in transcript_list)
    return " ".join(chunk.text for chunk in transcript)

def format_context(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)

# ─────────────────────────────────────────────────────────────────────────────
# FIX 2: Cache the vector store per video_id
# If the same video ID is passed again, FAISS is NOT rebuilt.
# @st.cache_data is fine here since FAISS can be pickled.
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_vector_store(video_id: str):
    transcript = get_transcript(video_id)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=lambda text: len(text.split()),
    )
    chunks = splitter.create_documents([transcript])
    return FAISS.from_documents(chunks, embedding), len(transcript.split())

def build_chain(retriever):
    parallel_chain = RunnableParallel({
        "context": retriever | RunnableLambda(format_context),
        "question": RunnablePassthrough()
    })
    return parallel_chain | prompt | model | StrOutputParser()

# ─────────────────────────────────────────────────────────────────────────────
# FIX 3: Session state for chat history and vector store
# Normal variables reset on every rerun — session_state persists.
# ─────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

# ── Sidebar: Video loader ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("📹 Load a Video")
    video_id_input = st.text_input("YouTube Video ID", placeholder="e.g. dQw4w9WgXcQ")
    load_btn = st.button("Load Video →", use_container_width=True)

    if load_btn and video_id_input.strip():
        vid = video_id_input.strip()

        # Only rebuild if it's a new video
        if vid != st.session_state.current_video_id:
            try:
                with st.spinner("Fetching transcript & building vector store..."):
                    vector_store, word_count = build_vector_store(vid)

                st.session_state.current_video_id = vid
                st.session_state.video_loaded = True
                st.session_state.messages = []  # clear chat for new video
                st.success(f"✅ Ready! ({word_count} words)")

            except (TranscriptsDisabled, NoTranscriptFound):
                st.error("❌ No English transcript found for this video.")
            except Exception as e:
                st.error(f"❌ Error: {e}")
        else:
            st.info("This video is already loaded.")

    if st.session_state.video_loaded:
        st.markdown("---")
        st.markdown(f"**Loaded:** `{st.session_state.current_video_id}`")
        st.image(
            f"https://img.youtube.com/vi/{st.session_state.current_video_id}/mqdefault.jpg",
            use_container_width=True
        )
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Main: Chat interface ──────────────────────────────────────────────────────
if not st.session_state.video_loaded:
    st.info("👈 Paste a YouTube video ID in the sidebar and click **Load Video** to start.")

else:
    # ─────────────────────────────────────────────────────────────────────────
    # FIX 4: Render full chat history on every rerun
    # This is how Streamlit chat works — no loop needed.
    # ─────────────────────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # st.chat_input stays at the bottom — returns value when user submits
    if question := st.chat_input("Ask anything about the video..."):

        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Build retriever + chain and get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                vector_store, _ = build_vector_store(st.session_state.current_video_id)
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 4}
                )
                chain = build_chain(retriever)
                answer = chain.invoke(question)

            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
