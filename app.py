import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# ✅ Page settings
st.set_page_config(page_title="Smart Resume Q&A Bot", layout="wide")
st.title("🧠 Resume Q&A Bot – GenAI Powered")

# ✅ Load models (only once)
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, qa_model

embed_model, qa_model = load_models()

# ✅ File upload
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Resume uploaded!")

    # ✅ Extract text from resume
    with fitz.open("resume.pdf") as doc:
        resume_text = " ".join([page.get_text() for page in doc])

    # ✅ Chunk text
    CHUNK_SIZE = 300
    chunks = [resume_text[i:i+CHUNK_SIZE] for i in range(0, len(resume_text), CHUNK_SIZE)]

    # ✅ Generate embeddings and store in FAISS index
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    # ✅ Ask a question
    question = st.text_input("Ask your question about the resume 👇")

    # ✅ Show example questions
    st.markdown("#### 💬 Example Questions:")
    st.markdown("""
    - ✅ Is this resume suitable for a Data Analyst role?
    - 💼 What roles is this resume best suited for?
    - 🧠 What are the candidate’s top skills?
    - 🧪 Does the candidate know Python or SQL?
    - 🎓 What is the educational background?
    - 📊 Are any data analysis projects mentioned?
    - 🔧 What tools or technologies are listed?
    - 🔍 Does this resume include internship or job experience?
    - 📝 How can this resume be improved?
    """)

    if question:
        # ✅ Retrieve top 3 most relevant chunks
        q_embedding = embed_model.encode([question])
        D, I = index.search(np.array(q_embedding), k=3)
        best_chunk = " ".join([chunks[i] for i in I[0]])

        # ✅ Build smart GenAI prompt
        prompt = (
            "You are an intelligent assistant helping review resumes.\n"
            f"Here is a section of the resume:\n{best_chunk}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # ✅ Generate answer with flan-t5
        with st.spinner("🤖 Thinking..."):
            answer = qa_model(prompt, max_new_tokens=100)[0]['generated_text']

        # ✅ Show answer
        st.markdown("### 🧠 Answer:")
        st.success(answer)
