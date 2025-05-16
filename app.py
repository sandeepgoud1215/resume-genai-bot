import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(page_title="Smart Resume Q&A Bot", layout="wide")
st.title("🧠 Resume Q&A Bot – GenAI (flan-t5-base Version)")

# ✅ Load embedding and Q&A model
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline("text2text-generation", model="google/flan-t5-base")
    return embed_model, qa_model

embed_model, qa_model = load_models()

# ✅ Upload Resume
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    with open("resume.pdf", "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Resume uploaded!")

    # ✅ Extract Resume Text
    with fitz.open("resume.pdf") as doc:
        resume_text = " ".join([page.get_text() for page in doc])

    # ✅ Break into chunks
    CHUNK_SIZE = 300
    chunks = [resume_text[i:i+CHUNK_SIZE] for i in range(0, len(resume_text), CHUNK_SIZE)]

    # ✅ Create FAISS Index
    embeddings = embed_model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings[0].shape[0])
    index.add(np.array(embeddings))

    # ✅ Question Box
    st.subheader("❓ Ask a question about your resume:")
    question = st.text_input("Example: 'Is this resume good for a Data Analyst role?'")

    if question:
        # Retrieve top 3 chunks
        q_embedding = embed_model.encode([question])
        D, I = index.search(np.array(q_embedding), k=3)
        best_chunk = " ".join([chunks[i] for i in I[0]])

        # ✅ Generate smart answer using flan-t5-base
        prompt = (
            "You are an intelligent assistant helping review resumes.\n"
            f"Here is a section of the resume:\n{best_chunk}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        with st.spinner("Thinking..."):
            answer = qa_model(prompt, max_new_tokens=100)[0]['generated_text']

        st.markdown(f"**🧠 Answer:** {answer}")
