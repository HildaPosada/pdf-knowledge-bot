import os
import tempfile
import streamlit as st

from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Make sure HF token is passed from Codespaces secrets
if "HUGGINGFACEHUB_API_TOKEN" in os.environ:
    st.sidebar.success("✅ Hugging Face token loaded")
else:
    st.sidebar.error("❌ No Hugging Face token found!")

# Always use Hugging Face API for Path A
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

llm = HuggingFaceInferenceAPI(
    model_name=LLM_MODEL,
    temperature=0.1,
    max_tokens=512,
)

# Embeddings (free + light)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
Settings.llm = llm

st.set_page_config(page_title="PDF Knowledge Bot", page_icon="⚡", layout="wide")
st.title("⚡ PDF Knowledge Bot (Path A: Hugging Face API)")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

persist_dir = "storage"
os.makedirs(persist_dir, exist_ok=True)

def build_or_load_index(file_bytes_list):
    if file_bytes_list:
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, fobj in enumerate(file_bytes_list):
                with open(os.path.join(tmpdir, f"doc_{i}.pdf"), "wb") as out:
                    out.write(fobj.read())
            docs = SimpleDirectoryReader(tmpdir).load_data()

        index = VectorStoreIndex.from_documents(docs, show_progress=True)
        index.storage_context.persist(persist_dir=persist_dir)
        return index
    elif os.path.isdir(persist_dir) and len(os.listdir(persist_dir)) > 0:
        storage = StorageContext.from_defaults(persist_dir=persist_dir)
        return load_index_from_storage(storage)
    return None

index = build_or_load_index(uploaded_files)

if index is None:
    st.info("Upload a PDF to build the index.")
else:
    query_engine = index.as_query_engine(similarity_top_k=4, response_mode="compact")
    question = st.text_input("Ask a question about your PDF")
    if question:
        with st.spinner("Thinking..."):
            try:
                response = query_engine.query(question)
                st.markdown("### Answer")
                st.write(response.response)

                if hasattr(response, "source_nodes"):
                    st.markdown("### Sources")
                    for i, node in enumerate(response.source_nodes, start=1):
                        st.write(f"{i}. score={getattr(node, 'score', None)}")
                        st.code(node.node.get_content()[:500])
            except Exception as e:
                st.error(f"Error: {e}")
