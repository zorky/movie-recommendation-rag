from dataclasses import dataclass
from functools import lru_cache
import json
import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
import uuid
import torch

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "movies"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
MODEL_RERANKING = "BAAI/bge-reranker-v2-m3"
LLM_API = "http://localhost:11434/v1"
MODEL_LLM = "mistral"
MAX_TOKENS = 500
TEMPERATURE = 0.3
TOP_P = 0.9

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_EMBEDDING)

@st.cache_resource
def load_cross_encoder_model():
    def _get_device_cpu_gpu() -> str:
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return CrossEncoder(MODEL_RERANKING, device=_get_device_cpu_gpu())

embedding_model = load_embedding_model()
reranking_model = load_cross_encoder_model()

def rerank(query, documents):
    BATCH_SIZE = 25
    query_document_pairs = [(query, doc) for doc in documents]
    scores = reranking_model.predict(query_document_pairs, batch_size=BATCH_SIZE, apply_softmax=False)
    return list(zip(documents, scores))

@st.cache_data
def load_movies(file_path="data/data.json"):
    with open(file_path, "r") as f:
        return json.load(f)

@st.cache_resource
def initialize_qdrant():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    collections = client.get_collections().collections
    collection_exists = any(col.name == COLLECTION_NAME for col in collections)

    if not collection_exists:
        create_collection(client, COLLECTION_NAME)
        movies = load_movies()
        index_movies(movies, client, collection_name=COLLECTION_NAME)

    return client

def generate_uuid(imdb_id):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, imdb_id))

def create_collection(client, collection_name=COLLECTION_NAME):
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=MODEL_DIMENSION, distance=models.Distance.COSINE),
        )

def index_movies(movies, client, collection_name=COLLECTION_NAME):
    points = []
    for movie in movies:
        doc_id = generate_uuid(movie["imdbID"])
        title = movie["Title"]
        plot = movie["Plot"]
        content = f"{title}: {plot}"
        points.append(
            models.PointStruct(
                id=doc_id,
                vector=embedding_model.encode(content).tolist(),
                payload={"title": title, "content": content, "year": movie["Year"], "plot": plot},
            )
        )
    client.upsert(collection_name=collection_name, points=points)

def perform_query_and_rerank(client, query, collection_name=COLLECTION_NAME):
    query_vector = embedding_model.encode(query).tolist()
    results = client.query_points(collection_name=collection_name, query=query_vector, limit=3)
    points = results.points

    if not points:
        return []

    documents = [point.payload["plot"] for point in points]
    plot_to_point = {point.payload["plot"]: point for point in points}
    reranked_pairs = rerank(query, documents)

    reranked_points = []
    for text, score in reranked_pairs:
        point = plot_to_point[text]
        reranked_points.append(
            models.ScoredPoint(
                id=point.id,
                payload=point.payload,
                vector=point.vector,
                score=score,
                version=point.version if hasattr(point, "version") else None,
            )
        )

    return reranked_points

def generate_prompt(context, query):
    return f"""
You are a knowledgeable DVD salesperson with expertise in movies. Your task is to recommend movies to customers, but you can only suggest films that are available in the store's inventory. Make sure your recommendations are based solely on the list of movies provided.
Answer in French.

Context: Below is a list of movies currently available in the store:
{context}

Customer's Question: Ask for a movie about {query}

Your Movie Recommendations (only from the available list):
"""

@st.cache_data()
def cached_query_ollama(prompt, model_name=MODEL_LLM, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS):
    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_p": top_p,
        }
    )
    return response["message"]["content"]

def main():
    st.set_page_config(layout="centered")  # Force le centrage
    st.title("🎬 Movie Recommendations")

    if "qdrant_client" not in st.session_state:
        st.session_state.qdrant_client = initialize_qdrant()
    if "results" not in st.session_state:
        st.session_state.results = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "do_inference" not in st.session_state:
        st.session_state.do_inference = False

    # Limite la largeur du contenu
    st.markdown(
        """
        <style>
            .st-emotion-cache-1v0mbdj {
                max-width: 800px;
                margin: 0 auto;
            }
            .stButton>button {
                width: 100%;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Barre de recherche et options
    query = st.text_input(
        "Ask for a movie about...",
        "a wormhole in space",
        placeholder="e.g. a serie with drugs",
        label_visibility="collapsed",
    )
    do_inference = st.checkbox("Inference", value=st.session_state.do_inference, key="inference_checkbox")
    st.session_state.do_inference = do_inference

    # Bouton de recherche
    if st.button("Search"):
        st.session_state.response = None  # Reset previous response
        if query:
            with st.spinner("Searching..."):
                try:
                    st.session_state.results = perform_query_and_rerank(st.session_state.qdrant_client, query)
                    if st.session_state.do_inference and st.session_state.results:
                        context = "\n".join([f"- {r.payload['title']}: {r.payload['plot']}" for r in st.session_state.results])
                        st.session_state.prompt = generate_prompt(context, query)
                        st.session_state.response = cached_query_ollama(
                            st.session_state.prompt,
                            model_name=MODEL_LLM,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_TOKENS
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    # Résultats
    if st.session_state.results:
        st.subheader("📽️ Movies Found")
        for result in st.session_state.results:
            payload = result.payload
            with st.container():
                st.markdown(f"**{payload['title']}** ({payload['year']})")
                st.caption(payload["plot"])
                st.divider()

    # Réponse LLM
    if st.session_state.response:
        st.subheader("🤖 Recommendation")
        st.markdown(st.session_state.response)

    # Détails techniques (masqués par défaut)
    with st.expander("Technical Details", expanded=False):
        if hasattr(st.session_state, "prompt") and st.session_state.prompt:
            st.text_area("Prompt sent to Ollama", st.session_state.prompt, height=150)

if __name__ == "__main__":
    main()
