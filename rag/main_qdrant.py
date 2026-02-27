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

# available_rerankers = {
#     "BAAI/bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3",
#     "cross-encoder/ms-marco-MiniLM-L-6-v2": "cross-encoder/ms-marco-MiniLM-L-6-v2",
# }
# selected_reranker = st.selectbox("Choose reranking model", options=list(available_rerankers.keys()))
# MODEL_RERANKING = available_rerankers[selected_reranker]

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_EMBEDDING)

@st.cache_resource
def load_cross_encoder_model():
    def _get_device_cpu_gpu() -> str:
        if torch.cuda.is_available():
            # gpu_name = torch.cuda.get_device_name(0)
            # st.info(f"Using GPU: {gpu_name}")
            return "cuda"
        # st.info("Using CPU for reranking.")
        return "cpu"
    # st.info(f"Loading reranking model '{MODEL_RERANKING}'...")
    return CrossEncoder(MODEL_RERANKING, device=_get_device_cpu_gpu())

embedding_model = load_embedding_model()
reranking_model = load_cross_encoder_model()

def rerank(query, documents):
    BATCH_SIZE = 25
    # st.info(f"Reranking {len(documents)} documents with model '{MODEL_RERANKING}'...")
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
        st.info("Creating collection and indexing movies...")
        create_collection(client, COLLECTION_NAME)
        movies = load_movies()
        index_movies(movies, client, collection_name=COLLECTION_NAME)
    else:
        st.info("The movie index already exists.")

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
        st.info(f"Collection '{collection_name}' created.")

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
    st.info("Data indexed in Qdrant.")

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

def query_ollama(prompt, model_name=MODEL_LLM):
    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

@st.cache_data()
def cached_query_ollama(prompt, model_name=MODEL_LLM, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_TOKENS):
    """Version cacheable de query_ollama, utilisant prompt, model_name, temperature, top_p et max_tokens comme clé de cache."""
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
    if "qdrant_client" not in st.session_state:
        st.session_state.qdrant_client = initialize_qdrant()

    if "results" not in st.session_state:
        st.session_state.results = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "prompt" not in st.session_state:
        st.session_state.prompt = None
    if "do_inference" not in st.session_state:
        st.session_state.do_inference = False

    st.title("Movie Recommendations")
    # Example queries:
    #   a wormhole in space
    #   a serie with drugs
    query = st.text_input("Ask for a movie about...", "a wormhole in space")

    col_inference, col_button = st.columns([1, 1])
    with col_inference:
        st.session_state.do_inference = st.checkbox("Inference", value=st.session_state.do_inference)
    with col_button:
        st.write("")
        ask = st.button("Ask")

    if ask and query:
        with st.spinner("Searching..."):
            try:
                st.session_state.results = perform_query_and_rerank(st.session_state.qdrant_client, query)
            except Exception as e:
                st.error(f"Error during reranking: {e}")
                st.session_state.results = None

        if st.session_state.do_inference and st.session_state.results:
            context = "\n".join([f"- {r.payload['title']}: {r.payload['plot']}" for r in st.session_state.results])
            st.session_state.prompt = generate_prompt(context, query)
            with st.spinner("Generating recommendation..."):
                try:
                    st.session_state.response = cached_query_ollama(
                        st.session_state.prompt,
                        model_name=MODEL_LLM,
                        temperature=TEMPERATURE,
                        top_p=TOP_P,
                        max_tokens=MAX_TOKENS
                    )
                except Exception as e:
                    st.error(f"Error during inference: {e}")
                    st.session_state.response = None

    if st.session_state.results:
        st.subheader("Movies Found")
        for result in st.session_state.results:
            payload = result.payload
            st.markdown(f"**{payload['title']}** ({payload['year']})")
            # st.markdown(f"**{payload['title']}** ({payload['year']}) — reranked score: `{result.score:.2f}` original score: `{result.score:.2f}`")
            st.write(payload["plot"])
            st.divider()    

    if st.session_state.response:
        st.subheader("Ollama's Recommendation")
        st.write(st.session_state.response)

    if st.session_state.prompt:
        st.subheader("Prompt Sent to Ollama")
        st.text_area("Here is the prompt sent to Ollama", st.session_state.prompt, height=200)

if __name__ == "__main__":
    main()
