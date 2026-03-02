from dataclasses import dataclass
from functools import lru_cache
import json
import ollama
from qdrant_client import QdrantClient, models
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
import uuid
import torch

## Qdrant
QDRANT_IN_MEMORY = (
    True  # True : no persistence : in memory, False : persistence on server Qdrant
)

# Qdrant with server: QDRANT_HOST = "localhost"; QDRANT_PORT = 6333
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

# Qdrant collection name for movies : data.json to be indexed in this collection
COLLECTION_NAME = "movies"

# Embedding and Reranking Models
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
MODEL_RERANKING = "BAAI/bge-reranker-v2-m3"

# Ollama
LLM_API = "http://localhost:11434/v1"
LLM_MODEL = "mistral"

# inference parameters
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

    _device = _get_device_cpu_gpu()
    print(f"Loading CrossEncoder model {MODEL_RERANKING} on device {_device}...")
    return CrossEncoder(MODEL_RERANKING, device=_device)


embedding_model = load_embedding_model()
reranking_model = load_cross_encoder_model()


def rerank(query, documents):
    BATCH_SIZE = 25
    query_document_pairs = [(query, doc) for doc in documents]
    scores = reranking_model.predict(
        query_document_pairs, batch_size=BATCH_SIZE, apply_softmax=False
    )
    return list(zip(documents, scores))


@st.cache_data
def load_movies(file_path="data/data.json"):
    with open(file_path, "r") as f:
        return json.load(f)


@st.cache_resource
def initialize_qdrant():
    if QDRANT_IN_MEMORY:
        client = QdrantClient(":memory:")
        create_collection(client, COLLECTION_NAME)
        movies = load_movies()
        index_movies(movies, client, collection_name=COLLECTION_NAME)
    else:
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
            vectors_config=models.VectorParams(
                size=MODEL_DIMENSION, distance=models.Distance.COSINE
            ),
        )


def index_movies(movies, client, collection_name=COLLECTION_NAME):
    """Indexes movies data/data.json in Qdrant."""
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
                payload={
                    "title": title,
                    "content": content,
                    "year": movie["Year"],
                    "plot": plot,
                    "poster": movie.get("Poster", ""),
                },
            )
        )
    client.upsert(collection_name=collection_name, points=points)


def perform_query_and_rerank(client, query, collection_name=COLLECTION_NAME):
    """Performs a search in Qdrant and returns the reranked results."""
    query_vector = embedding_model.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name, query=query_vector, limit=3
    )
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
def cached_query_ollama(
    prompt,
    model_name=LLM_MODEL,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
):
    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": max_tokens,
        },
    )
    return response["message"]["content"]


@lru_cache(maxsize=128)
def fetch_image(url: str) -> bytes | None:
    try:
        # 403 Forbidden = IMDB bloque les requêtes non-navigateur -> utilisation de m.media-amazon.com
        url = url.replace("http://ia.media-imdb.com", "https://m.media-amazon.com")
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"fetch_image error for {url}: {e}")  # voir dans le terminal
        return None


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
    do_inference = st.checkbox(
        "Inference", value=st.session_state.do_inference, key="inference_checkbox"
    )
    st.session_state.do_inference = do_inference

    # Bouton de recherche
    if st.button("Search"):
        st.session_state.response = None
        if query:
            with st.spinner("Searching..."):
                try:
                    st.session_state.results = perform_query_and_rerank(
                        st.session_state.qdrant_client, query
                    )
                    if st.session_state.do_inference and st.session_state.results:
                        context = "\n".join(
                            [
                                f"- {r.payload['title']}: {r.payload['plot']}"
                                for r in st.session_state.results
                            ]
                        )
                        st.session_state.prompt = generate_prompt(context, query)
                        st.session_state.response = cached_query_ollama(
                            st.session_state.prompt,
                            model_name=LLM_MODEL,
                            temperature=TEMPERATURE,
                            top_p=TOP_P,
                            max_tokens=MAX_TOKENS,
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

    # Résultats
    if st.session_state.results:
        st.subheader("📽️ Movies Found")
        for result in st.session_state.results:
            payload = result.payload
            with st.container():
                col_img, col_info = st.columns([1, 3])
                with col_img:
                    poster_url = payload.get("poster", "")
                    if poster_url:
                        img_bytes = fetch_image(poster_url)
                        if img_bytes:
                            st.image(img_bytes, width=100)
                        else:
                            st.write("❌ fetch failed")
                with col_info:
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
