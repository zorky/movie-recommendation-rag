from functools import lru_cache
import json
import ollama
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import streamlit as st
import uuid

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "movies"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
LLM_API = "http://localhost:11434/v1"
MODEL_LLM = "mistral"
MAX_TOKENS = 500
TEMPERATURE = 0.3
TOP_P = 0.9

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_EMBEDDING)

embedding_model = load_embedding_model()

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
        print("Creating collection and indexing movies...")
        create_collection(client, COLLECTION_NAME)
        movies = load_movies()
        index_movies(movies, client, collection_name=COLLECTION_NAME)
    else:
        print("The movie index already exists. It was not regenerated.")

    return client

def generate_uuid(imdb_id):
    return str(uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_DNS, imdb_id)))

def create_collection(client, collection_name=COLLECTION_NAME):
    existing_collections = [c.name for c in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=MODEL_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Collection '{collection_name}' created.")

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
    print("Data indexed in Qdrant.")

def perform_query(client, query, collection_name=COLLECTION_NAME):
    query_vector = embedding_model.encode(query).tolist()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
    )
    return results.points

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

def main():
    # Initialisation unique de Qdrant
    if "qdrant_client" not in st.session_state:
        st.session_state.qdrant_client = initialize_qdrant()

    # Gestion des états
    if "results" not in st.session_state:
        st.session_state.results = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "prompt" not in st.session_state:
        st.session_state.prompt = None
    if "do_inference" not in st.session_state:
        st.session_state.do_inference = False

    st.title("Movie Recommendations")
    query = st.text_input("Ask for a movie about...", "a wormhole in space")

    # Layout pour le checkbox et le bouton
    col_inference, col_button = st.columns([1, 1])
    with col_inference:
        st.session_state.do_inference = st.checkbox("Inference", value=st.session_state.do_inference)
    with col_button:
        st.write("")
        ask = st.button("Ask")

    # Action uniquement si le bouton est cliqué
    if ask and query:
        st.session_state.results = perform_query(st.session_state.qdrant_client, query)
        print(f"Search results: {st.session_state.results}")

        if st.session_state.do_inference:
            context = "\n".join([
                f"- {r.payload['title']}: {r.payload['plot']}"
                for r in st.session_state.results
            ])
            st.session_state.prompt = generate_prompt(context, query)
            print(f"Prompt sent to Ollama:\n{st.session_state.prompt}")

            with st.spinner("Searching..."):
                st.session_state.response = query_ollama(st.session_state.prompt)

    # Affichage des résultats
    if st.session_state.results:
        st.subheader("Movies Found")
        for result in st.session_state.results:
            payload = result.payload
            st.markdown(f"**{payload['title']}** ({payload['year']}) — score: `{result.score:.2f}`")
            st.write(payload["plot"])
            st.divider()

    if st.session_state.prompt:
        st.subheader("Prompt Sent to Ollama")
        st.text_area("Here is the prompt sent to Ollama", st.session_state.prompt, height=200)

    if st.session_state.response:
        st.subheader("Ollama's Recommendation")
        st.write(st.session_state.response)

if __name__ == "__main__":
    main()
