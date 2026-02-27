from functools import lru_cache
import json
import ollama
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import SentenceTransformer
import streamlit as st
import uuid

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "movies"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384

DO_INFERENCE = False
LLM_API = "http://localhost:11434/v1"
MODEL_LLM = "mistral"
# MODEL_LLM = "llama3.2"
MAX_TOKENS=500
TEMPERATURE=0.3
TOP_P=0.9

embedding_model = SentenceTransformer(MODEL_EMBEDDING)
    
def generate_uuid(imdb_id):
    """Generates a UUID from a string (e.g., imdbID)."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, imdb_id))

@st.cache_data
def load_movies(file_path="data/data.json"):
    """Loads movie data from the JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)

@st.cache_resource
def initialize_qdrant():
    """Initializes, indexes if needed, and returns a Qdrant client."""
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

def index_movies(movies, client, collection_name=COLLECTION_NAME):
    """Indexes movies in Qdrant."""    
    points = []

    for movie in movies:
        doc_id = generate_uuid(movie["imdbID"])
        title = movie["Title"]
        plot = movie["Plot"]
        content = f"{title}: {plot}"

        points.append(
            models.PointStruct(
                id=doc_id,
                vector=models.Document(text=content, model=MODEL_EMBEDDING),
                payload={"title": title, "content": content, "year": movie["Year"], "plot": plot},
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )

    print("Data indexed in Qdrant.")

def create_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME):
    """Creates a Qdrant collection if it doesn't already exist."""
    existing_collections = [c.name for c in client.get_collections().collections]

    if collection_name in existing_collections:
        print(f"Collection '{collection_name}' already exists, skipping creation.")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=MODEL_DIMENSION,
            distance=models.Distance.COSINE,
        ),
    )

    print(f"Collection '{collection_name}' created.")

def perform_query(client, query, collection_name=COLLECTION_NAME):
    """Performs a search in Qdrant and returns the results."""    
    query_vector = embedding_model.encode(query).tolist()

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=3,
    )
    return results.points

def generate_prompt(context, query):
    """Generates a prompt for Ollama where the model acts as a DVD salesperson."""
    return f"""
You are a knowledgeable DVD salesperson with expertise in movies. Your task is to recommend movies to customers, but you can only suggest films that are available in the store's inventory. Make sure your recommendations are based solely on the list of movies provided.
Answer in French.

Context: Below is a list of movies currently available in the store:
{context}

Customer's Question: Ask for a movie about {query}

Your Movie Recommendations (only from the available list):
"""

def init_llm_chat():    
    from langchain_openai import ChatOpenAI    
    
    return ChatOpenAI(
        model=MODEL_LLM,
        openai_api_base=LLM_API,
        openai_api_key="dummy-key-ollama",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

def query_ollama(prompt, model_name="myllama"):
    """Queries Ollama with the given prompt."""
    client = ollama.Client(host="http://localhost:11434")
    response = client.chat(model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

def main():
    client = initialize_qdrant()

    # Initialisation du session_state pour éviter de recharger à chaque interaction
    if "results" not in st.session_state:
        st.session_state.results = None
    if "response" not in st.session_state:
        st.session_state.response = None
    if "prompt" not in st.session_state:
        st.session_state.prompt = None

    st.title("Movie Recommendations")
    query = st.text_input("Ask for a movie about...", "a wormhole in space")

    # Inference radio + bouton Ask côte à côte
    col_inference, col_button = st.columns([1, 1])
    with col_inference:
        do_inference = st.checkbox("Inference", value=DO_INFERENCE)

    with col_button:
        st.write("")  # petit espace pour aligner verticalement
        ask = st.button("Ask")

    if ask and query:
        st.session_state.results = perform_query(client, query)
        print(f"Search results: {st.session_state.results}")

        if do_inference:
            context = "\n".join([
                f"- {r.payload['title']}: {r.payload['plot']}"
                for r in st.session_state.results
            ])
            st.session_state.prompt = generate_prompt(context, query)
            print(f"Prompt sent to Ollama:\n{st.session_state.prompt}")

            with st.spinner("Searching..."):
                st.session_state.response = query_ollama(st.session_state.prompt, model_name=MODEL_LLM)
        else:
            st.session_state.prompt = None
            st.session_state.response = None

    # Affichage persistant des résultats
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
