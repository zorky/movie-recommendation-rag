from dataclasses import dataclass
from functools import lru_cache
import json
import ollama
from qdrant_client import QdrantClient, models
from qdrant_client.models import ScoredPoint
from sentence_transformers import SentenceTransformer, CrossEncoder
import streamlit as st
import uuid

# Configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "movies"
MODEL_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIMENSION = 384
MODEL_RERANKING = "BAAI/bge-reranker-v2-m3"
# MODEL_RERANKING = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_API = "http://localhost:11434/v1"
MODEL_LLM = "mistral"
MAX_TOKENS = 500
TEMPERATURE = 0.3
TOP_P = 0.9

@dataclass
class RerankedScoredPoint:
    scored_point: ScoredPoint  # Le point original avec le nouveau score reranké
    original_score: float  # Le score original de Qdrant

@dataclass
class RerankerResult:
    """Résultats après reranking"""
    reranked_points: list[RerankedScoredPoint]  # nouveaux points avec scores rerankés
    original_points: list[ScoredPoint]  # points originaux avant reranking


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(MODEL_EMBEDDING)

embedding_model = load_embedding_model()

@st.cache_resource
def load_cross_encoder_model():    
    def _get_device_cpu_gpu() -> str:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)            
            return gpu_name
            # return "cuda"
        return "cpu"
    
    return CrossEncoder(MODEL_RERANKING,
                        device=_get_device_cpu_gpu())

reranking_model = load_cross_encoder_model()

def rerank(query, documents):
    BATCH_SIZE = 25
    print(f"Reranking {documents} documents with model '{MODEL_RERANKING}' on device '{reranking_model.device}'...")
    query_document_pairs = [(query, doc) for doc in documents]
    _scores = reranking_model.predict(query_document_pairs,
                                   batch_size=BATCH_SIZE,
                                   apply_softmax=False, # si le modèle n'était pas normalisé entre 0 et 1 (probas)
                                   show_progress_bar=False)
    print(f"Reranking scores: {_scores}")
    _ranked_results = list(zip(documents, _scores))
    _ranked_results.sort(key=lambda x: x[1], reverse=True)
    return _ranked_results

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
    # Example queries:
    #   a wormhole in space
    #   a serie with drugs
    query = st.text_input("Ask for a movie about...", "a wormhole in space")

    # Layout pour le checkbox et le bouton
    col_inference, col_button = st.columns([1, 1])
    with col_inference:
        st.session_state.do_inference = st.checkbox("Inference", value=st.session_state.do_inference)

    with col_button:
        st.write("")
        ask = st.button("Ask")

    # Action uniquement si le bouton est cliqué
    st.session_state.results = None
    if ask and query:  
        with st.spinner("Searching imb..."):              
            points = perform_query(st.session_state.qdrant_client, query)            
            print(f"Search results: {points}")            
            if points:                
                _documents = [
                        f"{r.payload['plot']}"
                        for r in points
                ]
                _results = rerank(query, _documents)
                print(f"Reranking scores: {_results}")

                reranked_points = []            
                for _, (text, score) in enumerate(_results):
                    for point in points:
                        if point.payload.get("plot", "") == text:
                            new_point = models.ScoredPoint(
                                id=point.id,
                                payload=point.payload,
                                vector=point.vector,
                                score=score,
                                version=point.version if hasattr(point, "version") else None,
                            )
                            reranked_points.append(
                                RerankedScoredPoint(scored_point=new_point, original_score=point.score)
                            )
                            break
                
                # reranker_results = RerankerResult(reranked_points=reranked_points, original_points=points)

                st.subheader("Movies Found")
                st.session_state.results = reranked_points
                for result in st.session_state.results:
                    payload = result.scored_point.payload
                    st.markdown(f"**{payload['title']}** ({payload['year']}) — reranked score: `{result.scored_point.score:.2f}` (original score: `{result.original_score:.2f}`)")
                    st.write(payload["plot"])
                    st.divider()                

        if st.session_state.do_inference:
            context = ""
            for result in st.session_state.results:
                payload = result.scored_point.payload
                context += f"\n- {payload['title']}: {payload['plot']}"
            print(f"Context for prompt:\n{context}")
            st.session_state.prompt = generate_prompt(context, query)
            print(f"Prompt sent to Ollama:\n{st.session_state.prompt}")

            with st.spinner("Searching..."):
                st.session_state.response = query_ollama(st.session_state.prompt)
        else:
            st.session_state.response = None

    if st.session_state.do_inference and st.session_state.prompt and st.session_state.response:
        st.subheader("Prompt Sent to Ollama")
        st.text_area("Here is the prompt sent to Ollama", st.session_state.prompt, height=200)
        st.subheader("Ollama's Recommendation")
        st.write(st.session_state.response)

if __name__ == "__main__":
    main()
