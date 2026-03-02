# Movie Recommendation System with ChromaDB or Qdrant and Ollama

This project implements a movie recommendation system using ChromaDB or Qdrant for indexing and searching movie data, and Ollama for generating personalized movie recommendations. It leverages machine learning models for generating embeddings and querying relevant results based on user input.

## Features
- **Movie Indexing**: Movies are indexed in Chroma or Qdrant with embeddings generated using SentenceTransformers.
- **Search**: Users can search for movies based on a query (e.g., "a wormhole in space").
- **Movie Recommendations**: Ollama generates personalized movie recommendations based on the search results.
- **Streamlit UI**: A simple web interface built with Streamlit to interact with the system.

## Prerequisites

- Python 3.12+
- UV (for dependency management)
- Qdrant or Chroma, Sentence-Transformers, Ollama, Streamlit

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/dotmobo/movie-recommendation-rag.git
cd movie-recommendation-rag
```

### 2. Activate the virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate # or under Win : source .venv/Scripts/activate
```

with GPU torch

```bash
uv sync --extra gpu
```

with only CPU torch

```bash
uv sync --extra cpu
```

## Running the Application

Qdrant in memory by default (QDRANT_IN_MEMORY=True)

Launch Ollama

```bash
docker compose -f ollama.yml up -d
```

With Qdrant server (QDRANT_IN_MEMORY=False) and Ollama, launch both

```bash
docker compose -f qdrant.yml -f ollama.yml up -d
```

To run the application and start the Streamlit UI:

With Qdrant

```bash
streamlit run rag/main_qdrant.py
```

With Chroma

```bash
streamlit run rag/main_chroma.py
```

This will open a web interface where you can search for movies and get recommendations.

## How It Works

1. **Movie Data**: Movies are loaded from a `data/data.json` file.
2. **Indexing**: Movie titles and plots are embedded using SentenceTransformer, then indexed in ChromaDB.
3. **Search**: Upon user query, the system retrieves the closest matches from ChromaDB.
4. **Recommendation**: A prompt is sent to Ollama to generate a movie recommendation based on the search results.

## Testing

This project includes unit tests for the core functions.

To run the tests, execute:

```bash
uv run pytest
```

## License

MIT License. See `LICENSE` for more details.
