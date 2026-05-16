# SHL Assessment Recommendation Assistant

Conversational SHL product recommender using:

- Streamlit for the frontend
- ChromaDB for vector search
- Gemini API for natural language responses
- SHL product catalog JSON as the dataset
- Local hashed embeddings, so Chroma works without downloading a model

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and add your Gemini key:

```text
GEMINI_API_KEY=your_key_here
```

For faster local or free-tier API responses, you can skip Gemini wording and use
the deterministic fallback answer:

```text
USE_GEMINI=false
```

## Run

```powershell
streamlit run streamlit_app.py
```

## Run API

```powershell
uvicorn api:app --reload --port 8000
```

Health check:

```text
GET http://localhost:8000/health
```

Chat:

```text
POST http://localhost:8000/chat
```

Example body:

```json
{
  "message": "I want to hire a junior AI developer"
}
```

The first run builds the local Chroma index from the SHL catalog. If the catalog URL is unavailable, place a downloaded copy at:

```text
data/shl_product_catalog.json
```

By default, Chroma runs in memory because Windows OneDrive folders can cause
SQLite disk I/O errors. To enable persistence, set:

```text
CHROMA_PERSIST=true
CHROMA_PATH=data/chroma
```

You can change this with `CHROMA_PATH` in `.env`.

## How It Works

1. Loads the SHL catalog JSON.
2. Builds searchable text from product name, description, keys, job levels, languages, duration, remote, and adaptive fields.
3. Stores catalog entries in ChromaDB using sentence-transformer embeddings.
4. Retrieves the most relevant SHL assessments for each user request.
5. Uses Gemini to create a conversational answer while preserving structured recommendations.
