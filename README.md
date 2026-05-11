# Bocconi Buddy

Bocconi Buddy is a full-stack AI assistant built during the Bocconi AI Hackathon. It helps students ask practical questions about moving to Milan, campus life, study abroad, and career readiness, then answers with grounded citations from a local Bocconi-focused knowledge base.

The product is intentionally conservative: when the available sources do not clearly support an answer, it abstains instead of guessing.

## What It Does

- Answers student questions in English or Italian.
- Routes questions into four verticals:
  - `relocation`
  - `life_on_campus`
  - `study_abroad`
  - `career_readiness`
- Uses hybrid retrieval over a local document corpus.
- Returns answers with source file paths.
- Provides a polished dark-mode React interface for asking questions and reviewing cited sources.

## Architecture

```text
frontend/  Vite + React chat UI
backend/   FastAPI RAG service
data/      Bocconi and public student-support sources
```

The backend exposes:

- `GET /health`
- `POST /ask`

The `/ask` response schema is:

```json
{
  "answer": "string",
  "sources": ["string"],
  "verticale": "relocation | life_on_campus | study_abroad | career_readiness"
}
```

## Retrieval Pipeline

The RAG pipeline combines:

- Dense retrieval with OpenAI embeddings and FAISS.
- Sparse retrieval with BM25.
- Rule-based vertical routing.
- Query expansion for key bilingual administrative terms.
- Deterministic abstention gates.
- Strict grounding prompts to avoid unsupported answers.

The vector index is generated offline from the files in `backend/data/`.

## Tech Stack

### Backend

- Python
- FastAPI
- OpenAI API
- FAISS
- BM25
- Tenacity retries
- Language detection

### Frontend

- React
- TypeScript
- Vite
- CSS design system
- Railway deployment

## Local Setup

### 1. Backend environment

Create a local `.env` file from the template:

```bash
cp .env.example .env
```

Then add your own key locally:

```bash
OPENAI_API_KEY=sk-...
```

Never commit `.env`.

### 2. Install backend dependencies

```bash
cd backend
uv sync
```

### 3. Build the retrieval index

The generated index is intentionally not committed to GitHub because it is a large binary artifact.

```bash
cd backend
uv run python scripts/build_index.py
```

This creates:

```text
backend/data/index/
```

### 4. Run the backend

```bash
cd backend
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

Set the frontend backend URL with:

```bash
VITE_BACKEND_URL=http://localhost:8000
```

## Deployment

The app is designed as two Railway services:

- Backend service: FastAPI app.
- Frontend service: Vite static app.

Set these environment variables in Railway:

```text
Backend:
OPENAI_API_KEY=your_key_here

Frontend:
VITE_BACKEND_URL=https://your-backend-url
```

Do not commit API keys or `.env` files.

## Security Notes

- `.env` and `.env.*` are ignored.
- Generated vector indexes are ignored.
- Railway upload folders are ignored.
- No OpenAI API key is stored in source code.

Before publishing, you can verify:

```bash
git status --ignored
git grep "sk-"
```

## Screenshots

Add a landing page screenshot here after uploading it to GitHub:

```md
![Bocconi Buddy landing page](./docs/landing-page.png)
```

## Project Context

This project was built for the Bocconi AI Hackathon with a focus on practical usefulness, source-grounded answers, and a student-facing product experience.
