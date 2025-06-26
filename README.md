# Aptos RAG Server

A simple Retrieval-Augmented Generation (RAG) server for managing and querying a knowledge base of Aptos documentation and DeFi snippets. Features a FastAPI backend and a web UI for snippet management.

## Quick Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd aptos.dev
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

3. **Set environment variable:**
   - `PINECONE_API_KEY`: Your Pinecone API key

4. **Run the server locally:**
   ```bash
   uvicorn rag_server:app --reload
   ```
   The server will be available at [http://localhost:8000](http://localhost:8000).

5. **Access the web UI:**
   Open your browser and go to [http://localhost:8000](http://localhost:8000) to manage and query snippets.

---

## API Documentation

All endpoints are under the root path (`/`).

### `GET /`
- Returns the web UI for snippet management.

### `POST /add`
- Add a new snippet.
- **Request Body:**
  ```json
  { "text": "<snippet text>" }
  ```
- **Response:**
  ```json
  { "id": "<snippet_id>", ... }
  ```

### `POST /edit`
- Edit an existing snippet.
- **Request Body:**
  ```json
  { "id": "<snippet_id>", "text": "<new text>" }
  ```
- **Response:**
  ```json
  { "success": true }
  ```

### `POST /delete`
- Delete a snippet.
- **Request Body:**
  ```json
  { "id": "<snippet_id>" }
  ```
- **Response:**
  ```json
  { "success": true }
  ```

### `GET /list`
- List all snippets.
- **Response:**
  ```json
  [ { "id": ..., "text": ..., "created": ... }, ... ]
  ```

### `GET /download`
- Download all snippets as JSON.

### `POST /query`
- Query the knowledge base for relevant snippets (RAG retrieval).
- **Request Body:**
  ```json
  { "query": "<your question>" }
  ```
- **Response:**
  ```json
  { "results": [ { "id": ..., "text": ..., "score": ... }, ... ] }
  ```

---

## Embedding Model & Vector Details

- **Model:** [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) (Hugging Face)
- **Vector Dimension:** 768
- **Vector Type:** Dense
- **How it works:**
  - Each snippet is embedded using the BAAI/bge-base-en-v1.5 model to produce a 768-dimensional dense vector.
  - These vectors are stored and indexed in Pinecone for fast similarity search and retrieval.
  - When querying, the user question is embedded with the same model, and the most similar snippets are retrieved based on vector similarity.

---

For more details, see the code and comments in the repository. 
