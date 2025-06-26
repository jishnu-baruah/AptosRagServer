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

For more details, see the code and comments in the repository. 