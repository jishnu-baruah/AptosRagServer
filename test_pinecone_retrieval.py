import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# Load env
load_dotenv()
api_key = os.environ["PINECONE_API_KEY"]

INDEX_NAME = "aptos-whitepaper"
MODEL_NAME = "BAAI/bge-base-en-v1.5"

# Connect to Pinecone
pc = Pinecone(api_key=api_key)
desc = pc.describe_index(INDEX_NAME)
index = pc.Index(host=desc.host)

# Load embedding model
model = SentenceTransformer(MODEL_NAME)

# Sample query
query = input("Enter your test query: ")
query_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

# Query Pinecone
results = index.query(vector=query_emb, top_k=3, include_metadata=True)

print("\nTop 3 results:")
for match in results.matches:
    print(f"Score: {match.score:.4f}")
    print(f"Section: {match.metadata.get('section')}")
    print(f"Text: {match.metadata.get('text')[:300]}...")
    print("-"*60) 