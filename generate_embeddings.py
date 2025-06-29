import json
import os
import requests
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

INPUT_FILE = 'whitepaper_chunks.jsonl'
OUTPUT_FILE = 'whitepaper_embeddings.jsonl'
HUGGINGFACE_API_KEY = os.environ["HUGGINGFACE_API_KEY"]
HUGGINGFACE_API_URL = os.environ.get(
    "HUGGINGFACE_API_URL",
    "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
)
BATCH_SIZE = 8

headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

def get_embeddings(texts):
    # Hugging Face API supports batching
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json={"inputs": texts})
    response.raise_for_status()
    return response.json()

# Read all chunks
chunks = []
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        chunks.append(json.loads(line))

# Prepare texts for embedding
texts = [chunk['text'] for chunk in chunks]

# Generate embeddings in batches
print(f"Generating embeddings for {len(texts)} chunks ...")
embeddings = []
for i in tqdm(range(0, len(texts), BATCH_SIZE)):
    batch = texts[i:i+BATCH_SIZE]
    batch_emb = get_embeddings(batch)
    embeddings.extend(batch_emb)

# Write output
print(f"Writing embeddings to {OUTPUT_FILE}")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
    for chunk, emb in zip(chunks, embeddings):
        out.write(json.dumps({
            'embedding': emb,
            'text': chunk['text'],
            'section': chunk['section']
        }, ensure_ascii=False) + '\n')
print("Done.") 