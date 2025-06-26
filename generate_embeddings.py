import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

INPUT_FILE = 'whitepaper_chunks.jsonl'
OUTPUT_FILE = 'whitepaper_embeddings.jsonl'
MODEL_NAME = 'BAAI/bge-base-en-v1.5'  # You can change to 'all-MiniLM-L6-v2' if you prefer
BATCH_SIZE = 8

# Load model
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

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
    batch_emb = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
    embeddings.extend(batch_emb.tolist())

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