import os
import json
from pinecone import Pinecone, ServerlessSpec, CloudProvider, Metric, VectorType
from tqdm import tqdm
from dotenv import load_dotenv

print("Script started")
print(f"Current working directory: {os.getcwd()}")
print(f".env exists: {os.path.exists('.env')}")

load_dotenv()

INDEX_NAME = "aptos-whitepaper"
DIMENSION = 384  # for all-MiniLM-L6-v2
BATCH_SIZE = 32

try:
    api_key = os.environ["PINECONE_API_KEY"]
    print(f"Loaded Pinecone API key: {api_key[:8]}...{'*' * (len(api_key)-8)}")
    pc = Pinecone(api_key=api_key)

    # Create index if it doesn't exist
    print("Checking for existing indexes...")
    index_names = [idx.name for idx in pc.list_indexes()]
    if INDEX_NAME not in index_names:
        print(f"Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=Metric.COSINE,
            spec=ServerlessSpec(cloud=CloudProvider.AWS, region="us-east-1"),
            vector_type=VectorType.DENSE,
        )
        print("Index created.")
    else:
        print(f"Index '{INDEX_NAME}' already exists.")

    desc = pc.describe_index(INDEX_NAME)
    print(f"Index host: {desc.host}")
    index = pc.Index(host=desc.host)

    # Read all vectors
    vectors = []
    with open("whitepaper_embeddings.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            obj = json.loads(line)
            vectors.append((
                f"chunk-{i}",
                obj["embedding"],
                {"text": obj["text"], "section": obj["section"]}
            ))
    print(f"Loaded {len(vectors)} vectors from whitepaper_embeddings.jsonl")

    # Upsert in batches
    print(f"Upserting {len(vectors)} vectors to Pinecone...")
    for i in tqdm(range(0, len(vectors), BATCH_SIZE)):
        batch = vectors[i:i+BATCH_SIZE]
        index.upsert(vectors=batch)
    print("Done.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 