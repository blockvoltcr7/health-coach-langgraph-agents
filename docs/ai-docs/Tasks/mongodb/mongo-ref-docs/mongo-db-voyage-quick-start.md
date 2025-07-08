🔹 1. MongoDB Official Quick Start (Voyage AI + Auto Quantization)

A recent tutorial walks through how to embed text programmatically, index it with Atlas Vector Search, and perform semantic queries:

import os
import pymongo
import voyageai

# Setup
os.environ["VOYAGE_API_KEY"] = "<your_key>"
vo = voyageai.Client()
client = pymongo.MongoClient(os.getenv("MONGODB_URI"))
col = client.get_database("testdb").embeddings

def get_embedding(text):
    emb = vo.embed([text], model="voyage-3-large", input_type="document").embeddings[0]
    return pymongo.Binary.from_vector(emb, pymongo.BinaryVectorDtype.FLOAT32)

# Insert text and embedding
doc = {"text": "Your full markdown or large document too", "embedding": get_embedding(text)}
col.insert_one(doc)

# Create vector index
col.create_search_index({
  "definition": {"mappings": {
    "dynamic": True,
    "fields": {"embedding": {"type": "knnVector", "dimensions": len(emb), "similarity": "cosine"}}
}},"name":"vec_idx"})

# Query
qemb = get_embedding("search query")
results = col.aggregate([
  {"$vectorSearch": {"queryVector": qemb, "path": "embedding", "k": 5}},
  {"$sort": {"score": -1}}
])
for r in results:
    print(r["text"], r["score"])

This incorporates automatic quantization support and follows the official Atlas guidelines  ￼ ￼ ￼ ￼.

⸻

🔹 2. LangChain: Auto-Chunk, Embed & Index in Python

Using LangChain’s MongoDBAtlasVectorSearch, it’s possible to ingest full documents (PDF/Markdown/text), automatically split them into chunks, embed, and store them:

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import TextLoader  # e.g. Markdown loader

loader = TextLoader("post.md", encoding="utf8")
docs = loader.load_and_split()  # auto chunking

vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),  # you can swap for Voyage AI embeddings
    collection=your_pyMongo_collection,
    index_name="vec_idx"
)

# Now search semantically
results = vector_search.similarity_search("Explain the main topic", k=5)
for d in results:
    print(d.page_content[:200])

This auto-chunks and handles embedding + indexing in a few lines—clean and production-ready  ￼ ￼.

⸻

✅ What’s Best for You?

Approach	Pros	Cons
Direct pymongo + Voyage	Full control, direct use of latest Voyage features (e.g. quantization)	You manage embeddings / chunking manually
LangChain wrapper	One-liner chunk/embed/index, cleaner pipeline	Requires managing two embedding systems if mixing Voyage


⸻

🧭 Next Steps
	1.	Pilot the pymongo + Voyage code for precise control over models like voyage-3-large, float32, and quantization.
	2.	Prototype with LangChain to quickly spin up a semantic search pipeline; replace OpenAIEmbeddings with a Voyage adapter once available.
	3.	Monitor MongoDB announcements—Atlas is actively building native auto-ingestion & chunking, which will streamline future workflows  ￼ ￼ ￼.

⸻

