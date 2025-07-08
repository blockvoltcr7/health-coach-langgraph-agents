# How to Create a Proper Atlas Vector Search Index

## Step-by-Step Instructions

### 1. Go to MongoDB Atlas
- Log into your MongoDB Atlas account
- Navigate to your cluster (Cluster0-health-coach-ai)

### 2. Go to Atlas Search
- Click on the "Atlas Search" tab in your cluster

### 3. Create New Index (or Edit "default")
- Click "Create Search Index" or edit your existing "default" index
- Choose "JSON Editor" mode

### 4. Use This Exact Configuration

For **Voyage AI embeddings** (1024 dimensions):
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1024,
        "similarity": "cosine"
      }
    }
  }
}
```

For **OpenAI embeddings** (1536 dimensions):
```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

### 5. Important Settings
- **Index Name**: Keep as "default" (since that's what we're using in the code)
- **Database**: health_coach_ai
- **Collection**: medspa_services

### 6. Create/Update the Index
- Click "Create Search Index" or "Save Changes"
- Wait for the index to build (usually takes 1-2 minutes)

## Verify Your Index

After creating the index, you should see:
- Status: "READY"
- Type shows "vectorSearch" capabilities

## Test in Atlas Search Tester

Once the index is ready, you can test it in Atlas:

1. Go to "Search Tester" tab
2. Use this query:
```json
{
  "$vectorSearch": {
    "index": "default",
    "path": "embedding",
    "queryVector": [/* your test vector here */],
    "numCandidates": 50,
    "limit": 5
  }
}
```

## Common Issues

### "embedding is not indexed as knnVector"
This error means your index isn't configured for vector search. Follow the steps above to fix it.

### Wrong Dimensions
Make sure the dimensions match your embeddings:
- Voyage AI: 1024
- OpenAI: 1536

### Index Not Ready
Wait for the index status to show "READY" before testing.

## After Creating the Index

Once your index is properly configured:
1. Restart your Gradio app
2. The vector search should work automatically
3. You'll get much better search results than text search

The key is that the "embedding" field MUST be defined as type "knnVector" in the index configuration!