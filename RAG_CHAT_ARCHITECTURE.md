# RAG Chat Architecture with Voyage AI and MongoDB Vector Search

## Overview

This document explains how the Retrieval-Augmented Generation (RAG) chat system works in our Gradio application, combining Voyage AI embeddings, MongoDB vector search, and OpenAI's GPT models to create an intelligent chat interface.

## Architecture Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│  Voyage AI API   │────▶│ Query Embedding │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Chat Response   │◀────│   OpenAI GPT     │◀────│ MongoDB Vector  │
└─────────────────┘     └──────────────────┘     │     Search      │
                              ▲                   └─────────────────┘
                              │                           ▲
                              │                           │
                        ┌─────┴──────────┐                │
                        │Retrieved Context│◀───────────────┘
                        └─────────────────┘
```

## How It Works: Step-by-Step

### 1. Document Preparation Phase (One-time setup)

When documents are uploaded to the system:

```python
# Document Upload Flow
1. User uploads document (e.g., med-spa-test-data.md)
   ↓
2. Text Splitting
   - Markdown headers split
   - Chunk size: 1000 characters
   - Overlap: 200 characters
   ↓
3. Embedding Generation (Voyage AI)
   - Model: voyage-3-large
   - Dimensions: 1024
   - Input type: "document"
   ↓
4. Storage in MongoDB
   {
     "content": "chunk text...",
     "embedding": [0.123, -0.456, ...],  # 1024 dimensions
     "metadata": {
       "source": "med-spa-test-data.md",
       "chunk_index": 0,
       "embedding_provider": "voyage"
     }
   }
```

### 2. Chat Query Phase (Real-time)

When a user sends a chat message:

#### Step 1: Query Embedding
```python
# User asks: "What treatments help with energy?"
query_embedding = embedding_provider.embed_query(user_message)
# Voyage AI converts this to a 1024-dimensional vector
```

#### Step 2: Vector Search
```python
# MongoDB aggregation pipeline
pipeline = [{
    "$vectorSearch": {
        "index": "vector_index",
        "path": "embedding",
        "queryVector": query_embedding,  # Voyage AI embedding
        "numCandidates": 30,  # Search broader set
        "limit": 3  # Return top 3
    }
}]
```

#### Step 3: Context Retrieval
The system retrieves the most semantically similar document chunks:
```
Context 1 (Score: 0.923, Source: med-spa-test-data.md):
"### The CEO Drip
Price: $299
Duration: 60 minutes
Key Ingredients: High-dose B-Complex, Vitamin C (2000mg)...
Benefits: Maximum cognitive enhancement and mental clarity..."

Context 2 (Score: 0.891, Source: med-spa-test-data.md):
"### Athletic Recovery Plus
Benefits: Enhanced athletic performance, Improved endurance..."
```

#### Step 4: LLM Response Generation
```python
# OpenAI GPT receives:
messages = [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "system", "content": "Available Context:\n[retrieved documents]"},
    {"role": "user", "content": "What treatments help with energy?"}
]

# GPT generates response using the context
response = "Based on your needs for energy, I recommend the CEO Drip..."
```

## Key Components

### 1. EmbeddingProvider Class
Manages embedding generation with automatic fallback:

```python
class EmbeddingProvider:
    def __init__(self):
        # Try Voyage AI first (premium choice)
        # Fall back to OpenAI if unavailable
        
    def embed_query(self, text: str):
        # Voyage AI: voyage-3-large, input_type="query"
        # OpenAI fallback: text-embedding-ada-002
```

### 2. Vector Search Configuration

MongoDB Atlas Vector Index:
```json
{
  "mappings": {
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1024,  // Voyage AI dimensions
        "similarity": "cosine"
      }
    }
  }
}
```

### 3. Chat Function Flow

```python
def chat_with_rag(message, history, collection_name, ...):
    1. Search for relevant context (Voyage AI)
    2. Format retrieved documents
    3. Build conversation with context
    4. Generate response (OpenAI GPT)
    5. Track analytics
    6. Return response
```

## Why This Architecture?

### 1. **Voyage AI for Embeddings**
- **Specialized for Retrieval**: voyage-3-large is optimized for semantic search
- **Better Accuracy**: Captures nuanced meanings better than general embeddings
- **Consistent Vectors**: Same model for documents and queries ensures compatibility

### 2. **MongoDB Vector Search**
- **Scalable**: Handles millions of documents efficiently
- **Integrated**: No separate vector database needed
- **Flexible**: Supports filters and complex queries

### 3. **OpenAI GPT for Generation**
- **Natural Responses**: Generates human-like answers
- **Context-Aware**: Uses retrieved documents to provide accurate information
- **Configurable**: Different models and temperatures for various use cases

## Configuration Options

### Embedding Settings
- **Primary**: Voyage AI (voyage-3-large, 1024 dims)
- **Fallback**: OpenAI (text-embedding-ada-002, 1536 dims)
- **Rate Limiting**: 0.5s delay between Voyage API calls

### Chat Settings
- **Models**: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- **Temperature**: 0-1 (creativity control)
- **Context Documents**: 1-10 (retrieval count)
- **Max Tokens**: 1000 (response length)

## Performance Considerations

### 1. **Embedding Consistency**
- Always use the same embedding model for indexing and querying
- Mixing Voyage AI and OpenAI embeddings reduces accuracy

### 2. **Rate Limiting**
- Voyage AI: 3 RPM (free tier), higher with paid plans
- Automatic fallback to OpenAI prevents failures
- Built-in delays prevent hitting limits

### 3. **Cost Optimization**
- Voyage AI: ~$0.0002/1k tokens (embeddings)
- OpenAI GPT-4: ~$0.03/1k tokens (chat)
- Cache frequent queries to reduce costs

## Example Use Cases

### 1. Medical Information Query
```
User: "What's the best treatment for athletic recovery?"
System: 
1. Embeds query with Voyage AI
2. Finds "Athletic Recovery Plus" chunk
3. GPT explains the treatment using retrieved info
```

### 2. Pricing Information
```
User: "How much does NAD+ therapy cost?"
System:
1. Vector search finds NAD+ pricing section
2. GPT provides specific pricing from context
3. Can compare with other treatments
```

### 3. Treatment Recommendations
```
User: "I need help with energy and focus for work"
System:
1. Semantic search understands intent
2. Retrieves CEO Drip and related treatments
3. GPT provides personalized recommendations
```

## Troubleshooting

### Common Issues

1. **"No embedding provider available"**
   - Set VOYAGE_AI_API_KEY or OPENAI_API_KEY
   - Check API key validity

2. **Poor retrieval quality**
   - Ensure consistent embedding model usage
   - Increase context documents (top_k)
   - Check if vector index exists in MongoDB

3. **Rate limiting errors**
   - Upgrade Voyage AI plan
   - System auto-falls back to OpenAI
   - Add caching for frequent queries

## Best Practices

### 1. **Document Preparation**
- Use meaningful chunk sizes (500-2000 chars)
- Include overlap for context continuity
- Add descriptive metadata

### 2. **Query Optimization**
- Be specific in questions
- Use domain terminology
- Increase context for complex topics

### 3. **System Prompts**
- Customize for your domain
- Include citation requirements
- Set appropriate tone

## Future Enhancements

1. **Hybrid Search**: Combine vector and keyword search
2. **Caching Layer**: Redis for frequent queries
3. **Multi-Modal**: Support image embeddings
4. **Fine-Tuning**: Custom Voyage AI models
5. **Streaming**: Real-time response generation

## Conclusion

This RAG architecture combines the best of three technologies:
- **Voyage AI**: Superior semantic understanding for retrieval
- **MongoDB**: Scalable vector storage and search
- **OpenAI GPT**: Natural language generation

Together, they create an intelligent chat system that understands context, retrieves relevant information, and provides accurate, conversational responses based on your knowledge base.