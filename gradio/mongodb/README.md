# MongoDB Vector Search Gradio Application

A powerful web interface for managing MongoDB vector search functionality with support for document embeddings and semantic search.

## Features

- **🔌 Connection Management**: Test and monitor MongoDB and embedding provider connections
- **📁 Collection Management**: View, create, and manage MongoDB collections with vector indexes
- **📄 Document Upload**: Process and embed documents for semantic search
- **🔍 Semantic Search**: Search documents using vector similarity or text matching
- **📊 Analytics Dashboard**: Track usage, searches, uploads, and costs
- **💬 AI Chat**: Chat with AI using your knowledge base (RAG - Retrieval Augmented Generation)
- **⚙️ Settings**: View configuration and requirements

## Prerequisites

1. **MongoDB Atlas Account**: You need a MongoDB Atlas cluster with vector search enabled
2. **API Keys**: At least one of:
   - `VOYAGE_AI_API_KEY` (recommended, for Voyage AI embeddings)
   - `OPENAI_API_KEY` (fallback option)
3. **MongoDB Password**: `MONGO_DB_PASSWORD` environment variable

## Installation

1. Make sure all dependencies are installed:
```bash
uv sync
```

2. Set up your environment variables in `.env`:
```bash
MONGO_DB_PASSWORD=your_mongodb_password
VOYAGE_AI_API_KEY=your_voyage_key  # Optional but recommended
OPENAI_API_KEY=your_openai_key     # Fallback if Voyage not available
```

## Running the Application

```bash
# Run directly with Python
uv run python gradio/mongodb_vector_search_app.py

# Or make it executable and run
chmod +x gradio/mongodb_vector_search_app.py
./gradio/mongodb_vector_search_app.py
```

The application will start on `http://localhost:7860` and provide a shareable link.

## Usage Guide

### 1. Connection Tab
- Click "Test Connection" to verify MongoDB and embedding provider status
- Shows MongoDB version, active embedding provider, and dimensions

### 2. Collections Tab
- View all collections with document counts and index status
- Create new collections with optional vector index
- Refresh to see updates

### 3. Documents Tab
- Upload `.txt`, `.md`, `.pdf`, or `.docx` files
- Choose target collection and chunk size (500-2000 characters)
- Documents are automatically chunked and embedded

### 4. Search Tab
- Enter natural language queries
- Select collection and number of results
- View ranked results with scores and previews
- Try example queries for quick testing

### 5. Analytics Tab
- View recent searches and uploads
- Track usage statistics and estimated costs
- Monitor system performance

### 6. AI Chat Tab
- Chat with AI using your knowledge base
- Select collection to search from
- Choose AI model (GPT-3.5, GPT-4)
- Adjust temperature for creativity
- Set number of context documents
- Custom system prompts supported
- Example questions provided
- Full conversation history

### 7. Settings Tab
- View current configuration
- See required environment variables
- Copy vector index configuration for Atlas

## MongoDB Atlas Vector Index Setup

To enable vector search in MongoDB Atlas:

1. Go to your cluster in MongoDB Atlas
2. Navigate to the "Search" tab
3. Create a new search index
4. Use this configuration:

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

Note: Use `dimensions: 1024` if using Voyage AI embeddings.

## Embedding Providers

The application supports two embedding providers with automatic fallback:

1. **Voyage AI** (Primary)
   - Model: `voyage-3-large`
   - Dimensions: 1024
   - Better for semantic accuracy

2. **OpenAI** (Fallback)
   - Model: `text-embedding-ada-002`
   - Dimensions: 1536
   - Reliable fallback option

## Troubleshooting

### Connection Issues
- Verify `MONGO_DB_PASSWORD` is set correctly
- Check MongoDB Atlas network access settings
- Ensure your IP is whitelisted

### Vector Search Not Working
- Create the vector index in Atlas (see setup above)
- The app will fall back to text search if vector search fails
- Check the index name matches "vector_index"

### Rate Limiting
- The app handles Voyage AI rate limits by switching to OpenAI
- Built-in delays prevent hitting limits
- Consider upgrading your API plan for heavy usage

## RAG Chat Interface

The AI Chat feature implements Retrieval-Augmented Generation (RAG):

### How it Works
1. When you send a message, it searches your knowledge base for relevant context
2. The top K most relevant documents are retrieved using vector search
3. These documents are included as context for the AI model
4. The AI generates a response based on both the context and its training

### Configuration Options
- **Collection**: Choose which knowledge base to search
- **Model**: Select between GPT-3.5-turbo, GPT-4, or GPT-4-turbo
- **Temperature**: Control creativity (0 = focused, 1 = creative)
- **Context Documents**: Number of documents to retrieve (1-10)
- **System Prompt**: Customize the AI's behavior and role

### Best Practices
- Use specific questions for better context retrieval
- Increase context documents for complex topics
- Lower temperature for factual responses
- Higher temperature for creative tasks
- Custom prompts can improve domain-specific responses

## Cost Estimation

The Analytics tab provides cost estimates based on:
- Voyage AI: ~$0.0002 per 1k tokens
- OpenAI: ~$0.0001 per 1k tokens
- Chat API: Varies by model (GPT-3.5 ~$0.002/1k, GPT-4 ~$0.03/1k)
- Actual costs may vary based on your API pricing

## Development

To modify or extend the application:

1. The main application is in `mongodb_vector_search_app.py`
2. Global state is managed through module-level variables
3. Each tab is a separate function for modularity
4. Analytics are tracked in memory (resets on restart)

## Security Notes

- Never commit `.env` files with API keys
- Use environment variables for all secrets
- The app uses SSL/TLS for MongoDB connections
- Consider adding authentication for production use