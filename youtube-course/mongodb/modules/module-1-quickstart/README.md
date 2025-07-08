# Module 1: Quick Start Guide

## 🎯 Module Overview
Get up and running with MongoDB vector search and RAG in just 30 minutes! This module provides hands-on experience building your first semantic search system.

## 📚 Learning Objectives
By the end of this module, you will:
- ✅ Set up your development environment
- ✅ Create your first vector search index
- ✅ Generate embeddings and perform semantic search
- ✅ Build a complete RAG Q&A system
- ✅ Understand the basic components of production RAG

## 🎬 Video Structure

### Video 1.1: Environment Setup (5 minutes)
**File**: `01_environment_setup.py`

**What you'll do**:
- Install required packages
- Configure API keys
- Test MongoDB connection
- Verify environment is ready

**Key Concepts**:
- Environment variables best practices
- API key management
- MongoDB Atlas basics

### Video 1.2: First Vector Search (10 minutes)
**File**: `02_first_vector_search.py`

**What you'll do**:
- Create MongoDB collection
- Generate OpenAI embeddings
- Insert documents with embeddings
- Perform semantic searches
- See search results in action

**Key Concepts**:
- Vector embeddings
- Semantic vs keyword search
- MongoDB aggregation pipelines
- Cosine similarity

### Video 1.3: Complete RAG Demo (15 minutes)
**File**: `03_complete_rag_demo.py`

**What you'll do**:
- Implement document chunking
- Build knowledge base
- Retrieve relevant context
- Generate AI responses
- Create interactive Q&A system

**Key Concepts**:
- Document chunking strategies
- Context retrieval
- Prompt engineering
- RAG architecture

## 🛠️ Prerequisites
- Python 3.8+
- MongoDB Atlas account (free tier)
- OpenAI API key
- Basic Python knowledge

## 📦 Installation
```bash
# Clone the course repository
git clone [course-repo]
cd youtube-course/modules/module-1-quickstart

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

## 🚀 Running the Code

### Step 1: Environment Setup
```bash
python 01_environment_setup.py
```
This will verify your setup and create necessary files.

### Step 2: First Vector Search
```bash
python 02_first_vector_search.py
```
**Important**: After running, create the vector index in MongoDB Atlas UI:
1. Go to your cluster in Atlas
2. Click "Search" → "Create Search Index"
3. Use JSON Editor with this configuration:
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

### Step 3: Complete RAG System
```bash
python 03_complete_rag_demo.py
```
Follow the interactive prompts to see RAG in action!

## 📊 What's Next?
After completing this module, you'll be ready for:
- **Module 2**: Deep dive into embedding strategies
- **Module 3**: Building production applications
- **Module 4**: Advanced patterns and optimization

## 🆘 Troubleshooting

### MongoDB Connection Issues
- Verify your connection string includes username/password
- Check IP whitelist in Atlas (use 0.0.0.0/0 for development)
- Ensure cluster is active (not paused)

### API Key Errors
- OpenAI: Check key starts with 'sk-'
- Verify keys have necessary permissions
- Check for rate limits

### Vector Search Not Working
- Ensure index is created and active
- Index name must match code (default: "vector_index")
- Wait 1-2 minutes after index creation

## 💡 Pro Tips
1. **Save money**: Use OpenAI's ada-002 for learning (cheapest)
2. **Debug easily**: Add print statements to see embeddings
3. **Start small**: Test with few documents first
4. **Monitor costs**: Check API usage dashboards

## 📝 Exercises
1. Add more documents to the knowledge base
2. Try different chunk sizes and see the impact
3. Implement a simple filter on the search results
4. Add document metadata and use it in responses

## 🎯 Success Criteria
You've mastered this module when you can:
- [ ] Explain what vector embeddings are
- [ ] Create a vector search index
- [ ] Build a basic RAG system from scratch
- [ ] Debug common issues

Ready for the next module? Let's dive deeper! 🚀