## Complete MongoDB Vector Search Implementation with Markdown Processing

### Step 1: Install Required Dependencies

```bash
pip install pymongo langchain langchain-text-splitters langchain-openai openai python-dotenv
```

### Step 2: Complete Implementation

```python
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import certifi
from pymongo.server_api import ServerApi

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

class MongoDBVectorProcessor:
    def __init__(self):
        # MongoDB setup
        self.mongo_password = os.getenv("MONGO_DB_PASSWORD")
        self.mongo_uri = f"mongodb+srv://health-coach-ai-sami:{self.mongo_password}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
        self.client = MongoClient(self.mongo_uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())
        self.db = self.client["health_coach_ai"]
        self.collection = self.db["documents"]
        
        # OpenAI embeddings setup
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",  # 1536 dimensions
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
    def load_and_chunk_markdown(self, markdown_path: str) -> List[Document]:
        """Load and chunk a markdown file using LangChain splitters"""
        
        # Step 1: Load the markdown file
        loader = UnstructuredMarkdownLoader(markdown_path, mode="elements")
        raw_documents = loader.load()
        
        # Step 2: Split by markdown headers first
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False  # Keep headers for context
        )
        
        # Process each document through the markdown splitter
        header_split_docs = []
        for doc in raw_documents:
            splits = markdown_splitter.split_text(doc.page_content)
            header_split_docs.extend(splits)
        
        # Step 3: Further split by size if needed
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,      # Adjust based on your needs
            chunk_overlap=200,    # Overlap for context preservation
            add_start_index=True  # Track position in original document
        )
        
        final_chunks = text_splitter.split_documents(header_split_docs)
        
        print(f"Created {len(final_chunks)} chunks from markdown file")
        return final_chunks
    
    def generate_embeddings(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks"""
        
        # Extract texts for embedding
        texts = [doc.page_content for doc in documents]
        
        # Generate embeddings in batches
        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embeddings.embed_documents(texts)
        
        # Prepare documents for MongoDB
        mongo_docs = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            mongo_doc = {
                "content": doc.page_content,
                "embedding": embedding,  # List of floats
                "metadata": {
                    **doc.metadata,
                    "chunk_index": i,
                    "source": "markdown_import",
                    "embedding_model": "text-embedding-ada-002",
                    "embedding_dimensions": len(embedding)
                }
            }
            mongo_docs.append(mongo_doc)
        
        return mongo_docs
    
    def insert_to_mongodb(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Insert documents with embeddings into MongoDB"""
        
        try:
            result = self.collection.insert_many(documents)
            print(f"Successfully inserted {len(result.inserted_ids)} documents")
            return result.inserted_ids
        except Exception as e:
            print(f"Error inserting documents: {e}")
            raise
    
    def process_markdown_file(self, markdown_path: str) -> List[str]:
        """Complete pipeline: load, chunk, embed, and store markdown file"""
        
        # Step 1: Load and chunk the markdown
        chunks = self.load_and_chunk_markdown(markdown_path)
        
        # Step 2: Generate embeddings
        documents_with_embeddings = self.generate_embeddings(chunks)
        
        # Step 3: Insert into MongoDB
        inserted_ids = self.insert_to_mongodb(documents_with_embeddings)
        
        return inserted_ids
    
    def vector_search(self, query: str, limit: int = 5, filters: Dict = None):
        """Perform vector search on the stored documents"""
        
        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query)
        
        # Perform vector search
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": filters or {}
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "content": 1,
                    "metadata": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the processor
    processor = MongoDBVectorProcessor()
    
    # Process a markdown file
    markdown_file = "path/to/your/document.md"
    
    try:
        # Process the markdown file
        inserted_ids = processor.process_markdown_file(markdown_file)
        print(f"Successfully processed {len(inserted_ids)} chunks")
        
        # Test search
        query = "What is MongoDB Vector Search?"
        results = processor.vector_search(query, limit=3)
        
        print("\nSearch Results:")
        for result in results:
            print(f"Score: {result['score']:.4f}")
            print(f"Content: {result['content'][:200]}...")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")
```

### Step 3: Alternative Using MongoDB's Native Embedding (if available)

MongoDB doesn't have a built-in embedding model, but you can use various embedding providers:

```python
# Option 1: OpenAI (most common)
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Option 2: Cohere
from langchain_community.embeddings import CohereEmbeddings
embeddings = CohereEmbeddings(model="embed-english-v3.0")

# Option 3: HuggingFace
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Option 4: Local embeddings with Ollama
from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="llama2")
```

### Step 4: Create Vector Search Index in Atlas

Before running searches, create the vector index in MongoDB Atlas:

```json
{
  "name": "vector_index",
  "type": "vectorSearch",
  "fields": [{
    "type": "vector",
    "path": "embedding",
    "numDimensions": 1536,
    "similarity": "cosine"
  }, {
    "type": "filter",
    "path": "metadata.source"
  }, {
    "type": "filter",
    "path": "metadata.Header 1"
  }, {
    "type": "filter",
    "path": "metadata.Header 2"
  }]
}
```

### Key Features of This Implementation:

1. **Markdown-aware chunking**: Uses LangChain's `MarkdownHeaderTextSplitter` to preserve document structure
2. **Two-stage splitting**: First by headers, then by size for optimal chunks
3. **Metadata preservation**: Keeps header hierarchy and chunk position
4. **Flexible embeddings**: Easy to switch between embedding providers
5. **Batch processing**: Efficient embedding generation
6. **Error handling**: Proper exception handling throughout

### Best Practices:

1. **Chunk size**: Adjust based on your content and use case (500-2000 tokens typically)
2. **Overlap**: Include 10-20% overlap between chunks for context
3. **Metadata**: Store as much metadata as possible for filtering
4. **Batch size**: Process embeddings in batches of 50-100 for efficiency
5. **Index configuration**: Include filter fields for all metadata you want to search by

This implementation provides a complete pipeline from markdown files to searchable vector embeddings in MongoDB Atlas.