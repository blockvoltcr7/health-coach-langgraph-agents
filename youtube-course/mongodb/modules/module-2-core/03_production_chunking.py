"""
Module 2.3: Production Chunking Strategies
Time: 15 minutes
Goal: Master document chunking for optimal RAG performance
"""

import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import hashlib
import re
from datetime import datetime
import tiktoken
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Chunk:
    """Represents a document chunk with metadata"""
    content: str
    metadata: Dict
    chunk_id: str
    parent_id: str
    position: int
    token_count: int
    char_count: int

class ProductionChunker:
    """
    Advanced chunking strategies for production RAG systems
    Handles multiple document formats and preserves context
    """
    
    def __init__(self, model: str = "gpt-3.5-turbo"):
        # Initialize tokenizer for accurate token counting
        self.encoding = tiktoken.encoding_for_model(model)
        
    def count_tokens(self, text: str) -> int:
        """Accurate token counting using tiktoken"""
        return len(self.encoding.encode(text))
    
    def chunk_by_tokens(
        self,
        text: str,
        max_tokens: int = 400,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 100
    ) -> List[Chunk]:
        """
        Token-based chunking for precise control
        Better than character-based for LLM consumption
        """
        chunks = []
        tokens = self.encoding.encode(text)
        
        # Generate document ID
        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        position = 0
        start_idx = 0
        
        while start_idx < len(tokens):
            # Calculate end index
            end_idx = min(start_idx + max_tokens, len(tokens))
            
            # Extract chunk tokens
            chunk_tokens = tokens[start_idx:end_idx]
            
            # Skip if chunk is too small (unless it's the last chunk)
            if len(chunk_tokens) < min_chunk_tokens and end_idx < len(tokens):
                break
            
            # Decode back to text
            chunk_text = self.encoding.decode(chunk_tokens)
            
            # Create chunk object
            chunk = Chunk(
                content=chunk_text,
                metadata={
                    "chunking_method": "token_based",
                    "max_tokens": max_tokens,
                    "overlap_tokens": overlap_tokens,
                    "timestamp": datetime.utcnow().isoformat()
                },
                chunk_id=f"{doc_id}_chunk_{position}",
                parent_id=doc_id,
                position=position,
                token_count=len(chunk_tokens),
                char_count=len(chunk_text)
            )
            
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            start_idx += max_tokens - overlap_tokens
            position += 1
            
            # Prevent infinite loop
            if start_idx >= len(tokens) - min_chunk_tokens:
                break
        
        return chunks
    
    def chunk_markdown_aware(
        self,
        markdown_text: str,
        max_tokens: int = 400,
        overlap_tokens: int = 50
    ) -> List[Chunk]:
        """
        Markdown-aware chunking that preserves document structure
        Respects headers, code blocks, and lists
        """
        chunks = []
        doc_id = hashlib.md5(markdown_text.encode()).hexdigest()[:8]
        
        # Split by headers while preserving them
        header_pattern = r'(^#{1,6}\s+.*$)'
        sections = re.split(header_pattern, markdown_text, flags=re.MULTILINE)
        
        current_header_stack = []
        position = 0
        
        for i in range(0, len(sections), 2):
            # Get header and content
            header = sections[i] if i > 0 else ""
            content = sections[i + 1] if i + 1 < len(sections) else sections[i]
            
            # Update header stack
            if header:
                header_level = len(re.match(r'^(#+)', header).group(1))
                # Pop headers of same or lower level
                while current_header_stack and current_header_stack[-1][1] >= header_level:
                    current_header_stack.pop()
                current_header_stack.append((header.strip(), header_level))
            
            # Skip empty content
            if not content.strip():
                continue
            
            # Prepare section with context
            header_context = "\n".join([h[0] for h in current_header_stack])
            full_section = f"{header_context}\n\n{content.strip()}"
            
            # Check if section needs further chunking
            section_tokens = self.count_tokens(full_section)
            
            if section_tokens <= max_tokens:
                # Section fits in one chunk
                chunk = Chunk(
                    content=full_section,
                    metadata={
                        "chunking_method": "markdown_aware",
                        "headers": [h[0] for h in current_header_stack],
                        "section_type": self._detect_section_type(content),
                        "has_code": "```" in content,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    chunk_id=f"{doc_id}_chunk_{position}",
                    parent_id=doc_id,
                    position=position,
                    token_count=section_tokens,
                    char_count=len(full_section)
                )
                chunks.append(chunk)
                position += 1
            else:
                # Section needs sub-chunking
                sub_chunks = self._chunk_large_section(
                    full_section,
                    max_tokens,
                    overlap_tokens,
                    doc_id,
                    position,
                    current_header_stack
                )
                chunks.extend(sub_chunks)
                position += len(sub_chunks)
        
        return chunks
    
    def chunk_code_aware(
        self,
        text: str,
        max_tokens: int = 400,
        language: Optional[str] = None
    ) -> List[Chunk]:
        """
        Code-aware chunking that preserves function/class boundaries
        """
        chunks = []
        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Patterns for different languages
        patterns = {
            "python": {
                "class": r'^class\s+\w+.*?(?=^class\s+|\Z)',
                "function": r'^def\s+\w+.*?(?=^def\s+|^class\s+|\Z)',
                "split_points": [r'^\s*$', r'^import\s+', r'^from\s+']
            },
            "javascript": {
                "class": r'class\s+\w+\s*{[^}]*}',
                "function": r'function\s+\w+\s*\([^)]*\)\s*{[^}]*}',
                "split_points": [r'^\s*$', r'^import\s+', r'^const\s+', r'^let\s+']
            }
        }
        
        # Detect language if not specified
        if not language:
            language = self._detect_language(text)
        
        # Use language-specific patterns or fall back to line-based
        if language in patterns:
            # Extract code blocks
            blocks = self._extract_code_blocks(text, patterns[language])
            
            position = 0
            for block_type, block_content, start_line in blocks:
                block_tokens = self.count_tokens(block_content)
                
                if block_tokens <= max_tokens:
                    chunk = Chunk(
                        content=block_content,
                        metadata={
                            "chunking_method": "code_aware",
                            "language": language,
                            "block_type": block_type,
                            "start_line": start_line,
                            "timestamp": datetime.utcnow().isoformat()
                        },
                        chunk_id=f"{doc_id}_chunk_{position}",
                        parent_id=doc_id,
                        position=position,
                        token_count=block_tokens,
                        char_count=len(block_content)
                    )
                    chunks.append(chunk)
                    position += 1
                else:
                    # Block too large, use smart splitting
                    sub_chunks = self._split_large_code_block(
                        block_content,
                        max_tokens,
                        doc_id,
                        position,
                        language,
                        block_type
                    )
                    chunks.extend(sub_chunks)
                    position += len(sub_chunks)
        else:
            # Fallback to token-based chunking
            chunks = self.chunk_by_tokens(text, max_tokens)
        
        return chunks
    
    def chunk_semantic(
        self,
        text: str,
        max_tokens: int = 400,
        min_sentence_tokens: int = 10
    ) -> List[Chunk]:
        """
        Semantic chunking based on sentence boundaries and topics
        """
        chunks = []
        doc_id = hashlib.md5(text.encode()).hexdigest()[:8]
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        position = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # Skip very short sentences
            if sentence_tokens < min_sentence_tokens and current_chunk:
                current_chunk[-1] += " " + sentence
                current_tokens += sentence_tokens
                continue
            
            # Check if adding sentence exceeds limit
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                # Create chunk
                chunk_content = " ".join(current_chunk)
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        "chunking_method": "semantic",
                        "sentence_count": len(current_chunk),
                        "avg_sentence_tokens": current_tokens / len(current_chunk),
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    chunk_id=f"{doc_id}_chunk_{position}",
                    parent_id=doc_id,
                    position=position,
                    token_count=current_tokens,
                    char_count=len(chunk_content)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap (last sentence)
                current_chunk = [current_chunk[-1], sentence] if current_chunk else [sentence]
                current_tokens = self.count_tokens(" ".join(current_chunk))
                position += 1
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_content = " ".join(current_chunk)
            chunk = Chunk(
                content=chunk_content,
                metadata={
                    "chunking_method": "semantic",
                    "sentence_count": len(current_chunk),
                    "timestamp": datetime.utcnow().isoformat()
                },
                chunk_id=f"{doc_id}_chunk_{position}",
                parent_id=doc_id,
                position=position,
                token_count=current_tokens,
                char_count=len(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _detect_section_type(self, content: str) -> str:
        """Detect the type of content in a section"""
        if "```" in content:
            return "code"
        elif re.search(r'^\d+\.', content, re.MULTILINE):
            return "list"
        elif re.search(r'^\|.*\|', content, re.MULTILINE):
            return "table"
        else:
            return "text"
    
    def _chunk_large_section(
        self,
        content: str,
        max_tokens: int,
        overlap_tokens: int,
        doc_id: str,
        start_position: int,
        headers: List[Tuple[str, int]]
    ) -> List[Chunk]:
        """Sub-chunk large sections while preserving context"""
        # Use paragraph-aware splitting for large sections
        paragraphs = content.split('\n\n')
        chunks = []
        current_content = []
        current_tokens = 0
        position = start_position
        
        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)
            
            if current_tokens + para_tokens > max_tokens and current_content:
                # Create chunk
                chunk_text = '\n\n'.join(current_content)
                chunk = Chunk(
                    content=chunk_text,
                    metadata={
                        "chunking_method": "markdown_aware_sub",
                        "headers": [h[0] for h in headers],
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    chunk_id=f"{doc_id}_chunk_{position}",
                    parent_id=doc_id,
                    position=position,
                    token_count=current_tokens,
                    char_count=len(chunk_text)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                current_content = [current_content[-1]] if current_content else []
                current_content.append(paragraph)
                current_tokens = self.count_tokens('\n\n'.join(current_content))
                position += 1
            else:
                current_content.append(paragraph)
                current_tokens += para_tokens
        
        # Last chunk
        if current_content:
            chunk_text = '\n\n'.join(current_content)
            chunk = Chunk(
                content=chunk_text,
                metadata={
                    "chunking_method": "markdown_aware_sub",
                    "headers": [h[0] for h in headers],
                    "timestamp": datetime.utcnow().isoformat()
                },
                chunk_id=f"{doc_id}_chunk_{position}",
                parent_id=doc_id,
                position=position,
                token_count=current_tokens,
                char_count=len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _detect_language(self, code: str) -> Optional[str]:
        """Simple language detection based on patterns"""
        if "def " in code and "import " in code:
            return "python"
        elif "function " in code or "const " in code:
            return "javascript"
        elif "#include" in code:
            return "c"
        elif "public class" in code:
            return "java"
        return None
    
    def _extract_code_blocks(self, text: str, patterns: Dict) -> List[Tuple[str, str, int]]:
        """Extract code blocks based on language patterns"""
        blocks = []
        lines = text.split('\n')
        
        # Simple extraction - in production, use proper AST parsing
        current_block = []
        current_type = None
        start_line = 0
        
        for i, line in enumerate(lines):
            if re.match(patterns.get('class', ''), line):
                if current_block:
                    blocks.append((current_type, '\n'.join(current_block), start_line))
                current_block = [line]
                current_type = 'class'
                start_line = i
            elif re.match(patterns.get('function', ''), line):
                if current_block and current_type == 'function':
                    blocks.append((current_type, '\n'.join(current_block), start_line))
                    current_block = [line]
                    start_line = i
                else:
                    current_block.append(line)
            else:
                current_block.append(line)
        
        if current_block:
            blocks.append((current_type or 'code', '\n'.join(current_block), start_line))
        
        return blocks
    
    def _split_large_code_block(
        self,
        code: str,
        max_tokens: int,
        doc_id: str,
        start_position: int,
        language: str,
        block_type: str
    ) -> List[Chunk]:
        """Split large code blocks intelligently"""
        # For now, use line-based splitting
        # In production, use AST-based splitting
        lines = code.split('\n')
        chunks = []
        current_lines = []
        current_tokens = 0
        position = start_position
        
        for line in lines:
            line_tokens = self.count_tokens(line + '\n')
            
            if current_tokens + line_tokens > max_tokens and current_lines:
                chunk_content = '\n'.join(current_lines)
                chunk = Chunk(
                    content=chunk_content,
                    metadata={
                        "chunking_method": "code_aware_split",
                        "language": language,
                        "block_type": block_type,
                        "timestamp": datetime.utcnow().isoformat()
                    },
                    chunk_id=f"{doc_id}_chunk_{position}",
                    parent_id=doc_id,
                    position=position,
                    token_count=current_tokens,
                    char_count=len(chunk_content)
                )
                chunks.append(chunk)
                
                current_lines = []
                current_tokens = 0
                position += 1
            
            current_lines.append(line)
            current_tokens += line_tokens
        
        if current_lines:
            chunk_content = '\n'.join(current_lines)
            chunk = Chunk(
                content=chunk_content,
                metadata={
                    "chunking_method": "code_aware_split",
                    "language": language,
                    "block_type": block_type,
                    "timestamp": datetime.utcnow().isoformat()
                },
                chunk_id=f"{doc_id}_chunk_{position}",
                parent_id=doc_id,
                position=position,
                token_count=current_tokens,
                char_count=len(chunk_content)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting - use spaCy or NLTK in production
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

def demonstrate_chunking_strategies():
    """Compare different chunking strategies"""
    print("🎓 CHUNKING STRATEGY COMPARISON\n")
    
    # Sample documents
    sample_markdown = """# MongoDB Vector Search Guide

## Introduction
MongoDB Atlas provides powerful vector search capabilities that enable semantic search in your applications. This guide covers the essential concepts and implementation details.

## Setting Up Vector Search

### Prerequisites
Before you begin, ensure you have:
- A MongoDB Atlas cluster (M10 or higher for production)
- Python 3.8 or later
- Required Python packages

### Creating Your First Index
To create a vector search index, navigate to your Atlas cluster and follow these steps:

```python
def create_vector_index(collection_name, index_name):
    index_definition = {
        "mappings": {
            "dynamic": True,
            "fields": {
                "embedding": {
                    "type": "knnVector",
                    "dimensions": 1536,
                    "similarity": "cosine"
                }
            }
        }
    }
    return index_definition
```

This function defines a basic vector index configuration.

## Best Practices
1. Choose appropriate embedding dimensions
2. Use batching for large datasets
3. Implement caching for frequently accessed embeddings
4. Monitor index performance
"""
    
    sample_code = """import os
from pymongo import MongoClient
from openai import OpenAI

class VectorSearchEngine:
    def __init__(self, connection_string, database_name):
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        self.openai = OpenAI()
    
    def generate_embedding(self, text):
        \"\"\"Generate embedding for given text\"\"\"
        response = self.openai.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def search(self, query, collection_name, limit=5):
        \"\"\"Perform vector search on collection\"\"\"
        query_embedding = self.generate_embedding(query)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 100,
                    "limit": limit
                }
            }
        ]
        
        results = self.db[collection_name].aggregate(pipeline)
        return list(results)
"""
    
    # Initialize chunker
    chunker = ProductionChunker()
    
    # Test different strategies
    strategies = [
        ("Token-based", lambda: chunker.chunk_by_tokens(sample_markdown, max_tokens=200)),
        ("Markdown-aware", lambda: chunker.chunk_markdown_aware(sample_markdown, max_tokens=200)),
        ("Code-aware", lambda: chunker.chunk_code_aware(sample_code, max_tokens=150)),
        ("Semantic", lambda: chunker.chunk_semantic(sample_markdown, max_tokens=200))
    ]
    
    for strategy_name, chunk_func in strategies:
        print(f"\n{'='*60}")
        print(f"📋 {strategy_name} Chunking")
        print(f"{'='*60}")
        
        chunks = chunk_func()
        
        print(f"Generated {len(chunks)} chunks\n")
        
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"Chunk {i + 1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Tokens: {chunk.token_count}")
            print(f"  Position: {chunk.position}")
            print(f"  Metadata: {chunk.metadata.get('chunking_method', 'unknown')}")
            print(f"  Preview: {chunk.content[:100]}...")
            print()

def chunking_best_practices():
    """Display chunking best practices"""
    print("\n📚 CHUNKING BEST PRACTICES\n")
    
    practices = [
        {
            "title": "Choose the Right Strategy",
            "guidelines": [
                "Use token-based for general text",
                "Use markdown-aware for documentation",
                "Use code-aware for source code",
                "Use semantic for narrative content"
            ]
        },
        {
            "title": "Optimize Chunk Size",
            "guidelines": [
                "200-500 tokens for most use cases",
                "Smaller chunks (100-200) for precise retrieval",
                "Larger chunks (500-800) for context-heavy content",
                "Consider model context window limits"
            ]
        },
        {
            "title": "Handle Overlap Properly",
            "guidelines": [
                "10-20% overlap for general content",
                "Higher overlap for technical content",
                "No overlap for structured data",
                "Sliding window for continuous text"
            ]
        },
        {
            "title": "Preserve Metadata",
            "guidelines": [
                "Track chunk position and parent document",
                "Include section headers and context",
                "Store chunking parameters",
                "Add timestamps for versioning"
            ]
        }
    ]
    
    for practice in practices:
        print(f"🎯 {practice['title']}:")
        for guideline in practice['guidelines']:
            print(f"   • {guideline}")
        print()

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Production Chunking\n")
    
    try:
        # Demo different chunking strategies
        demonstrate_chunking_strategies()
        
        # Show best practices
        chunking_best_practices()
        
        print("\n🎉 Key Takeaways:")
        print("✅ Different content types need different chunking strategies")
        print("✅ Token-based chunking provides precise control")
        print("✅ Preserve document structure and metadata")
        print("✅ Test and optimize based on your specific use case")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Install required packages: pip install tiktoken")