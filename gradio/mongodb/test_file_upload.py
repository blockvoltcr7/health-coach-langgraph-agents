#!/usr/bin/env python3
"""
Test file upload functionality
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mongodb_vector_search_app import upload_and_process_document, test_connection

load_dotenv()

# Initialize connection
print("Initializing connection...")
success, msg, details = test_connection()
print(f"Connection: {msg}")

# Create a mock file object similar to what Gradio passes
class MockNamedString:
    def __init__(self, content, name):
        self.value = content
        self.name = name
        self.orig_name = name

# Test with a simple text content
test_content = """# Test Document

This is a test document for MongoDB Vector Search.

## Section 1
Some content about treatments and services.

## Section 2
More information about pricing and availability.
"""

print("\n" + "="*60)
print("Testing file upload with NamedString object")
print("="*60)

# Create mock file object
mock_file = MockNamedString(test_content, "test_document.md")

# Test upload
result = upload_and_process_document(
    file=mock_file,
    collection_name="vector_search_test",
    chunk_size=500
)

print("\nResult:")
print(result)

# Also test with a file-like object
class MockFileObject:
    def __init__(self, content, name):
        self.content = content
        self.name = name
        self.position = 0
    
    def read(self):
        return self.content

print("\n" + "="*60)
print("Testing file upload with file-like object")
print("="*60)

mock_file2 = MockFileObject(test_content.encode('utf-8'), "test_document2.txt")
result2 = upload_and_process_document(
    file=mock_file2,
    collection_name="vector_search_test",
    chunk_size=500
)

print("\nResult:")
print(result2)