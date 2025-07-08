# Troubleshooting MongoDB Vector Search

## Issue: "System not initialized" or Empty Search Results

### Quick Fixes:

1. **Restart the Gradio App**
   ```bash
   # Stop the current app (Ctrl+C) and restart:
   ./gradio/launch.sh
   ```

2. **Click "Test Connection" First**
   - Go to the Connection tab
   - Click "Test Connection" button
   - Verify all systems show green checkmarks

3. **Check Your Atlas Vector Index Name**
   
   From your Atlas screenshot, I can see you have an index called "default". However, the app is looking for "vector_index". You have two options:

   **Option A: Update the index name in Atlas** (Recommended)
   - In Atlas, go to Atlas Search tab
   - Delete the "default" index
   - Create a new index named "vector_index" with this configuration:
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

   **Option B: Update the code to use "default"**
   - Change all occurrences of "vector_index" to "default" in the code

4. **Run the Debug Script**
   ```bash
   uv run python gradio/debug_search.py
   ```
   This will show you exactly what's working and what's not.

## Common Issues and Solutions:

### 1. Vector Index Name Mismatch
**Problem**: The code expects "vector_index" but Atlas has "default"
**Solution**: Rename the index in Atlas or update the code

### 2. Embedding Dimension Mismatch
**Problem**: Atlas index configured for wrong dimensions
**Solution**: 
- If using Voyage AI: Set dimensions to 1024
- If using OpenAI: Set dimensions to 1536

### 3. No Embeddings in Documents
**Problem**: Documents were inserted without embeddings
**Solution**: Re-run the data insertion:
```bash
uv run pytest tests/db-tests/test_mongo_medspa_data.py::TestMedSpaDataVectorSearch::test_process_medspa_data -v
```

### 4. Rate Limiting
**Problem**: Voyage AI rate limits (3 RPM for free tier)
**Solution**: The app automatically falls back to OpenAI

## Testing the Search:

1. **Test with Simple Query**:
   - Try: "CEO Drip"
   - Try: "$299"
   - Try: "mental clarity"

2. **Check Atlas Search Tester**:
   - Your Atlas screenshot shows it's working!
   - The issue is likely the index name mismatch

## Quick Test Commands:

```bash
# Check if data exists
uv run python tests/db-tests/view_medspa_data.py

# Test search directly
uv run python tests/db-tests/search_medspa_demo.py

# Debug the Gradio app
uv run python gradio/debug_search.py
```

## The CEO Drip Price

From your Atlas data, the CEO Drip costs **$299**. The full entry includes:
- Price: $299
- Duration: 60 minutes
- Key Ingredients: High-dose B-Complex, Vitamin C (2000mg), Magnesium, Zinc, B12, Amino Acids, Glutathione Push
- Benefits: Maximum cognitive enhancement and mental clarity, Sustained energy for 5-7 days

The search should return this information once the index name issue is resolved!