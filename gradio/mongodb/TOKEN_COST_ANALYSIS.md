# Token Usage & Cost Analysis

## Overview
Added token calculation and cost estimation to the Gradio chat interface to help track OpenAI API costs when using different numbers of context documents.

## Features Added

### 1. Token Calculator
- Calculates total input tokens sent to OpenAI
- Breaks down tokens by:
  - Context documents (from MongoDB)
  - System prompt
  - Conversation history
  - User message

### 2. Cost Estimator
- Real-time cost calculation based on OpenAI pricing
- Supports multiple models:
  - gpt-4o-mini: $0.15 per 1M input tokens
  - gpt-4o: $2.50 per 1M input tokens (17x more expensive)

### 3. UI Improvements
- Token usage display after each message
- Cost comparison guide in the sidebar
- Shows number of documents used

## Cost Analysis for "All IV Treatments" Query

### Using gpt-4o-mini (Recommended)
| Documents | Tokens | Cost | Treatments Found |
|-----------|--------|------|------------------|
| 3 docs | 456 | $0.000068 | 1 treatment |
| 5 docs | 633 | $0.000095 | 1 treatment |
| 10 docs | 1,030 | $0.000155 | 3 treatments |
| 15 docs | 1,594 | $0.000239 | 4+ treatments |

### Using gpt-4o
| Documents | Tokens | Cost | Treatments Found |
|-----------|--------|------|------------------|
| 5 docs | 633 | $0.001582 | 1 treatment |
| 10 docs | 1,030 | $0.002575 | 2 treatments |

## Recommendations

1. **Default Settings (Balanced)**:
   - 5 documents with gpt-4o-mini
   - Cost: ~$0.0001 per query
   - Good for most queries

2. **Comprehensive Queries** (like "all treatments"):
   - 10-15 documents with gpt-4o-mini
   - Cost: ~$0.0002-0.0003 per query
   - Better coverage for broad questions

3. **Cost Optimization**:
   - Always use gpt-4o-mini unless you need advanced reasoning
   - gpt-4o costs 17x more for the same query
   - Adjust document count based on query type

## Implementation Details

### Token Estimation
```python
def calculate_tokens(text: str) -> int:
    """Rough estimation of tokens (1 token ≈ 4 characters)"""
    return len(text) // 4
```

### Cost Calculation
```python
def calculate_cost(tokens: int, model: str) -> float:
    """Calculate cost based on OpenAI pricing"""
    pricing = {
        "gpt-4o": {"input": 2.50},      # per 1M tokens
        "gpt-4o-mini": {"input": 0.15},  # per 1M tokens
    }
    cost = (tokens / 1_000_000) * pricing[model]["input"]
    return cost
```

## UI Display
The token information appears below the chat interface showing:
- Total input tokens with breakdown
- Number of documents retrieved
- Estimated cost for the query
- Cost per 1,000 tokens

This helps users make informed decisions about:
- How many context documents to retrieve
- Which model to use
- Expected costs for their usage patterns