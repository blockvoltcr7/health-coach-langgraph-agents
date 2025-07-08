#!/bin/bash

echo "🚀 Launching MongoDB Vector Search Gradio App..."
echo "=================================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  Warning: .env file not found in project root"
    echo "Please create a .env file with:"
    echo "  MONGO_DB_PASSWORD=your_password"
    echo "  VOYAGE_AI_API_KEY=your_key (optional)"
    echo "  OPENAI_API_KEY=your_key (optional)"
    echo ""
fi

# Navigate to project root
cd "$(dirname "$0")/.." || exit

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "Please install uv first: https://github.com/astral-sh/uv"
    exit 1
fi

# Run the Gradio app
echo "Starting Gradio app on http://localhost:7860"
echo "Press Ctrl+C to stop"
echo ""

uv run python gradio/mongodb_vector_search_app.py