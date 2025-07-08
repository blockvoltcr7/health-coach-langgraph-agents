"""
Module 1.1: Environment Setup
Time: 5 minutes
Goal: Get your development environment ready for MongoDB vector search
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check required environment variables
def check_environment():
    """Verify all required API keys and connections are set"""
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for embeddings and chat",
        "VOYAGE_AI_API_KEY": "Voyage AI key for better embeddings",
        "MONGODB_URI": "MongoDB connection string",
        "MONGODB_DATABASE": "Database name (e.g., 'rag_course')"
    }
    
    missing_vars = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(f"{var}: {description}")
    
    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\n📝 Create a .env file with these variables")
        
        # Create sample .env file
        with open(".env.example", "w") as f:
            f.write("# MongoDB RAG Course Environment Variables\n\n")
            for var in required_vars:
                f.write(f"{var}=your_{var.lower()}_here\n")
        print("✅ Created .env.example file as template")
    else:
        print("✅ All environment variables configured!")
        print("\n🚀 Ready to start building RAG systems!")

# Install required packages
def create_requirements():
    """Create requirements.txt for the course"""
    requirements = """# MongoDB RAG Course Requirements
pymongo==4.6.1
openai==1.12.0
voyageai==0.2.3
langchain==0.1.7
langchain-community==0.0.21
python-dotenv==1.0.0
gradio==4.19.1
fastapi==0.109.2
uvicorn==0.27.1
pydantic==2.6.1
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    print("📦 Created requirements.txt")
    print("Run: pip install -r requirements.txt")

# Quick MongoDB connection test
def test_mongodb_connection():
    """Test MongoDB connection"""
    try:
        from pymongo import MongoClient
        
        client = MongoClient(os.getenv("MONGODB_URI"))
        db = client[os.getenv("MONGODB_DATABASE", "rag_course")]
        
        # Test connection
        client.admin.command('ping')
        print("✅ MongoDB connection successful!")
        
        # List collections
        collections = db.list_collection_names()
        if collections:
            print(f"📊 Existing collections: {collections}")
        else:
            print("📊 No collections yet - we'll create them next!")
            
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("🔧 Check your connection string in .env file")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Environment Setup\n")
    
    # Step 1: Check environment
    check_environment()
    
    # Step 2: Create requirements file
    print("\n" + "="*50 + "\n")
    create_requirements()
    
    # Step 3: Test MongoDB
    print("\n" + "="*50 + "\n")
    test_mongodb_connection()
    
    print("\n🎉 Setup complete! Ready for Module 1.2")