from pymongo.mongo_client import MongoClient
import certifi
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
load_dotenv()
import os

MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")

uri = f"mongodb+srv://health-coach-ai-sami:{MONGO_DB_PASSWORD}@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"
client = MongoClient(uri, server_api=ServerApi('1'), tlsCAFile=certifi.where())

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    raise AssertionError(f"MongoDB connection failed: {e}")