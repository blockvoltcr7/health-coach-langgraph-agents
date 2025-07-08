as a developer i want to test my mongo db connection

from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://health-coach-ai-sami:<db_password>@cluster0-health-coach-a.69bhzsd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0-health-coach-ai"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

- [x] Create smoke test for MongoDB connection using pytest