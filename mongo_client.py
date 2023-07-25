
from pymongo import MongoClient


def get_database():
    uri = "mongodb+srv://vectordb-admin:ILy2oEjFS3LoHxk8@gen-ai-cluster.zo2kezn.mongodb.net/?retryWrites=true&w=majority"
    # Create a new client and connect to the server
    # Send a ping to confirm a successful connection
    client = MongoClient(uri)
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
        return client['vector-db']
    except Exception as e:
        print(e)
