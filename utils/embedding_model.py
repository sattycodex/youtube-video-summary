import requests
import numpy as np
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
import os

load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")

def generate_embeddings(text):
    print("Generating embeddings...")
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()

    if "data" not in data:
        print("can not find data in response")
        return np.zeros(1536)
    
    embedding = np.array(data['data'][0]['embedding'])
    
    return embedding



class EuriEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [generate_embeddings(text) for text in texts]

    def embed_query(self, text):
        return generate_embeddings(text)