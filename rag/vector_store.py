from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorStore:
    def __init__(self, embeddings):
        self.store = None
        self.embeddings = embeddings
    
    def create_from_documents(self, documents):
        self.store = FAISS.from_documents(documents, self.embeddings)
    
    def save(self, path):
        self.store.save_local(path)
    
    def load(self, path):
        self.store = FAISS.load_local(path, self.embeddings)
    
    def similarity_search(self, query: str, k: int = 3):
        return self.store.similarity_search(query, k=k)