import numpy as np
from langchain_community.vectorstores import FAISS
from config.settings import settings

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self):
        self.vector_store = None
    
    def create_vector_store(self, docs, embeddings):
        """
        Create FAISS vector store from documents and embeddings
        
        Args:
            docs: List of Document objects
            embeddings: List of embedding arrays
            
        Returns:
            FAISS vector store instance
        """
        try:
            if not docs or not embeddings:
                raise ValueError("Documents and embeddings cannot be empty")
            
            if len(docs) != len(embeddings):
                raise ValueError("Number of documents must match number of embeddings")
            
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings)
            
            # Create FAISS vector store
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=[(doc.page_content, emb) for doc, emb in zip(docs, embeddings_array)],
                embedding=None,  # Using precomputed embeddings
                metadatas=[doc.metadata for doc in docs]
            )
            
            print(f"Vector store created successfully with {len(docs)} documents")
            return self.vector_store
            
        except Exception as e:
            print(f"Error creating vector store: {e}")
            raise
    
    def search_by_vector(self, query_embedding, k=None):
        """
        Search vector store using embedding vector
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or settings.DEFAULT_K
        
        try:
            results = self.vector_store.similarity_search_by_vector(
                embedding=query_embedding,
                k=k
            )
            return results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            raise
    
    def search_by_text(self, query_text, k=None):
        """
        Search vector store using text query
        
        Args:
            query_text: Text query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        k = k or settings.DEFAULT_K
        
        try:
            results = self.vector_store.similarity_search(
                query=query_text,
                k=k
            )
            return results
            
        except Exception as e:
            print(f"Error searching vector store: {e}")
            raise
    
    def get_vector_store(self):
        """Get the current vector store instance"""
        return self.vector_store
    
    def is_initialized(self):
        """Check if vector store is initialized"""
        return self.vector_store is not None
    
    def save_vector_store(self, path):
        """Save vector store to disk"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            self.vector_store.save_local(path)
            print(f"Vector store saved to {path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
            raise
    
    def load_vector_store(self, path):
        """Load vector store from disk"""
        try:
            self.vector_store = FAISS.load_local(path, embeddings=None)
            print(f"Vector store loaded from {path}")
            return self.vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            raise

# Global vector store manager instance
vector_store_manager = VectorStoreManager()