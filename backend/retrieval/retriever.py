from backend.processing.embeddings import embedding_generator
from backend.processing.vector_store import vector_store_manager
from config.settings import settings

class MultimodalRetriever:
    """Handles multimodal retrieval using CLIP embeddings"""
    
    def __init__(self):
        self.embedding_generator = embedding_generator
        self.vector_store_manager = vector_store_manager
    
    def retrieve_by_text(self, query_text, k=None):
        """
        Retrieve documents using text query
        
        Args:
            query_text: Text query string
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        k = k or settings.DEFAULT_K
        
        try:
            # Embed the query text
            query_embedding = self.embedding_generator.embed_text(query_text)
            
            # Search vector store
            results = self.vector_store_manager.search_by_vector(query_embedding, k)
            
            return results
            
        except Exception as e:
            print(f"Error in text retrieval: {e}")
            raise
    
    def retrieve_by_image(self, image_data, k=None):
        """
        Retrieve documents using image query
        
        Args:
            image_data: PIL Image or image path
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        k = k or settings.DEFAULT_K
        
        try:
            # Embed the query image
            query_embedding = self.embedding_generator.embed_image(image_data)
            
            # Search vector store
            results = self.vector_store_manager.search_by_vector(query_embedding, k)
            
            return results
            
        except Exception as e:
            print(f"Error in image retrieval: {e}")
            raise
    
    def retrieve_multimodal(self, query, query_type="text", k=None):
        """
        Generic multimodal retrieval method
        
        Args:
            query: Query data (text string or image)
            query_type: Type of query ("text" or "image")
            k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        if query_type == "text":
            return self.retrieve_by_text(query, k)
        elif query_type == "image":
            return self.retrieve_by_image(query, k)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
    
    def get_similar_documents(self, embedding_vector, k=None):
        """
        Get similar documents using pre-computed embedding
        
        Args:
            embedding_vector: Pre-computed embedding vector
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        k = k or settings.DEFAULT_K
        
        try:
            results = self.vector_store_manager.search_by_vector(embedding_vector, k)
            return results
            
        except Exception as e:
            print(f"Error retrieving similar documents: {e}")
            raise

# Global retriever instance
multimodal_retriever = MultimodalRetriever()