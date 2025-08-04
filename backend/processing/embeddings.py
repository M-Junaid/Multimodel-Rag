import torch
import numpy as np
from PIL import Image
from backend.models.clip_model import clip_manager

class EmbeddingGenerator:
    """Handles text and image embedding generation using CLIP"""
    
    def __init__(self):
        self.clip_model = clip_manager.get_model()
        self.clip_processor = clip_manager.get_processor()
    
    def embed_image(self, image_data):
        """
        Embed image using CLIP
        
        Args:
            image_data: PIL Image or string path to image
            
        Returns:
            numpy.ndarray: Normalized image embedding
        """
        try:
            # Handle different input types
            if isinstance(image_data, str):
                image = Image.open(image_data).convert("RGB")
            else:
                image = image_data
            
            # Process image through CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings to unit vector
                features = features / features.norm(dim=-1, keepdim=True)
                return features.squeeze().numpy()
                
        except Exception as e:
            print(f"Error embedding image: {e}")
            raise
    
    def embed_text(self, text):
        """
        Embed text using CLIP
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy.ndarray: Normalized text embedding
        """
        try:
            inputs = self.clip_processor(
                text=text, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max token length
            )
            
            with torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
                # Normalize embeddings
                features = features / features.norm(dim=-1, keepdim=True)
                return features.squeeze().numpy()
                
        except Exception as e:
            print(f"Error embedding text: {e}")
            raise
    
    def embed_batch_texts(self, texts):
        """
        Embed multiple texts at once for efficiency
        
        Args:
            texts: List of text strings
            
        Returns:
            List of numpy arrays
        """
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def embed_batch_images(self, images):
        """
        Embed multiple images at once for efficiency
        
        Args:
            images: List of PIL Images or image paths
            
        Returns:
            List of numpy arrays
        """
        embeddings = []
        for image in images:
            embedding = self.embed_image(image)
            embeddings.append(embedding)
        return embeddings

# Global embedding generator instance
embedding_generator = EmbeddingGenerator()