import base64
import io
from PIL import Image
from langchain.schema.messages import HumanMessage
from backend.models.llm_model import llm_manager

class ResponseGenerator:
    """Generates responses using multimodal LLM"""
    
    def __init__(self):
        self.llm = llm_manager.get_llm()
    
    def create_text_query_message(self, query, retrieved_docs, image_data_store):
        """
        Create message for text-based query
        
        Args:
            query: Text query string
            retrieved_docs: List of retrieved documents
            image_data_store: Dictionary of image data
            
        Returns:
            HumanMessage for LLM
        """
        content = []
        
        # Add the query
        content.append({
            "type": "text",
            "text": f"Question: {query}\n\nContext:\n"
        })
        
        # Separate text and image documents
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
        
        # Add text context
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {doc.metadata['page'] + 1}]: {doc.page_content}"
                for doc in text_docs
            ])
            content.append({
                "type": "text",
                "text": f"Text excerpts:\n{text_context}\n"
            })
        
        # Add images
        for doc in image_docs:
            image_id = doc.metadata.get("image_id")
            if image_id and image_id in image_data_store:
                content.append({
                    "type": "text",
                    "text": f"\n[Image from page {doc.metadata['page'] + 1}]:\n"
                })
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_data_store[image_id]}"
                    }
                })
        
        # Add instruction
        content.append({
            "type": "text",
            "text": "\n\nPlease answer the question based on the provided text and images."
        })
        
        return HumanMessage(content=content)
    
    def create_image_query_message(self, input_image, retrieved_docs, image_data_store):
        """
        Create message for image-based query
        
        Args:
            input_image: PIL Image object
            retrieved_docs: List of retrieved documents
            image_data_store: Dictionary of image data
            
        Returns:
            HumanMessage for LLM
        """
        content = []
        
        # Add the input image
        buffered = io.BytesIO()
        input_image.save(buffered, format="PNG")
        input_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        content.append({
            "type": "text",
            "text": "I have provided you with an input image and relevant content from a PDF document. Please analyze the input image and provide information based on the related content found in the PDF.\n\nInput Image:"
        })
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{input_img_base64}"
            }
        })
        
        # Separate text and image documents from PDF
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
        
        content.append({
            "type": "text",
            "text": "\n\nRelated content from PDF:\n"
        })
        
        # Add text context
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {doc.metadata['page'] + 1}]: {doc.page_content}"
                for doc in text_docs
            ])
            content.append({
                "type": "text",
                "text": f"Text content:\n{text_context}\n"
            })
        
        # Add related images from PDF
        if image_docs:
            content.append({
                "type": "text",
                "text": "\nRelated images from PDF:\n"
            })
            
            for doc in image_docs:
                image_id = doc.metadata.get("image_id")
                if image_id and image_id in image_data_store:
                    content.append({
                        "type": "text",
                        "text": f"[Image from page {doc.metadata['page'] + 1}]:\n"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data_store[image_id]}"
                        }
                    })
        
        content.append({
            "type": "text",
            "text": "\n\nBased on the input image and the related content from the PDF, please provide a comprehensive answer about what the input image shows and how it relates to the information in the PDF document."
        })
        
        return HumanMessage(content=content)
    
    def generate_response(self, message):
        """
        Generate response using LLM
        
        Args:
            message: HumanMessage object
            
        Returns:
            Response content string
        """
        try:
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            raise
    
    def process_text_query(self, query, retrieved_docs, image_data_store):
        """
        Process text query and generate response
        
        Args:
            query: Text query string
            retrieved_docs: List of retrieved documents
            image_data_store: Dictionary of image data
            
        Returns:
            Response content string
        """
        message = self.create_text_query_message(query, retrieved_docs, image_data_store)
        return self.generate_response(message)
    
    def process_image_query(self, input_image, retrieved_docs, image_data_store):
        """
        Process image query and generate response
        
        Args:
            input_image: PIL Image object
            retrieved_docs: List of retrieved documents
            image_data_store: Dictionary of image data
            
        Returns:
            Response content string
        """
        message = self.create_image_query_message(input_image, retrieved_docs, image_data_store)
        return self.generate_response(message)
    
    def create_image_question_message(self, input_image, question, retrieved_docs, image_data_store):
        """
        Create message for image question with specific user question
        
        Args:
            input_image: PIL Image object
            question: User's specific question about the image
            retrieved_docs: List of retrieved documents
            image_data_store: Dictionary of image data
            
        Returns:
            HumanMessage for LLM
        """
        content = []
        
        # Add the input image
        buffered = io.BytesIO()
        input_image.save(buffered, format="PNG")
        input_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        content.append({
            "type": "text",
            "text": f"I have provided you with an input image and relevant content from a PDF document. Please answer the following specific question about the image:\n\nQuestion: {question}\n\nInput Image:"
        })
        
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{input_img_base64}"
            }
        })
        
        # Separate text and image documents from PDF
        text_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "text"]
        image_docs = [doc for doc in retrieved_docs if doc.metadata.get("type") == "image"]
        
        content.append({
            "type": "text",
            "text": "\n\nRelated content from PDF:\n"
        })
        
        # Add text context
        if text_docs:
            text_context = "\n\n".join([
                f"[Page {doc.metadata['page'] + 1}]: {doc.page_content}"
                for doc in text_docs
            ])
            content.append({
                "type": "text",
                "text": f"Text content:\n{text_context}\n"
            })
        
        # Add related images from PDF
        if image_docs:
            content.append({
                "type": "text",
                "text": "\nRelated images from PDF:\n"
            })
            
            for doc in image_docs:
                image_id = doc.metadata.get("image_id")
                if image_id and image_id in image_data_store:
                    content.append({
                        "type": "text",
                        "text": f"[Image from page {doc.metadata['page'] + 1}]:\n"
                    })
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_data_store[image_id]}"
                        }
                    })
        
        content.append({
            "type": "text",
            "text": f"\n\nPlease provide a detailed answer to the question: '{question}' based on the input image and the related content from the PDF document."
        })
        
        return HumanMessage(content=content)
    
    def process_image_question(self, input_image, question, retrieved_docs, image_data_store):
        """
        Process image question and generate response
        
        Args:
            input_image: PIL Image object
            question: User's specific question about the image
            retrieved_docs: List of retrieved documents
            image_data_store: Dictionary of image data
            
        Returns:
            Response content string
        """
        message = self.create_image_question_message(input_image, question, retrieved_docs, image_data_store)
        return self.generate_response(message)

# Global response generator instance
response_generator = ResponseGenerator()