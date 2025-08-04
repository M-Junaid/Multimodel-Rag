import pymupdf
import io
import base64
from PIL import Image
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.processing.embeddings import embedding_generator
from config.settings import settings

class PDFProcessor:
    """Processes PDF files to extract text and images"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
    
    def process_pdf(self, pdf_path_or_bytes, progress_callback=None):
        """
        Process PDF file and extract text/images with embeddings
        
        Args:
            pdf_path_or_bytes: Path to PDF file or bytes data
            progress_callback: Optional callback function for progress updates
            
        Returns:
            tuple: (all_docs, all_embeddings, image_data_store)
        """
        try:
            # Open PDF
            if isinstance(pdf_path_or_bytes, str):
                doc = pymupdf.open(pdf_path_or_bytes)
            else:
                doc = pymupdf.open(stream=pdf_path_or_bytes, filetype="pdf")
            
            all_docs = []
            all_embeddings = []
            image_data_store = {}
            
            total_pages = len(doc)
            
            for i, page in enumerate(doc):
                # Update progress if callback provided
                if progress_callback:
                    progress_callback((i + 1) / total_pages)
                
                # Process text from page
                text_docs, text_embeddings = self._process_page_text(page, i)
                all_docs.extend(text_docs)
                all_embeddings.extend(text_embeddings)
                
                # Process images from page
                image_docs, image_embeddings, page_image_store = self._process_page_images(page, i)
                all_docs.extend(image_docs)
                all_embeddings.extend(image_embeddings)
                image_data_store.update(page_image_store)
            
            doc.close()
            return all_docs, all_embeddings, image_data_store
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
    
    def _process_page_text(self, page, page_num):
        """Process text content from a PDF page"""
        docs = []
        embeddings = []
        
        text = page.get_text()
        if text.strip():
            # Create temporary document for splitting
            temp_doc = Document(
                page_content=text, 
                metadata={"page": page_num, "type": "text"}
            )
            text_chunks = self.text_splitter.split_documents([temp_doc])
            
            # Embed each chunk
            for chunk in text_chunks:
                try:
                    embedding = embedding_generator.embed_text(chunk.page_content)
                    embeddings.append(embedding)
                    docs.append(chunk)
                except Exception as e:
                    print(f"Error embedding text chunk on page {page_num}: {e}")
                    continue
        
        return docs, embeddings
    
    def _process_page_images(self, page, page_num):
        """Process images from a PDF page"""
        docs = []
        embeddings = []
        image_store = {}
        
        for img_index, img in enumerate(page.get_images(full=True)):
            try:
                # Extract image data
                xref = img[0]
                base_image = page.parent.extract_image(xref)
                image_bytes = base_image["image"]
                
                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Resize if too large
                if pil_image.size[0] > settings.MAX_IMAGE_SIZE[0] or pil_image.size[1] > settings.MAX_IMAGE_SIZE[1]:
                    pil_image.thumbnail(settings.MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)
                
                # Create unique identifier
                image_id = f"page_{page_num}_img_{img_index}"
                
                # Store image as base64 for GPT-4V
                buffered = io.BytesIO()
                pil_image.save(buffered, format=settings.IMAGE_FORMAT)
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                image_store[image_id] = img_base64
                
                # Embed image using CLIP
                embedding = embedding_generator.embed_image(pil_image)
                embeddings.append(embedding)
                
                # Create document for image
                image_doc = Document(
                    page_content=f"[Image: {image_id}]",
                    metadata={"page": page_num, "type": "image", "image_id": image_id}
                )
                docs.append(image_doc)
                
            except Exception as e:
                print(f"Error processing image {img_index} on page {page_num}: {e}")
                continue
        
        return docs, embeddings, image_store

# Global PDF processor instance
pdf_processor = PDFProcessor()