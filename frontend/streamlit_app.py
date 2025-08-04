import streamlit as st
import tempfile
import base64
from PIL import Image
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processing.pdf_processor import pdf_processor
from backend.processing.vector_store import vector_store_manager
from backend.retrieval.retriever import multimodal_retriever
from backend.retrieval.response_generator import response_generator
from config.settings import settings

# Page configuration
st.set_page_config(
    page_title=settings.PAGE_TITLE,
    page_icon=settings.PAGE_ICON,
    layout=settings.LAYOUT,
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .image-input-section {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #2E86AB;
        color: white;
        border-radius: 5px;
    }
    .image-preview {
        border: 2px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 10px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'vector_store_ready' not in st.session_state:
        st.session_state.vector_store_ready = False
    if 'image_data_store' not in st.session_state:
        st.session_state.image_data_store = {}
    if 'processed_docs_count' not in st.session_state:
        st.session_state.processed_docs_count = 0
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None

def process_uploaded_pdf(uploaded_file):
    """Process uploaded PDF file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(progress):
            progress_bar.progress(progress)
            status_text.text(f"Processing... {int(progress * 100)}%")
        
        # Process PDF
        all_docs, all_embeddings, image_data_store = pdf_processor.process_pdf(
            tmp_path, progress_callback
        )
        
        if all_docs and all_embeddings:
            # Create vector store
            status_text.text("Creating vector store...")
            vector_store_manager.create_vector_store(all_docs, all_embeddings)
            
            # Update session state
            st.session_state.vector_store_ready = True
            st.session_state.image_data_store = image_data_store
            st.session_state.processed_docs_count = len(all_docs)
            st.session_state.current_pdf_name = uploaded_file.name
            
            # Clean up
            os.unlink(tmp_path)
            progress_bar.empty()
            status_text.empty()
            
            return True, f"Successfully processed {len(all_docs)} document chunks!"
        else:
            return False, "No content found in the PDF file."
            
    except Exception as e:
        return False, f"Error processing PDF: {str(e)}"

def display_retrieved_context(retrieved_docs, image_data_store):
    """Display retrieved context in an expandable section"""
    with st.expander("📚 Retrieved Context", expanded=False):
        st.write(f"Found {len(retrieved_docs)} relevant items:")
        
        for i, doc in enumerate(retrieved_docs):
            doc_type = doc.metadata.get("type", "unknown")
            page = doc.metadata.get("page", 0)
            
            if doc_type == "text":
                preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                st.write(f"**{i+1}. Text from page {page+1}:**")
                st.write(preview)
            else:
                st.write(f"**{i+1}. Image from page {page+1}:**")
                image_id = doc.metadata.get("image_id")
                if image_id and image_id in image_data_store:
                    img_data = base64.b64decode(image_data_store[image_id])
                    st.image(img_data, caption=f"Image from page {page+1}", width=300)
            st.write("---")

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ Multimodal RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown("Upload a PDF document, then search using text queries or images!")
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📁 PDF Document")
        
        uploaded_pdf = st.file_uploader(
            "Upload PDF Document",
            type="pdf",
            help="Upload a PDF document containing both text and images"
        )
        
        if uploaded_pdf is not None:
            st.success(f"PDF uploaded: {uploaded_pdf.name}")
            
            if st.button("🔄 Process PDF", type="primary"):
                with st.spinner("Processing PDF... This may take a few minutes."):
                    success, message = process_uploaded_pdf(uploaded_pdf)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        # Display processing status
        if st.session_state.vector_store_ready:
            st.success("✅ PDF processed and ready!")
            st.info(f"📄 {st.session_state.processed_docs_count} document chunks")
            st.info(f"📋 Current PDF: {st.session_state.current_pdf_name}")
            
            # Search settings
            st.subheader("🔧 Search Settings")
            k_value = st.slider("Number of results:", 1, 10, settings.DEFAULT_K)
        else:
            st.info("👆 Upload and process a PDF document first.")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.vector_store_ready:
            # Query type selection
            query_type = st.radio(
                "Choose search method:",
                ["Text Query", "Image Search"],
                horizontal=True
            )
            
            if query_type == "Text Query":
                st.markdown('<div class="image-input-section">', unsafe_allow_html=True)
                st.subheader("💬 Text Search")
                
                query_text = st.text_area(
                    "Enter your question:",
                    placeholder="What does the chart show? Summarize the document. What visual elements are present?",
                    height=100
                )
                
                if st.button("🔍 Search with Text", type="primary", disabled=not query_text.strip()):
                    if query_text.strip():
                        with st.spinner("Searching and generating response..."):
                            try:
                                # Retrieve relevant documents
                                retrieved_docs = multimodal_retriever.retrieve_by_text(
                                    query_text, k=k_value
                                )
                                
                                # Generate response
                                response = response_generator.process_text_query(
                                    query_text,
                                    retrieved_docs,
                                    st.session_state.image_data_store
                                )
                                
                                # Display results
                                st.subheader("📝 Answer:")
                                st.write(response)
                                
                                # Show retrieved context
                                display_retrieved_context(
                                    retrieved_docs, 
                                    st.session_state.image_data_store
                                )
                                
                            except Exception as e:
                                st.error(f"Error during search: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            else:  # Image Search
                st.markdown('<div class="image-input-section">', unsafe_allow_html=True)
                st.subheader("🖼️ Image Search")
                
                # Image input options
                input_method = st.radio(
                    "Choose input method:",
                    ["Upload Image", "Take Photo"],
                    horizontal=True
                )
                
                input_image = None
                
                if input_method == "Upload Image":
                    uploaded_image = st.file_uploader(
                        "Upload an image to search in the PDF:",
                        type=["png", "jpg", "jpeg", "gif", "bmp"],
                        help="Upload an image to find related content in the PDF"
                    )
                    
                    if uploaded_image is not None:
                        input_image = Image.open(uploaded_image).convert("RGB")
                        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                        st.image(input_image, caption="Input Image", width=400)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                else:  # Take Photo
                    camera_image = st.camera_input("Take a photo to search in the PDF:")
                    
                    if camera_image is not None:
                        input_image = Image.open(camera_image).convert("RGB")
                        st.markdown('<div class="image-preview">', unsafe_allow_html=True)
                        st.image(input_image, caption="Captured Image", width=400)
                        st.markdown('</div>', unsafe_allow_html=True)
                
                # Add text input for image questions
                if input_image is not None:
                    st.markdown("---")
                    st.subheader("💬 Ask a question about the image:")
                    
                    image_question = st.text_area(
                        "Enter your question about the image:",
                        placeholder="What does this image show? What are the key elements? How does this relate to the PDF content?",
                        height=100,
                        help="Ask specific questions about the uploaded image and its relationship to the PDF content"
                    )
                    
                    # Search button
                    col1_search, col2_search = st.columns([1, 1])
                    
                    with col1_search:
                        if st.button("🔍 Search with Image", type="primary", disabled=not image_question.strip()):
                            with st.spinner("Searching PDF for related content..."):
                                try:
                                    # Retrieve relevant documents
                                    retrieved_docs = multimodal_retriever.retrieve_by_image(
                                        input_image, k=k_value
                                    )
                                    
                                    # Generate response
                                    response = response_generator.process_image_query(
                                        input_image,
                                        retrieved_docs,
                                        st.session_state.image_data_store
                                    )
                                    
                                    # Display results
                                    st.subheader("📝 Analysis Results:")
                                    st.write(response)
                                    
                                    # Show retrieved context
                                    display_retrieved_context(
                                        retrieved_docs, 
                                        st.session_state.image_data_store
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error during search: {str(e)}")
                    
                    with col2_search:
                        if st.button("🤖 Ask AI about Image", type="secondary", disabled=not image_question.strip()):
                            with st.spinner("Analyzing image and generating response..."):
                                try:
                                    # Retrieve relevant documents
                                    retrieved_docs = multimodal_retriever.retrieve_by_image(
                                        input_image, k=k_value
                                    )
                                    
                                    # Generate response with the specific question
                                    response = response_generator.process_image_question(
                                        input_image,
                                        image_question,
                                        retrieved_docs,
                                        st.session_state.image_data_store
                                    )
                                    
                                    # Display results
                                    st.subheader("🤖 AI Response:")
                                    st.write(response)
                                    
                                    # Show retrieved context
                                    display_retrieved_context(
                                        retrieved_docs, 
                                        st.session_state.image_data_store
                                    )
                                    
                                except Exception as e:
                                    st.error(f"Error generating response: {str(e)}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        else:
            st.info("Please upload and process a PDF document first.")
    
    with col2:
        st.subheader("ℹ️ How it works")
        st.markdown("""
        ### Text Search
        1. **Ask questions** about the PDF content
        2. **Get answers** based on both text and images
        3. **View context** that influenced the response
        
        ### Image Search  
        1. **Upload/capture** an image
        2. **Ask specific questions** about the image
        3. **Get AI analysis** with PDF context
        
        **Example Image Questions:**
        - "What does this image show?"
        - "How does this relate to the PDF content?"
        - "What are the key elements in this image?"
        - "Explain the data or information displayed"
        """)
        
        st.subheader("🔧 Technical Details")
        st.markdown("""
        - **CLIP Model**: Unified text/image embeddings
        - **FAISS**: Fast similarity search
        - **GPT-4 Vision**: Multimodal responses
        - **PyMuPDF**: PDF processing
        """)
        
        if st.session_state.vector_store_ready:
            st.subheader("📊 Current Document")
            st.success("✅ Ready for queries")
            st.info(f"Document chunks: {st.session_state.processed_docs_count}")
            st.info(f"Images stored: {len(st.session_state.image_data_store)}")

if __name__ == "__main__":
    main()