# 🖼️ Multimodal RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) application that works with both text and images in PDF documents. Upload PDFs containing charts, diagrams, and text, then search using either text queries or images to get intelligent responses.

## ✨ Features

- **📄 PDF Processing**: Extract and process both text and images from PDF documents
- **🔍 Dual Search Modes**: Search using text queries or image inputs
- **🤖 AI-Powered Responses**: Get intelligent answers using GPT-4 Vision
- **📊 Visual Context**: View retrieved text and images that influenced the response
- **🎯 Semantic Search**: Uses CLIP embeddings for unified text/image search
- **📱 User-Friendly Interface**: Clean Streamlit web interface

## 🏗️ Architecture

```
Input (PDF) → Text/Image Extraction → CLIP Embeddings → FAISS Vector Store
                                                              ↓
Query (Text/Image) → CLIP Embedding → Similarity Search → GPT-4 Vision → Response
```

**Key Components:**
- **CLIP Model**: Creates unified embeddings for text and images
- **FAISS**: Fast similarity search in vector space
- **GPT-4 Vision**: Multimodal language model for responses
- **PyMuPDF**: PDF processing and image extraction

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd multimodal-rag-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application

```bash
python run.py
```

The application will start and open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
multimodal-rag-project/
│
├── backend/                    # Core processing logic
│   ├── models/                # Model management
│   │   ├── clip_model.py      # CLIP model initialization
│   │   └── llm_model.py       # LLM initialization
│   │
│   ├── processing/            # Data processing
│   │   ├── pdf_processor.py   # PDF text/image extraction
│   │   ├── embeddings.py      # Embedding generation
│   │   └── vector_store.py    # Vector store operations
│   │
│   └── retrieval/             # Search and response
│       ├── retriever.py       # Multimodal retrieval
│       └── response_generator.py # Response generation
│
├── frontend/                  # User interface
│   └── streamlit_app.py       # Main Streamlit app
│
├── config/                    # Configuration
│   └── settings.py           # Application settings
│
├── data/                     # Data storage
│   └── sample_pdfs/          # Sample files
│
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables
├── run.py                   # Main entry point
└── README.md               # This file
```

## 💡 Usage Examples

### Text Search
1. Upload a PDF with charts and text
2. Ask questions like:
   - "What does the revenue chart show?"
   - "Summarize the main findings"
   - "What are the key metrics?"

### Image Search
1. Upload a PDF manual or report
2. Take a photo or upload an image
3. Get analysis of similar content in the PDF

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `CLIP_MODEL_NAME` | CLIP model name | `openai/clip-vit-base-patch32` |
| `LLM_MODEL_NAME` | LLM model name | `gpt-4` |
| `CHUNK_SIZE` | Text chunk size | `500` |
| `CHUNK_OVERLAP` | Text chunk overlap | `100` |
| `DEFAULT_K` | Default search results | `5` |

### Customization

Modify `config/settings.py` to adjust:
- Model configurations
- Processing parameters
- UI settings
- File handling options

## 🛠️ Development

### Adding New Features

1. **Backend Logic**: Add new modules in `backend/`
2. **Frontend Components**: Extend `frontend/streamlit_app.py`
3. **Configuration**: Update `config/settings.py`

### Testing

```bash
# Test with sample PDF
python -c "
from backend.processing.pdf_processor import pdf_processor
docs, embeddings, images = pdf_processor.process_pdf('data/sample_pdfs/test.pdf')
print(f'Processed {len(docs)} documents and {len(images)} images')
"
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for CLIP model)
- Internet connection (for OpenAI API)

### Python Dependencies
- `torch>=2.0.0` - PyTorch for CLIP model
- `transformers>=4.30.0` - Hugging Face transformers
- `langchain>=0.1.0` - LangChain framework
- `streamlit>=1.28.0` - Web interface
- `PyMuPDF>=1.23.0` - PDF processing
- `faiss-cpu>=1.7.4` - Vector search
- `openai>=1.12.0` - OpenAI API

## 🚨 Troubleshooting

### Common Issues

**1. CLIP Model Loading Error**
```bash
# Clear cache and reinstall
pip uninstall transformers torch
pip install torch transformers --no-cache-dir
```

**2. OpenAI API Error**
- Check your API key in `.env`
- Verify API quota and billing
- Ensure GPT-4 access is enabled

**3. Memory Issues**
- Reduce `CHUNK_SIZE` in settings
- Process smaller PDFs
- Restart the application

**4. PDF Processing Error**
- Ensure PDF is not password-protected
- Check PDF contains extractable text/images
- Try with a different PDF file

### Performance Tips

- **Faster Processing**: Use smaller chunk sizes
- **Better Results**: Increase number of search results (k)
- **Memory Optimization**: Process PDFs one at a time
- **Speed**: Keep PDFs under 50 pages for best performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for CLIP and GPT-4 models
- **LangChain** for the RAG framework
- **Streamlit** for the web interface
- **PyMuPDF** for PDF processing

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Search existing issues on GitHub
3. Create a new issue with detailed description

---

**Happy searching! 🔍✨**