# AI Resume Assistant Backend

This is the backend service for the AI Resume Assistant. It uses Flask to serve a RAG (Retrieval-Augmented Generation) model that can answer questions about Moudad's resume.

## Setup Instructions

1. **Install Python Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Set Up Environment Variables**
   - Create a `.env` file in the `backend` directory
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

3. **Run the Backend Server**
   ```bash
   python app.py
   ```
   The server will start on `http://localhost:5000`

## API Endpoints

- `GET /api/health` - Health check endpoint
- `POST /api/chat` - Chat endpoint
  ```json
  {
    "question": "What is Moudad's experience with cybersecurity?"
  }
  ```
  
  Response:
  ```json
  {
    "answer": "Moudad has experience in...",
    "sources": ["CV_Anouar_Moudad_En.pdf"]
  }
  ```

## Development

- The backend uses FAISS for vector similarity search
- The resume PDF is loaded and processed on server startup
- All responses include source documents for verification

## Troubleshooting

- If you get rate-limited by OpenAI, consider adding rate limiting to the API
- For large resumes, you might need to adjust the chunk size in `app.py`
- Make sure the PDF file exists at the specified path in `app.py`
