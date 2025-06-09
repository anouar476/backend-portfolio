import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize global variables
qa_chain = None

class ResumeChatbot:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.qa_chain = self._initialize_qa_chain()
    
    def _initialize_qa_chain(self):
        # Load and process the PDF
        loader = PyPDFLoader(self.pdf_path)
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separator="\n"
        )
        chunks = text_splitter.split_documents(docs)
        
        # Create vector store with OpenAI embeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        vectorstore = FAISS.from_documents(chunks, embeddings)
        
        # Create QA chain
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(
            temperature=0, 
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def query(self, question):
        try:
            result = self.qa_chain({"query": question})
            return {
                "answer": result['result'],
                "sources": [doc.metadata['source'] for doc in result.get('source_documents', [])]
            }
        except Exception as e:
            return {"error": str(e)}

# Initialize chatbot with the resume
chatbot = ResumeChatbot("public/CV_Anouar_Moudad_En.pdf")

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    response = chatbot.query(question)
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
