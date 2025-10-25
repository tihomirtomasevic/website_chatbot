import os
import glob
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
KNOWLEDGE_BASE_PATH = "data/knowledge_base"

def load_knowledge_base():
    documents = []
    md_files = glob.glob(os.path.join(KNOWLEDGE_BASE_PATH, "*.md"))
    
    if not md_files:
        raise ValueError(f"No markdown files found in {KNOWLEDGE_BASE_PATH}")
    
    for file_path in md_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
    
    return documents

def create_vector_store():
    print("Loading knowledge base documents...")
    documents = load_knowledge_base()
    print(f"Loaded {len(documents)} documents")
    
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} text chunks")
    
    print("Creating embeddings (this may take a moment on first run)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    print("Building FAISS vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store ready!")
    
    return vector_store

def create_conversation_chain(vector_store):
    llm = ChatOpenAI(
        temperature=0.7,
        model_name=MODEL_NAME,
        openai_api_key=OPENAI_API_KEY
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=False,
        combine_docs_chain_kwargs={
            "prompt": get_qa_prompt()
        }
    )
    
    return conversation_chain

def get_qa_prompt():
    from langchain.prompts import PromptTemplate
    
    template = """You are a helpful AI assistant for a company website, answering visitor questions based on the company's FAQ and information.

Use the following context to answer the question. If you cannot find the answer in the context, politely say that you don't have that information and suggest contacting support.

Be concise, friendly, and professional. If the context mentions "MISSING ANSWER", acknowledge that this information is being updated and suggest an alternative way to get help.

Context:
{context}

Question: {question}

Answer:"""
    
    return PromptTemplate(template=template, input_variables=["context", "question"])

def create_gradio_interface():
    print("Initializing vector store and conversation chain...")
    vector_store = create_vector_store()
    conversation_chain = create_conversation_chain(vector_store)
    
    def chat_function(message, history):
        try:
            if not OPENAI_API_KEY:
                return "Error: OpenAI API key not configured. Please set OPENAI_API_KEY in your .env file."
            
            if not message.strip():
                return ""
            
            response = conversation_chain({"question": message})
            return response["answer"]
        
        except Exception as e:
            error_message = str(e)
            if "rate_limit" in error_message.lower():
                return "I'm experiencing high demand right now. Please try again in a moment."
            elif "api_key" in error_message.lower():
                return "There's an issue with the API configuration. Please contact support."
            else:
                return f"I apologize, but I encountered an error. Please try rephrasing your question or contact support."
    
    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="slate",
    ).set(
        button_primary_background_fill="#2563eb",
        button_primary_background_fill_hover="#1d4ed8",
        button_primary_text_color="#ffffff",
    )
    
    demo = gr.ChatInterface(
        fn=chat_function,
        title="💬 Company FAQ Assistant",
        description="Ask me anything about our services, pricing, portfolio, or general questions!",
        theme=custom_theme,
        examples=[
            "What kind of services does T2 Software offer?",
            "How can implementing Agentic System help my business?",
            "How can we get in touch?",
        ],
        chatbot=gr.Chatbot(height=500),
        textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
        submit_btn=gr.Button("Send", scale=1, variant="primary"),
        retry_btn=None,
        undo_btn=None,
        clear_btn="Clear Chat",
    )
    
    return demo

if __name__ == "__main__":
    print("Starting FAQ Chatbot...")
    
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not found in environment variables!")
        print("Please create a .env file with your OpenAI API key.")
    
    demo = create_gradio_interface()
    demo.launch()

