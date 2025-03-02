from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import json
from typing import List, Dict

# Load environment variables
load_dotenv()

def preprocess_json(data: dict) -> str:
    """Preprocess JSON data to clearly differentiate between routers."""
    processed_text = []
    
    # Process each router separately
    for router, router_data in data.items():
        # Add clear router identifier
        processed_text.append(f"\nROUTER: {router}\n{'='*50}")
        
        # Process each major section
        for section, section_data in router_data.items():
            processed_text.append(f"\nSECTION: {section}\n{'-'*30}")
            section_str = json.dumps(section_data, indent=2)
            processed_text.append(section_str)
    
    return "\n".join(processed_text)

def load_and_split_json(file_path: str) -> List[str]:
    """Load JSON file and split into chunks with router context preservation."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Preprocess the JSON to add clear router boundaries
    processed_str = preprocess_json(data)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\nROUTER:", "\nSECTION:", "\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(processed_str)
    return chunks

def create_vectorstore(chunks: List[str]) -> Chroma:
    """Create vector store from text chunks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_dir = os.path.join(script_dir, "chroma_db")
    
    # Enhanced embeddings configuration
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        chunk_size=1000
    )
    
    # Create vector store with enhanced parameters
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="network_data_v1"
    )
    
    # Optional: Build HNSW index for better retrieval
    vectorstore.persist()
    return vectorstore

def enhance_prompt_template():
    """Create an enhanced prompt template that considers router context."""
    return """You are an expert network engineer analyzing network device data. 
    The data contains information about multiple routers (R1, R2, etc.).
    
    When answering questions:
    1. First analyze all the provided context. 
    2. Then identify which router(s) or device the question is about. When in doubt consider all routers.
    3. Only use information from the relevant router(s)
    4. Clearly indicate which router you are referring to in your answer
    5. If the router is not specified in the question, provide information for all relevant routers
    
    Context: {context}
    
    Question: {question}
    
    Answer in this format:
    - Router: [router name]
    - Details: [your detailed answer]
    """

def setup_rag_chain(vectorstore: Chroma) -> Dict:
    """Set up the RAG chain with enhanced retriever and prompt."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 12,  # Retrieve more documents
            "fetch_k": 24,  # Fetch more documents before filtering
            "lambda_mult": 0.7  # Balance between relevance and diversity
        }
    )
    
    llm = ChatOpenAI(
        temperature=0.2,
        model="gpt-4o-mini",
        max_tokens=2000
    )
    
    # Use enhanced prompt template
    prompt = ChatPromptTemplate.from_template(enhance_prompt_template())
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return {"chain": chain, "retriever": retriever}

def main():
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct absolute path to data.json relative to script location
    json_file = os.path.join(script_dir, "data.json")
    
    # Load and split document
    print(f"Loading and splitting document from: {json_file}")
    chunks = load_and_split_json(json_file)
    
    # Create vector store
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)
    
    # Setup RAG chain
    print("Setting up RAG chain...")
    rag_components = setup_rag_chain(vectorstore)
    chain = rag_components["chain"]
    
    # Interactive query loop
    print("\nRAG system ready! Enter your questions (type 'quit' to exit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'quit':
            break
            
        # Get answer
        try:
            answer = chain.invoke(question)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()