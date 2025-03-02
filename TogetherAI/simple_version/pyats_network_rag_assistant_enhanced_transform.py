"""
Network Configuration Analysis RAG System

This code implements a Retrieval-Augmented Generation (RAG) system specialized for network device configurations.
It processes JSON-formatted network device data, creates embeddings using HuggingFace's sentence transformers,
and uses the Together AI API with Llama 3.3 70B to provide detailed analysis of network configurations.

Key features:
- Processes structured network device configuration data from JSON files
- Maintains hierarchical context during text chunking
- Uses vector similarity search for relevant context retrieval
- Provides detailed technical analysis of network configurations
- Supports interactive Q&A about network device configurations

Dependencies:
- langchain-community
- langchain-huggingface
- together
- chromadb
- sentence-transformers

Usage:
    python pyats_network_rag_assistant.py

Environment variables required:
    TOGETHER_API_KEY: API key for Together AI platform
"""

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
from typing import List
import shutil

# Configure tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

def transform_json_to_flat(data: dict, max_depth: int = 10) -> str: # Adjust depth as needed
    """
    Transform network device JSON data into a flat, YAML-like format for better processing.
    Handles arbitrary JSON structures with configurable depth control.
    
    Args:
        data (dict): Raw JSON network device data
        max_depth (int): Maximum recursion depth to prevent infinite loops
        
    Returns:
        str: Formatted string with hierarchical information
    """
    def format_value(value) -> str:
        """Format different value types appropriately"""
        if isinstance(value, bool):
            return str(value).lower()
        elif value is None:
            return "null"
        else:
            return str(value)

    def process_dict(d: dict, prefix: str = "", depth: int = 0, indent: str = "  ") -> list:
        """Recursively process dictionary items with depth control"""
        if depth >= max_depth:
            return [f"{prefix}Max depth ({max_depth}) reached"]
        
        output = []
        
        # Sort keys for consistent output
        for key in sorted(d.keys()):
            value = d[key]
            current_prefix = f"{prefix}{indent}" if prefix else indent
            
            # Handle different value types
            if isinstance(value, dict):
                if not value:  # Empty dict
                    output.append(f"{current_prefix}{key}: {{}}")
                else:
                    output.append(f"{current_prefix}{key}:")
                    output.extend(process_dict(value, current_prefix, depth + 1))
            elif isinstance(value, (list, tuple)):
                if not value:  # Empty list
                    output.append(f"{current_prefix}{key}: []")
                else:
                    output.append(f"{current_prefix}{key}:")
                    for item in value:
                        if isinstance(item, dict):
                            output.extend(process_dict(item, current_prefix + indent, depth + 1))
                        else:
                            output.append(f"{current_prefix}{indent}- {format_value(item)}")
            else:
                output.append(f"{current_prefix}{key}: {format_value(value)}")
                
        return output

    # Process each device separately
    formatted_sections = []
    
    for device_name, device_data in sorted(data.items()):
        device_section = [f"\n### DEVICE: {device_name}"]
        
        # Group data by major categories
        categories = {
            "BASIC INFO": ["hostname", "version", "service"],
            "INTERFACES": ["interface", "GigabitEthernet", "FastEthernet", "Loopback", "Port-channel"],
            "ROUTING": ["ospf", "bgp", "eigrp", "static_routing", "routing"],
            "SECURITY": ["acl", "aaa", "authentication", "authorization"],
            "PROTOCOLS": ["lldp", "cdp", "stp", "ntp"],
            "OPERATIONAL": ["arp", "mac", "statistics", "counters"]
            # Add more categories as needed
        }
        
        # Process data by categories
        for category, keywords in categories.items():
            category_data = {}
            for key, value in device_data.items():
                if any(keyword.lower() in key.lower() for keyword in keywords):
                    category_data[key] = value
            
            if category_data:
                device_section.append(f"\n=== {category} ===")
                device_section.extend(process_dict(category_data))
        
        # Process remaining uncategorized data
        uncategorized = {k: v for k, v in device_data.items() 
                        if not any(keyword.lower() in k.lower() 
                                 for keywords in categories.values() 
                                 for keyword in keywords)}
        if uncategorized:
            device_section.append("\n=== OTHER ===")
            device_section.extend(process_dict(uncategorized))
        
        device_section.append(f"\n### END_DEVICE: {device_name}\n")
        formatted_sections.append("\n".join(device_section))
    
    return "\n".join(formatted_sections)

def load_and_split_json(file_path: str) -> List[str]:
    """
    Load JSON data and split it into manageable chunks while preserving context.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        List[str]: List of text chunks
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    processed_str = transform_json_to_flat(data)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        length_function=len,
        separators=["\n### DEVICE:", "\n=== ", "\n  ", "\n", ". "]
    )
    
    return text_splitter.split_text(processed_str)

def create_vectorstore(chunks: List[str]) -> Chroma:
    """
    Create and initialize the vector store from text chunks.
    
    Args:
        chunks (List[str]): List of text chunks to be embedded
        
    Returns:
        Chroma: Initialized vector store
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_dir = os.path.join(script_dir, "chroma_db")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="network_data_together"
    )

def get_context_and_query(retriever, question: str) -> str:
    """
    Generate a context-aware prompt based on the question and retrieved documents.
    
    Args:
        retriever: Document retriever instance
        question (str): User's question
        
    Returns:
        str: Formatted prompt with context
    """
    context_docs = retriever.invoke(question)
    formatted_sections = []
    current_device = None
    current_content = []
    
    for doc in context_docs:
        content = doc.page_content
        device_sections = content.split("### DEVICE:")
        
        for section in device_sections:
            if not section.strip():
                continue
            
            lines = section.strip().split("\n", 1)
            if len(lines) == 2:
                device_name = lines[0].strip()
                section_content = lines[1].strip()
                
                if current_device != device_name:
                    if current_device and current_content:
                        formatted_sections.append(
                            f"### DEVICE: {current_device}\n" + 
                            "\n".join(current_content)
                        )
                    current_device = device_name
                    current_content = []
                
                current_content.append(section_content)
    
    if current_device and current_content:
        formatted_sections.append(
            f"### DEVICE: {current_device}\n" + 
            "\n".join(current_content)
        )
    
    formatted_context = "\n\n".join(formatted_sections)
    
    return f"""As an expert network engineer, analyze the following network configuration and answer the question.
    The configuration is structured in a YAML-like format with clear section markers.

    Configuration Context:
    {formatted_context}

    Question: {question}

    Provide a comprehensive analysis following these guidelines:

    1. Device Information Analysis:
       * Identify relevant devices and their roles
       * Review configuration sections thoroughly
       * Note any important settings or states
       * Highlight critical parameters

    2. Technical Details:
       * Use `code` format for configuration lines
       * Include specific values and parameters
       * Reference exact configuration sections
       * Quote relevant operational states

    3. Relationship Analysis:
       * Examine device interconnections
       * Check protocol relationships
       * Verify network compatibility
       * Analyze end-to-end paths

    4. Implementation Impact:
       * Assess configuration effects
       * Identify potential issues
       * Suggest optimizations if relevant
       * Note security implications

    Format your response as follows:
    ### Device Information
    - Device: [device name(s)]
    - Relevant Configuration:
      * [specific technical details with values in `code` format]
      * [relationships between configurations if applicable]
      * [analysis]
      * [implications only if relevant or critical]
      * [additional context or recommendations only if relevant]

    ### In-a-nutshell:
    - [concise summary]
    """
    
    return prompt

def debug_print_transformed_data(file_path: str):
    """Debug function to print the transformed data"""
    with open(file_path, 'r') as file:
        data = json.load(file)
    transformed = transform_json_to_flat(data)
    print("\nTransformed Configuration:")
    print("-" * 80)
    print(transformed)
    print("-" * 80)

# Update the main function to include debug printing
def main():
    """Main function to run the RAG system."""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "data.json")
    chroma_dir = os.path.join(script_dir, "chroma_db")
    
    # Clean existing vector store if present
    if os.path.exists(chroma_dir):
        print(f"Cleaning up existing vector store in {chroma_dir}")
        shutil.rmtree(chroma_dir)
    
    print(f"Loading and splitting document from: {json_file}")
    chunks = load_and_split_json(json_file)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.7}
    )
    
    print("\nRAG system ready! Enter your questions (type 'quit' to exit):")
    while True:
        question = input("\n🔍 Question: ")
        if question.lower() == 'quit':
            break
            
        try:
            prompt = get_context_and_query(retriever, question)
            response = client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert network engineer with deep knowledge of:
        - Network protocols and their interdependencies
        - Device configuration analysis and troubleshooting
        - Network security and access control
        - Routing and switching technologies
        
        Your responses should:
        - Be technically precise and detailed
        - Include exact configuration lines when relevant
        - Consider relationships between different configurations
        - Verify all possible configuration sections
        - State explicitly if information is unavailable, but only after thorough verification
        """
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2048,
                temperature=0.3,
                top_p=0.8,
                top_k=50,
                repetition_penalty=1.1,
                timeout=40,
                stop=["[/INST]", "</s>"],
                stream=False
            )
            
            if response and response.choices:
                answer = response.choices[0].message.content
                print("\n📝 Answer:\n", answer)
            else:
                print("No response received")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()