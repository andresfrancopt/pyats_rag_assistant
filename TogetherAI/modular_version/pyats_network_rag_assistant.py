"""
Network Configuration Analysis RAG System
------------------------------------------
A RAG-based assistant that analyzes Cisco pyATS JSON output using Together AI's API.
Uses meta-llama/Llama-3.3-70B-Instruct-Turbo-Free model for generating responses.

The system:
- Processes pyATS JSON network data
- Creates embeddings using HuggingFace's all-MiniLM-L6-v2
- Uses Chroma for vector storage
- Implements MMR for relevant context retrieval
- Provides detailed network configuration analysis
"""

# Standard library imports
import os
import json
import shutil
from typing import List

# Third-party imports
from dotenv import load_dotenv
from together import Together
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Local imports
from dynamic_transformer import DynamicConfigTransformer

# Initialize environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

def setup_paths() -> dict:
    """Initialize file paths"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'json_file': os.path.join(script_dir, "data.json"),
        'chroma_dir': os.path.join(script_dir, "chroma_db"),
        'config_path': os.path.join(script_dir, 'section_config.yaml'),
        'output_path': os.path.join(script_dir, 'flattened_output.yaml')
    }

def load_and_split_json(file_path: str, config_path: str, output_path: str) -> List[str]:
    """Load JSON and split into chunks while preserving section context."""
    # Create transformer with the full path
    transformer = DynamicConfigTransformer(config_path)
    
    # Process and transform data
    with open(file_path, 'r') as f:
        data = json.load(f)
    processed_str = transformer.transform(data)
    
    # Save processed output
    with open(output_path, 'w') as f:
        f.write(processed_str)
    print(f"\n✨ Flattened output saved to: {output_path}")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=500,
        length_function=len,
        separators=[
            "\n### DEVICE:",
            "\n=== ",
            "\n  ",
            "\n",
            ". ",
        ]
    )
    
    return text_splitter.split_text(processed_str)

def create_vectorstore(chunks: List[str], chroma_dir: str) -> Chroma:
    """Create vector store from text chunks."""
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

def format_context_sections(context_docs) -> str:
    """Format retrieved context documents into sections."""
    formatted_sections = []
    current_device = None
    current_content = []
    
    for doc in context_docs:
        for section in doc.page_content.split("### DEVICE:"):
            if not section.strip():
                continue
            
            lines = section.strip().split("\n", 1)
            if len(lines) == 2:
                device_name, section_content = lines[0].strip(), lines[1].strip()
                
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
    
    return "\n\n".join(formatted_sections)

def get_context_and_query(retriever, question: str) -> str:
    """Get context and create prompt for the question."""
    context_docs = retriever.invoke(question)
    formatted_context = format_context_sections(context_docs)
    
    # Your existing detailed prompt structure
    prompt = f"""As an expert network engineer, analyze the following network configuration and answer the question.
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

def get_llm_response(client: Together, system_prompt: str, user_prompt: str) -> str:
    """Get response from LLM model."""
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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
        return response.choices[0].message.content
    return "No response received"

def main():
    """Main application entry point."""
    # Initialize
    paths = setup_paths()
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    # Setup vector store
    if os.path.exists(paths['chroma_dir']):
        print(f"Cleaning up existing vector store in {paths['chroma_dir']}")
        shutil.rmtree(paths['chroma_dir'])
    
    # Process documents
    print(f"Loading and splitting document from: {paths['json_file']}")
    chunks = load_and_split_json(paths['json_file'], paths['config_path'], paths['output_path'])
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks, paths['chroma_dir'])
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8, "fetch_k": 15, "lambda_mult": 0.7}
    )
    
    # Your existing system prompt
    system_prompt = """You are an expert network engineer with deep knowledge of:
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
    
    # Main interaction loop
    print("\nRAG system ready! Enter your questions (type 'quit' to exit):")
    while True:
        question = input("\n🔍 Question: ")
        if question.lower() == 'quit':
            break
        
        try:
            prompt = get_context_and_query(retriever, question)
            answer = get_llm_response(client, system_prompt, prompt)
            print("\n📝 Answer:\n", answer)
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()