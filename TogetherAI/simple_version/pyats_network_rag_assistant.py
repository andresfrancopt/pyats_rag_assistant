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

def transform_json_to_flat(data: dict) -> str:
    """
    Transform network device JSON data into a flat, YAML-like format for better processing.
    
    Args:
        data (dict): Raw JSON network device data
        
    Returns:
        str: Formatted string with hierarchical network information
    """
    output = []
    
    for device_name, device_data in data.items():
        output.append(f"\n### DEVICE: {device_name}")
        
        # Device Configuration Section
        output.append("=== CONFIGURATION ===")
        if "config" in device_data:
            # Basic system configuration
            output.append("SYSTEM:")
            for key, value in device_data["config"].items():
                if not isinstance(value, dict) or not value:
                    output.append(f"  {key}")
            
            # Interface configurations
            output.append("INTERFACES:")
            for key, value in device_data["config"].items():
                if key.startswith("interface"):
                    output.append(f"  {key}:")
                    for subkey, _ in value.items():
                        output.append(f"    {subkey}")
            
            # Routing protocol configurations
            output.append("ROUTING PROTOCOLS:")
            for key, value in device_data["config"].items():
                if key.startswith("router"):
                    output.append(f"  {key}:")
                    for subkey, _ in value.items():
                        output.append(f"    {subkey}")
        
        # Operational State Section
        output.append("=== OPERATIONAL STATE ===")
        if "interface" in device_data:
            output.append("INTERFACE STATE:")
            for intf, intf_data in device_data["interface"].get("info", {}).items():
                output.append(f"  {intf}:")
                if "enabled" in intf_data:
                    output.append(f"    Status: {'enabled' if intf_data['enabled'] else 'disabled'}")
                if "oper_status" in intf_data:
                    output.append(f"    Operational: {intf_data['oper_status']}")
                if "ipv4" in intf_data:
                    for ip_data in intf_data["ipv4"].values():
                        output.append(f"    IPv4: {ip_data['ip']}/{ip_data['prefix_length']}")
        
        # Routing Information Section
        output.append("=== ROUTING INFORMATION ===")
        
        # Process OSPF information
        if "ospf" in device_data:
            output.append("OSPF STATE:")
            ospf_info = device_data["ospf"].get("info", {})
            for vrf in ospf_info.get("vrf", {}).values():
                for af in vrf.get("address_family", {}).values():
                    for instance, inst_data in af.get("instance", {}).items():
                        output.append(f"  Process: {instance}")
                        output.append(f"  Router ID: {inst_data.get('router_id', 'N/A')}")
                        if "areas" in inst_data:
                            for area_id, area_data in inst_data["areas"].items():
                                output.append(f"  Area: {area_id}")
                                if "interfaces" in area_data:
                                    for intf, intf_data in area_data["interfaces"].items():
                                        output.append(f"    Interface: {intf}")
                                        output.append(f"    State: {intf_data.get('state', 'unknown')}")
        
        # Process static routes
        if "static_routing" in device_data:
            output.append("STATIC ROUTES:")
            static_info = device_data["static_routing"].get("info", {})
            for vrf in static_info.get("vrf", {}).values():
                for af in vrf.get("address_family", {}).values():
                    for route, route_data in af.get("routes", {}).items():
                        output.append(f"  Route: {route}")
                        if "next_hop" in route_data:
                            for hop in route_data["next_hop"].get("next_hop_list", {}).values():
                                output.append(f"    Next-hop: {hop.get('next_hop')}")
                                output.append(f"    Preference: {hop.get('preference')}")
        
        # Network Relationships Section
        output.append("=== NETWORK RELATIONSHIPS ===")
        if "arp" in device_data:
            output.append("ARP ENTRIES:")
            arp_info = device_data["arp"].get("info", {})
            for intf, intf_data in arp_info.get("interfaces", {}).items():
                output.append(f"  Interface: {intf}")
                if "ipv4" in intf_data:
                    for ip, details in intf_data["ipv4"].get("neighbors", {}).items():
                        output.append(f"    Neighbor IP: {ip}")
                        output.append(f"    MAC: {details.get('link_layer_address')}")
                        output.append(f"    Type: {details.get('origin')}")
        
        output.append(f"### END_DEVICE: {device_name}\n")
    
    return "\n".join(output)

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