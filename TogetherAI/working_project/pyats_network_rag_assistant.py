from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
from typing import List, Dict
import shutil

# Add this after imports to handle tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

def transform_json_to_flat(data: dict) -> str:
    """Transform JSON data into a flattened, easily queryable format."""
    output = []
    
    for device_name, device_data in data.items():
        output.append(f"\n### DEVICE: {device_name}")
        
        # SECTION 1: Device Configuration
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
                    for subkey, subvalue in value.items():
                        output.append(f"    {subkey}")
            
            # Routing protocol configurations
            output.append("ROUTING PROTOCOLS:")
            for key, value in device_data["config"].items():
                if key.startswith("router"):
                    output.append(f"  {key}:")
                    for subkey, subvalue in value.items():
                        output.append(f"    {subkey}")
        
        # SECTION 2: Network State
        output.append("=== OPERATIONAL STATE ===")
        
        # Interface state
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
        
        # SECTION 3: Routing Information
        output.append("=== ROUTING INFORMATION ===")
        
        # OSPF state
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
        
        # Static routes
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
        
        # SECTION 4: Network Relationships
        output.append("=== NETWORK RELATIONSHIPS ===")
        
        # ARP information
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
    """Load JSON and split into chunks while preserving section context."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    processed_str = transform_json_to_flat(data)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,         # Increased chunk size
        chunk_overlap=500,       # Increased overlap
        length_function=len,
        separators=[
            "\n### DEVICE:",     # Primary separator
            "\n=== ",           # Section separator
            "\n  ",            # Subsection separator
            "\n",
            ". ",
        ]
    )
    
    chunks = text_splitter.split_text(processed_str)
    return chunks

def create_vectorstore(chunks: List[str]) -> Chroma:
    """Create vector store from text chunks."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chroma_dir = os.path.join(script_dir, "chroma_db")
    
    # Initialize embeddings with kwargs to suppress warnings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name="network_data_together"
    )
    
    return vectorstore

def get_context_and_query(retriever, question: str) -> str:
    context_docs = retriever.invoke(question)
    
    # Improved context extraction
    formatted_sections = []
    current_device = None
    current_content = []
    
    for doc in context_docs:
        content = doc.page_content
        # Split by device sections
        device_sections = content.split("### DEVICE:")
        
        for section in device_sections:
            if not section.strip():
                continue
                
            # Extract device name and content
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
    
    # Add last device if exists
    if current_device and current_content:
        formatted_sections.append(
            f"### DEVICE: {current_device}\n" + 
            "\n".join(current_content)
        )
    
    formatted_context = "\n\n".join(formatted_sections)
    
    # Debug print commented out
    # print("\nDEBUG: Formatted context sample:")
    # print(formatted_context[:500] + "...")
    
    # Rest of the function remains the same...
    
    # Create the prompt with the improved context
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
    # Initialize Together AI client
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "data.json")
    chroma_dir = os.path.join(script_dir, "chroma_db")
    
    # Debug print the transformed data (commented out)
    # print("\nDEBUG: Printing transformed configuration...")
    # debug_print_transformed_data(json_file)
    
    # Clean up existing vector store if it exists
    if os.path.exists(chroma_dir):
        print(f"Cleaning up existing vector store in {chroma_dir}")
        shutil.rmtree(chroma_dir)
    
    print(f"Loading and splitting document from: {json_file}")
    chunks = load_and_split_json(json_file)
    
    # In load_and_split_json function, comment out debug prints:
    """
    # Debug print to verify transformation
    print("\nDEBUG: Sample of processed text:")
    print(processed_str[:500] + "...")
    
    # Debug print first chunk
    print("\nDEBUG: First chunk sample:")
    print(chunks[0][:500] + "...")
    """
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)
    
    # In get_context_and_query function, comment out debug prints:
    """
    # Debug print
    print("\nDEBUG: Formatted context sample:")
    print(formatted_context[:500] + "...")
    """
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,           # Increased for better coverage
            "fetch_k": 15,    # Increased to fetch more relevant docs
            "lambda_mult": 0.7  # Adjusted for better relevance/diversity balance
        }
    )
    
    print("\nRAG system ready! Enter your questions (type 'quit' to exit):")
    while True:
        question = input("\n🔍 Question: ")
        if question.lower() == 'quit':
            break
            
        try:
            # Get context and create prompt
            prompt = get_context_and_query(retriever, question)
            
            # Get response from Together AI
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
                max_tokens=2048,  # Increased for more detailed responses
                temperature=0.3,  # Lower temperature for more focused answers
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