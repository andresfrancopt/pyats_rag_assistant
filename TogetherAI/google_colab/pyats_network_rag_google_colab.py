from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from together import Together
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import json
from typing import List
from google.colab import drive
import shutil

# Configure tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

def preprocess_json(data: dict) -> str:
    """Enhanced JSON preprocessing for network devices with better structure."""
    processed_text = []

    for device_name, device_data in data.items():
        # Add clear device boundary without assuming it's a router
        processed_text.append(f"\n### DEVICE: {device_name} ###")

        # Add device type if available in the data
        if "config" in device_data and "hostname" in device_data["config"]:
            hostname_entry = next((value for key, value in device_data["config"].items()
                                if "hostname" in key.lower()), {})
            if hostname_entry:
                processed_text.append(f"HOSTNAME: {next(iter(hostname_entry))}")

        # Process each section with better formatting
        for section, section_data in device_data.items():
            processed_text.append(f"\n>> SECTION: {section}")

            # Special handling for important sections
            if section in ["interface", "config", "routing", "ospf"]:
                processed_text.append("IMPORTANT CONFIGURATION SECTION")

            if isinstance(section_data, dict):
                # Process nested dictionaries with clear key paths
                def process_nested_dict(d, prefix=""):
                    for key, value in d.items():
                        if isinstance(value, dict):
                            processed_text.append(f"{prefix}{key}:")
                            process_nested_dict(value, prefix + "  ")
                        else:
                            processed_text.append(f"{prefix}{key}: {json.dumps(value, indent=2)}")

                process_nested_dict(section_data)
            else:
                processed_text.append(json.dumps(section_data, indent=2))

            processed_text.append("-" * 50)

    return "\n".join(processed_text)

def transform_json_to_flat(data: dict) -> str:
    """
    Transform network device JSON data into a flat, YAML-like format optimized for Colab.
    """
    output = []
    
    for device_name, device_data in data.items():
        output.append(f"\n### DEVICE: {device_name}")
        
        # Device Configuration Section
        output.append("=== CONFIGURATION ===")
        if "config" in device_data:
            output.append("SYSTEM:")
            for key, value in device_data["config"].items():
                if not isinstance(value, dict) or not value:
                    output.append(f"  {key}")
            
            output.append("INTERFACES:")
            for key, value in device_data["config"].items():
                if key.startswith("interface"):
                    output.append(f"  {key}:")
                    for subkey, _ in value.items():
                        output.append(f"    {subkey}")
            
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
        
        # End of device processing     
        output.append(f"### END_DEVICE: {device_name}\n")
    
    return "\n".join(output)

def load_and_split_json(file_path: str) -> List[str]:
    """
    Load JSON data and split it into manageable chunks while preserving context.
    Optimized for Google Colab environment.
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

def create_vectorstore(chunks: List[str]) -> Chroma | None:
    """Create vector store optimized for Colab environment."""
    try:
        # Use Colab-specific temporary directory
        chroma_dir = "/tmp/chroma_db"
        if os.path.exists(chroma_dir):
            shutil.rmtree(chroma_dir)
        os.makedirs(chroma_dir, exist_ok=True)
        
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
        
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

def get_context_and_query(retriever, question: str) -> str:
    """Get enhanced context and create better prompt for any network device."""
    context_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)

    prompt = f"""You are an expert network engineer providing detailed analysis of network device configurations.
    Analyze the following network data and provide a comprehensive answer.

    IMPORTANT GUIDELINES:
    1. CONTEXT ANALYSIS:
       - Carefully examine all provided network information
       - Identify specific device(s) mentioned in the context
       - Focus on relevant configuration sections
       - Consider the device type and role in the network

    2. RESPONSE STRUCTURE:
       - Start with identifying the relevant device(s)
       - Provide specific configuration details
       - Include IP addresses, interfaces, protocols where relevant
       - Use technical terminology accurately
       - Specify device roles if apparent (router, switch, etc.)

    3. FORMATTING:
       - Use clear markdown formatting
       - Structure information in bullet points
       - Highlight important technical details using code blocks
       - For configuration values, use `code` formatting

    CONTEXT:
    {context}

    QUESTION:
    {question}

    Provide your answer using this format:
    ### Device: [device_name]
    - **Device Role**: [if identifiable]
    - **Configuration Details**: [specific details]
    - **Technical Parameters**: [parameters]
    - **Additional Information**: [if relevant]

    ### In a nutshell:
    **[summary]**

    Use \`code\` formatting for:
    - IP addresses
    - Interface names
    - Configuration commands
    - Protocol parameters
    - VLAN information
    - Switching/routing specific parameters
    """

    return prompt

def main():
    """Main function for Google Colab environment."""
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Initialize Together AI client
    client = Together(api_key='a087fe63d505d9a5b487e5b5bbe97d059311249f6f285837efc1748a7e3eaaec')
    
    # Use Colab-specific path
    json_file = "/content/data.json"
    
    print(f"Loading and splitting document from: {json_file}")
    chunks = load_and_split_json(json_file)
    
    print("Creating vector store...")
    vectorstore = create_vectorstore(chunks)
    
    if vectorstore is None:
        print("Vector store creation failed. Exiting.")
        return
    
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 15,
            "lambda_mult": 0.7
        }
    )
    
    # Use the enhanced system message from RAG assistant
    system_message = """You are an expert network engineer with deep knowledge of:
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
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
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