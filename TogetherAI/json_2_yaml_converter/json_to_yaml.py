import json
import yaml
import sys
import argparse
from pathlib import Path

def convert_json_to_yaml(input_file: str, output_file: str = None) -> bool:
    """
    Convert a JSON file to YAML format.
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str, optional): Path to the output YAML file. 
                                   If not provided, will use the same name with .yaml extension
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    try:
        # Read JSON file
        with open(input_file, 'r') as json_file:
            data = json.load(json_file)
        
        # If no output file specified, create one with same name but .yaml extension
        if not output_file:
            output_file = str(Path(input_file).with_suffix('.yaml'))
        
        # Write YAML file
        with open(output_file, 'w') as yaml_file:
            yaml.dump(data, yaml_file, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Successfully converted '{input_file}' to '{output_file}'")
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ Error reading JSON file: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"❌ Error writing YAML file: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert JSON files to YAML format')
    parser.add_argument('input', help='Input JSON file path')
    parser.add_argument('-o', '--output', help='Output YAML file path (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input).exists():
        print(f"❌ Input file '{args.input}' does not exist")
        sys.exit(1)
    
    # Convert the file
    success = convert_json_to_yaml(args.input, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()