from typing import Dict, Any, List
import yaml
from section_processors import SectionProcessor, SectionMetadata

class DynamicConfigTransformer:
    """Dynamically transform network configuration data"""
    
    def __init__(self, config_path: str = None):
        self.processors: Dict[str, SectionProcessor] = {}
        self.section_metadata: Dict[str, SectionMetadata] = {}
        self._load_config(config_path)
        self._register_default_processors()

    def _load_config(self, config_path: str = None):
        """Load section configuration from YAML file if provided"""
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                for section, metadata in config.get('sections', {}).items():
                    self.section_metadata[section] = SectionMetadata(
                        name=section,
                        priority=metadata.get('priority', 50),
                        section_type=metadata.get('type', 'operational')
                    )

    def _register_default_processors(self):
        """Register built-in section processors"""
        from section_processors import (
            InterfaceProcessor,
            ConfigProcessor,
            RoutingProcessor,
            SecurityProcessor,
            NetworkProcessor
        )
        
        default_processors = [
            InterfaceProcessor(),
            ConfigProcessor(),
            RoutingProcessor(),
            SecurityProcessor(),
            NetworkProcessor()
        ]
        
        for processor in default_processors:
            self.processors[processor.__class__.__name__] = processor

    def transform(self, data: Dict[str, Any]) -> str:
        """Transform configuration data into flattened format"""
        output = []
        
        for device_name, device_data in data.items():
            output.append(f"\n### DEVICE: {device_name}")
            
            # Process all available sections in device_data
            for section_name, section_data in device_data.items():
                # Skip empty or null sections
                if not section_data or section_data.get('raw_data') is False and not section_data.get('info'):
                    continue
                
                # Get processor or use generic processor
                processor = self._get_processor(section_name)
                if processor:
                    section_output = processor.process(section_name, section_data)
                else:
                    # Generic processing for unknown sections
                    section_output = self._process_generic_section(section_name, section_data)
                
                if section_output:
                    output.extend(section_output)
            
            output.append(f"### END_DEVICE: {device_name}\n")
        
        return "\n".join(output)

    def _process_generic_section(self, section_name: str, section_data: Any) -> List[str]:
        """Process sections without specific processors"""
        output = [f"=== {section_name.upper()} ==="]
        
        # Handle info subsection if present
        if isinstance(section_data, dict):
            if 'info' in section_data:
                self._format_dict(section_data['info'], output, indent="  ")
            else:
                self._format_dict(section_data, output, indent="  ")
                
        return output

    def _format_dict(self, data: Dict, output: List[str], indent: str = "") -> None:
        """Recursively format dictionary data"""
        if not isinstance(data, dict):
            return
            
        for key, value in data.items():
            if isinstance(value, dict):
                output.append(f"{indent}{key}:")
                self._format_dict(value, output, indent + "  ")
            else:
                output.append(f"{indent}{key}: {value}")

    def _get_sections_by_type(self, data: Dict[str, Any], section_type: str) -> List[tuple]:
        """Get sections of specified type sorted by priority"""
        sections = []
        for key, value in data.items():
            metadata = self.section_metadata.get(key, SectionMetadata(key, 50))
            if metadata.section_type == section_type:
                sections.append((key, value))
        return sorted(sections, key=lambda x: self.section_metadata.get(x[0], SectionMetadata(x[0], 50)).priority)

    def _get_processor(self, section_name: str) -> SectionProcessor:
        """Get appropriate processor for section"""
        for processor in self.processors.values():
            if processor.can_process(section_name, None):
                return processor
        return None