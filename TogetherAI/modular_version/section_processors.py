from typing import Any, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class SectionMetadata:
    name: str
    priority: int
    section_type: str = "operational"

class SectionProcessor(ABC):
    @abstractmethod
    def can_process(self, key: str, data: Any) -> bool:
        pass

    @abstractmethod
    def process(self, key: str, data: Any) -> List[str]:
        pass

class ConfigProcessor(SectionProcessor):
    def can_process(self, key: str, data: Any) -> bool:
        return key == "config"

    def process(self, key: str, data: Any) -> List[str]:
        output = ["=== CONFIGURATION ==="]
        if isinstance(data, dict):
            for cmd, value in data.items():
                if isinstance(value, dict):
                    output.append(f"  {cmd}:")
                    for subcmd, subvalue in value.items():
                        output.append(f"    {subcmd}")
                else:
                    output.append(f"  {cmd}")
        return output

class InterfaceProcessor(SectionProcessor):
    def can_process(self, key: str, data: Any) -> bool:
        return key == "interface"

    def process(self, key: str, data: Any) -> List[str]:
        output = ["=== INTERFACE INFORMATION ==="]
        if isinstance(data, dict) and "info" in data:
            for intf, intf_data in data["info"].items():
                output.append(f"  {intf}:")
                if "enabled" in intf_data:
                    output.append(f"    Status: {'enabled' if intf_data['enabled'] else 'disabled'}")
                if "oper_status" in intf_data:
                    output.append(f"    Operational: {intf_data['oper_status']}")
                if "ipv4" in intf_data:
                    for ip_data in intf_data["ipv4"].values():
                        output.append(f"    IPv4: {ip_data['ip']}/{ip_data['prefix_length']}")
        return output

class RoutingProcessor(SectionProcessor):
    def can_process(self, key: str, data: Any) -> bool:
        return key in ["ospf", "bgp", "static_routing", "routing"]

    def process(self, key: str, data: Any) -> List[str]:
        output = [f"=== {key.upper()} INFORMATION ==="]
        if isinstance(data, dict):
            if 'info' in data:
                self._process_routing_info(data['info'], output, "  ")
            elif 'vrf' in data:
                self._process_routing_info({'vrf': data['vrf']}, output, "  ")
        return output

    def _process_routing_info(self, info: dict, output: list, indent: str):
        for key, value in info.items():
            if isinstance(value, dict):
                output.append(f"{indent}{key}:")
                self._process_routing_info(value, output, indent + "  ")
            else:
                output.append(f"{indent}{key}: {value}")

class SecurityProcessor(SectionProcessor):
    def can_process(self, key: str, data: Any) -> bool:
        return key in ["aaa", "acl", "authentication"]

    def process(self, key: str, data: Any) -> List[str]:
        output = ["=== SECURITY CONFIGURATION ==="]
        if isinstance(data, dict):
            for setting, value in data.items():
                output.append(f"  {setting}: {value}")
        return output

class NetworkProcessor(SectionProcessor):
    def can_process(self, key: str, data: Any) -> bool:
        return key in ["arp", "lldp", "stp", "vlan", "dot1x"]

    def process(self, key: str, data: Any) -> List[str]:
        output = [f"=== {key.upper()} INFORMATION ==="]
        if isinstance(data, dict) and 'info' in data:
            self._process_info(data['info'], output, "  ")
        return output

    def _process_info(self, info: dict, output: list, indent: str):
        for key, value in info.items():
            if isinstance(value, dict):
                output.append(f"{indent}{key}:")
                self._process_info(value, output, indent + "  ")
            else:
                output.append(f"{indent}{key}: {value}")

__all__ = [
    'SectionMetadata',
    'SectionProcessor',
    'ConfigProcessor',
    'InterfaceProcessor',
    'RoutingProcessor',
    'SecurityProcessor',
    'NetworkProcessor'
]