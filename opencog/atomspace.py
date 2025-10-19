"""
AtomSpace implementation for OpenCog integration.
The AtomSpace is a hypergraph database that stores and manages atoms and their relationships.
"""

import time
from typing import Dict, List, Set, Optional, Iterator, Tuple, Any
from collections import defaultdict
import json

from .atoms import Atom, Node, Link


class AtomSpace:
    """
    AtomSpace is the central hypergraph database for storing OpenCog knowledge.
    It manages atoms, their relationships, and provides query capabilities.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize an AtomSpace.
        
        Args:
            name: Name of this AtomSpace instance
        """
        self.name = name
        self.atoms: Dict[str, Atom] = {}  # UUID -> Atom mapping
        self.atom_index: Dict[str, Set[str]] = defaultdict(set)  # Type -> Set of UUIDs
        self.name_index: Dict[str, Set[str]] = defaultdict(set)  # Name -> Set of UUIDs
        self.creation_time = time.time()
        self.modification_count = 0
        
        # Statistics
        self.stats = {
            'total_atoms': 0,
            'nodes': 0,
            'links': 0,
            'queries': 0,
            'insertions': 0,
            'deletions': 0
        }
    
    def add_atom(self, atom: Atom) -> Atom:
        """
        Add an atom to the AtomSpace.
        
        Args:
            atom: The atom to add
            
        Returns:
            The added atom (or existing atom if duplicate)
        """
        # Check if atom already exists
        existing = self._find_identical_atom(atom)
        if existing:
            # Merge truth values if needed
            if atom.truth_value > existing.truth_value:
                existing.truth_value = atom.truth_value
            return existing
        
        # Add new atom
        atom.creation_time = time.time()
        atom.last_access_time = time.time()
        
        self.atoms[atom.uuid] = atom
        self.atom_index[atom.get_type()].add(atom.uuid)
        self.name_index[atom.name].add(atom.uuid)
        
        # Update statistics
        self.stats['total_atoms'] += 1
        self.stats['insertions'] += 1
        if isinstance(atom, Node):
            self.stats['nodes'] += 1
        elif isinstance(atom, Link):
            self.stats['links'] += 1
        
        self.modification_count += 1
        return atom
    
    def remove_atom(self, atom: Atom) -> bool:
        """
        Remove an atom from the AtomSpace.
        
        Args:
            atom: The atom to remove
            
        Returns:
            True if atom was removed, False if not found
        """
        if atom.uuid not in self.atoms:
            return False
        
        # Remove from indices
        self.atom_index[atom.get_type()].discard(atom.uuid)
        self.name_index[atom.name].discard(atom.uuid)
        
        # Remove from atoms
        del self.atoms[atom.uuid]
        
        # Update statistics
        self.stats['total_atoms'] -= 1
        self.stats['deletions'] += 1
        if isinstance(atom, Node):
            self.stats['nodes'] -= 1
        elif isinstance(atom, Link):
            self.stats['links'] -= 1
        
        self.modification_count += 1
        return True
    
    def get_atom(self, uuid: str) -> Optional[Atom]:
        """
        Get an atom by its UUID.
        
        Args:
            uuid: The UUID of the atom
            
        Returns:
            The atom if found, None otherwise
        """
        atom = self.atoms.get(uuid)
        if atom:
            atom.last_access_time = time.time()
        return atom
    
    def get_atoms_by_type(self, atom_type: str) -> List[Atom]:
        """
        Get all atoms of a specific type.
        
        Args:
            atom_type: The type of atoms to retrieve
            
        Returns:
            List of atoms of the specified type
        """
        self.stats['queries'] += 1
        uuids = self.atom_index[atom_type]
        return [self.atoms[uuid] for uuid in uuids if uuid in self.atoms]
    
    def get_atoms_by_name(self, name: str) -> List[Atom]:
        """
        Get all atoms with a specific name.
        
        Args:
            name: The name to search for
            
        Returns:
            List of atoms with the specified name
        """
        self.stats['queries'] += 1
        uuids = self.name_index[name]
        return [self.atoms[uuid] for uuid in uuids if uuid in self.atoms]
    
    def query_atoms(self, 
                   atom_type: Optional[str] = None,
                   name: Optional[str] = None,
                   min_truth_value: Optional[float] = None,
                   min_attention_value: Optional[float] = None) -> List[Atom]:
        """
        Query atoms based on various criteria.
        
        Args:
            atom_type: Filter by atom type
            name: Filter by atom name
            min_truth_value: Minimum truth value
            min_attention_value: Minimum attention value
            
        Returns:
            List of atoms matching the criteria
        """
        self.stats['queries'] += 1
        
        # Start with all atoms or filter by type/name
        if atom_type:
            candidates = self.get_atoms_by_type(atom_type)
        elif name:
            candidates = self.get_atoms_by_name(name)
        else:
            candidates = list(self.atoms.values())
        
        # Apply additional filters
        result = []
        for atom in candidates:
            if min_truth_value and atom.truth_value < min_truth_value:
                continue
            if min_attention_value and atom.attention_value < min_attention_value:
                continue
            result.append(atom)
        
        return result
    
    def get_incoming_set(self, atom: Atom) -> List[Atom]:
        """
        Get all atoms that reference the given atom.
        
        Args:
            atom: The atom to find incoming references for
            
        Returns:
            List of atoms that reference the given atom
        """
        incoming = []
        for uuid in atom.incoming_set:
            if uuid in self.atoms:
                incoming.append(self.atoms[uuid])
        return incoming
    
    def find_links_containing(self, atom: Atom) -> List[Link]:
        """
        Find all links that contain the given atom in their outgoing set.
        
        Args:
            atom: The atom to search for
            
        Returns:
            List of links containing the atom
        """
        links = []
        for link_atom in self.atoms.values():
            if isinstance(link_atom, Link):
                if atom in link_atom.get_outgoing():
                    links.append(link_atom)
        return links
    
    def get_neighbors(self, atom: Atom, depth: int = 1) -> Set[Atom]:
        """
        Get all atoms connected to the given atom within specified depth.
        
        Args:
            atom: The starting atom
            depth: Maximum depth to traverse
            
        Returns:
            Set of connected atoms
        """
        if depth <= 0:
            return set()
        
        neighbors = set()
        
        # Get directly connected atoms
        # From incoming set
        for uuid in atom.incoming_set:
            if uuid in self.atoms:
                neighbor = self.atoms[uuid]
                neighbors.add(neighbor)
                if depth > 1:
                    neighbors.update(self.get_neighbors(neighbor, depth - 1))
        
        # From outgoing set (if this is a link)
        if isinstance(atom, Link):
            for outgoing_atom in atom.get_outgoing():
                neighbors.add(outgoing_atom)
                if depth > 1:
                    neighbors.update(self.get_neighbors(outgoing_atom, depth - 1))
        
        return neighbors
    
    def clear(self) -> None:
        """Clear all atoms from the AtomSpace."""
        self.atoms.clear()
        self.atom_index.clear()
        self.name_index.clear()
        self.stats = {
            'total_atoms': 0,
            'nodes': 0,
            'links': 0,
            'queries': 0,
            'insertions': 0,
            'deletions': 0
        }
        self.modification_count += 1
    
    def get_size(self) -> int:
        """Get the total number of atoms in the AtomSpace."""
        return len(self.atoms)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the AtomSpace."""
        return {
            'name': self.name,
            'size': self.get_size(),
            'creation_time': self.creation_time,
            'modification_count': self.modification_count,
            'stats': self.stats.copy()
        }
    
    def _find_identical_atom(self, atom: Atom) -> Optional[Atom]:
        """
        Find an identical atom in the AtomSpace.
        
        Args:
            atom: The atom to find
            
        Returns:
            Identical atom if found, None otherwise
        """
        candidates = self.get_atoms_by_name(atom.name)
        for candidate in candidates:
            if (candidate.get_type() == atom.get_type() and
                isinstance(candidate, type(atom))):
                # For links, also check outgoing atoms
                if isinstance(atom, Link) and isinstance(candidate, Link):
                    if len(atom.outgoing) == len(candidate.outgoing):
                        if all(a.uuid == b.uuid for a, b in zip(atom.outgoing, candidate.outgoing)):
                            return candidate
                else:
                    return candidate
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AtomSpace to dictionary representation."""
        atoms_data = []
        for atom in self.atoms.values():
            atom_data = {
                'uuid': atom.uuid,
                'name': atom.name,
                'type': atom.get_type(),
                'truth_value': atom.truth_value,
                'attention_value': atom.attention_value,
                'creation_time': atom.creation_time,
                'last_access_time': atom.last_access_time
            }
            
            if isinstance(atom, Link):
                atom_data['outgoing'] = [a.uuid for a in atom.get_outgoing()]
            
            atoms_data.append(atom_data)
        
        return {
            'name': self.name,
            'creation_time': self.creation_time,
            'modification_count': self.modification_count,
            'atoms': atoms_data,
            'statistics': self.get_statistics()
        }
    
    def __len__(self) -> int:
        return len(self.atoms)
    
    def __iter__(self) -> Iterator[Atom]:
        return iter(self.atoms.values())
    
    def __contains__(self, atom: Atom) -> bool:
        return atom.uuid in self.atoms
    
    def __str__(self) -> str:
        return f"AtomSpace(name='{self.name}', size={len(self.atoms)})"
    
    def __repr__(self) -> str:
        return f"AtomSpace(name='{self.name}', atoms={len(self.atoms)}, nodes={self.stats['nodes']}, links={self.stats['links']})"