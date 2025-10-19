"""
Basic Atom classes for OpenCog integration.
Implements the fundamental building blocks of OpenCog's knowledge representation.
"""

import uuid
from typing import Any, List, Optional, Set, Dict
from abc import ABC, abstractmethod


class Atom(ABC):
    """
    Base class for all OpenCog atoms.
    Atoms are the fundamental units of knowledge representation in OpenCog.
    """
    
    def __init__(self, name: str, truth_value: float = 1.0, attention_value: float = 0.0):
        """
        Initialize an Atom.
        
        Args:
            name: The name/label of the atom
            truth_value: Truth value between 0.0 and 1.0
            attention_value: Attention value for ECAN (Economic Attention Network)
        """
        self.uuid = str(uuid.uuid4())
        self.name = name
        self.truth_value = max(0.0, min(1.0, truth_value))
        self.attention_value = max(0.0, attention_value)
        self.incoming_set: Set[str] = set()  # UUIDs of atoms that reference this atom
        self.creation_time = None
        self.last_access_time = None
        
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.name})"
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', tv={self.truth_value}, av={self.attention_value})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Atom):
            return False
        return self.uuid == other.uuid
    
    def __hash__(self) -> int:
        return hash(self.uuid)
    
    @abstractmethod
    def get_type(self) -> str:
        """Return the type of this atom."""
        pass
    
    def add_incoming(self, atom_uuid: str) -> None:
        """Add an atom to the incoming set."""
        self.incoming_set.add(atom_uuid)
    
    def remove_incoming(self, atom_uuid: str) -> None:
        """Remove an atom from the incoming set."""
        self.incoming_set.discard(atom_uuid)


class Node(Atom):
    """
    Node atoms represent concepts, entities, or individual things.
    They are terminal nodes in the hypergraph with no outgoing connections.
    """
    
    def __init__(self, name: str, node_type: str = "ConceptNode", **kwargs):
        """
        Initialize a Node.
        
        Args:
            name: The name of the node
            node_type: The specific type of node (e.g., ConceptNode, PredicateNode)
            **kwargs: Additional arguments passed to Atom
        """
        super().__init__(name, **kwargs)
        self.node_type = node_type
        
    def get_type(self) -> str:
        return self.node_type


class Link(Atom):
    """
    Link atoms represent relationships between other atoms.
    They can connect nodes and other links in the hypergraph.
    """
    
    def __init__(self, name: str, outgoing: List[Atom], link_type: str = "InheritanceLink", **kwargs):
        """
        Initialize a Link.
        
        Args:
            name: The name of the link
            outgoing: List of atoms this link connects
            link_type: The specific type of link (e.g., InheritanceLink, SimilarityLink)
            **kwargs: Additional arguments passed to Atom
        """
        super().__init__(name, **kwargs)
        self.link_type = link_type
        self.outgoing = outgoing
        
        # Add this link to the incoming set of all outgoing atoms
        for atom in self.outgoing:
            atom.add_incoming(self.uuid)
    
    def get_type(self) -> str:
        return self.link_type
    
    def get_outgoing(self) -> List[Atom]:
        """Get the atoms this link connects."""
        return self.outgoing
    
    def get_arity(self) -> int:
        """Get the number of atoms this link connects."""
        return len(self.outgoing)


# Common node types
class ConceptNode(Node):
    """Node representing a concept or category."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "ConceptNode", **kwargs)


class PredicateNode(Node):
    """Node representing a predicate or property."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "PredicateNode", **kwargs)


class WordNode(Node):
    """Node representing a word in natural language."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "WordNode", **kwargs)


# Common link types
class InheritanceLink(Link):
    """Link representing an inheritance relationship (A is-a B)."""
    
    def __init__(self, child: Atom, parent: Atom, **kwargs):
        name = f"{child.name}_inherits_{parent.name}"
        super().__init__(name, [child, parent], "InheritanceLink", **kwargs)


class SimilarityLink(Link):
    """Link representing similarity between atoms."""
    
    def __init__(self, atom1: Atom, atom2: Atom, **kwargs):
        name = f"{atom1.name}_similar_{atom2.name}"
        super().__init__(name, [atom1, atom2], "SimilarityLink", **kwargs)


class EvaluationLink(Link):
    """Link representing the evaluation of a predicate on arguments."""
    
    def __init__(self, predicate: Atom, arguments: List[Atom], **kwargs):
        name = f"eval_{predicate.name}"
        super().__init__(name, [predicate] + arguments, "EvaluationLink", **kwargs)


class ListLink(Link):
    """Link representing an ordered list of atoms."""
    
    def __init__(self, atoms: List[Atom], **kwargs):
        name = f"list_of_{len(atoms)}"
        super().__init__(name, atoms, "ListLink", **kwargs)