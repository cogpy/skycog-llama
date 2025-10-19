"""
Cognitive primitives for OpenCog LLM integration.
This module provides higher-level cognitive functions that enhance LLM reasoning.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Set, Any
from collections import defaultdict, Counter

from .atomspace import AtomSpace
from .atoms import (
    Atom, Node, Link, ConceptNode, PredicateNode, WordNode,
    InheritanceLink, SimilarityLink, EvaluationLink, ListLink
)


class CognitivePrimitives:
    """
    High-level cognitive functions that operate on the AtomSpace
    to enhance LLM reasoning and knowledge management.
    """
    
    # Configuration constants
    STOP_WORDS = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 
        'should', 'may', 'might', 'can', 'must'
    }
    
    # Importance calculation weights
    ATTENTION_WEIGHT = 0.4
    TRUTH_WEIGHT = 0.3
    CONNECTION_WEIGHT = 0.3
    MAX_CONNECTIONS_NORMALIZE = 10.0
    
    def __init__(self, atomspace: AtomSpace):
        """
        Initialize cognitive primitives with an AtomSpace.
        
        Args:
            atomspace: The AtomSpace to operate on
        """
        self.atomspace = atomspace
        self.concept_cache = {}
        self.similarity_cache = {}
    
    def extract_concepts_from_text(self, text: str, add_to_atomspace: bool = True) -> List[ConceptNode]:
        """
        Extract concepts from natural language text and optionally add them to AtomSpace.
        
        Args:
            text: Input text to analyze
            add_to_atomspace: Whether to add extracted concepts to AtomSpace
            
        Returns:
            List of concept nodes extracted from text
        """
        # Simple concept extraction using patterns
        # In a real implementation, this would use NLP libraries
        
        concepts = []
        
        # Extract nouns and noun phrases (simplified)
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        
        # Filter for potential concepts (words > 2 chars, not common stop words)
        concept_words = [word for word in words if len(word) > 2 and word not in self.STOP_WORDS]
        
        # Create concept nodes
        for word in set(concept_words):  # Remove duplicates
            concept = ConceptNode(word, truth_value=0.8)
            concepts.append(concept)
            
            if add_to_atomspace:
                self.atomspace.add_atom(concept)
        
        return concepts
    
    def create_word_nodes_from_text(self, text: str, add_to_atomspace: bool = True) -> List[WordNode]:
        """
        Create word nodes from text for language processing.
        
        Args:
            text: Input text
            add_to_atomspace: Whether to add to AtomSpace
            
        Returns:
            List of word nodes
        """
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        word_nodes = []
        
        for word in words:
            word_node = WordNode(word, truth_value=1.0)
            word_nodes.append(word_node)
            
            if add_to_atomspace:
                self.atomspace.add_atom(word_node)
        
        return word_nodes
    
    def create_inheritance_relationship(self, child_concept: str, parent_concept: str, 
                                      truth_value: float = 0.9) -> InheritanceLink:
        """
        Create an inheritance relationship between concepts.
        
        Args:
            child_concept: Name of the child concept
            parent_concept: Name of the parent concept
            truth_value: Strength of the relationship
            
        Returns:
            The inheritance link
        """
        # Get or create concept nodes
        child_nodes = self.atomspace.get_atoms_by_name(child_concept)
        parent_nodes = self.atomspace.get_atoms_by_name(parent_concept)
        
        if not child_nodes:
            child_node = ConceptNode(child_concept)
            self.atomspace.add_atom(child_node)
        else:
            child_node = child_nodes[0]
        
        if not parent_nodes:
            parent_node = ConceptNode(parent_concept)
            self.atomspace.add_atom(parent_node)
        else:
            parent_node = parent_nodes[0]
        
        # Create inheritance link
        inheritance_link = InheritanceLink(child_node, parent_node, truth_value=truth_value)
        self.atomspace.add_atom(inheritance_link)
        
        return inheritance_link
    
    def create_similarity_relationship(self, concept1: str, concept2: str, 
                                     similarity_score: float = 0.8) -> SimilarityLink:
        """
        Create a similarity relationship between concepts.
        
        Args:
            concept1: Name of first concept
            concept2: Name of second concept
            similarity_score: Similarity strength
            
        Returns:
            The similarity link
        """
        # Get or create concept nodes
        nodes1 = self.atomspace.get_atoms_by_name(concept1)
        nodes2 = self.atomspace.get_atoms_by_name(concept2)
        
        if not nodes1:
            node1 = ConceptNode(concept1)
            self.atomspace.add_atom(node1)
        else:
            node1 = nodes1[0]
        
        if not nodes2:
            node2 = ConceptNode(concept2)
            self.atomspace.add_atom(node2)
        else:
            node2 = nodes2[0]
        
        # Create similarity link
        similarity_link = SimilarityLink(node1, node2, truth_value=similarity_score)
        self.atomspace.add_atom(similarity_link)
        
        return similarity_link
    
    def find_related_concepts(self, concept_name: str, max_depth: int = 2, 
                            min_truth_value: float = 0.5) -> Dict[str, List[Atom]]:
        """
        Find concepts related to the given concept through various relationship types.
        
        Args:
            concept_name: Name of the concept to find relations for
            max_depth: Maximum traversal depth
            min_truth_value: Minimum truth value threshold
            
        Returns:
            Dictionary mapping relationship types to lists of related atoms
        """
        concepts = self.atomspace.get_atoms_by_name(concept_name)
        if not concepts:
            return {}
        
        concept = concepts[0]
        related = defaultdict(list)
        
        # Find direct relationships
        neighbors = self.atomspace.get_neighbors(concept, depth=max_depth)
        
        for neighbor in neighbors:
            if isinstance(neighbor, Link) and neighbor.truth_value >= min_truth_value:
                link_type = neighbor.get_type()
                
                # Find the other atoms in the link
                for atom in neighbor.get_outgoing():
                    if atom != concept:
                        related[link_type].append(atom)
        
        return dict(related)
    
    def calculate_concept_importance(self, concept_name: str) -> float:
        """
        Calculate the importance of a concept based on its connections and usage.
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            Importance score (0.0 to 1.0)
        """
        concepts = self.atomspace.get_atoms_by_name(concept_name)
        if not concepts:
            return 0.0
        
        concept = concepts[0]
        
        # Factors for importance calculation
        attention_score = concept.attention_value
        truth_score = concept.truth_value
        connection_count = len(concept.incoming_set)
        
        # Weighted importance score
        importance = (attention_score * self.ATTENTION_WEIGHT + 
                     truth_score * self.TRUTH_WEIGHT + 
                     min(connection_count / self.MAX_CONNECTIONS_NORMALIZE, 1.0) * self.CONNECTION_WEIGHT)
        
        return min(importance, 1.0)
    
    def enhance_response_with_knowledge(self, query: str, response: str) -> Dict[str, Any]:
        """
        Enhance an LLM response with knowledge from the AtomSpace.
        
        Args:
            query: The original query
            response: The LLM's response
            
        Returns:
            Enhanced response with additional context and knowledge
        """
        # Extract concepts from query and response
        query_concepts = self.extract_concepts_from_text(query, add_to_atomspace=False)
        response_concepts = self.extract_concepts_from_text(response, add_to_atomspace=False)
        
        enhancement = {
            'original_response': response,
            'query_concepts': [c.name for c in query_concepts],
            'response_concepts': [c.name for c in response_concepts],
            'related_knowledge': [],
            'confidence_score': 0.0
        }
        
        # Find related knowledge for key concepts
        all_concepts = set([c.name for c in query_concepts + response_concepts])
        
        for concept_name in all_concepts:
            related = self.find_related_concepts(concept_name, max_depth=1)
            if related:
                enhancement['related_knowledge'].append({
                    'concept': concept_name,
                    'relations': {k: [atom.name for atom in v] for k, v in related.items()}
                })
        
        # Calculate confidence based on knowledge availability
        concept_importance_scores = [self.calculate_concept_importance(c) for c in all_concepts]
        if concept_importance_scores:
            enhancement['confidence_score'] = sum(concept_importance_scores) / len(concept_importance_scores)
        
        return enhancement
    
    def learn_from_interaction(self, query: str, response: str, feedback_score: float = 0.8) -> None:
        """
        Learn from a query-response interaction by updating the AtomSpace.
        
        Args:
            query: The user query
            response: The system response
            feedback_score: Quality score for the interaction (0.0 to 1.0)
        """
        # Extract and add concepts
        query_concepts = self.extract_concepts_from_text(query, add_to_atomspace=True)
        response_concepts = self.extract_concepts_from_text(response, add_to_atomspace=True)
        
        # Create relationships between query and response concepts
        for q_concept in query_concepts:
            for r_concept in response_concepts:
                if q_concept.name != r_concept.name:
                    # Create a weak similarity link
                    similarity = SimilarityLink(
                        q_concept, r_concept, 
                        truth_value=feedback_score * 0.6
                    )
                    self.atomspace.add_atom(similarity)
        
        # Update attention values based on usage
        for concept in query_concepts + response_concepts:
            existing = self.atomspace.get_atoms_by_name(concept.name)
            if existing:
                existing[0].attention_value = min(existing[0].attention_value + 0.1, 1.0)
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the knowledge stored in the AtomSpace.
        
        Returns:
            Dictionary with knowledge statistics and insights
        """
        stats = self.atomspace.get_statistics()
        
        # Get most important concepts
        concept_nodes = self.atomspace.get_atoms_by_type("ConceptNode")
        important_concepts = sorted(
            concept_nodes, 
            key=lambda c: self.calculate_concept_importance(c.name),
            reverse=True
        )[:10]
        
        # Get most connected concepts
        most_connected = sorted(
            concept_nodes,
            key=lambda c: len(c.incoming_set),
            reverse=True
        )[:10]
        
        return {
            'atomspace_stats': stats,
            'top_concepts_by_importance': [
                {
                    'name': c.name,
                    'importance': self.calculate_concept_importance(c.name),
                    'connections': len(c.incoming_set)
                }
                for c in important_concepts
            ],
            'most_connected_concepts': [
                {
                    'name': c.name,
                    'connections': len(c.incoming_set),
                    'truth_value': c.truth_value
                }
                for c in most_connected
            ]
        }