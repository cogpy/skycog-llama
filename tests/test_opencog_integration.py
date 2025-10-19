#!/usr/bin/env python3
"""
Basic tests for OpenCog integration with SkyPilot LLaMA chatbot.
These tests validate the core OpenCog functionality without requiring the full LLaMA model.
"""

import sys
import os
import unittest
import tempfile
from pathlib import Path

# Add the parent directory to sys.path to import opencog modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from opencog import AtomSpace, Atom, Node, Link, ConceptNode, PredicateNode, WordNode
from opencog import InheritanceLink, SimilarityLink, EvaluationLink, CognitivePrimitives


class TestAtomSpace(unittest.TestCase):
    """Test AtomSpace functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atomspace = AtomSpace("test_space")
    
    def test_atomspace_creation(self):
        """Test AtomSpace creation and basic properties."""
        self.assertEqual(self.atomspace.name, "test_space")
        self.assertEqual(len(self.atomspace), 0)
        self.assertIsNotNone(self.atomspace.creation_time)
    
    def test_add_node(self):
        """Test adding nodes to AtomSpace."""
        concept = ConceptNode("test_concept")
        added_concept = self.atomspace.add_atom(concept)
        
        self.assertEqual(len(self.atomspace), 1)
        self.assertIn(added_concept, self.atomspace)
        self.assertEqual(added_concept.name, "test_concept")
        self.assertEqual(added_concept.get_type(), "ConceptNode")
    
    def test_add_link(self):
        """Test adding links to AtomSpace."""
        parent = ConceptNode("parent")
        child = ConceptNode("child")
        
        self.atomspace.add_atom(parent)
        self.atomspace.add_atom(child)
        
        inheritance = InheritanceLink(child, parent)
        added_link = self.atomspace.add_atom(inheritance)
        
        self.assertEqual(len(self.atomspace), 3)  # parent, child, inheritance
        self.assertIn(added_link, self.atomspace)
        self.assertEqual(added_link.get_type(), "InheritanceLink")
        self.assertEqual(len(added_link.get_outgoing()), 2)
    
    def test_query_by_type(self):
        """Test querying atoms by type."""
        concept1 = ConceptNode("concept1")
        concept2 = ConceptNode("concept2")
        predicate = PredicateNode("predicate1")
        
        self.atomspace.add_atom(concept1)
        self.atomspace.add_atom(concept2)
        self.atomspace.add_atom(predicate)
        
        concepts = self.atomspace.get_atoms_by_type("ConceptNode")
        predicates = self.atomspace.get_atoms_by_type("PredicateNode")
        
        self.assertEqual(len(concepts), 2)
        self.assertEqual(len(predicates), 1)
        self.assertIn(concept1, concepts)
        self.assertIn(concept2, concepts)
        self.assertIn(predicate, predicates)
    
    def test_query_by_name(self):
        """Test querying atoms by name."""
        concept = ConceptNode("test_name")
        word = WordNode("test_name")
        
        self.atomspace.add_atom(concept)
        self.atomspace.add_atom(word)
        
        atoms_with_name = self.atomspace.get_atoms_by_name("test_name")
        
        self.assertEqual(len(atoms_with_name), 2)
        self.assertIn(concept, atoms_with_name)
        self.assertIn(word, atoms_with_name)
    
    def test_remove_atom(self):
        """Test removing atoms from AtomSpace."""
        concept = ConceptNode("to_remove")
        added_concept = self.atomspace.add_atom(concept)
        
        self.assertEqual(len(self.atomspace), 1)
        
        removed = self.atomspace.remove_atom(added_concept)
        
        self.assertTrue(removed)
        self.assertEqual(len(self.atomspace), 0)
        self.assertNotIn(added_concept, self.atomspace)
    
    def test_statistics(self):
        """Test AtomSpace statistics."""
        concept = ConceptNode("concept")
        predicate = PredicateNode("predicate")
        inheritance = InheritanceLink(concept, predicate)
        
        self.atomspace.add_atom(concept)
        self.atomspace.add_atom(predicate)
        self.atomspace.add_atom(inheritance)
        
        stats = self.atomspace.get_statistics()
        
        self.assertEqual(stats['size'], 3)
        self.assertEqual(stats['stats']['nodes'], 2)
        self.assertEqual(stats['stats']['links'], 1)
        self.assertEqual(stats['stats']['total_atoms'], 3)


class TestAtoms(unittest.TestCase):
    """Test Atom, Node, and Link functionality."""
    
    def test_node_creation(self):
        """Test creating different types of nodes."""
        concept = ConceptNode("test_concept")
        predicate = PredicateNode("test_predicate")
        word = WordNode("test_word")
        
        self.assertEqual(concept.name, "test_concept")
        self.assertEqual(concept.get_type(), "ConceptNode")
        self.assertEqual(concept.truth_value, 1.0)
        
        self.assertEqual(predicate.get_type(), "PredicateNode")
        self.assertEqual(word.get_type(), "WordNode")
    
    def test_link_creation(self):
        """Test creating different types of links."""
        concept1 = ConceptNode("concept1")
        concept2 = ConceptNode("concept2")
        
        inheritance = InheritanceLink(concept1, concept2)
        similarity = SimilarityLink(concept1, concept2)
        
        self.assertEqual(inheritance.get_type(), "InheritanceLink")
        self.assertEqual(inheritance.get_arity(), 2)
        self.assertIn(concept1, inheritance.get_outgoing())
        self.assertIn(concept2, inheritance.get_outgoing())
        
        self.assertEqual(similarity.get_type(), "SimilarityLink")
        self.assertEqual(similarity.get_arity(), 2)
    
    def test_atom_equality(self):
        """Test atom equality and hashing."""
        concept1 = ConceptNode("same_name")
        concept2 = ConceptNode("same_name")
        
        # Atoms with same content should not be equal (different UUIDs)
        self.assertNotEqual(concept1, concept2)
        self.assertNotEqual(hash(concept1), hash(concept2))
        
        # Atom should be equal to itself
        self.assertEqual(concept1, concept1)
        self.assertEqual(hash(concept1), hash(concept1))
    
    def test_truth_values(self):
        """Test truth value handling."""
        concept = ConceptNode("test", truth_value=0.7)
        self.assertEqual(concept.truth_value, 0.7)
        
        # Test bounds
        concept_high = ConceptNode("high", truth_value=1.5)
        concept_low = ConceptNode("low", truth_value=-0.5)
        
        self.assertEqual(concept_high.truth_value, 1.0)
        self.assertEqual(concept_low.truth_value, 0.0)
    
    def test_incoming_set(self):
        """Test incoming set management."""
        concept1 = ConceptNode("concept1")
        concept2 = ConceptNode("concept2")
        
        inheritance = InheritanceLink(concept1, concept2)
        
        # Check that the link is added to incoming sets
        self.assertIn(inheritance.uuid, concept1.incoming_set)
        self.assertIn(inheritance.uuid, concept2.incoming_set)


class TestCognitivePrimitives(unittest.TestCase):
    """Test cognitive primitives functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.atomspace = AtomSpace("test_cognitive_space")
        self.cognitive = CognitivePrimitives(self.atomspace)
    
    def test_concept_extraction(self):
        """Test concept extraction from text."""
        text = "Artificial intelligence and machine learning are important technologies."
        concepts = self.cognitive.extract_concepts_from_text(text, add_to_atomspace=True)
        
        # Should extract meaningful concepts
        concept_names = [c.name for c in concepts]
        self.assertIn("artificial", concept_names)
        self.assertIn("intelligence", concept_names)
        self.assertIn("machine", concept_names)
        self.assertIn("learning", concept_names)
        
        # Should be added to atomspace
        self.assertGreater(len(self.atomspace), 0)
    
    def test_word_node_creation(self):
        """Test word node creation from text."""
        text = "Hello world"
        word_nodes = self.cognitive.create_word_nodes_from_text(text, add_to_atomspace=True)
        
        self.assertEqual(len(word_nodes), 2)
        word_names = [w.name for w in word_nodes]
        self.assertIn("hello", word_names)
        self.assertIn("world", word_names)
        
        # Check they were added to atomspace
        hello_nodes = self.atomspace.get_atoms_by_name("hello")
        self.assertEqual(len(hello_nodes), 1)
        self.assertEqual(hello_nodes[0].get_type(), "WordNode")
    
    def test_inheritance_relationship(self):
        """Test creating inheritance relationships."""
        inheritance = self.cognitive.create_inheritance_relationship(
            "dog", "animal", truth_value=0.9
        )
        
        self.assertEqual(inheritance.get_type(), "InheritanceLink")
        self.assertEqual(inheritance.truth_value, 0.9)
        
        # Check that concepts were created
        dog_concepts = self.atomspace.get_atoms_by_name("dog")
        animal_concepts = self.atomspace.get_atoms_by_name("animal")
        
        self.assertEqual(len(dog_concepts), 1)
        self.assertEqual(len(animal_concepts), 1)
        self.assertEqual(dog_concepts[0].get_type(), "ConceptNode")
        self.assertEqual(animal_concepts[0].get_type(), "ConceptNode")
    
    def test_similarity_relationship(self):
        """Test creating similarity relationships."""
        similarity = self.cognitive.create_similarity_relationship(
            "cat", "dog", similarity_score=0.7
        )
        
        self.assertEqual(similarity.get_type(), "SimilarityLink")
        self.assertEqual(similarity.truth_value, 0.7)
        
        # Check that both concepts exist
        cat_concepts = self.atomspace.get_atoms_by_name("cat")
        dog_concepts = self.atomspace.get_atoms_by_name("dog")
        
        self.assertEqual(len(cat_concepts), 1)
        self.assertEqual(len(dog_concepts), 1)
    
    def test_find_related_concepts(self):
        """Test finding related concepts."""
        # Create some relationships
        self.cognitive.create_inheritance_relationship("dog", "animal")
        self.cognitive.create_inheritance_relationship("cat", "animal")
        self.cognitive.create_similarity_relationship("dog", "cat")
        
        # Find related concepts for "dog"
        related = self.cognitive.find_related_concepts("dog")
        
        self.assertIsInstance(related, dict)
        # Should find inheritance and similarity relationships
        self.assertIn("InheritanceLink", related.keys())
        self.assertIn("SimilarityLink", related.keys())
    
    def test_concept_importance(self):
        """Test concept importance calculation."""
        # Create a concept with some relationships
        self.cognitive.create_inheritance_relationship("important_concept", "category")
        self.cognitive.create_similarity_relationship("important_concept", "related_concept")
        
        importance = self.cognitive.calculate_concept_importance("important_concept")
        
        self.assertGreater(importance, 0.0)
        self.assertLessEqual(importance, 1.0)
        
        # Non-existent concept should have 0 importance
        no_importance = self.cognitive.calculate_concept_importance("non_existent")
        self.assertEqual(no_importance, 0.0)
    
    def test_enhance_response(self):
        """Test response enhancement with knowledge."""
        # Add some knowledge first
        self.cognitive.create_inheritance_relationship("python", "programming_language")
        self.cognitive.create_inheritance_relationship("java", "programming_language")
        
        query = "What is Python?"
        response = "Python is a programming language."
        
        enhancement = self.cognitive.enhance_response_with_knowledge(query, response)
        
        self.assertIn('original_response', enhancement)
        self.assertIn('query_concepts', enhancement)
        self.assertIn('response_concepts', enhancement)
        self.assertIn('confidence_score', enhancement)
        
        self.assertEqual(enhancement['original_response'], response)
        self.assertIsInstance(enhancement['confidence_score'], float)
    
    def test_learning_from_interaction(self):
        """Test learning from interactions."""
        initial_size = len(self.atomspace)
        
        query = "What is machine learning?"
        response = "Machine learning is a subset of artificial intelligence."
        
        self.cognitive.learn_from_interaction(query, response, feedback_score=0.8)
        
        # AtomSpace should have grown
        self.assertGreater(len(self.atomspace), initial_size)
        
        # Should be able to find concepts
        ml_concepts = self.atomspace.get_atoms_by_name("machine")
        ai_concepts = self.atomspace.get_atoms_by_name("artificial")
        
        self.assertGreater(len(ml_concepts), 0)
        self.assertGreater(len(ai_concepts), 0)


class TestIntegration(unittest.TestCase):
    """Test integration between components."""
    
    def test_complete_workflow(self):
        """Test a complete workflow from concept extraction to enhancement."""
        atomspace = AtomSpace("integration_test")
        cognitive = CognitivePrimitives(atomspace)
        
        # Step 1: Extract concepts from initial knowledge
        knowledge_text = "Dogs are animals. Cats are animals. Dogs and cats are pets."
        cognitive.extract_concepts_from_text(knowledge_text, add_to_atomspace=True)
        
        # Step 2: Create explicit relationships
        cognitive.create_inheritance_relationship("dogs", "animals")
        cognitive.create_inheritance_relationship("cats", "animals")
        cognitive.create_inheritance_relationship("dogs", "pets")
        cognitive.create_inheritance_relationship("cats", "pets")
        
        # Step 3: Process a query
        query = "Tell me about dogs"
        response = "Dogs are loyal pets and animals."
        
        # Step 4: Enhance response
        enhancement = cognitive.enhance_response_with_knowledge(query, response)
        
        # Step 5: Learn from interaction
        cognitive.learn_from_interaction(query, response, feedback_score=0.9)
        
        # Verify the workflow worked
        self.assertGreater(len(atomspace), 0)
        self.assertIn('confidence_score', enhancement)
        
        # Check that relationships exist
        related_to_dogs = cognitive.find_related_concepts("dogs")
        self.assertIsInstance(related_to_dogs, dict)
        self.assertGreater(len(related_to_dogs), 0)
        
        # Get knowledge summary
        summary = cognitive.get_knowledge_summary()
        self.assertIn('atomspace_stats', summary)
        self.assertIn('top_concepts_by_importance', summary)


def run_tests():
    """Run all tests."""
    print("Running OpenCog integration tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestAtomSpace))
    suite.addTests(loader.loadTestsFromTestCase(TestAtoms))
    suite.addTests(loader.loadTestsFromTestCase(TestCognitivePrimitives))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)