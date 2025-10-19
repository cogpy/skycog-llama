#!/usr/bin/env python3
"""
OpenCog Demonstration Script
This script demonstrates the OpenCog functionality without requiring the full LLaMA model.
"""

import sys
import json
from pathlib import Path

# Add the opencog module to path
sys.path.insert(0, str(Path(__file__).parent))

from opencog import (
    AtomSpace, ConceptNode, PredicateNode, WordNode,
    InheritanceLink, SimilarityLink, CognitivePrimitives
)


class OpenCogDemo:
    """Demonstration of OpenCog functionality for the SkyPilot LLaMA chatbot."""
    
    def __init__(self):
        """Initialize the demonstration."""
        print("🧠 OpenCog SkyPilot LLaMA Integration Demo")
        print("=" * 50)
        
        # Initialize OpenCog components
        self.atomspace = AtomSpace("demo_space")
        self.cognitive = CognitivePrimitives(self.atomspace)
        
        print(f"✅ AtomSpace initialized: {self.atomspace.name}")
        print(f"✅ Cognitive primitives ready")
        print()
    
    def demo_basic_knowledge_representation(self):
        """Demonstrate basic knowledge representation."""
        print("📚 DEMO 1: Basic Knowledge Representation")
        print("-" * 40)
        
        # Create basic concepts
        ai_concept = ConceptNode("artificial_intelligence", truth_value=0.9)
        ml_concept = ConceptNode("machine_learning", truth_value=0.85)
        chatbot_concept = ConceptNode("chatbot", truth_value=0.8)
        
        # Add to AtomSpace
        self.atomspace.add_atom(ai_concept)
        self.atomspace.add_atom(ml_concept)
        self.atomspace.add_atom(chatbot_concept)
        
        print(f"Created concept: {ai_concept}")
        print(f"Created concept: {ml_concept}")
        print(f"Created concept: {chatbot_concept}")
        
        # Create relationships
        ml_inherits_ai = InheritanceLink(ml_concept, ai_concept, truth_value=0.9)
        chatbot_uses_ai = InheritanceLink(chatbot_concept, ai_concept, truth_value=0.8)
        
        self.atomspace.add_atom(ml_inherits_ai)
        self.atomspace.add_atom(chatbot_uses_ai)
        
        print(f"Created relationship: {ml_inherits_ai}")
        print(f"Created relationship: {chatbot_uses_ai}")
        
        print(f"\n📊 AtomSpace now contains {len(self.atomspace)} atoms")
        print()
    
    def demo_text_processing(self):
        """Demonstrate text processing and concept extraction."""
        print("🔤 DEMO 2: Text Processing and Concept Extraction")
        print("-" * 40)
        
        sample_texts = [
            "OpenCog is a framework for artificial general intelligence.",
            "Machine learning algorithms can process natural language.",
            "Chatbots use neural networks for conversation.",
            "SkyPilot enables cloud deployment of large language models."
        ]
        
        for i, text in enumerate(sample_texts, 1):
            print(f"Text {i}: {text}")
            
            # Extract concepts
            concepts = self.cognitive.extract_concepts_from_text(text, add_to_atomspace=True)
            concept_names = [c.name for c in concepts]
            print(f"  Extracted concepts: {', '.join(concept_names)}")
            
            # Create word nodes
            words = self.cognitive.create_word_nodes_from_text(text, add_to_atomspace=True)
            print(f"  Created {len(words)} word nodes")
            print()
        
        print(f"📊 AtomSpace now contains {len(self.atomspace)} atoms")
        print()
    
    def demo_relationship_creation(self):
        """Demonstrate automatic relationship creation."""
        print("🔗 DEMO 3: Relationship Creation and Discovery")
        print("-" * 40)
        
        # Create hierarchical knowledge
        relationships = [
            ("python", "programming_language"),
            ("java", "programming_language"),
            ("tensorflow", "machine_learning_library"),
            ("pytorch", "machine_learning_library"),
            ("programming_language", "technology"),
            ("machine_learning_library", "technology"),
        ]
        
        print("Creating inheritance relationships:")
        for child, parent in relationships:
            link = self.cognitive.create_inheritance_relationship(child, parent, truth_value=0.9)
            print(f"  {child} ➡ {parent}")
        
        # Create similarity relationships
        similarities = [
            ("python", "java", 0.7),
            ("tensorflow", "pytorch", 0.8),
        ]
        
        print("\nCreating similarity relationships:")
        for concept1, concept2, score in similarities:
            link = self.cognitive.create_similarity_relationship(concept1, concept2, score)
            print(f"  {concept1} ↔ {concept2} (similarity: {score})")
        
        print(f"\n📊 AtomSpace now contains {len(self.atomspace)} atoms")
        print()
    
    def demo_knowledge_queries(self):
        """Demonstrate knowledge queries and reasoning."""
        print("🔍 DEMO 4: Knowledge Queries and Reasoning")
        print("-" * 40)
        
        # Query by type
        concepts = self.atomspace.get_atoms_by_type("ConceptNode")
        print(f"Found {len(concepts)} concept nodes:")
        for concept in concepts[:10]:  # Show first 10
            print(f"  • {concept.name} (truth: {concept.truth_value:.2f})")
        
        if len(concepts) > 10:
            print(f"  ... and {len(concepts) - 10} more")
        
        # Find related concepts
        test_concepts = ["python", "artificial_intelligence", "chatbot"]
        print(f"\nFinding related concepts:")
        
        for concept_name in test_concepts:
            related = self.cognitive.find_related_concepts(concept_name, max_depth=1)
            if related:
                print(f"  {concept_name}:")
                for link_type, atoms in related.items():
                    atom_names = [atom.name for atom in atoms[:3]]  # Show first 3
                    print(f"    {link_type}: {', '.join(atom_names)}")
            else:
                print(f"  {concept_name}: No direct relationships found")
        
        print()
    
    def demo_response_enhancement(self):
        """Demonstrate response enhancement with knowledge."""
        print("🚀 DEMO 5: Response Enhancement with OpenCog Knowledge")
        print("-" * 40)
        
        test_cases = [
            {
                "query": "What is machine learning?",
                "response": "Machine learning is a subset of artificial intelligence that enables computers to learn."
            },
            {
                "query": "How do chatbots work?",
                "response": "Chatbots use natural language processing and neural networks to understand and respond to users."
            },
            {
                "query": "What programming languages are good for AI?",
                "response": "Python and Java are popular programming languages for artificial intelligence development."
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"Test Case {i}:")
            print(f"  Query: {case['query']}")
            print(f"  Response: {case['response']}")
            
            # Enhance with OpenCog knowledge
            enhancement = self.cognitive.enhance_response_with_knowledge(
                case['query'], case['response']
            )
            
            print(f"  Query concepts: {', '.join(enhancement['query_concepts'])}")
            print(f"  Response concepts: {', '.join(enhancement['response_concepts'])}")
            print(f"  Confidence score: {enhancement['confidence_score']:.3f}")
            
            if enhancement['related_knowledge']:
                print("  Related knowledge:")
                for knowledge in enhancement['related_knowledge'][:2]:  # Show first 2
                    print(f"    {knowledge['concept']}: {list(knowledge['relations'].keys())}")
            
            print()
    
    def demo_learning_simulation(self):
        """Demonstrate learning from interactions."""
        print("🎓 DEMO 6: Learning from Interactions")
        print("-" * 40)
        
        initial_size = len(self.atomspace)
        print(f"Initial AtomSpace size: {initial_size}")
        
        # Simulate learning from conversations
        interactions = [
            {
                "query": "Tell me about deep learning",
                "response": "Deep learning is a subset of machine learning using neural networks with multiple layers.",
                "feedback": 0.9
            },
            {
                "query": "What is SkyPilot?",
                "response": "SkyPilot is a framework for running AI workloads on any cloud platform.",
                "feedback": 0.8
            },
            {
                "query": "How does OpenCog work?",
                "response": "OpenCog uses a hypergraph database called AtomSpace for knowledge representation and reasoning.",
                "feedback": 0.85
            }
        ]
        
        print("Learning from interactions:")
        for i, interaction in enumerate(interactions, 1):
            print(f"  Interaction {i}: Learning from query about '{interaction['query'][:30]}...'")
            
            self.cognitive.learn_from_interaction(
                interaction['query'],
                interaction['response'],
                interaction['feedback']
            )
            
            new_size = len(self.atomspace)
            print(f"    AtomSpace grew by {new_size - initial_size} atoms")
            initial_size = new_size
        
        print(f"\nFinal AtomSpace size: {len(self.atomspace)}")
        print()
    
    def demo_knowledge_summary(self):
        """Demonstrate knowledge base analysis."""
        print("📈 DEMO 7: Knowledge Base Analysis")
        print("-" * 40)
        
        summary = self.cognitive.get_knowledge_summary()
        
        print("AtomSpace Statistics:")
        stats = summary['atomspace_stats']['stats']
        print(f"  Total atoms: {stats['total_atoms']}")
        print(f"  Concept nodes: {stats['nodes']}")
        print(f"  Relationship links: {stats['links']}")
        print(f"  Queries processed: {stats['queries']}")
        
        print("\nTop concepts by importance:")
        for concept in summary['top_concepts_by_importance'][:5]:
            print(f"  • {concept['name']:<20} (importance: {concept['importance']:.3f}, connections: {concept['connections']})")
        
        print("\nMost connected concepts:")
        for concept in summary['most_connected_concepts'][:5]:
            print(f"  • {concept['name']:<20} ({concept['connections']} connections, truth: {concept['truth_value']:.2f})")
        
        print()
    
    def run_full_demo(self):
        """Run the complete demonstration."""
        demos = [
            self.demo_basic_knowledge_representation,
            self.demo_text_processing,
            self.demo_relationship_creation,
            self.demo_knowledge_queries,
            self.demo_response_enhancement,
            self.demo_learning_simulation,
            self.demo_knowledge_summary,
        ]
        
        for demo in demos:
            try:
                demo()
                input("Press Enter to continue to next demo...")
                print()
            except KeyboardInterrupt:
                print("\nDemo interrupted by user.")
                break
            except Exception as e:
                print(f"Error in demo: {e}")
                continue
        
        print("🎉 OpenCog Integration Demo Complete!")
        print("=" * 50)
        print("This demonstration showed how OpenCog enhances the SkyPilot LLaMA chatbot with:")
        print("• Knowledge representation using hypergraph AtomSpace")
        print("• Automatic concept extraction from natural language")
        print("• Relationship discovery and reasoning")
        print("• Response enhancement with stored knowledge")
        print("• Learning and adaptation from interactions")
        print("• Cognitive analysis and attention allocation")
        print()
        print("The full implementation integrates these capabilities with the LLaMA model")
        print("to create a more intelligent and adaptive conversational AI system.")


def main():
    """Main function to run the OpenCog demonstration."""
    demo = OpenCogDemo()
    
    try:
        demo.run_full_demo()
    except KeyboardInterrupt:
        print("\n\nDemo terminated by user. Goodbye!")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())