#!/usr/bin/env python3
"""
OpenCog-enhanced SkyPilot LLaMA Chatbot
This script integrates OpenCog cognitive architecture with the LLaMA chatbot
for enhanced reasoning, memory, and learning capabilities.
"""

import fire
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Import original LLaMA components
from chat import ChatLLaMA, setup_model_parallel, load, sample_top_p

# Import OpenCog components
from opencog import AtomSpace, OpenCogChatLLaMA, CognitivePrimitives


class OpenCogLLaMAServer:
    """
    Enhanced LLaMA chatbot server with OpenCog cognitive architecture.
    """
    
    def __init__(self, ckpt_dir: str, tokenizer_path: str, 
                 temperature: float = 0.8, top_p: float = 0.99,
                 enable_opencog: bool = True, knowledge_file: Optional[str] = None,
                 seed: int = 42):
        """
        Initialize the OpenCog-enhanced LLaMA chatbot.
        
        Args:
            ckpt_dir: Directory containing model checkpoints
            tokenizer_path: Path to tokenizer model
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            enable_opencog: Whether to enable OpenCog enhancements
            knowledge_file: Path to knowledge persistence file
            seed: Random seed
        """
        print("Initializing OpenCog-enhanced LLaMA chatbot...")
        
        self.enable_opencog = enable_opencog
        self.knowledge_file = knowledge_file or "opencog_knowledge.json"
        
        # Initialize model parallel processing
        self.local_rank, self.world_size = setup_model_parallel(seed)
        if self.local_rank > 0:
            sys.stdout = open(os.devnull, 'w')
        
        # Load LLaMA model and tokenizer
        print("Loading LLaMA model...")
        self.model, self.tokenizer = load(ckpt_dir, tokenizer_path, 
                                         self.local_rank, self.world_size)
        
        # Create base ChatLLaMA instance
        self.base_chatbot = ChatLLaMA(
            local_rank=self.local_rank,
            world_size=self.world_size,
            model=self.model,
            tokenizer=self.tokenizer,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Initialize OpenCog enhancement if enabled
        if self.enable_opencog:
            print("Initializing OpenCog cognitive architecture...")
            self.opencog_chatbot = OpenCogChatLLaMA(
                base_chatbot=self.base_chatbot,
                enable_learning=True,
                memory_persistence=True,
                knowledge_file=self.knowledge_file
            )
            print("OpenCog integration completed.")
        else:
            self.opencog_chatbot = None
        
        print("System ready!")
    
    def interactive_chat(self):
        """Start an interactive chat session."""
        
        if self.local_rank == 0:
            print("\n" + "="*60)
            print("OpenCog-Enhanced LLaMA Chatbot")
            print("="*60)
            
            if self.enable_opencog:
                print("🧠 OpenCog cognitive architecture: ENABLED")
                print("📚 Knowledge persistence: ENABLED")
                print("🎯 Learning from interactions: ENABLED")
            else:
                print("🔧 Basic LLaMA mode: ENABLED")
            
            print("\nSampling parameters:")
            print(f"  Temperature: {self.base_chatbot.temperature}")
            print(f"  Top-p: {self.base_chatbot.top_p}")
            print(f"  Max response length: {self.base_chatbot.max_gen_len}")
            
            if self.enable_opencog:
                knowledge_state = self.opencog_chatbot.get_knowledge_state()
                print(f"\nKnowledge base:")
                print(f"  Total atoms: {knowledge_state['atomspace_stats']['size']}")
                print(f"  Concepts: {knowledge_state['atomspace_stats']['stats']['nodes']}")
                print(f"  Relations: {knowledge_state['atomspace_stats']['stats']['links']}")
            
            print("\nType 'quit', 'exit', or press Ctrl+C to end the conversation.")
            print("Type '/help' for additional commands.")
            print("Type '/stats' to see cognitive statistics.")
            print("Type '/knowledge' to see knowledge summary.")
            print("-"*60)
        
        try:
            while True:
                if self.local_rank == 0:
                    user_input = input("\n👤 You: ").strip()
                    
                    # Handle special commands
                    if user_input.lower() in ['quit', 'exit']:
                        break
                    elif user_input == '/help':
                        self._show_help()
                        continue
                    elif user_input == '/stats' and self.enable_opencog:
                        self._show_stats()
                        continue
                    elif user_input == '/knowledge' and self.enable_opencog:
                        self._show_knowledge()
                        continue
                    elif user_input == '/reset' and self.enable_opencog:
                        self._reset_session()
                        continue
                    elif not user_input:
                        continue
                    
                    # Process the input
                    response_data = self._process_input(user_input)
                    
                    # Display response
                    print(f"\n🤖 LLaMA: {response_data['response']}")
                    
                    # Show OpenCog enhancements if enabled
                    if self.enable_opencog and 'insights' in response_data:
                        self._display_cognitive_insights(response_data)
                
        except KeyboardInterrupt:
            if self.local_rank == 0:
                print("\n\nGoodbye!")
        
        # Save knowledge before exiting
        if self.enable_opencog and self.local_rank == 0:
            print("Saving knowledge base...")
            self.opencog_chatbot.save_knowledge()
            print("Knowledge saved.")
    
    def _process_input(self, user_input: str) -> Dict[str, Any]:
        """Process user input and generate response."""
        
        if self.enable_opencog:
            # Use OpenCog-enhanced processing
            return self.opencog_chatbot.enhanced_chat_step(user_input)
        else:
            # Use basic LLaMA processing
            # TODO: Integrate with the original chat method from base_chatbot
            # For demonstration, we return a placeholder indicating basic mode
            response = f"[Basic LLaMA mode - would process: {user_input[:50]}...]"
            return {'response': response}
    
    def _display_cognitive_insights(self, response_data: Dict[str, Any]):
        """Display cognitive insights from OpenCog processing."""
        
        insights = response_data.get('insights', {})
        enhancement = response_data.get('enhancement', {})
        
        # Show confidence score
        confidence = enhancement.get('confidence_score', 0.0)
        print(f"🎯 Confidence: {confidence:.2f}")
        
        # Show key concepts identified
        concepts = enhancement.get('query_concepts', []) + enhancement.get('response_concepts', [])
        if concepts:
            print(f"🔑 Key concepts: {', '.join(set(concepts[:5]))}")
        
        # Show knowledge activation
        knowledge_activation = insights.get('knowledge_activation', {})
        if knowledge_activation:
            top_activated = sorted(knowledge_activation.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
            activated_concepts = [f"{concept} ({score:.2f})" 
                                for concept, score in top_activated]
            print(f"🧠 Knowledge activated: {', '.join(activated_concepts)}")
        
        # Show learning opportunities
        learning_opps = insights.get('learning_opportunities', [])
        if learning_opps:
            print(f"📚 Learning: {len(learning_opps)} new concept(s)")
    
    def _show_help(self):
        """Show help information."""
        print("\n" + "="*40)
        print("Available Commands:")
        print("="*40)
        print("/help      - Show this help message")
        print("/stats     - Show cognitive statistics")
        print("/knowledge - Show knowledge summary")
        print("/reset     - Reset current session")
        print("quit/exit  - End conversation")
        print("="*40)
    
    def _show_stats(self):
        """Show cognitive statistics."""
        if not self.enable_opencog:
            print("OpenCog not enabled.")
            return
        
        session_summary = self.opencog_chatbot.get_session_summary()
        stats = session_summary['stats']
        
        print("\n" + "="*40)
        print("Cognitive Statistics:")
        print("="*40)
        print(f"Session duration: {time.time() - stats['session_start']:.1f}s")
        print(f"Interactions: {stats['interactions']}")
        print(f"Concepts learned: {stats['concepts_learned']}")
        print(f"Knowledge enhancements: {stats['knowledge_enhancements']}")
        print(f"Active concepts: {session_summary['active_concepts']}")
        print(f"AtomSpace size: {session_summary['atomspace_size']}")
        print("="*40)
    
    def _show_knowledge(self):
        """Show knowledge summary."""
        if not self.enable_opencog:
            print("OpenCog not enabled.")
            return
        
        knowledge_state = self.opencog_chatbot.get_knowledge_state()
        
        print("\n" + "="*50)
        print("Knowledge Base Summary:")
        print("="*50)
        
        stats = knowledge_state['atomspace_stats']['stats']
        print(f"Total atoms: {stats['total_atoms']}")
        print(f"Concept nodes: {stats['nodes']}")
        print(f"Relationship links: {stats['links']}")
        print(f"Queries processed: {stats['queries']}")
        
        print("\nTop concepts by importance:")
        for concept in knowledge_state['top_concepts_by_importance'][:5]:
            print(f"  • {concept['name']} (importance: {concept['importance']:.3f})")
        
        print("\nMost connected concepts:")
        for concept in knowledge_state['most_connected_concepts'][:5]:
            print(f"  • {concept['name']} ({concept['connections']} connections)")
        
        print("="*50)
    
    def _reset_session(self):
        """Reset the current session."""
        if self.enable_opencog:
            self.opencog_chatbot.reset_session()
            print("Session reset. Knowledge base preserved.")
        else:
            print("OpenCog not enabled.")


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.99,
    enable_opencog: bool = True,
    knowledge_file: Optional[str] = None,
    seed: int = 42,
) -> None:
    """
    Main function to start the OpenCog-enhanced LLaMA chatbot.
    
    Args:
        ckpt_dir: Directory containing model checkpoints
        tokenizer_path: Path to tokenizer model  
        temperature: Sampling temperature for generation
        top_p: Top-p sampling parameter
        enable_opencog: Enable OpenCog cognitive enhancements
        knowledge_file: Path to knowledge persistence file
        seed: Random seed for reproducibility
    """
    
    # Initialize and start the chatbot server
    server = OpenCogLLaMAServer(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        temperature=temperature,
        top_p=top_p,
        enable_opencog=enable_opencog,
        knowledge_file=knowledge_file,
        seed=seed
    )
    
    # Start interactive chat
    server.interactive_chat()


if __name__ == "__main__":
    fire.Fire(main)