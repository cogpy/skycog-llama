# OpenCog-Enhanced SkyPilot LLaMA Chatbot

This repository contains an OpenCog-enhanced implementation of the [LLaMA](https://github.com/facebookresearch/llama/tree/main) model as an intelligent chatbot, deployable on any cloud platform using [SkyPilot](https://github.com/skypilot-org/skypilot).

## 🧠 OpenCog Integration

This implementation integrates [OpenCog](https://github.com/opencog/opencog) cognitive architecture with the LLaMA language model to provide:

- **Knowledge Representation**: Hypergraph-based AtomSpace for storing and managing knowledge
- **Cognitive Reasoning**: Enhanced reasoning capabilities using OpenCog's cognitive primitives
- **Learning & Adaptation**: Continuous learning from conversations with memory persistence
- **Context Awareness**: Improved understanding through concept relationships and attention allocation
- **Response Enhancement**: Augmented responses using stored knowledge and relationship discovery

## 🚀 Features

### Core LLaMA Capabilities
- Large Language Model inference using Meta's LLaMA
- SkyPilot integration for cloud deployment
- Interactive conversational interface
- Configurable sampling parameters

### OpenCog Enhancements
- **AtomSpace**: Hypergraph database for knowledge storage
- **Concept Extraction**: Automatic concept identification from text
- **Relationship Learning**: Discovery and creation of semantic relationships
- **Cognitive Insights**: Analysis of reasoning processes and knowledge activation
- **Memory Persistence**: Save and restore learned knowledge between sessions
- **Attention Allocation**: Dynamic resource allocation using ECAN-inspired mechanisms

## 📦 Installation

### Requirements
- Python 3.8+
- PyTorch
- Additional dependencies in `requirements.txt`

### Setup
```bash
# Clone the repository
git clone https://github.com/cogpy/skycog-llama.git
cd skycog-llama

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## 🎯 Usage

### Basic LLaMA Chatbot
```bash
python chat.py --ckpt_dir /path/to/llama/checkpoints --tokenizer_path /path/to/tokenizer.model
```

### OpenCog-Enhanced Chatbot
```bash
python opencog_chat.py --ckpt_dir /path/to/llama/checkpoints --tokenizer_path /path/to/tokenizer.model
```

### Demo Without LLaMA Model
To explore OpenCog functionality without requiring the full LLaMA model:
```bash
python opencog_demo.py
```

### Command Line Options
- `--temperature`: Sampling temperature (default: 0.8)
- `--top_p`: Top-p sampling parameter (default: 0.99)
- `--enable_opencog`: Enable OpenCog enhancements (default: True)
- `--knowledge_file`: Path to knowledge persistence file
- `--seed`: Random seed for reproducibility

## 🧪 Testing

Run the test suite to verify OpenCog integration:
```bash
python tests/test_opencog_integration.py
```

## 💡 Interactive Commands

When using the OpenCog-enhanced chatbot:
- `/help` - Show available commands
- `/stats` - Display cognitive statistics
- `/knowledge` - Show knowledge base summary
- `/reset` - Reset current session (preserves learned knowledge)
- `quit` or `exit` - End conversation

## 🏗️ Architecture

### OpenCog Components

1. **AtomSpace**: Central hypergraph database storing all knowledge as atoms (nodes and links)
2. **Cognitive Primitives**: High-level functions for concept extraction, relationship creation, and reasoning
3. **Enhanced Chatbot**: Integration layer that combines LLaMA with OpenCog capabilities

### Knowledge Representation

```
ConceptNode("artificial_intelligence")
ConceptNode("machine_learning") 
InheritanceLink(machine_learning, artificial_intelligence)
SimilarityLink(chatbot, assistant)
```

### Processing Flow

1. **Input Analysis**: Extract concepts and analyze context
2. **Knowledge Activation**: Identify relevant stored knowledge
3. **Response Generation**: Generate base response using LLaMA
4. **Enhancement**: Augment response with OpenCog knowledge
5. **Learning**: Update knowledge base from interaction
6. **Persistence**: Save learned knowledge for future sessions

## 📚 Examples

### Knowledge Building
The system automatically builds knowledge from conversations:
```
User: "What is machine learning?"
System: Extracts concepts [machine, learning, artificial, intelligence]
        Creates relationships: machine_learning ↔ artificial_intelligence
        Enhances response with stored knowledge
        Learns new connections for future interactions
```

### Relationship Discovery
```python
# Automatically discovered relationships
python → programming_language → technology
tensorflow → machine_learning_library → technology
chatbot → artificial_intelligence → technology
```

### Cognitive Insights
```
🎯 Confidence: 0.85
🔑 Key concepts: machine, learning, algorithm, neural, network
🧠 Knowledge activated: artificial_intelligence (0.92), technology (0.78)
📚 Learning: 3 new concept(s)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines and:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project incorporates code from:
- [LLaMA](https://github.com/facebookresearch/llama) - Meta Platforms, Inc. (GPL v3)
- [OpenCog](https://github.com/opencog/opencog) - OpenCog Foundation (AGPL v3)

See [LICENSE](LICENSE) for full license details.

## 🔗 Related Projects

- [SkyPilot](https://github.com/skypilot-org/skypilot) - Multi-cloud deployment framework
- [OpenCog](https://github.com/opencog/opencog) - Artificial General Intelligence framework
- [LLaMA](https://github.com/facebookresearch/llama) - Large Language Model from Meta

For comprehensive SkyPilot deployment instructions, see the [SkyPilot LLaMA example](https://github.com/skypilot-org/skypilot/tree/master/llm/llama-chatbots).
