# Agentic RAG for FAQ Mapping: Complete Setup Guide

This guide provides step-by-step instructions for setting up and running the Agentic RAG for FAQ Mapping system described in the research paper. The system uses a hierarchical multi-agent architecture to accurately map user queries to relevant FAQs.

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [File Structure](#file-structure)
5. [Configuration](#configuration)
6. [Running the System](#running-the-system)
7. [Evaluation and Testing](#evaluation-and-testing)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Usage](#advanced-usage)

## System Overview

The Agentic RAG for FAQ Mapping system is a novel multi-agent framework for FAQ mapping that extends traditional Retrieval-Augmented Generation (RAG) systems with agentic capabilities. Key components include:

- Query Planning Agent: Analyzes and expands user queries
- Multiple Retrieval Agents: Specialized for different retrieval strategies
- Multi-Reranker System: Evaluates candidates from different perspectives
- Judge Agent: Makes final ranking decisions using consistency-based multi-dimensional evaluation
- Response Generation Agent: Creates natural language responses
- Memory System: Learns from past interactions for self-improvement
- Orchestration Layer: Coordinates all components

## Prerequisites

Before beginning, ensure you have:

- Python 3.8 or higher
- pip package manager
- OpenAI API key (for embeddings and LLM access)
- 5GB+ of free disk space
- Active internet connection

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/agentic-rag-faq-mapper.git
cd agentic-rag-faq-mapper
```

### Step 2: Set Up Python Environment

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install pandas numpy openai tqdm scikit-learn
```

### Step 3: Set Up OpenAI API Key

```bash
# For Linux/Mac
export OPENAI_API_KEY=your_api_key_here

# For Windows (Command Prompt)
set OPENAI_API_KEY=your_api_key_here

# For Windows (PowerShell)
$env:OPENAI_API_KEY="your_api_key_here"
```

For persistent storage, add this to your `.bashrc`, `.zshrc`, or equivalent.

### Step 4: Fix File Names and Structure

Run the file structure fix script to ensure all filenames match the expected import structure:

```bash
python file_structure_fix.py
```

### Step 5: Check Missing Dependencies

Run the implementation check script to verify all required components are present:

```bash
python implementation_check.py
```

If any components are missing, the script will create template files for you to customize.

## File Structure

After setup, your directory should have the following structure:

```
agentic-rag-faq-mapper/
├── main_integration.py         # Main entry point and CLI
├── enhanced_faq_mapper.py      # Multi-agent FAQ mapper implementation
├── enhanced_judge_agent.py     # Judge agent for reranking
├── memory_system.py            # Memory system for learning
├── multi_agent_orchestrator.py # Multi-agent coordination
├── evaluation_framework.py     # Testing and evaluation tools
├── faq_mapper_implementation.py # Base mapper implementation
├── enhanced_faq_mapper_with_answers.py # Extended mapper with answer context
├── integration_test.py         # Integration testing script
├── file_structure_fix.py       # Helper for file naming
├── implementation_check.py     # Dependency verification
├── data/                       # Directory for data files
│   ├── aem_faqs.csv            # Main FAQ dataset
│   └── test_set.csv            # Test queries and ground truth
└── README.md                   # Project documentation
```

## Configuration

### Preparing FAQ Data

The system expects FAQs in CSV format with at least these columns:

- `question`: The FAQ question text
- `answer`: The answer text (optional but recommended)

Example structure:

```csv
question,answer
How do I reset my password?,Visit the login page and click on "Forgot Password". Follow the instructions sent to your email.
Where can I view my account balance?,You can see your account balance on the dashboard after logging in.
```

### Preparing Test Data (for Evaluation)

For evaluation, test data should be in CSV format with these columns:

- `utterance`: The user query
- `FAQ`: The correct FAQ that matches this query
- `Rank`: The rank of this FAQ (used for multiple correct answers)

Example structure:

```csv
utterance,FAQ,Rank
how to change my password,How do I reset my password?,1
where is my balance shown,Where can I view my account balance?,1
```

## Running the System

### Interactive Mode

To run the system in interactive mode:

```bash
python main_integration.py --faqs data/aem_faqs.csv --mode interactive
```

You'll be prompted to enter queries, and the system will display the ranked FAQs and generated responses.

### Single Query Processing

To process a single query directly:

```bash
python main_integration.py --faqs data/aem_faqs.csv --mode interactive --query "how do I reset my password"
```

### Batch Processing

To process multiple queries from a file:

```bash
python main_integration.py --faqs data/aem_faqs.csv --mode batch --batch-file queries.txt --output-dir results
```

The `queries.txt` file should contain one query per line.

## Evaluation and Testing

### Integration Test

To verify your setup is working correctly:

```bash
python integration_test.py
```

This creates a small test dataset and validates the core functionality.

### System Evaluation

To evaluate system performance against a test dataset:

```bash
python main_integration.py --faqs data/aem_faqs.csv --test data/test_set.csv --mode evaluate --num-samples 100
```

This generates a comprehensive evaluation report including metrics like:

- Top-k accuracy (k=1,3,5)
- Mean Reciprocal Rank (MRR)
- Precision@k, Recall@k, F1@k
- NDCG (Normalized Discounted Cumulative Gain)

### Baseline Comparison

To compare against a baseline implementation:

```bash
python main_integration.py --faqs data/aem_faqs.csv --test data/test_set.csv --mode compare --baseline implementations/baseline.py
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**:

   - Check that your API key is correctly set in the environment
   - Verify API key validity and rate limits

2. **Import Errors**:

   - Run `python file_structure_fix.py` to fix filename inconsistencies
   - Ensure all dependencies are installed with `pip install -r requirements.txt`

3. **Memory Usage Issues**:

   - For large FAQ datasets, use `--max-workers` flag to limit concurrency
   - Consider using chunking with batch processing for very large datasets

4. **Performance Issues**:
   - Use `--no-memory` and `--no-self-improve` flags for faster processing during testing
   - Cache embeddings for frequently used queries

### Logs

Check these log files for detailed error information:

- `faq_mapping_system.log` - Main integration logs
- `faq_mapper.log` - FAQ mapper component logs
- `judge_agent.log` - Judge agent logs
- `memory_system.log` - Memory system logs
- `evaluation.log` - Evaluation framework logs
- `integration_test.log` - Integration test logs

## Advanced Usage

### System Components Configuration

To disable memory or self-improvement:

```bash
python main_integration.py --faqs data/aem_faqs.csv --no-memory --no-self-improve
```

### Adjusting Concurrency

To adjust the number of concurrent workers:

```bash
python main_integration.py --faqs data/aem_faqs.csv --max-workers 8
```

### Custom Output Directory

To specify a custom output directory for results:

```bash
python main_integration.py --faqs data/aem_faqs.csv --output-dir custom_results
```

### Using the System Programmatically

Example of using the system in your own code:

```python
from main_integration import FAQMappingSystem

# Initialize the system
system = FAQMappingSystem(
    faqs_file='data/aem_faqs.csv',
    test_file='data/test_set.csv',
    use_memory=True,
    use_self_improvement=True
)

# Process a query
result = system.process_query(
    query="how do I reset my password",
    return_details=True
)

# Access ranked FAQs and response
ranked_faqs = result["ranked_faqs"]
response = result["response"]

print(f"Top FAQ: {ranked_faqs[0][0]}")
print(f"Response: {response}")
```

# Enhanced Multi-Agent FAQ Mapping System

# Usage Guide and Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [System Architecture](#system-architecture)
3. [Installation](#installation)
4. [Data Preparation](#data-preparation)
5. [Basic Usage](#basic-usage)
6. [Advanced Usage](#advanced-usage)
7. [Evaluation and Benchmarking](#evaluation-and-benchmarking)
8. [System Components](#system-components)
9. [Extending the System](#extending-the-system)
10. [Troubleshooting](#troubleshooting)

## Introduction

The Enhanced Multi-Agent FAQ Mapping System is a cutting-edge solution for mapping user queries to relevant Frequently Asked Questions (FAQs). By leveraging multiple specialized agents, advanced retrieval techniques, and a sophisticated orchestration framework, the system delivers more accurate and contextually relevant results than traditional approaches.

### Key Features

- **Multi-Agent Architecture**: Specialized agents for different aspects of FAQ mapping
- **Advanced Judge System**: Uses consistency-based reranking for more accurate results
- **Memory System**: Learns from historical interactions to improve over time
- **Self-Improvement**: Automatically refines its strategies based on performance
- **Comprehensive Evaluation**: Detailed metrics and analysis for system performance

### Use Cases

- Customer support automation
- Knowledge base search enhancement
- Conversational AI applications
- Enterprise help desk systems
- Banking and financial services support

## System Architecture

The system follows a modular architecture with several key components:

```
┌────────────────────────────────────────────────────────┐
│                  User Query Interface                  │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│                Query Planning Agent                    │
│  ┌─────────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │ Query Analysis  │→│Query Expansion│→│Task Division│  │
│  └─────────────────┘  └─────────────┘  └────────────┘  │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│                Retrieval Agent Network                 │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Basic Retrieval│ │Enhanced Agent│  │Answer-Context │  │
│  │    Agent     │ │              │  │    Agent      │  │
│  └──────┬──────┘  └───────┬──────┘  └───────┬───────┘  │
└─────────┼───────────────┬─┼────────────────┬───────────┘
          ↓               ↓ ↓                ↓
┌─────────┴───────────────┴─┴────────────────┴───────────┐
│                 Multi-Reranker System                  │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Semantic     │  │Relevance     │  │Intent-Match   │  │
│  │Reranker     │  │Reranker      │  │Reranker       │  │
│  └──────┬──────┘  └───────┬──────┘  └───────┬───────┘  │
└─────────┼───────────────┬─┼────────────────┬───────────┘
          ↓               ↓ ↓                ↓
┌─────────┴───────────────┴─┴────────────────┴───────────┐
│                  Judge Agent System                    │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Consistency  │  │Multi-Dimension│ │Preference     │  │
│  │Verification │  │Evaluation    │  │Optimization   │  │
│  └──────┬──────┘  └───────┬──────┘  └───────┬───────┘  │
└─────────┼───────────────┬─┼────────────────┬───────────┘
          ↓               ↓ ↓                ↓
┌─────────┴───────────────┴─┴────────────────┴───────────┐
│               Response Generation Agent                │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Content      │  │Format        │  │Confidence     │  │
│  │Synthesis    │  │Optimization  │  │Assessment     │  │
│  └──────┬──────┘  └───────┬──────┘  └───────┬───────┘  │
└─────────┼───────────────┬─┼────────────────┬───────────┘
          ↓               ↓ ↓                ↓
┌─────────┴───────────────┴─┴────────────────┴───────────┐
│                  Memory System                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │Workflow     │  │Performance   │  │Self-Improvement│  │
│  │Recording    │  │Analytics     │  │Module         │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Required Python packages (see `requirements.txt`)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/enhanced-faq-mapper.git
cd enhanced-faq-mapper
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set up OpenAI API Key

The system uses OpenAI's API for embeddings and language model inference. Set your API key as an environment variable:

```bash
export OPENAI_API_KEY=your_api_key_here
```

For persistent storage, add this to your `.bashrc` or `.zshrc` file.

## Data Preparation

### FAQ Data Format

The system expects FAQs in CSV format with at least the following columns:

- `question`: The FAQ question text
- `answer`: The answer text (optional but recommended)

Example:

```csv
question,answer
How do I reset my password?,Visit the login page and click on "Forgot Password". Follow the instructions sent to your email.
Where can I view my account balance?,You can see your account balance on the dashboard after logging in.
```

### Test Data Format

For evaluation, test data should be in CSV format with these columns:

- `utterance`: The user query
- `FAQ`: The correct FAQ that matches this query
- `Rank`: The rank of this FAQ (used for multiple correct answers)

Example:

```csv
utterance,FAQ,Rank
how to change my password,How do I reset my password?,1
where is my balance shown,Where can I view my account balance?,1
```

## Basic Usage

### Interactive Mode

For testing individual queries:

```bash
python main.py --faqs data/aem_faqs.csv --mode interactive
```

You'll be prompted to enter queries, and the system will display the ranked FAQs and generated responses.

### Single Query Processing

To process a single query directly:

```bash
python main.py --faqs data/aem_faqs.csv --mode interactive --query "how do I reset my password"
```

### Batch Processing

To process multiple queries from a file:

```bash
python main.py --faqs data/aem_faqs.csv --mode batch --batch-file queries.txt --output-dir results
```

The `queries.txt` file should contain one query per line.

## Advanced Usage

### Evaluation Mode

To evaluate the system against test data:

```bash
python main.py --faqs data/aem_faqs.csv --test data/test_set.csv --mode evaluate --num-samples 100
```

This will generate a comprehensive evaluation report in the output directory.

### Comparison Mode

To compare against a baseline implementation:

```bash
python main.py --faqs data/aem_faqs.csv --test data/test_set.csv --mode compare --baseline implementations/baseline.py
```

### Disabling Features

To disable memory or self-improvement:

```bash
python main.py --faqs data/aem_faqs.csv --no-memory --no-self-improve
```

### Adjusting Concurrency

To adjust the number of concurrent workers:

```bash
python main.py --faqs data/aem_faqs.csv --max-workers 8
```

## Evaluation and Benchmarking

The system includes a comprehensive evaluation framework that analyzes:

1. **Traditional Metrics**:

   - Top-k accuracy (k=1,3,5)
   - Mean Reciprocal Rank (MRR)

2. **Advanced Metrics**:

   - Precision@k, Recall@k, F1@k
   - NDCG@k (Normalized Discounted Cumulative Gain)
   - Hit rate and recall metrics

3. **Efficiency Metrics**:

   - Processing time
   - Memory usage
   - API call efficiency

4. **Component Analysis**:
   - Individual agent performance
   - Reranker impact
   - Judge consistency

### Generating Evaluation Reports

```bash
python main.py --faqs data/aem_faqs.csv --test data/test_set.csv --mode evaluate --output-dir eval_results
```

The evaluation framework will generate:

- Detailed metrics in JSON format
- Comparative visualizations
- Error analysis
- Component contribution analysis
- HTML report summarizing all findings

## System Components

### Query Planning Agent

The Query Planning Agent analyzes and expands user queries to improve retrieval performance:

- **Query Analysis**: Understands the intent behind the query
- **Query Expansion**: Adds relevant terms to improve recall
- **Agent Selection**: Determines which specialized agents to use

### Retrieval Agents

Multiple specialized agents retrieve candidate FAQs:

- **Basic Retrieval Agent**: Simple semantic matching
- **Enhanced Retrieval Agent**: More sophisticated matching with intent understanding
- **Answer-Context Agent**: Considers both questions and answers

### Multi-Reranker System

Multiple rerankers evaluate candidates from different perspectives:

- **Semantic Reranker**: Based on semantic similarity
- **Intent Reranker**: Based on intent matching
- **Relevance Reranker**: Based on relevance to the query

### Judge Agent System

The Judge Agent makes the final ranking decision:

- **Consistency Verification**: Ensures consistent judgments over time
- **Multi-Dimensional Evaluation**: Evaluates candidates across multiple dimensions
- **Preference Optimization**: Learns from past judgments

### Response Generation Agent

Generates natural language responses based on the top FAQ:

- **Content Synthesis**: Combines FAQ information into a coherent response
- **Format Optimization**: Formats the response appropriately
- **Confidence Assessment**: Provides confidence scores

### Memory System

Records and learns from past interactions:

- **Workflow Recording**: Saves successful patterns
- **Performance Analytics**: Analyzes what works best
- **Self-Improvement Module**: Updates strategies based on performance

### Multi-Agent Orchestrator

Coordinates all components:

- **Task Management**: Creates and tracks tasks
- **Dependency Resolution**: Handles task dependencies
- **Workflow Execution**: Executes complex workflows

## Extending the System

### Adding New Agents

To add a new specialized agent:

1. Create a new agent class implementing the required methods
2. Register it with the orchestrator:

```python
orchestrator.register_agent(
    agent_name="NewAgent",
    agent_instance=new_agent,
    agent_info={"type": "custom", "description": "New agent description"}
)
```

### Creating Custom Workflows

Define custom workflows for specific tasks:

```python
workflow = {
    "name": "custom_workflow",
    "description": "A custom workflow",
    "steps": {
        "step1": {
            "agent": "QueryPlanningAgent",
            "function": "plan_query",
            "args": {
                "query": "$input.query"
            }
        },
        # Add more steps...
    },
    "output": {
        "result": "$task.step1"
    }
}

orchestrator.register_workflow("custom_workflow", workflow)
```

### Implementing Custom Rerankers

Add specialized rerankers for your domain:

```python
def custom_reranker(query, candidates):
    # Implement custom reranking logic
    return reranked_candidates

# Add to the faq_mapper's rerankers
faq_mapper.rerankers.append(("Custom", custom_reranker))
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**: If you encounter rate limiting with OpenAI API:

   - Implement backoff strategies
   - Increase the delay between API calls
   - Use API key rotation

2. **Memory Usage**: For large FAQ datasets:

   - Use chunking to process in batches
   - Enable disk caching for embeddings
   - Increase max_workers for higher parallelism

3. **Evaluation Errors**: If evaluation fails:
   - Check test data format
   - Ensure all required columns are present
   - Verify that FAQs in test data exist in FAQ data

### Logging

The system uses Python's logging module. To increase verbosity:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Log files are saved in the current directory and can be used for troubleshooting.

### Getting Help

If you encounter issues:

1. Check the logs for detailed error messages
2. Consult the documentation for your specific component
3. Submit issues on GitHub with detailed reproduction steps
