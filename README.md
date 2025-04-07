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
