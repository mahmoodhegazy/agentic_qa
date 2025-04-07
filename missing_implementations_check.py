#!/usr/bin/env python
"""
This script checks for missing implementations and dependencies
for the Agentic RAG for FAQ Mapping system.
"""

import os
import importlib
import sys

# Required modules
REQUIRED_MODULES = [
    "pandas", 
    "numpy", 
    "openai", 
    "tqdm", 
    "sklearn"
]

# Required implementation files
REQUIRED_FILES = [
    "enhanced_faq_mapper.py",
    "enhanced_judge_agent.py", 
    "memory_system.py",
    "multi_agent_orchestrator.py",
    "evaluation_framework.py",
    "main_integration.py",
    "faq_mapper_implementation.py",  # Base implementation
    "enhanced_faq_mapper_with_answers.py"  # Extended implementation
]

def check_modules():
    """Check if required Python modules are installed."""
    missing_modules = []
    
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"✅ {module} is installed")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module} is NOT installed")
    
    return missing_modules

def check_files():
    """Check if required implementation files exist."""
    missing_files = []
    
    for file in REQUIRED_FILES:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            missing_files.append(file)
            print(f"❌ {file} is missing")
    
    return missing_files

def check_required_classes():
    """
    Check if required classes are implemented in their respective files.
    This is a more detailed check than just file existence.
    """
    class_issues = []
    file_to_class = {
        "faq_mapper_implementation.py": ["FAQMapper"],
        "enhanced_faq_mapper.py": ["EnhancedFAQMapper"],
        "enhanced_faq_mapper_with_answers.py": ["EnhancedFAQMapperWithAnswers"],
        "enhanced_judge_agent.py": ["EnhancedJudgeAgent"],
        "memory_system.py": ["MemorySystem"],
        "multi_agent_orchestrator.py": ["MultiAgentOrchestrator"],
        "evaluation_framework.py": ["EvaluationFramework"]
    }
    
    for file, classes in file_to_class.items():
        if not os.path.exists(file):
            # Skip if file doesn't exist (already reported in check_files)
            continue
        
        # Try to import the module dynamically
        try:
            module_name = os.path.splitext(file)[0]  # Remove .py extension
            module = importlib.import_module(module_name)
            
            for class_name in classes:
                if not hasattr(module, class_name):
                    class_issues.append(f"{class_name} not found in {file}")
                    print(f"❌ {class_name} not found in {file}")
                else:
                    print(f"✅ {class_name} found in {file}")
        except Exception as e:
            class_issues.append(f"Error checking {file}: {str(e)}")
            print(f"❌ Error checking {file}: {str(e)}")
    
    return class_issues

def check_environment_variables():
    """Check if required environment variables are set."""
    required_env_vars = ["OPENAI_API_KEY"]
    missing_env_vars = []
    
    for var in required_env_vars:
        if var not in os.environ:
            missing_env_vars.append(var)
            print(f"❌ Environment variable {var} is NOT set")
        else:
            print(f"✅ Environment variable {var} is set")
    
    return missing_env_vars

def create_missing_implementation_templates():
    """
    Create template implementation files for any missing required files.
    This gives the user a starting point to implement missing components.
    """
    templates = {
        "faq_mapper_implementation.py": """
import pandas as pd
import numpy as np
import openai
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FAQMapper")

class FAQMapper:
    """
    A basic FAQ mapping implementation using embeddings or direct comparison.
    """
    
    def __init__(self, faqs_df, test_df=None, use_embeddings=True):
        """
        Initialize the FAQ mapper.
        
        Args:
            faqs_df: DataFrame containing FAQs with 'question' and 'answer' columns
            test_df: Optional test dataset
            use_embeddings: Whether to use embeddings for similarity matching
        """
        self.faqs_df = faqs_df
        self.test_df = test_df
        self.use_embeddings = use_embeddings
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        openai.api_key = self.api_key
    
    def get_faq_mapping_embeddings(self, utterance):
        """
        Map an utterance to FAQs using embedding similarity.
        
        Args:
            utterance: The user query to map
            
        Returns:
            Response from the OpenAI API
        """
        # Get embedding for the user utterance
        utterance_embedding = self._get_embedding(utterance)
        
        # Get embeddings for all FAQs (in a real implementation, these would be cached)
        faq_embeddings = []
        for _, row in self.faqs_df.iterrows():
            faq_embedding = self._get_embedding(row['question'])
            faq_embeddings.append(faq_embedding)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(faq_embeddings):
            # Calculate cosine similarity
            similarity = sum(a*b for a, b in zip(utterance_embedding, embedding))
            similarities.append((self.faqs_df.iloc[i]['question'], similarity))
        
        # Sort by similarity
        sorted_faqs = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Format as API-like response
        response = {
            "mappings": [
                {
                    "faq": faq,
                    "score": float(score) * 100  # Scale to 0-100
                }
                for faq, score in sorted_faqs[:5]  # Top 5 results
            ]
        }
        
        return response
    
    def get_faq_mapping_direct(self, utterance):
        """
        Map an utterance to FAQs using direct comparison.
        
        Args:
            utterance: The user query to map
            
        Returns:
            Response in the same format as the embedding method
        """
        # A simple direct comparison method
        # In a real implementation, this would use a more sophisticated approach
        
        scores = []
        for _, row in self.faqs_df.iterrows():
            # Calculate a simple score based on word overlap
            query_words = set(utterance.lower().split())
            faq_words = set(row['question'].lower().split())
            overlap = len(query_words.intersection(faq_words))
            total = len(query_words.union(faq_words))
            jaccard = overlap / total if total > 0 else 0
            
            scores.append((row['question'], jaccard))
        
        # Sort by score
        sorted_faqs = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Format as API-like response
        response = {
            "mappings": [
                {
                    "faq": faq,
                    "score": float(score) * 100  # Scale to 0-100
                }
                for faq, score in sorted_faqs[:5]  # Top 5 results
            ]
        }
        
        return response
    
    def _get_embedding(self, text, model="text-embedding-3-large"):
        """
        Get embedding for text.
        
        Args:
            text: Text to get embedding for
            model: Embedding model to use
            
        Returns:
            Embedding vector as a list of floats
        """
        try:
            response = openai.Embedding.create(
                model=model,
                input=text
            )
            return response['data'][0]['embedding']
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default size for text-embedding-3-large
    
    def extract_ranked_faqs(self, api_response):
        """
        Extract ranked FAQs from an API response.
        
        Args:
            api_response: Response from get_faq_mapping_*
            
        Returns:
            List of (FAQ, score) tuples
        """
        mappings = api_response.get("mappings", [])
        return [(m["faq"], m["score"]) for m in mappings]
    
    def evaluate(self, num_samples=None, save_path=None):
        """
        Evaluate the FAQ mapper on the test set.
        
        Args:
            num_samples: Number of samples to evaluate
            save_path: Path to save evaluation results
            
        Returns:
            DataFrame of results and metrics dictionary
        """
        if self.test_df is None:
            logger.error("No test data available for evaluation")
            return None, {}
        
        # Use a subset of the test data if specified
        test_samples = self.test_df if num_samples is None else self.test_df.head(num_samples)
        
        results = []
        
        for _, row in test_samples.iterrows():
            utterance = row['utterance']
            actual_faq = row['FAQ']
            
            # Get predictions
            if self.use_embeddings:
                api_response = self.get_faq_mapping_embeddings(utterance)
            else:
                api_response = self.get_faq_mapping_direct(utterance)
            
            # Extract ranked FAQs
            ranked_faqs = self.extract_ranked_faqs(api_response)
            
            # Create the result record
            result = {
                'utterance': utterance,
                'actual_faq': actual_faq,
                'predicted_rank': None,
                'correct_top1': 0,
                'correct_top3': 0,
                'correct_top5': 0
            }
            
            # Check if the actual FAQ is in the predictions
            for i, (faq, score) in enumerate(ranked_faqs):
                if faq == actual_faq:
                    result['predicted_rank'] = i + 1
                    result['correct_top1'] = 1 if i == 0 else 0
                    result['correct_top3'] = 1 if i < 3 else 0
                    result['correct_top5'] = 1 if i < 5 else 0
                    break
            
            # Add predictions to the result
            for i, (faq, score) in enumerate(ranked_faqs, 1):
                result[f'pred_faq_{i}'] = faq
                result[f'pred_score_{i}'] = score
            
            results.append(result)
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Calculate metrics
        metrics = {
            'top1_accuracy': results_df['correct_top1'].mean(),
            'top3_accuracy': results_df['correct_top3'].mean(),
            'top5_accuracy': results_df['correct_top5'].mean(),
            'mean_reciprocal_rank': self._calculate_mrr(results_df)
        }
        
        # Save results if a path is provided
        if save_path:
            results_df.to_csv(save_path, index=False)
        
        return results_df, metrics
    
    def _calculate_mrr(self, results_df):
        """
        Calculate Mean Reciprocal Rank from results.
        
        Args:
            results_df: DataFrame with evaluation results
            
        Returns:
            Mean Reciprocal Rank as a float
        """
        reciprocal_ranks = []
        
        for _, row in results_df.iterrows():
            if pd.notna(row['predicted_rank']):
                reciprocal_ranks.append(1.0 / row['predicted_rank'])
            else:
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
""",

        "enhanced_faq_mapper.py": """
import pandas as pd
import numpy as np
import openai
import os
import json
import logging

# Import the base mapper
from faq_mapper_implementation import FAQMapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedFAQMapper")

class EnhancedFAQMapper(FAQMapper):
    """
    An enhanced version of the FAQ mapper with improved query understanding.
    """
    
    def __init__(self, faqs_df, test_df=None, use_embeddings=True):
        """
        Initialize the enhanced FAQ mapper.
        
        Args:
            faqs_df: DataFrame containing FAQs with 'question' and 'answer' columns
            test_df: Optional test dataset
            use_embeddings: Whether to use embeddings for similarity matching
        """
        super().__init__(faqs_df, test_df, use_embeddings)
    
    def map_utterance(self, utterance):
        """
        Map an utterance to FAQs with enhanced understanding.
        
        Args:
            utterance: The user query to map
            
        Returns:
            Response with mappings
        """
        # Enhanced version can do query expansion before mapping
        expanded_query = self._expand_query(utterance)
        
        # Use the expanded query with the base mapper method
        if self.use_embeddings:
            return self.get_faq_mapping_embeddings(expanded_query)
        else:
            return self.get_faq_mapping_direct(expanded_query)
    
    def _expand_query(self, query):
        """
        Expand the query with additional context based on intent understanding.
        
        Args:
            query: The original user query
            
        Returns:
            Expanded query
        """
        # In a real implementation, this would use a more sophisticated approach
        # Here we'll just make a simple extension
        
        # Check for banking-specific terms and add synonyms
        expanded = query
        
        term_mappings = {
            "pin": "PIN password security code",
            "card": "card debit credit bank card",
            "account": "account bank account banking account",
            "transfer": "transfer money transfer wire transfer",
            "balance": "balance account balance available funds",
            "deposit": "deposit make a deposit add money",
            "withdraw": "withdraw withdrawal take money out",
            "statement": "statement bank statement transaction history"
        }
        
        for term, expansion in term_mappings.items():
            if term in query.lower():
                expanded = expanded + " " + expansion
        
        return expanded
    
    # Other methods inherited from the base class
""",

        "enhanced_faq_mapper_with_answers.py": """
import pandas as pd
import numpy as np
import openai
import os
import json
import logging

# Import the base mapper
from faq_mapper_implementation import FAQMapper

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EnhancedFAQMapperWithAnswers")

class EnhancedFAQMapperWithAnswers(FAQMapper):
    """
    An enhanced version of the FAQ mapper that also considers answer content.
    """
    
    def __init__(self, faqs_df, test_df=None, use_embeddings=True):
        """
        Initialize the enhanced FAQ mapper with answers.
        
        Args:
            faqs_df: DataFrame containing FAQs with 'question' and 'answer' columns
            test_df: Optional test dataset
            use_embeddings: Whether to use embeddings for similarity matching
        """
        super().__init__(faqs_df, test_df, use_embeddings)
        
        # Check if the FAQ DataFrame has an 'answer' column
        if 'answer' not in self.faqs_df.columns:
            logger.warning("FAQ DataFrame does not have an 'answer' column. This mapper may not work as expected.")
    
    def map_utterance_with_answers(self, utterance):
        """
        Map an utterance to FAQs considering both questions and answers.
        
        Args:
            utterance: The user query to map
            
        Returns:
            Response with mappings
        """
        if 'answer' not in self.faqs_df.columns:
            # Fall back to regular mapping if no answers are available
            logger.warning("No answer column found, falling back to regular mapping")
            if self.use_embeddings:
                return self.get_faq_mapping_embeddings(utterance)
            else:
                return self.get_faq_mapping_direct(utterance)
        
        # Calculate combined similarity scores using both questions and answers
        if self.use_embeddings:
            # Get embedding for the user utterance
            utterance_embedding = self._get_embedding(utterance)
            
            # Calculate scores considering both questions and answers
            scores = []
            for _, row in self.faqs_df.iterrows():
                # Get embeddings for both question and answer
                question = row['question']
                answer = row['answer'] or ""
                
                question_embedding = self._get_embedding(question)
                answer_embedding = self._get_embedding(answer[:1000])  # Limit long answers
                
                # Calculate cosine similarity for both
                question_sim = sum(a*b for a, b in zip(utterance_embedding, question_embedding))
                answer_sim = sum(a*b for a, b in zip(utterance_embedding, answer_embedding))
                
                # Weighted combination of similarities (question given more weight)
                combined_sim = 0.7 * question_sim + 0.3 * answer_sim
                
                scores.append((question, combined_sim))
        else:
            # Simple word overlap method for both questions and answers
            scores = []
            for _, row in self.faqs_df.iterrows():
                question = row['question']
                answer = row['answer'] or ""
                
                query_words = set(utterance.lower().split())
                question_words = set(question.lower().split())
                answer_words = set(answer.lower().split())
                
                # Calculate Jaccard similarity for both
                question_overlap = len(query_words.intersection(question_words))
                question_total = len(query_words.union(question_words))
                question_jaccard = question_overlap / question_total if question_total > 0 else 0
                
                answer_overlap = len(query_words.intersection(answer_words))
                answer_total = len(query_words.union(answer_words))
                answer_jaccard = answer_overlap / answer_total if answer_total > 0 else 0
                
                # Weighted combination (question given more weight)
                combined_sim = 0.7 * question_jaccard + 0.3 * answer_jaccard
                
                scores.append((question, combined_sim))
        
        # Sort by score
        sorted_faqs = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Format as API-like response
        response = {
            "mappings": [
                {
                    "faq": faq,
                    "score": float(score) * 100  # Scale to 0-100
                }
                for faq, score in sorted_faqs[:5]  # Top 5 results
            ]
        }
        
        return response
    
    # Other methods inherited from the base class
"""
    }
    
    created_files = []
    for file, template in templates.items():
        if not os.path.exists(file) and file in missing_files:
            print(f"Creating template for {file}...")
            with open(file, 'w') as f:
                f.write(template.strip())
            created_files.append(file)
    
    return created_files

def generate_installation_guide(missing_modules, missing_files, missing_env_vars):
    """Generate installation guide based on missing components."""
    guide = "# Installation Guide for FAQ Mapping System\n\n"
    
    if missing_modules:
        guide += "## Required Python Modules\n\n"
        guide += "Install the following Python modules:\n\n"
        guide += "```bash\npip install " + " ".join(missing_modules) + "\n```\n\n"
    
    if missing_files:
        guide += "## Required Implementation Files\n\n"
        guide += "The following files are missing and need to be implemented:\n\n"
        for file in missing_files:
            guide += f"- `{file}`\n"
        guide += "\n"
    
    if missing_env_vars:
        guide += "## Required Environment Variables\n\n"
        guide += "Set the following environment variables:\n\n"
        guide += "```bash\n"
        for var in missing_env_vars:
            guide += f"export {var}=your_{var.lower()}_here\n"
        guide += "```\n\n"
    
    guide += "## Testing the System\n\n"
    guide += "Once all dependencies are installed, you can test the system with:\n\n"
    guide += "```bash\npython main_integration.py --faqs your_faqs.csv --mode interactive\n```\n"
    
    return guide

def main():
    """Main function to run all checks and generate output."""
    print("Checking for required Python modules...")
    missing_modules = check_modules()
    
    print("\nChecking for required implementation files...")
    missing_files = check_files()
    
    print("\nChecking for required environment variables...")
    missing_env_vars = check_environment_variables()
    
    print("\nChecking for required classes in existing files...")
    class_issues = check_required_classes()
    
    # Create missing implementation templates
    if missing_files:
        print("\nCreating templates for missing implementation files...")
        created_files = create_missing_implementation_templates()
        if created_files:
            print(f"Created template files: {', '.join(created_files)}")
            # Update missing files list
            missing_files = [f for f in missing_files if f not in created_files]
    
    # Generate installation guide
    guide = generate_installation_guide(missing_modules, missing_files, missing_env_vars)
    with open("INSTALLATION_GUIDE.md", "w") as f:
        f.write(guide)
    print("\nCreated INSTALLATION_GUIDE.md with instructions for missing components")
    
    # Final summary
    print("\n=== System Check Summary ===")
    if not (missing_modules or missing_files or missing_env_vars or class_issues):
        print("✅ All required components are present and ready to use!")
    else:
        print("⚠️ Some components are missing or have issues:")
        if missing_modules:
            print(f"- Missing Python modules: {', '.join(missing_modules)}")
        if missing_files:
            print(f"- Missing files: {', '.join(missing_files)}")
        if missing_env_vars:
            print(f"- Missing environment variables: {', '.join(missing_env_vars)}")
        if class_issues:
            print(f"- Class implementation issues: {len(class_issues)} issues found")
        print("\nSee INSTALLATION_GUIDE.md for installation instructions")

if __name__ == "__main__":
    main()