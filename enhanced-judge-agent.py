import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import openai
import logging
from typing import List, Dict, Tuple, Any, Optional, Union
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("judge_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedJudgeAgent")

# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class EnhancedJudgeAgent:
    """
    An enhanced judge agent for reranking FAQ candidates using advanced techniques
    such as judge consistency and multi-dimensional evaluation.
    """
    
    def __init__(self, faqs_df, test_df=None, judge_consistency=True):
        """
        Initialize the enhanced judge agent.
        
        Args:
            faqs_df: DataFrame containing the FAQs with 'question' and 'answer' columns
            test_df: Optional DataFrame containing example mappings for few-shot learning
            judge_consistency: Whether to use judge consistency methods
        """
        self.faqs_df = faqs_df
        self.test_df = test_df
        self.judge_consistency = judge_consistency
        
        # Performance history for judge consistency
        self.judgement_history = []
        
        # Initialize memory for preferred judgments
        self.preferred_judgments = {}
        
        # Load previous preferred judgments if available
        self._load_preferred_judgments()
    
    def _load_preferred_judgments(self):
        """
        Load preferred judgments from disk if available.
        """
        preferred_judgments_path = "preferred_judgments.json"
        if os.path.exists(preferred_judgments_path):
            try:
                with open(preferred_judgments_path, 'r') as f:
                    self.preferred_judgments = json.load(f)
                logger.info(f"Loaded {len(self.preferred_judgments)} preferred judgments")
            except Exception as e:
                logger.error(f"Error loading preferred judgments: {e}")
    
    def _save_preferred_judgments(self):
        """
        Save preferred judgments to disk.
        """
        try:
            with open("preferred_judgments.json", 'w') as f:
                json.dump(self.preferred_judgments, f)
        except Exception as e:
            logger.error(f"Error saving preferred judgments: {e}")
    
    def rerank_candidates(self, utterance, candidates, agent_predictions=None):
        """
        Rerank the candidate FAQs using multi-dimensional evaluation and judge consistency.
        
        Args:
            utterance: The user query
            candidates: List of (FAQ, score) tuples
            agent_predictions: Optional dictionary of agent_name -> list of (FAQ, score) tuples
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for faq, score in candidates:
            if faq not in seen:
                unique_candidates.append((faq, score))
                seen.add(faq)
        
        # If we have 5 or fewer unique candidates, return them
        if len(unique_candidates) <= 5:
            return sorted(unique_candidates, key=lambda x: x[1], reverse=True)
        
        # Check if we have a preferred judgment for this utterance
        utterance_key = self._get_utterance_key(utterance)
        if self.judge_consistency and utterance_key in self.preferred_judgments:
            logger.info(f"Using preferred judgment for '{utterance_key}'")
            
            # Get the preferred judgment
            preferred_judgment = self.preferred_judgments[utterance_key]
            
            # Filter candidates to match the preferred judgment
            filtered_candidates = []
            for faq in preferred_judgment['top_faqs']:
                # Find the matching candidate
                for candidate_faq, candidate_score in unique_candidates:
                    if candidate_faq == faq:
                        filtered_candidates.append((candidate_faq, candidate_score))
                        break
            
            # Add any remaining candidates if needed
            remaining_candidates = [c for c in unique_candidates if c[0] not in preferred_judgment['top_faqs']]
            filtered_candidates.extend(remaining_candidates)
            
            # Return the top candidates
            return filtered_candidates[:5]
        
        # Use different evaluation dimensions for judge consistency
        if self.judge_consistency:
            return self._rerank_with_consistency(utterance, unique_candidates, agent_predictions)
        else:
            return self._rerank_standard(utterance, unique_candidates, agent_predictions)
    
    def _get_utterance_key(self, utterance):
        """
        Create a key for the utterance to use in the preferred judgments dictionary.
        
        Args:
            utterance: The user query
            
        Returns:
            A string key for the utterance
        """
        # Normalize the utterance for consistency
        normalized = utterance.lower().strip()
        
        # Remove common stop words and punctuation
        stop_words = ['and', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']
        for word in stop_words:
            normalized = normalized.replace(f' {word} ', ' ')
        
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        
        return normalized
    
    def _rerank_standard(self, utterance, candidates, agent_predictions=None):
        """
        Rerank candidates using the standard judge approach.
        
        Args:
            utterance: The user query
            candidates: List of (FAQ, score) tuples
            agent_predictions: Optional dictionary of agent_name -> list of (FAQ, score) tuples
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Create a prompt for the standard judge
        prompt = self._create_standard_judge_prompt(utterance, candidates, agent_predictions)
        
        try:
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": self._get_standard_system_message()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Parse the response to get reranked FAQs
            reranked_faqs = self._parse_standard_judge_response(response.choices[0].message.content)
            
            return reranked_faqs
        except Exception as e:
            logger.error(f"Error in standard reranking: {e}")
            # Fall back to simple aggregation
            sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            return sorted_candidates[:5]
    
    def _rerank_with_consistency(self, utterance, candidates, agent_predictions=None):
        """
        Rerank candidates using judge consistency across multiple evaluation dimensions.
        
        Args:
            utterance: The user query
            candidates: List of (FAQ, score) tuples
            agent_predictions: Optional dictionary of agent_name -> list of (FAQ, score) tuples
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Create multiple judgments using different dimensions
        judgments = []
        
        # Dimensions to evaluate
        dimensions = [
            "semantic_relevance",
            "intent_matching",
            "completeness",
            "coherence"
        ]
        
        # Get judgments for each dimension
        for dimension in dimensions:
            try:
                judgment = self._get_dimensional_judgment(utterance, candidates, dimension)
                judgments.append({
                    "dimension": dimension,
                    "judgment": judgment
                })
            except Exception as e:
                logger.error(f"Error in {dimension} judgment: {e}")
        
        # If no judgments were successful, fall back to standard reranking
        if not judgments:
            logger.warning("No dimensional judgments successful, falling back to standard reranking")
            return self._rerank_standard(utterance, candidates, agent_predictions)
        
        # Create multiple JudgeRank aggregation methods
        aggregation_methods = [
            self._aggregate_judgments_by_average,
            self._aggregate_judgments_by_borda_count,
            self._aggregate_judgments_by_weighted_average
        ]
        
        # Get aggregated rankings from each method
        aggregated_rankings = []
        for method in aggregation_methods:
            try:
                ranking = method(judgments)
                aggregated_rankings.append(ranking)
            except Exception as e:
                logger.error(f"Error in aggregation method: {e}")
        
        # If no aggregation methods were successful, take the first judgment
        if not aggregated_rankings:
            logger.warning("No aggregation methods successful, using first judgment")
            if judgments and judgments[0]["judgment"]:
                return judgments[0]["judgment"]
            else:
                # Fall back to simple sorting
                sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
                return sorted_candidates[:5]
        
        # Use meta-judge to select the best aggregation
        try:
            preferred_ranking = self._get_meta_judgment(utterance, aggregated_rankings)
            
            # Save the preferred judgment
            utterance_key = self._get_utterance_key(utterance)
            self.preferred_judgments[utterance_key] = {
                "top_faqs": [faq for faq, _ in preferred_ranking[:5]]
            }
            self._save_preferred_judgments()
            
            return preferred_ranking
        except Exception as e:
            logger.error(f"Error in meta-judgment: {e}")
            # Fall back to first aggregation method
            return aggregated_rankings[0]
    
    def _get_dimensional_judgment(self, utterance, candidates, dimension):
        """
        Get judgment for a specific evaluation dimension.
        
        Args:
            utterance: The user query
            candidates: List of (FAQ, score) tuples
            dimension: The evaluation dimension (semantic_relevance, intent_matching, etc.)
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Create a prompt for the dimensional judgment
        prompt = self._create_dimensional_judge_prompt(utterance, candidates, dimension)
        
        # Call the LLM API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": self._get_dimensional_system_message(dimension)},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Extract the rankings
        rankings = []
        for item in result.get("ranked_faqs", []):
            faq = item.get("faq", "")
            score = item.get("score", 0)
            rankings.append((faq, float(score)))
        
        # Sort by score
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _aggregate_judgments_by_average(self, judgments):
        """
        Aggregate judgments by simple averaging.
        
        Args:
            judgments: List of judgments from different dimensions
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Collect all FAQs
        all_faqs = set()
        for judgment_data in judgments:
            judgment = judgment_data["judgment"]
            for faq, _ in judgment:
                all_faqs.add(faq)
        
        # Calculate average scores
        faq_scores = {}
        for faq in all_faqs:
            # Collect scores for this FAQ from all judgments
            scores = []
            for judgment_data in judgments:
                judgment = judgment_data["judgment"]
                for j_faq, j_score in judgment:
                    if j_faq == faq:
                        scores.append(j_score)
                        break
            
            # Calculate average
            faq_scores[faq] = sum(scores) / len(scores) if scores else 0
        
        # Convert to list and sort
        rankings = [(faq, score) for faq, score in faq_scores.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _aggregate_judgments_by_borda_count(self, judgments):
        """
        Aggregate judgments using Borda count.
        
        Args:
            judgments: List of judgments from different dimensions
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Collect all FAQs
        all_faqs = set()
        for judgment_data in judgments:
            judgment = judgment_data["judgment"]
            for faq, _ in judgment:
                all_faqs.add(faq)
        
        # Calculate Borda counts
        faq_scores = {faq: 0 for faq in all_faqs}
        
        for judgment_data in judgments:
            judgment = judgment_data["judgment"]
            # Convert judgment to ranks
            ranked_faqs = [faq for faq, _ in judgment]
            
            # Assign Borda points (rank-based)
            for rank, faq in enumerate(ranked_faqs):
                # Higher ranks get more points
                points = len(ranked_faqs) - rank
                faq_scores[faq] += points
        
        # Convert to list and sort
        rankings = [(faq, score) for faq, score in faq_scores.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize scores to 0-100 range
        max_score = max([score for _, score in rankings]) if rankings else 1
        normalized_rankings = [(faq, score / max_score * 100) for faq, score in rankings]
        
        return normalized_rankings
    
    def _aggregate_judgments_by_weighted_average(self, judgments):
        """
        Aggregate judgments using weighted averaging.
        
        Args:
            judgments: List of judgments from different dimensions
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        # Weights for different dimensions (can be adjusted)
        dimension_weights = {
            "semantic_relevance": 0.2,
            "intent_matching": 0.4,
            "completeness": 0.2,
            "coherence": 0.2
        }
        
        # Collect all FAQs
        all_faqs = set()
        for judgment_data in judgments:
            judgment = judgment_data["judgment"]
            for faq, _ in judgment:
                all_faqs.add(faq)
        
        # Calculate weighted average scores
        faq_scores = {}
        for faq in all_faqs:
            # Collect scores for this FAQ from all judgments
            weighted_scores = []
            total_weight = 0
            
            for judgment_data in judgments:
                dimension = judgment_data["dimension"]
                judgment = judgment_data["judgment"]
                weight = dimension_weights.get(dimension, 0.25)  # Default weight
                
                for j_faq, j_score in judgment:
                    if j_faq == faq:
                        weighted_scores.append(j_score * weight)
                        total_weight += weight
                        break
            
            # Calculate weighted average
            faq_scores[faq] = sum(weighted_scores) / total_weight if total_weight > 0 else 0
        
        # Convert to list and sort
        rankings = [(faq, score) for faq, score in faq_scores.items()]
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _get_meta_judgment(self, utterance, aggregated_rankings):
        """
        Use meta-judge to select the best aggregation method.
        
        Args:
            utterance: The user query
            aggregated_rankings: List of rankings from different aggregation methods
            
        Returns:
            The preferred ranking
        """
        # Create a prompt for the meta-judge
        prompt = f"""
Analyze the following user query and three different rankings of FAQs. Select the ranking that best addresses the user's needs.

User Query: "{utterance}"

Ranking 1:
{chr(10).join([f"{i+1}. {faq} - Score: {score:.2f}" for i, (faq, score) in enumerate(aggregated_rankings[0][:5])])}

Ranking 2:
{chr(10).join([f"{i+1}. {faq} - Score: {score:.2f}" for i, (faq, score) in enumerate(aggregated_rankings[1][:5])])}

Ranking 3:
{chr(10).join([f"{i+1}. {faq} - Score: {score:.2f}" for i, (faq, score) in enumerate(aggregated_rankings[2][:5])])}

Select the best ranking (1, 2, or 3) and explain your choice. Your response should be in JSON format:
{{
    "selected_ranking": 1/2/3,
    "explanation": "Your explanation for why this ranking is best"
}}
"""
        
        # Call the LLM API
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at evaluating FAQ rankings for banking queries."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Get the selected ranking
        selected_ranking_index = result.get("selected_ranking", 1) - 1
        selected_ranking_index = max(0, min(selected_ranking_index, len(aggregated_rankings) - 1))
        
        # Add to judgment history
        self.judgement_history.append({
            "timestamp": datetime.now().isoformat(),
            "utterance": utterance,
            "selected_ranking": selected_ranking_index + 1,
            "explanation": result.get("explanation", "")
        })
        
        return aggregated_rankings[selected_ranking_index]
    
    def _get_standard_system_message(self):
        """
        Get the system message for the standard judge.
        
        Returns:
            A string containing the system message
        """
        return """You are an expert judge of FAQ relevance for banking applications. 
Your task is to carefully analyze user utterances and determine which FAQs best address their needs.
Consider both semantic relevance and practical usefulness when making your judgments.
Provide detailed reasoning for each of your rankings to explain your decision-making process.
Your rankings must be accurate and reliable, as banking customers depend on them for critical information."""
    
    def _get_dimensional_system_message(self, dimension):
        """
        Get the system message for a dimensional judge.
        
        Args:
            dimension: The evaluation dimension
            
        Returns:
            A string containing the system message
        """
        dimension_descriptions = {
            "semantic_relevance": """You are an expert judge of semantic relevance for banking FAQs.
Your task is to evaluate how well each FAQ matches the language and terminology used in the user's query.
Focus solely on linguistic and semantic similarity, not on intent or completeness.
Look for keyword matches, synonyms, and related banking terminology.""",
            
            "intent_matching": """You are an expert judge of intent matching for banking FAQs.
Your task is to evaluate how well each FAQ addresses the underlying intention behind the user's query.
Look beyond the specific words to understand what the user is trying to accomplish.
Consider the practical purpose of the query in a banking context.""",
            
            "completeness": """You are an expert judge of completeness for banking FAQs.
Your task is to evaluate how thoroughly each FAQ addresses all aspects of the user's query.
Check if the FAQ covers all components of multi-part questions.
Consider whether the FAQ would leave the user needing to ask follow-up questions.""",
            
            "coherence": """You are an expert judge of coherence for banking FAQs.
Your task is to evaluate how clear, organized, and understandable each FAQ is in relation to the query.
Consider whether the FAQ provides a logically structured answer.
Evaluate whether the FAQ would be easy for the user to follow and apply."""
        }
        
        return dimension_descriptions.get(dimension, self._get_standard_system_message())
    
    def _create_standard_judge_prompt(self, utterance, candidates, agent_predictions=None):
        """
        Create a prompt for the standard judge.
        
        Args:
            utterance: The user query
            candidates: List of (FAQ, score) tuples
            agent_predictions: Optional dictionary of agent_name -> list of (FAQ, score) tuples
            
        Returns:
            A prompt string for the standard judge
        """
        # Create the instructions
        instructions = """
Analyze the user query and rank the candidate FAQs based on their relevance and usefulness.

For each FAQ, provide:
1. A relevance score (0-100) based on how well it addresses the user's query
2. A detailed explanation of your reasoning for this ranking

Your response must be in JSON format:
{
    "ranked_faqs": [
        {
            "faq": "FAQ Title",
            "score": 85,
            "reasoning": "Your detailed reasoning for this ranking"
        },
        ...
    ]
}

Return all FAQs, ranked by relevance to the user's query.
"""

        # Add the user utterance
        utterance_text = f"\nUser Query: \"{utterance}\"\n"
        
        # Add the candidate FAQs
        candidates_text = "Candidate FAQs:\n"
        for i, (faq, score) in enumerate(candidates, 1):
            candidates_text += f"{i}. {faq} - Original Score: {score:.2f}\n"
        
        # Add agent predictions if available
        agent_text = ""
        if agent_predictions:
            agent_text = "\nPredictions from individual agents:\n"
            for agent_name, predictions in agent_predictions.items():
                agent_text += f"\n{agent_name}:\n"
                for i, (faq, score) in enumerate(predictions[:3], 1):  # Show top 3 from each agent
                    agent_text += f"{i}. {faq} - Score: {score:.2f}\n"
        
        # Add FAQ context information for the candidates
        faq_context = "\nFAQ Details:\n"
        
        # Collect all candidate FAQ titles
        candidate_faq_titles = [faq for faq, _ in candidates]
        
        # Look up FAQ descriptions/answers if available
        if 'answer' in self.faqs_df.columns:
            for _, row in self.faqs_df[self.faqs_df['question'].isin(candidate_faq_titles)].iterrows():
                faq_context += f"FAQ: \"{row['question']}\"\n"
                # Truncate very long answers
                answer = row['answer']
                if len(answer) > 200:
                    answer = answer[:200] + "..."
                faq_context += f"Answer: {answer}\n\n"
        
        # Combine all parts into the final prompt
        full_prompt = instructions + utterance_text + candidates_text + agent_text + faq_context
        
        return full_prompt
    
    def _create_dimensional_judge_prompt(self, utterance, candidates, dimension):
        """
        Create a prompt for a dimensional judge.
        
        Args:
            utterance: The user query
            candidates: List of (FAQ, score) tuples
            dimension: The evaluation dimension
            
        Returns:
            A prompt string for the dimensional judge
        """
        dimension_prompts = {
            "semantic_relevance": """
Analyze the user query and rank the candidate FAQs based on SEMANTIC RELEVANCE ONLY.

Focus exclusively on how well the language and terminology of each FAQ matches the user's query.
Look for keyword matches, synonyms, and related banking terminology.
Do NOT consider intent, completeness, or coherence - only semantic similarity.

For each FAQ, provide:
1. A semantic relevance score (0-100)
2. A brief explanation focused purely on semantic matching

Your response must be in JSON format:
{
    "ranked_faqs": [
        {
            "faq": "FAQ Title",
            "score": 85,
            "reasoning": "Explanation of semantic relevance"
        },
        ...
    ]
}

Return all FAQs, ranked by semantic relevance to the user's query.
""",
            
            "intent_matching": """
Analyze the user query and rank the candidate FAQs based on INTENT MATCHING ONLY.

Focus exclusively on how well each FAQ addresses the underlying intention behind the user's query.
Look beyond specific words to understand what the user is trying to accomplish.
Do NOT consider semantic similarity, completeness, or coherence - only intent matching.

For each FAQ, provide:
1. An intent matching score (0-100)
2. A brief explanation focused purely on intent matching

Your response must be in JSON format:
{
    "ranked_faqs": [
        {
            "faq": "FAQ Title",
            "score": 85,
            "reasoning": "Explanation of intent matching"
        },
        ...
    ]
}

Return all FAQs, ranked by intent matching to the user's query.
""",
            
            "completeness": """
Analyze the user query and rank the candidate FAQs based on COMPLETENESS ONLY.

Focus exclusively on how thoroughly each FAQ addresses all aspects of the user's query.
Check if the FAQ covers all components of multi-part questions.
Do NOT consider semantic similarity, intent matching, or coherence - only completeness.

For each FAQ, provide:
1. A completeness score (0-100)
2. A brief explanation focused purely on completeness

Your response must be in JSON format:
{
    "ranked_faqs": [
        {
            "faq": "FAQ Title",
            "score": 85,
            "reasoning": "Explanation of completeness"
        },
        ...
    ]
}

Return all FAQs, ranked by completeness in addressing the user's query.
""",
            
            "coherence": """
Analyze the user query and rank the candidate FAQs based on COHERENCE ONLY.

Focus exclusively on how clear, organized, and understandable each FAQ is in relation to the query.
Consider whether the FAQ provides a logically structured answer.
Do NOT consider semantic similarity, intent matching, or completeness - only coherence.

For each FAQ, provide:
1. A coherence score (0-100)
2. A brief explanation focused purely on coherence

Your response must be in JSON format:
{
    "ranked_faqs": [
        {
            "faq": "FAQ Title",
            "score": 85,
            "reasoning": "Explanation of coherence"
        },
        ...
    ]
}

Return all FAQs, ranked by coherence in relation to the user's query.
"""
        }
        
        # Get the dimension-specific prompt instructions
        dimension_prompt = dimension_prompts.get(dimension, "")
        
        # Add the user utterance
        utterance_text = f"\nUser Query: \"{utterance}\"\n"
        
        # Add the candidate FAQs
        candidates_text = "Candidate FAQs:\n"
        for i, (faq, score) in enumerate(candidates, 1):
            candidates_text += f"{i}. {faq}\n"
        
        # Add FAQ context information for the candidates
        faq_context = "\nFAQ Details:\n"
        
        # Collect all candidate FAQ titles
        candidate_faq_titles = [faq for faq, _ in candidates]
        
        # Look up FAQ descriptions/answers if available
        if 'answer' in self.faqs_df.columns:
            for _, row in self.faqs_df[self.faqs_df['question'].isin(candidate_faq_titles)].iterrows():
                faq_context += f"FAQ: \"{row['question']}\"\n"
                # Truncate very long answers
                answer = row['answer']
                if len(answer) > 200:
                    answer = answer[:200] + "..."
                faq_context += f"Answer: {answer}\n\n"
        
        # Combine all parts into the final prompt
        full_prompt = dimension_prompt + utterance_text + candidates_text + faq_context
        
        return full_prompt
    
    def _parse_standard_judge_response(self, response_text):
        """
        Parse the standard judge's response to extract reranked FAQs.
        
        Args:
            response_text: The text response from the LLM judge
            
        Returns:
            A list of tuples containing (FAQ title, relevance score)
        """
        try:
            # Parse the JSON
            result = json.loads(response_text)
            
            # Extract the reranked FAQs
            reranked_faqs = []
            for item in result.get('ranked_faqs', []):
                faq = item.get('faq', '')
                score = item.get('score', 0)
                reranked_faqs.append((faq, float(score)))
            
            # Sort by score in descending order
            reranked_faqs.sort(key=lambda x: x[1], reverse=True)
            
            return reranked_faqs
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing judge response JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing judge response: {e}")
            return []
    
    def explain_rankings(self, utterance, ranked_faqs):
        """
        Generate a human-readable explanation of why the FAQs were ranked in this order.
        
        Args:
            utterance: The user query
            ranked_faqs: List of (FAQ, score) tuples after reranking
            
        Returns:
            A string containing the explanation
        """
        if not ranked_faqs:
            return "No relevant FAQs were found for your query."
        
        # Create a prompt for the explanation
        prompt = f"""
For the user query: "{utterance}"

You've ranked the following FAQs:
{chr(10).join([f"{i+1}. {faq} - Score: {score:.2f}" for i, (faq, score) in enumerate(ranked_faqs[:5])])}

Please provide a brief, user-friendly explanation of why these FAQs were ranked in this order. 
Explain how they address the user's query and why the top-ranked FAQ is most relevant.
Keep your explanation concise and focused.
"""

        try:
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert in explaining FAQ rankings in a helpful, conversational way."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,  # Higher temperature for more natural language
                max_tokens=300
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "These FAQs have been ranked based on their relevance to your query, with the most directly applicable one listed first."

# Example usage
def main():
    # Load the datasets
    faqs_df = pd.read_csv('aem_faqs.csv')
    test_df = pd.read_csv('test_set.csv')
    
    # Initialize the enhanced judge agent
    judge = EnhancedJudgeAgent(faqs_df, test_df, judge_consistency=True)
    
    # Example: Rerank some candidates
    utterance = "how can I lock my card"
    candidates = [
        ("Lock Card", 90),
        ("Report Lost or Stolen Card", 80),
        ("Card Security Features", 75),
        ("Mobile App Security Features", 70),
        ("Freeze Account", 65),
        ("Change Card PIN", 60),
        ("View Card Details", 55),
        ("Report Fraud", 50)
    ]
    
    # Rerank the candidates
    reranked_faqs = judge.rerank_candidates(utterance, candidates)
    
    print("\nReranked FAQs:")
    for i, (faq, score) in enumerate(reranked_faqs, 1):
        print(f"{i}. {faq} - Score: {score:.2f}")
    
    # Generate an explanation
    explanation = judge.explain_rankings(utterance, reranked_faqs)
    print("\nExplanation:")
    print(explanation)

if __name__ == "__main__":
    main()
