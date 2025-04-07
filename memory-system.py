import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("memory_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AdvancedMemorySystem")

class MemorySystem:
    """
    An advanced memory system for FAQ mapping that enables learning from past interactions
    and self-improvement over time.
    """
    
    def __init__(self, memory_file="faq_mapping_memory.json", embedding_cache_file="memory_embeddings.json"):
        """
        Initialize the memory system.
        
        Args:
            memory_file: Path to the memory file
            embedding_cache_file: Path to the embedding cache file
        """
        self.memory_file = memory_file
        self.embedding_cache_file = embedding_cache_file
        
        # Initialize memory structure
        self.memory = {
            "query_expansions": {},       # Maps original queries to successful expansions
            "successful_mappings": [],    # Historical successful mappings
            "workflow_patterns": {},      # Patterns in successful workflows
            "failed_mappings": [],        # Historical failed mappings
            "agent_performance": {},      # Performance metrics for each agent
            "reranker_performance": {},   # Performance metrics for each reranker
            "domain_knowledge": {},       # Domain-specific knowledge learned
            "query_clusters": {},         # Clusters of similar queries
            "faq_clusters": {},           # Clusters of similar FAQs
            "meta_data": {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_interactions": 0
            }
        }
        
        # Embedding cache for efficient similarity search
        self.embedding_cache = {
            "queries": {},                # Query -> embedding
            "faqs": {}                    # FAQ -> embedding
        }
        
        # Load existing memory if available
        self.load_memory()
        self.load_embedding_cache()
        
        # Initialize TF-IDF vectorizer for text similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Track current session metrics
        self.session_metrics = {
            "start_time": datetime.now().isoformat(),
            "interactions": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "agent_calls": {},
            "reranker_calls": {}
        }
        
        logger.info("Memory system initialized")
    
    def load_memory(self):
        """
        Load memory from disk if available.
        """
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    loaded_memory = json.load(f)
                    # Update memory with loaded data, preserving structure
                    for key in self.memory:
                        if key in loaded_memory:
                            self.memory[key] = loaded_memory[key]
                logger.info(f"Loaded memory with {self.memory['meta_data']['total_interactions']} historical interactions")
            except Exception as e:
                logger.error(f"Error loading memory: {e}")
    
    def save_memory(self):
        """
        Save memory to disk.
        """
        try:
            # Update metadata
            self.memory["meta_data"]["last_updated"] = datetime.now().isoformat()
            
            with open(self.memory_file, 'w') as f:
                json.dump(self.memory, f, indent=2)
            logger.info(f"Saved memory with {self.memory['meta_data']['total_interactions']} interactions")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def load_embedding_cache(self):
        """
        Load embedding cache from disk if available.
        """
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'r') as f:
                    self.embedding_cache = json.load(f)
                logger.info(f"Loaded embedding cache with {len(self.embedding_cache['queries'])} queries and {len(self.embedding_cache['faqs'])} FAQs")
            except Exception as e:
                logger.error(f"Error loading embedding cache: {e}")
    
    def save_embedding_cache(self):
        """
        Save embedding cache to disk.
        """
        try:
            with open(self.embedding_cache_file, 'w') as f:
                json.dump(self.embedding_cache, f)
            logger.info(f"Saved embedding cache with {len(self.embedding_cache['queries'])} queries and {len(self.embedding_cache['faqs'])} FAQs")
        except Exception as e:
            logger.error(f"Error saving embedding cache: {e}")
    
    def record_mapping(self, 
                     original_query: str, 
                     expanded_query: str, 
                     ranked_faqs: List[Tuple[str, float]], 
                     agent_performance: Dict[str, Any], 
                     reranker_performance: Dict[str, Any],
                     workflow_info: Dict[str, Any],
                     user_feedback: Optional[Dict[str, Any]] = None,
                     is_successful: bool = True):
        """
        Record a mapping interaction in memory.
        
        Args:
            original_query: The original user query
            expanded_query: The expanded query after query planning
            ranked_faqs: The list of ranked FAQs as (faq, score) tuples
            agent_performance: Performance data for each agent
            reranker_performance: Performance data for each reranker
            workflow_info: Information about the workflow used
            user_feedback: Optional user feedback about the mapping
            is_successful: Whether the mapping was successful
        """
        # Create a record of the mapping
        mapping_record = {
            "timestamp": datetime.now().isoformat(),
            "original_query": original_query,
            "expanded_query": expanded_query,
            "ranked_faqs": ranked_faqs[:5],  # Store top 5 results
            "agent_performance": agent_performance,
            "reranker_performance": reranker_performance,
            "workflow_info": workflow_info,
            "user_feedback": user_feedback,
            "metrics": {
                "latency": workflow_info.get("processing_time", 0),
                "agent_agreement": self._calculate_agent_agreement(agent_performance),
                "query_expansion_impact": self._calculate_expansion_impact(original_query, expanded_query)
            }
        }
        
        # Record in the appropriate list
        if is_successful:
            self.memory["successful_mappings"].append(mapping_record)
            self.session_metrics["successful_mappings"] += 1
            
            # Update query expansions if this was a successful mapping with expansion
            if original_query != expanded_query:
                self.memory["query_expansions"][original_query] = expanded_query
        else:
            self.memory["failed_mappings"].append(mapping_record)
            self.session_metrics["failed_mappings"] += 1
        
        # Update total interactions counter
        self.memory["meta_data"]["total_interactions"] += 1
        self.session_metrics["interactions"] += 1
        
        # Update agent performance metrics
        for agent_name, metrics in agent_performance.items():
            if agent_name not in self.memory["agent_performance"]:
                self.memory["agent_performance"][agent_name] = {
                    "calls": 0,
                    "successful_calls": 0,
                    "total_latency": 0,
                    "contributions": 0
                }
            
            self.memory["agent_performance"][agent_name]["calls"] += 1
            if metrics.get("success", False):
                self.memory["agent_performance"][agent_name]["successful_calls"] += 1
            self.memory["agent_performance"][agent_name]["total_latency"] += metrics.get("latency", 0)
            self.memory["agent_performance"][agent_name]["contributions"] += metrics.get("contribution", 0)
            
            # Update session metrics
            if agent_name not in self.session_metrics["agent_calls"]:
                self.session_metrics["agent_calls"][agent_name] = 0
            self.session_metrics["agent_calls"][agent_name] += 1
        
        # Update reranker performance metrics
        for reranker_name, metrics in reranker_performance.items():
            if reranker_name not in self.memory["reranker_performance"]:
                self.memory["reranker_performance"][reranker_name] = {
                    "calls": 0,
                    "total_latency": 0,
                    "impact_score": 0
                }
            
            self.memory["reranker_performance"][reranker_name]["calls"] += 1
            self.memory["reranker_performance"][reranker_name]["total_latency"] += metrics.get("latency", 0)
            self.memory["reranker_performance"][reranker_name]["impact_score"] += metrics.get("impact", 0)
            
            # Update session metrics
            if reranker_name not in self.session_metrics["reranker_calls"]:
                self.session_metrics["reranker_calls"][reranker_name] = 0
            self.session_metrics["reranker_calls"][reranker_name] += 1
        
        # Update workflow patterns
        self._update_workflow_patterns(mapping_record)
        
        # Save the updated memory
        if self.session_metrics["interactions"] % 10 == 0:  # Save every 10 interactions
            self.save_memory()
            self.save_embedding_cache()
        
        logger.info(f"Recorded {'successful' if is_successful else 'failed'} mapping for query: '{original_query}'")
    
    def _calculate_agent_agreement(self, agent_performance: Dict[str, Any]) -> float:
        """
        Calculate the level of agreement between agents.
        
        Args:
            agent_performance: Performance data for each agent
            
        Returns:
            A float representing the agreement level (0-1)
        """
        # Extract the top prediction from each agent
        top_predictions = {}
        for agent_name, metrics in agent_performance.items():
            if "predictions" in metrics and metrics["predictions"]:
                top_predictions[agent_name] = metrics["predictions"][0][0]  # First FAQ
        
        if not top_predictions:
            return 0.0
        
        # Count how many agents agree on their top prediction
        prediction_counts = {}
        for faq in top_predictions.values():
            prediction_counts[faq] = prediction_counts.get(faq, 0) + 1
        
        # Get the max agreement count
        max_agreement = max(prediction_counts.values()) if prediction_counts else 0
        
        # Normalize by number of agents
        return max_agreement / len(top_predictions) if top_predictions else 0.0
    
    def _calculate_expansion_impact(self, original_query: str, expanded_query: str) -> float:
        """
        Calculate the impact of query expansion.
        
        Args:
            original_query: The original user query
            expanded_query: The expanded query after query planning
            
        Returns:
            A float representing the expansion impact (0-1)
        """
        if original_query == expanded_query:
            return 0.0
        
        # Calculate the ratio of new terms added
        original_terms = set(original_query.lower().split())
        expanded_terms = set(expanded_query.lower().split())
        
        new_terms = expanded_terms - original_terms
        return len(new_terms) / len(expanded_terms) if expanded_terms else 0.0
    
    def _update_workflow_patterns(self, mapping_record: Dict[str, Any]):
        """
        Update workflow patterns based on the current mapping.
        
        Args:
            mapping_record: The record of the current mapping
        """
        # Extract workflow information
        workflow_info = mapping_record["workflow_info"]
        
        # Create a pattern key
        pattern_key = f"expansion={'yes' if mapping_record['original_query'] != mapping_record['expanded_query'] else 'no'}_" \
                      f"agents={len(mapping_record['agent_performance'])}_" \
                      f"top_faq={mapping_record['ranked_faqs'][0][0] if mapping_record['ranked_faqs'] else 'none'}"
        
        # Update the count for this pattern
        if pattern_key in self.memory["workflow_patterns"]:
            self.memory["workflow_patterns"][pattern_key]["count"] += 1
            
            # Update success metrics if available
            if "success_score" in workflow_info:
                self.memory["workflow_patterns"][pattern_key]["success_scores"].append(workflow_info["success_score"])
                self.memory["workflow_patterns"][pattern_key]["avg_success_score"] = sum(self.memory["workflow_patterns"][pattern_key]["success_scores"]) / len(self.memory["workflow_patterns"][pattern_key]["success_scores"])
        else:
            self.memory["workflow_patterns"][pattern_key] = {
                "count": 1,
                "success_scores": [workflow_info.get("success_score", 0)],
                "avg_success_score": workflow_info.get("success_score", 0),
                "first_seen": datetime.now().isoformat()
            }
    
    def get_query_expansion(self, query: str) -> Optional[str]:
        """
        Get a previously successful query expansion for a similar query.
        
        Args:
            query: The user query
            
        Returns:
            An expanded query if available, None otherwise
        """
        # Exact match
        if query in self.memory["query_expansions"]:
            return self.memory["query_expansions"][query]
        
        # Find similar queries using TF-IDF similarity
        similar_query = self.find_similar_query(query)
        if similar_query and similar_query in self.memory["query_expansions"]:
            return self.memory["query_expansions"][similar_query]
        
        return None
    
    def find_similar_query(self, query: str, similarity_threshold: float = 0.8) -> Optional[str]:
        """
        Find a similar query in memory.
        
        Args:
            query: The user query
            similarity_threshold: The minimum similarity score (0-1)
            
        Returns:
            A similar query if found above the threshold, None otherwise
        """
        if not self.memory["query_expansions"]:
            return None
        
        # Get all queries with expansions
        stored_queries = list(self.memory["query_expansions"].keys())
        
        try:
            # Fit TF-IDF vectorizer on stored queries
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(stored_queries)
            
            # Transform the input query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
            
            # Find the most similar query
            max_index = np.argmax(similarities)
            max_similarity = similarities[max_index]
            
            if max_similarity >= similarity_threshold:
                return stored_queries[max_index]
        except Exception as e:
            logger.error(f"Error finding similar query: {e}")
        
        return None
    
    def get_embedding(self, text: str, refresh: bool = False, model: str = "text-embedding-3-large") -> List[float]:
        """
        Get embedding for a text, using cache if available.
        
        Args:
            text: The text to get an embedding for
            refresh: Whether to refresh the cached embedding
            model: The embedding model to use
            
        Returns:
            The embedding vector
        """
        # Normalize the text
        normalized_text = text.strip().lower()
        
        # Check if we have the embedding in cache and don't need to refresh
        if not refresh and normalized_text in self.embedding_cache["queries"]:
            return self.embedding_cache["queries"][normalized_text]
        
        try:
            # Get embedding from OpenAI
            response = openai.Embedding.create(
                model=model,
                input=normalized_text
            )
            embedding = response['data'][0]['embedding']
            
            # Cache the embedding
            self.embedding_cache["queries"][normalized_text] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 1536  # Default embedding size for text-embedding-3-large
    
    def find_similar_embeddings(self, text: str, collection: str = "queries", top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar texts based on embedding similarity.
        
        Args:
            text: The text to find similar texts for
            collection: Which collection to search in ("queries" or "faqs")
            top_k: Number of similar texts to return
            
        Returns:
            A list of (text, similarity_score) tuples
        """
        if collection not in ["queries", "faqs"]:
            logger.error(f"Invalid collection: {collection}")
            return []
        
        # Get embedding for the input text
        query_embedding = self.get_embedding(text)
        
        # Calculate similarities with all cached embeddings
        similarities = []
        for cached_text, cached_embedding in self.embedding_cache[collection].items():
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(query_embedding, cached_embedding))
            magnitude1 = sum(a * a for a in query_embedding) ** 0.5
            magnitude2 = sum(b * b for b in cached_embedding) ** 0.5
            similarity = dot_product / (magnitude1 * magnitude2) if magnitude1 * magnitude2 > 0 else 0
            
            similarities.append((cached_text, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_best_performing_agents(self) -> Dict[str, float]:
        """
        Get the best performing agents based on historical data.
        
        Returns:
            A dictionary mapping agent names to performance scores
        """
        agent_scores = {}
        
        for agent_name, metrics in self.memory["agent_performance"].items():
            # Calculate success rate
            success_rate = metrics["successful_calls"] / metrics["calls"] if metrics["calls"] > 0 else 0
            
            # Calculate average latency
            avg_latency = metrics["total_latency"] / metrics["calls"] if metrics["calls"] > 0 else 0
            
            # Calculate overall score (higher is better)
            # Normalize latency to 0-1 range (1 being fastest)
            norm_latency = 1.0 / (1.0 + avg_latency) if avg_latency > 0 else 1.0
            
            # Calculate contribution score (how often this agent's results were used)
            contribution_score = metrics["contributions"] / metrics["calls"] if metrics["calls"] > 0 else 0
            
            # Combined score (weighted average)
            score = 0.4 * success_rate + 0.2 * norm_latency + 0.4 * contribution_score
            
            agent_scores[agent_name] = score
        
        return agent_scores
    
    def get_best_performing_rerankers(self) -> Dict[str, float]:
        """
        Get the best performing rerankers based on historical data.
        
        Returns:
            A dictionary mapping reranker names to performance scores
        """
        reranker_scores = {}
        
        for reranker_name, metrics in self.memory["reranker_performance"].items():
            # Calculate average latency
            avg_latency = metrics["total_latency"] / metrics["calls"] if metrics["calls"] > 0 else 0
            
            # Calculate average impact
            avg_impact = metrics["impact_score"] / metrics["calls"] if metrics["calls"] > 0 else 0
            
            # Calculate overall score (higher is better)
            # Normalize latency to 0-1 range (1 being fastest)
            norm_latency = 1.0 / (1.0 + avg_latency) if avg_latency > 0 else 1.0
            
            # Combined score (weighted average)
            score = 0.3 * norm_latency + 0.7 * avg_impact
            
            reranker_scores[reranker_name] = score
        
        return reranker_scores
    
    def get_recommended_workflow(self, query: str) -> Dict[str, Any]:
        """
        Get a recommended workflow for a query based on historical performance.
        
        Args:
            query: The user query
            
        Returns:
            A dictionary containing the recommended workflow
        """
        # Find similar queries in memory
        similar_mappings = []
        
        for mapping in self.memory["successful_mappings"]:
            original_query = mapping["original_query"]
            # Use TF-IDF similarity for simple comparison
            query_terms = set(query.lower().split())
            original_terms = set(original_query.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(query_terms.intersection(original_terms))
            union = len(query_terms.union(original_terms))
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.5:  # Threshold for similarity
                similar_mappings.append((mapping, similarity))
        
        if not similar_mappings:
            # No similar mappings found, return default workflow
            return {
                "recommended_agents": list(self.get_best_performing_agents().keys()),
                "recommended_rerankers": list(self.get_best_performing_rerankers().keys()),
                "expected_latency": 0.0,
                "confidence": "LOW",
                "reason": "No similar queries found in memory"
            }
        
        # Sort by similarity
        similar_mappings.sort(key=lambda x: x[1], reverse=True)
        
        # Get the most successful workflows
        top_mappings = similar_mappings[:5]
        
        # Count the frequency of each agent and reranker in top mappings
        agent_counts = {}
        reranker_counts = {}
        total_latency = 0
        
        for mapping, _ in top_mappings:
            # Count agents
            for agent_name in mapping["agent_performance"].keys():
                agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
            
            # Count rerankers
            for reranker_name in mapping["reranker_performance"].keys():
                reranker_counts[reranker_name] = reranker_counts.get(reranker_name, 0) + 1
            
            # Sum latency
            total_latency += mapping["metrics"]["latency"]
        
        # Calculate average latency
        avg_latency = total_latency / len(top_mappings)
        
        # Get the most frequent agents and rerankers
        recommended_agents = [agent for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)]
        recommended_rerankers = [reranker for reranker, count in sorted(reranker_counts.items(), key=lambda x: x[1], reverse=True)]
        
        # Calculate confidence based on similarity and number of samples
        avg_similarity = sum(sim for _, sim in top_mappings) / len(top_mappings)
        confidence = "HIGH" if avg_similarity > 0.8 and len(top_mappings) >= 3 else "MEDIUM" if avg_similarity > 0.6 else "LOW"
        
        return {
            "recommended_agents": recommended_agents,
            "recommended_rerankers": recommended_rerankers,
            "expected_latency": avg_latency,
            "confidence": confidence,
            "reason": f"Based on {len(top_mappings)} similar queries with average similarity {avg_similarity:.2f}"
        }
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate insights from the memory.
        
        Returns:
            A dictionary containing insights from memory
        """
        if not self.memory["successful_mappings"]:
            return {"status": "Not enough data to generate insights"}
        
        # Calculate overall system performance
        total_mappings = len(self.memory["successful_mappings"]) + len(self.memory["failed_mappings"])
        success_rate = len(self.memory["successful_mappings"]) / total_mappings if total_mappings > 0 else 0
        
        # Calculate average latency
        all_latencies = [mapping["metrics"]["latency"] for mapping in self.memory["successful_mappings"]]
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        
        # Identify common patterns in successful mappings
        top_patterns = sorted(
            [(pattern, data["count"]) for pattern, data in self.memory["workflow_patterns"].items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Identify frequent queries
        query_counts = {}
        for mapping in self.memory["successful_mappings"]:
            query = mapping["original_query"]
            query_counts[query] = query_counts.get(query, 0) + 1
        
        top_queries = sorted(
            [(query, count) for query, count in query_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Identify most frequently returned FAQs
        faq_counts = {}
        for mapping in self.memory["successful_mappings"]:
            for faq, _ in mapping["ranked_faqs"]:
                faq_counts[faq] = faq_counts.get(faq, 0) + 1
        
        top_faqs = sorted(
            [(faq, count) for faq, count in faq_counts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        # Calculate agent and reranker performance
        best_agents = self.get_best_performing_agents()
        best_rerankers = self.get_best_performing_rerankers()
        
        return {
            "status": "success",
            "system_performance": {
                "total_mappings": total_mappings,
                "success_rate": success_rate,
                "avg_latency": avg_latency
            },
            "top_patterns": top_patterns,
            "top_queries": top_queries,
            "top_faqs": top_faqs,
            "best_agents": best_agents,
            "best_rerankers": best_rerankers,
            "session_metrics": self.session_metrics
        }
    
    def clear_memory(self, confirm: bool = False):
        """
        Clear the memory.
        
        Args:
            confirm: Whether to confirm the clearing
        """
        if not confirm:
            logger.warning("Memory clearing aborted. Set confirm=True to clear memory.")
            return
        
        # Reset memory
        self.memory = {
            "query_expansions": {},
            "successful_mappings": [],
            "workflow_patterns": {},
            "failed_mappings": [],
            "agent_performance": {},
            "reranker_performance": {},
            "domain_knowledge": {},
            "query_clusters": {},
            "faq_clusters": {},
            "meta_data": {
                "last_updated": datetime.now().isoformat(),
                "version": "1.0.0",
                "total_interactions": 0
            }
        }
        
        # Reset embedding cache
        self.embedding_cache = {
            "queries": {},
            "faqs": {}
        }
        
        # Save empty memory
        self.save_memory()
        self.save_embedding_cache()
        
        logger.info("Memory cleared")

# Example usage
def main():
    # Initialize memory system
    memory_system = MemorySystem()
    
    # Example: Record a mapping
    memory_system.record_mapping(
        original_query="how do I lock my card",
        expanded_query="how can I lock or freeze my credit card",
        ranked_faqs=[
            ("Lock Card", 95.5),
            ("Report Lost or Stolen Card", 85.2),
            ("Card Security Features", 75.8),
            ("Mobile App Security Features", 65.3),
            ("Freeze Account", 60.0)
        ],
        agent_performance={
            "Basic_Emb": {
                "latency": 0.35,
                "success": True,
                "predictions": [("Lock Card", 90.0), ("Report Lost or Stolen Card", 85.2)],
                "contribution": 0.8
            },
            "Enhanced_Emb": {
                "latency": 0.72,
                "success": True,
                "predictions": [("Lock Card", 92.5), ("Card Security Features", 80.3)],
                "contribution": 0.9
            }
        },
        reranker_performance={
            "Semantic": {
                "latency": 0.15,
                "impact": 0.7
            },
            "Intent": {
                "latency": 0.25,
                "impact": 0.8
            }
        },
        workflow_info={
            "processing_time": 1.5,
            "success_score": 0.9
        }
    )
    
    # Example: Get insights
    insights = memory_system.generate_insights()
    print("Memory Insights:")
    print(json.dumps(insights, indent=2))
    
    # Example: Get recommended workflow
    workflow = memory_system.get_recommended_workflow("how to lock my debit card")
    print("\nRecommended Workflow:")
    print(json.dumps(workflow, indent=2))

if __name__ == "__main__":
    main()
