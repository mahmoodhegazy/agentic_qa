import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import tqdm
import openai
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from datetime import datetime

# Import EnhancedJudgeAgent instead of using internal JudgeAgent
from enhanced_judge_agent import EnhancedJudgeAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("faq_mapper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EnhancedFAQMapper")

# Set up the OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

class EnhancedMultiAgentFAQMapper:
    """
    An enhanced multi-agent framework for FAQ mapping that combines multiple approaches
    and uses a sophisticated reranking and judging system.
    """
    
    def __init__(self, faqs_df, test_df, use_memory=True, use_self_improvement=True, judge_agent=None, memory_system=None):
        """
        Initialize the enhanced multi-agent FAQ mapper.
        
        Args:
            faqs_df: DataFrame containing the FAQs with 'question' and 'answer' columns
            test_df: DataFrame containing the test utterances, FAQs, and ranks
            use_memory: Whether to use memory for improving over time
            use_self_improvement: Whether to use self-improvement mechanisms
            judge_agent: Optional external judge agent instance to use
            memory_system: Optional external memory system instance to use
        """
        self.faqs_df = faqs_df
        self.test_df = test_df
        self.use_memory = use_memory
        self.use_self_improvement = use_self_improvement
        self.external_judge = judge_agent
        self.external_memory = memory_system
        
        # Initialize the agent network
        self.agents = []
        self.rerankers = []
        self.agent_weights = {}  # For potential weighted ensemble
        
        # Internal memory if no external system provided
        if not self.external_memory:
            self.memory = {
                'successful_mappings': [],
                'workflow_patterns': {},
                'performance_metrics': {}
            }
        
        # Initialize the agents and components
        self.initialize_components()
        
        logger.info(f"Initialized EnhancedMultiAgentFAQMapper with {len(self.agents)} agents and {len(self.rerankers)} rerankers")
    
    def initialize_components(self):
        """
        Initialize all the components of the enhanced multi-agent FAQ mapper.
        """
        logger.info("Initializing system components...")
        
        # Initialize the specialized retrieval agents
        self._initialize_retrieval_agents()
        
        # Initialize the rerankers
        self._initialize_rerankers()
        
        # Initialize the judge component
        self._initialize_judge()
        
        # Initialize the memory system if enabled
        if self.use_memory:
            self._initialize_memory_system()
        
        # Initialize the query planning agent
        self._initialize_query_planner()
        
        # Initialize the response generation agent
        self._initialize_response_generator()
        
        # Set initial weights for all agents
        self._initialize_weights()
    
    def _initialize_retrieval_agents(self):
        """
        Initialize the specialized retrieval agents.
        """
        logger.info("Initializing retrieval agents...")
        
        # Import specialized agents from your existing implementation
        from faq_mapper_implementation import FAQMapper
        from enhanced_faq_mapper import EnhancedFAQMapper
        from enhanced_faq_mapper_with_answers import EnhancedFAQMapperWithAnswers
        
        # Create and register the agents
        # Basic FAQ mapper with embeddings
        basic_emb_agent = FAQMapper(self.faqs_df, self.test_df, use_embeddings=True)
        self.agents.append(("Basic_Emb", basic_emb_agent))
        
        # Enhanced FAQ mapper with embeddings
        enhanced_emb_agent = EnhancedFAQMapper(self.faqs_df, self.test_df, use_embeddings=True)
        self.agents.append(("Enhanced_Emb", enhanced_emb_agent))
        
        # Enhanced FAQ mapper with answer context
        enhanced_ans_agent = EnhancedFAQMapperWithAnswers(self.faqs_df, self.test_df, use_embeddings=True)
        self.agents.append(("Enhanced_Ans", enhanced_ans_agent))
    
    def _initialize_rerankers(self):
        """
        Initialize the specialized rerankers.
        """
        logger.info("Initializing specialized rerankers...")
        
        # Create semantic similarity reranker
        self.rerankers.append(("Semantic", self.semantic_reranker))
        
        # Create intent matching reranker
        self.rerankers.append(("Intent", self.intent_reranker))
        
        # Create relevance reranker
        self.rerankers.append(("Relevance", self.relevance_reranker))
    
    def _initialize_judge(self):
        """
        Initialize the enhanced judge component.
        """
        logger.info("Initializing enhanced judge component...")
        
        # Use provided judge agent or create a new one
        if self.external_judge:
            self.judge = self.external_judge
            logger.info("Using external judge agent")
        else:
            # Create the enhanced judge component
            self.judge = EnhancedJudgeAgent(
                faqs_df=self.faqs_df, 
                test_df=self.test_df,
                judge_consistency=self.use_self_improvement
            )
            logger.info("Created new judge agent")
    
    def _initialize_memory_system(self):
        """
        Initialize the memory system for tracking and learning from successful mappings.
        """
        logger.info("Initializing memory system...")
        
        # External memory system is already initialized in __init__
        if not self.external_memory and self.use_memory:
            # Load existing memory if available
            memory_path = "faq_mapping_memory.json"
            if os.path.exists(memory_path):
                try:
                    with open(memory_path, 'r') as f:
                        self.memory = json.load(f)
                    logger.info(f"Loaded {len(self.memory['successful_mappings'])} previous mappings from memory")
                except Exception as e:
                    logger.error(f"Error loading memory: {e}")
    
    def _initialize_query_planner(self):
        """
        Initialize the query planning agent.
        """
        logger.info("Initializing query planning agent...")
        
        # Create the query planning agent
        self.query_planner = QueryPlanningAgent()
    
    def _initialize_response_generator(self):
        """
        Initialize the response generation agent.
        """
        logger.info("Initializing response generation agent...")
        
        # Create the response generation agent
        self.response_generator = ResponseGenerationAgent(self.faqs_df)
    
    def _initialize_weights(self):
        """
        Initialize weights for all agents and rerankers.
        """
        # Initialize equal weights for all agents
        for name, _ in self.agents:
            self.agent_weights[name] = 1.0
        
        # Initialize equal weights for all rerankers
        self.reranker_weights = {}
        for name, _ in self.rerankers:
            self.reranker_weights[name] = 1.0
    
    def map_utterance(self, utterance, return_details=False):
        """
        Map a user utterance to FAQs using the enhanced multi-agent approach.
        
        Args:
            utterance: The user query to map to FAQs
            return_details: Whether to return detailed information about the mapping process
            
        Returns:
            A list of tuples containing (FAQ title, relevance score) if return_details is False
            A dictionary containing detailed mapping information if return_details is True
        """
        start_time = time.time()
        self.start_time = start_time  # Store for memory system
        
        # 1. Query Planning
        logger.info(f"Planning query: '{utterance}'")
        planning_result = self.query_planner.plan_query(utterance)
        expanded_query = planning_result.get('expanded_query', utterance)
        agent_selection = planning_result.get('selected_agents', None)
        logger.info(f"Expanded query: '{expanded_query}'")
        
        # 2. Retrieval from specialized agents
        all_candidates = []
        agent_predictions = {}
        
        logger.info(f"Getting predictions for: '{expanded_query}'")
        for agent_name, agent in self.agents:
            # Skip if agent selection is provided and this agent is not selected
            if agent_selection and agent_name not in agent_selection:
                continue
                
            logger.info(f"  Agent: {agent_name}")
            
            try:
                if agent_name.startswith("Enhanced_Ans"):
                    # Enhanced FAQ mapper with answers
                    mapping_result = agent.map_utterance_with_answers(expanded_query)
                    candidates = agent.extract_ranked_faqs(mapping_result)
                elif agent_name.startswith("Enhanced"):
                    # Enhanced FAQ mapper
                    mapping_result = agent.map_utterance(expanded_query)
                    candidates = agent.extract_ranked_faqs(mapping_result)
                else:
                    # Basic FAQ mapper
                    api_response = agent.get_faq_mapping_embeddings(expanded_query) if "Emb" in agent_name else agent.get_faq_mapping_direct(expanded_query)
                    candidates = agent.extract_ranked_faqs(api_response)
                
                # Apply agent weights to scores
                weight = self.agent_weights.get(agent_name, 1.0)
                weighted_candidates = [(faq, score * weight) for faq, score in candidates]
                
                # Store the agent's predictions
                agent_predictions[agent_name] = candidates
                
                # Add agent's candidates to the overall pool
                all_candidates.extend(weighted_candidates)
                
                logger.info(f"    Found {len(candidates)} candidates")
            except Exception as e:
                logger.error(f"    Error getting predictions from {agent_name}: {e}")
                # Continue with other agents if one fails
                continue
        
        # If we don't have any candidates, return an empty list
        if not all_candidates:
            logger.warning("  No candidates found from any agent")
            if return_details:
                return {
                    'ranked_faqs': [],
                    'processing_time': time.time() - start_time,
                    'query_planning': planning_result,
                    'agent_predictions': agent_predictions,
                    'reranker_results': {},
                    'judge_results': None
                }
            else:
                return []
        
        # 3. Multi-Reranker System
        reranker_results = {}
        all_reranked = []
        
        # Deduplicate candidates
        unique_candidates = []
        seen = set()
        for faq, score in all_candidates:
            if faq not in seen:
                unique_candidates.append((faq, score))
                seen.add(faq)
        
        logger.info(f"Running {len(self.rerankers)} rerankers on {len(unique_candidates)} unique candidates")
        for reranker_name, reranker_func in self.rerankers:
            try:
                reranked = reranker_func(expanded_query, unique_candidates)
                reranker_results[reranker_name] = reranked
                
                # Apply reranker weight
                weight = self.reranker_weights.get(reranker_name, 1.0)
                weighted_reranked = [(faq, score * weight) for faq, score in reranked]
                all_reranked.extend(weighted_reranked)
                
                logger.info(f"  Reranker {reranker_name} completed")
            except Exception as e:
                logger.error(f"  Error in reranker {reranker_name}: {e}")
        
        # 4. Judge Agent for final reranking
        logger.info("Running judge for final reranking")
        try:
            judge_results = self.judge.rerank_candidates(
                expanded_query, 
                all_reranked, 
                agent_predictions
            )
            final_ranked_faqs = judge_results
            logger.info(f"  Judge completed with {len(final_ranked_faqs)} ranked FAQs")
        except Exception as e:
            logger.error(f"  Error in judge: {e}")
            # Fall back to simple aggregation if the judge fails
            faq_scores = {}
            for faq, score in all_reranked:
                if faq in faq_scores:
                    faq_scores[faq] += score
                else:
                    faq_scores[faq] = score
            
            final_ranked_faqs = sorted(faq_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"  Fallback to aggregation: {len(final_ranked_faqs)} ranked FAQs")
        
        # 5. Response Generation
        try:
            if final_ranked_faqs:
                top_faq, top_score = final_ranked_faqs[0]
                response_info = self.response_generator.generate_response(
                    utterance, 
                    top_faq, 
                    self.faqs_df[self.faqs_df['question'] == top_faq].iloc[0]['answer'] if 'answer' in self.faqs_df.columns else None
                )
                logger.info(f"Generated response for top FAQ: {top_faq}")
            else:
                response_info = {"response": "I couldn't find a relevant FAQ for your query."}
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            response_info = {"response": "An error occurred while generating the response."}
        
        # Calculate the processing time
        processing_time = time.time() - start_time
        
        # 6. Update Memory System if enabled
        if self.use_memory and final_ranked_faqs:
            self._update_memory(
                utterance, expanded_query, final_ranked_faqs, 
                agent_predictions, reranker_results, 
                processing_time=processing_time
            )
        
        # Return the results
        logger.info(f"Total processing time: {processing_time:.2f} seconds")
        
        if return_details:
            return {
                'ranked_faqs': final_ranked_faqs[:5],  # Return top 5
                'processing_time': processing_time,
                'query_planning': planning_result,
                'agent_predictions': agent_predictions,
                'reranker_results': reranker_results,
                'judge_results': judge_results,
                'response': response_info
            }
        else:
            return final_ranked_faqs[:5]  # Return top 5
    
    def _update_memory(self, original_query, expanded_query, final_ranked_faqs, agent_predictions, reranker_results, processing_time=None):
        """
        Update the memory system with the current mapping.
        
        Args:
            original_query: The original user query
            expanded_query: The expanded query after query planning
            final_ranked_faqs: The final ranked FAQs
            agent_predictions: The predictions from each agent
            reranker_results: The results from each reranker
            processing_time: The processing time for this mapping
        """
        # Use provided processing time or calculate from start_time
        if processing_time is None:
            processing_time = time.time() - self.start_time if hasattr(self, 'start_time') else 0
        
        if self.external_memory:
            # Use the external memory system
            agent_performance_data = {}
            for agent_name, predictions in agent_predictions.items():
                agent_performance_data[agent_name] = {
                    "latency": 0.0,  # Would need to track per-agent timing
                    "success": len(predictions) > 0,
                    "predictions": predictions,
                    "contribution": 1.0 if any(faq in [f for f, _ in final_ranked_faqs[:3]] for faq, _ in predictions[:3]) else 0.0
                }
            
            reranker_performance_data = {}
            for reranker_name, results in reranker_results.items():
                reranker_performance_data[reranker_name] = {
                    "latency": 0.0,  # Would need to track per-reranker timing
                    "impact": 0.5  # Placeholder value
                }
            
            self.external_memory.record_mapping(
                original_query=original_query,
                expanded_query=expanded_query,
                ranked_faqs=final_ranked_faqs[:5],  # Store top 5 results
                agent_performance=agent_performance_data,
                reranker_performance=reranker_performance_data,
                workflow_info={
                    'processing_time': processing_time,
                    'query_expansion_successful': original_query != expanded_query,
                    'agent_agreement': self._calculate_agent_agreement(agent_predictions),
                    'reranking_impact': self._calculate_reranking_impact(final_ranked_faqs, list(agent_predictions.values()))
                }
            )
        else:
            # Use the internal memory dictionary
            # Original code for updating internal memory
            mapping_record = {
                'timestamp': datetime.now().isoformat(),
                'original_query': original_query,
                'expanded_query': expanded_query,
                'final_ranked_faqs': final_ranked_faqs[:5],  # Store top 5 results
                'best_performing_agents': self._identify_best_agents(final_ranked_faqs, agent_predictions),
                'workflow_pattern': {
                    'query_expansion_successful': original_query != expanded_query,
                    'agent_agreement': self._calculate_agent_agreement(agent_predictions),
                    'reranking_impact': self._calculate_reranking_impact(final_ranked_faqs, all_candidates=list(agent_predictions.values()))
                }
            }
            
            # Add to memory
            self.memory['successful_mappings'].append(mapping_record)
            
            # Update workflow patterns
            self._update_workflow_patterns(mapping_record)
            
            # Save memory to disk
            try:
                with open("faq_mapping_memory.json", 'w') as f:
                    json.dump(self.memory, f)
            except Exception as e:
                logger.error(f"Error saving memory: {e}")
    
    def _identify_best_agents(self, final_ranked_faqs, agent_predictions):
        """
        Identify which agents performed best by comparing their predictions to the final results.
        
        Args:
            final_ranked_faqs: The final ranked FAQs
            agent_predictions: The predictions from each agent
            
        Returns:
            A dictionary mapping agent names to performance scores
        """
        best_agents = {}
        
        # Extract the top 3 final FAQs
        top_final_faqs = [faq for faq, _ in final_ranked_faqs[:3]]
        
        # Calculate how many of the top final FAQs each agent predicted
        for agent_name, predictions in agent_predictions.items():
            agent_faqs = [faq for faq, _ in predictions]
            overlap = sum(1 for faq in top_final_faqs if faq in agent_faqs)
            best_agents[agent_name] = overlap / max(len(top_final_faqs), 1)  # Normalize
        
        return best_agents
    
    def _calculate_agent_agreement(self, agent_predictions):
        """
        Calculate the level of agreement between agents.
        
        Args:
            agent_predictions: The predictions from each agent
            
        Returns:
            A float representing the agreement level (0-1)
        """
        if not agent_predictions:
            return 0.0
        
        # Extract the top prediction from each agent
        top_predictions = {}
        for agent_name, predictions in agent_predictions.items():
            if predictions:
                top_predictions[agent_name] = predictions[0][0]  # First FAQ from first prediction
        
        # Count how many agents agree on their top prediction
        prediction_counts = {}
        for faq in top_predictions.values():
            prediction_counts[faq] = prediction_counts.get(faq, 0) + 1
        
        # Get the max agreement count
        max_agreement = max(prediction_counts.values()) if prediction_counts else 0
        
        # Normalize by number of agents
        return max_agreement / len(top_predictions) if top_predictions else 0.0
    
    def _calculate_reranking_impact(self, final_ranked_faqs, all_candidates):
        """
        Calculate the impact of reranking on the final results.
        
        Args:
            final_ranked_faqs: The final ranked FAQs
            all_candidates: List of lists of candidates from all agents
            
        Returns:
            A float representing the reranking impact (0-1)
        """
        # Flatten and deduplicate all candidates
        all_faqs = set()
        for candidates in all_candidates:
            for faq, _ in candidates:
                all_faqs.add(faq)
        
        # Check how many of the final top 3 FAQs were not top 3 in any of the original agent results
        top_final_faqs = [faq for faq, _ in final_ranked_faqs[:3]]
        
        original_top_faqs = set()
        for candidates in all_candidates:
            for faq, _ in candidates[:3]:  # Top 3 from each agent
                original_top_faqs.add(faq)
        
        # Count how many final top FAQs were not in the original top FAQs
        new_faqs = sum(1 for faq in top_final_faqs if faq not in original_top_faqs)
        
        # Normalize
        return new_faqs / max(len(top_final_faqs), 1)
    
    def _update_workflow_patterns(self, mapping_record):
        """
        Update the workflow patterns based on the current mapping.
        
        Args:
            mapping_record: The record of the current mapping
        """
        # Extract the workflow pattern from the mapping record
        pattern = mapping_record['workflow_pattern']
        
        # Convert to a string key for storage
        pattern_key = f"expansion={'yes' if pattern['query_expansion_successful'] else 'no'}_" \
                      f"agreement={'high' if pattern['agent_agreement'] > 0.5 else 'low'}_" \
                      f"reranking={'high' if pattern['reranking_impact'] > 0.3 else 'low'}"
        
        # Update the count for this pattern
        if pattern_key in self.memory['workflow_patterns']:
            self.memory['workflow_patterns'][pattern_key] += 1
        else:
            self.memory['workflow_patterns'][pattern_key] = 1
    
    def semantic_reranker(self, query, candidates):
        """
        Rerank candidates based on semantic similarity.
        
        Args:
            query: The user query
            candidates: List of (FAQ, score) tuples
            
        Returns:
            Reranked list of (FAQ, score) tuples
        """
        try:
            # Get embedding for the query
            query_embedding = self._get_embedding(query)
            
            # Get embeddings for the FAQs
            faq_embeddings = []
            faqs = []
            for faq, _ in candidates:
                faqs.append(faq)
                faq_embedding = self._get_embedding(faq)
                faq_embeddings.append(faq_embedding)
            
            # Calculate cosine similarity between query and FAQ embeddings
            faq_embeddings = np.array(faq_embeddings)
            similarities = cosine_similarity([query_embedding], faq_embeddings)[0]
            
            # Combine with original scores and rerank
            reranked = [(faqs[i], float(similarities[i]) * 100) for i in range(len(faqs))]
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
        except Exception as e:
            logger.error(f"Error in semantic reranker: {e}")
            return candidates
    
    def intent_reranker(self, query, candidates):
        """
        Rerank candidates based on intent matching using an LLM.
        
        Args:
            query: The user query
            candidates: List of (FAQ, score) tuples
            
        Returns:
            Reranked list of (FAQ, score) tuples
        """
        if not candidates:
            return []
        
        try:
            # Create a prompt for the LLM to analyze intent matching
            prompt = f"""
            Analyze the following user query and rate how well each FAQ matches the user's intent on a scale of 0-100:
            
            User Query: "{query}"
            
            FAQs to rate:
            {chr(10).join([f"{i+1}. {faq}" for i, (faq, _) in enumerate(candidates)])}
            
            For each FAQ, provide:
            1. A relevance score (0-100) based on how well it matches the user's intent
            2. A brief explanation of your rating
            
            Format your response as a JSON object with FAQ titles as keys and objects containing 'score' and 'explanation' as values.
            """
            
            # Call the LLM
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing user intent and matching it to FAQs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Extract scores and rerank
            reranked = []
            for faq, _ in candidates:
                # Find the matching FAQ in the result
                for key in result:
                    if faq in key or key in faq:  # Fuzzy matching
                        score = result[key].get('score', 0)
                        reranked.append((faq, float(score)))
                        break
                else:
                    # If no match found, keep original score
                    reranked.append((faq, 0))
            
            # Sort by score
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
        except Exception as e:
            logger.error(f"Error in intent reranker: {e}")
            return candidates
    
    def relevance_reranker(self, query, candidates):
        """
        Rerank candidates based on relevance to the query.
        
        Args:
            query: The user query
            candidates: List of (FAQ, score) tuples
            
        Returns:
            Reranked list of (FAQ, score) tuples
        """
        if not candidates or 'answer' not in self.faqs_df.columns:
            return candidates
        
        try:
            # Create list of (FAQ, Answer) pairs
            faq_answers = []
            for faq, _ in candidates:
                answer = self.faqs_df[self.faqs_df['question'] == faq].iloc[0]['answer'] if not self.faqs_df[self.faqs_df['question'] == faq].empty else ""
                faq_answers.append((faq, answer))
            
            # Create a prompt for the relevance reranker
            prompt = f"""
            Rate how well each FAQ and its answer addresses the user's query on a scale of 0-100:
            
            User Query: "{query}"
            
            FAQs to rate:
            """
            
            for i, (faq, answer) in enumerate(faq_answers):
                prompt += f"\n{i+1}. Question: {faq}\n   Answer: {answer[:200]}{'...' if len(answer) > 200 else ''}\n"
            
            prompt += """
            For each FAQ, provide a relevance score (0-100) based on how well the question and answer address the user's query.
            
            Format your response as a JSON object with FAQ numbers as keys and scores as values.
            """
            
            # Call the LLM
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at evaluating how well FAQs address user queries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # Extract scores and rerank
            reranked = []
            for i, (faq, _) in enumerate(candidates):
                score = result.get(str(i+1), 0)
                reranked.append((faq, float(score)))
            
            # Sort by score
            reranked.sort(key=lambda x: x[1], reverse=True)
            
            return reranked
        except Exception as e:
            logger.error(f"Error in relevance reranker: {e}")
            return candidates
    
    def _get_embedding(self, text, model="text-embedding-3-large"):
        """
        Get embedding for a text using OpenAI's embedding API.
        
        Args:
            text: The text to get an embedding for
            model: The embedding model to use
            
        Returns:
            The embedding as a numpy array
        """
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        return response['data'][0]['embedding']
    
    def process_dataset(self, dataset_df, batch_size=10, save_path='enhanced_multi_agent_faq_mappings.csv'):
        """
        Process a dataset of utterances to map them to FAQs using the enhanced multi-agent approach.
        
        Args:
            dataset_df: DataFrame containing the utterances to map
            batch_size: Number of utterances to process in each batch
            save_path: Path to save the results
            
        Returns:
            DataFrame containing the utterances and their mapped FAQs
        """
        results = []
        
        # Process the data in batches
        for i in tqdm(range(0, len(dataset_df), batch_size)):
            batch = dataset_df.iloc[i:i+batch_size]
            
            for _, row in batch.iterrows():
                utterance = row['query'] if 'query' in row.columns else row['utterance']
                
                # Map the utterance to FAQs
                mapping_result = self.map_utterance(utterance, return_details=True)
                ranked_faqs = mapping_result['ranked_faqs']
                
                # Store the results
                result = {
                    'utterance': utterance,
                    'processing_time': mapping_result['processing_time']
                }
                
                # Add the ranked FAQs
                for j, (faq, score) in enumerate(ranked_faqs, 1):
                    result[f'faq_{j}'] = faq
                    result[f'score_{j}'] = score
                
                # Add response if available
                if 'response' in mapping_result and 'response' in mapping_result['response']:
                    result['generated_response'] = mapping_result['response']['response']
                
                results.append(result)
                
            # Save the results after each batch
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            logger.info(f"Saved {len(results)} results to {save_path}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        return pd.DataFrame(results)
    
    def evaluate(self, num_samples=20, save_path='enhanced_multi_agent_evaluation_results.csv'):
        """
        Evaluate the enhanced multi-agent model on the test set.
        
        Args:
            num_samples: Number of test samples to evaluate
            save_path: Path to save the evaluation results
            
        Returns:
            DataFrame containing the evaluation results and metrics
        """
        # Skip the few-shot examples from evaluation
        test_samples = self.test_df.iloc[5:5+num_samples]  # Skip first 5 examples used for few-shot
        
        results = []
        
        # Process each unique utterance once and store the results
        utterance_mapping_cache = {}
        
        # Group test samples by utterance to get all relevant FAQs for each utterance
        utterance_groups = test_samples.groupby('utterance')
        
        # Track metrics for each utterance
        all_precision_at_3 = []
        all_precision_at_5 = []
        all_recall_at_3 = []
        all_recall_at_5 = []
        all_f1_at_3 = []
        all_f1_at_5 = []
        all_ndcg_at_3 = []
        all_ndcg_at_5 = []
        hit_rate_at_5 = 0
        recall_any_at_3 = 0
        recall_any_at_5 = 0
        recall_all_at_3 = 0
        
        # Process each utterance
        for utterance, group_df in tqdm(utterance_groups, desc="Evaluating utterances"):
            # Get all relevant FAQs for this utterance
            relevant_faqs = group_df['FAQ'].tolist()
            
            # Check if we've already processed this utterance
            if utterance not in utterance_mapping_cache:
                # Map the utterance to FAQs (only once per unique utterance)
                mapping_result = self.map_utterance(utterance, return_details=True)
                ranked_faqs = mapping_result['ranked_faqs']
                
                # Cache the results
                utterance_mapping_cache[utterance] = {
                    'ranked_faqs': ranked_faqs,
                    'processing_time': mapping_result['processing_time'],
                    'query_planning': mapping_result['query_planning'],
                    'agent_predictions': mapping_result['agent_predictions'],
                    'reranker_results': mapping_result['reranker_results'],
                    'judge_results': mapping_result['judge_results'],
                    'response': mapping_result.get('response', {})
                }
            else:
                # Retrieve from cache
                cached_result = utterance_mapping_cache[utterance]
                ranked_faqs = cached_result['ranked_faqs']
            
            # Get predicted FAQs
            predicted_faqs = [faq for faq, _ in ranked_faqs]
            predicted_faqs_at_3 = predicted_faqs[:3] if len(predicted_faqs) >= 3 else predicted_faqs
            predicted_faqs_at_5 = predicted_faqs[:5] if len(predicted_faqs) >= 5 else predicted_faqs
            
            # Calculate Hit Rate and Recall Any at 5
            if any(faq in predicted_faqs_at_5 for faq in relevant_faqs):
                hit_rate_at_5 += 1
                recall_any_at_5 += 1
            
            # Calculate Recall Any at 3
            if any(faq in predicted_faqs_at_3 for faq in relevant_faqs):
                recall_any_at_3 += 1
            
            # Calculate Recall All at 3
            if all(faq in predicted_faqs_at_3 for faq in relevant_faqs):
                recall_all_at_3 += 1
            
            # Create binary relevance vectors for precision, recall, F1
            relevance_at_3 = [1 if faq in relevant_faqs else 0 for faq in predicted_faqs_at_3]
            relevance_at_5 = [1 if faq in relevant_faqs else 0 for faq in predicted_faqs_at_5]
            
            # Pad predictions if needed (for consistent array lengths)
            relevance_at_3_padded = relevance_at_3 + [0] * (3 - len(relevance_at_3))
            relevance_at_5_padded = relevance_at_5 + [0] * (5 - len(relevance_at_5))
            
            # Calculate precision at k
            precision_at_3 = np.sum(relevance_at_3) / len(relevance_at_3) if len(relevance_at_3) > 0 else 0
            precision_at_5 = np.sum(relevance_at_5) / len(relevance_at_5) if len(relevance_at_5) > 0 else 0
            
            # Calculate recall at k
            recall_at_3 = np.sum(relevance_at_3) / len(relevant_faqs) if len(relevant_faqs) > 0 else 0
            recall_at_5 = np.sum(relevance_at_5) / len(relevant_faqs) if len(relevant_faqs) > 0 else 0
            
            # Calculate F1 at k
            f1_at_3 = 2 * (precision_at_3 * recall_at_3) / (precision_at_3 + recall_at_3) if (precision_at_3 + recall_at_3) > 0 else 0
            f1_at_5 = 2 * (precision_at_5 * recall_at_5) / (precision_at_5 + recall_at_5) if (precision_at_5 + recall_at_5) > 0 else 0
            
            # Calculate NDCG at k
            def calculate_ndcg(relevance_scores, k):
                # Create ideal DCG (sorted relevance)
                ideal_relevance = sorted(relevance_scores, reverse=True)
                
                # Calculate DCG
                dcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(relevance_scores[:k])])
                
                # Calculate ideal DCG
                idcg = np.sum([rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance[:k])])
                
                # Calculate NDCG
                return dcg / idcg if idcg > 0 else 0
            
            # Create graded relevance (assuming all relevant FAQs are equally relevant)
            graded_relevance_at_3 = relevance_at_3_padded
            graded_relevance_at_5 = relevance_at_5_padded
            
            ndcg_at_3 = calculate_ndcg(graded_relevance_at_3, 3)
            ndcg_at_5 = calculate_ndcg(graded_relevance_at_5, 5)
            
            # Append metrics for this utterance
            all_precision_at_3.append(precision_at_3)
            all_precision_at_5.append(precision_at_5)
            all_recall_at_3.append(recall_at_3)
            all_recall_at_5.append(recall_at_5)
            all_f1_at_3.append(f1_at_3)
            all_f1_at_5.append(f1_at_5)
            all_ndcg_at_3.append(ndcg_at_3)
            all_ndcg_at_5.append(ndcg_at_5)
            
            # Create individual result records for each relevant FAQ
            for _, row in group_df.iterrows():
                actual_faq = row['FAQ']
                actual_rank = row['Rank']
                
                # Check if this specific FAQ is in the predicted FAQs
                predicted_rank = None
                if actual_faq in predicted_faqs:
                    predicted_rank = predicted_faqs.index(actual_faq) + 1
                
                # Get the cached result
                cached_result = utterance_mapping_cache[utterance]
                
                # Store the results for this (utterance, FAQ) pair
                result = {
                    'utterance': utterance,
                    'actual_faq': actual_faq,
                    'actual_rank': actual_rank,
                    'predicted_rank': predicted_rank,
                    'correct_top1': 1 if predicted_rank == 1 else 0,
                    'correct_top3': 1 if predicted_rank and predicted_rank <= 3 else 0,
                    'correct_top5': 1 if predicted_rank and predicted_rank <= 5 else 0,
                    'precision_at_3': precision_at_3,
                    'precision_at_5': precision_at_5,
                    'recall_at_3': recall_at_3,
                    'recall_at_5': recall_at_5,
                    'f1_at_3': f1_at_3,
                    'f1_at_5': f1_at_5,
                    'ndcg_at_3': ndcg_at_3,
                    'ndcg_at_5': ndcg_at_5,
                    'processing_time': cached_result['processing_time']
                }
                
                # Add the ranked FAQs
                for j, (faq, score) in enumerate(ranked_faqs, 1):
                    result[f'pred_faq_{j}'] = faq
                    result[f'pred_score_{j}'] = score
                
                # Add generated response if available
                if 'response' in cached_result and 'response' in cached_result['response']:
                    result['generated_response'] = cached_result['response']['response']
                
                results.append(result)
                
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
        
        # Create DataFrame with results
        results_df = pd.DataFrame(results)
        
        # Calculate aggregated metrics
        num_utterances = len(utterance_groups)
        
        metrics = {
            # Traditional metrics
            'top1_accuracy': results_df['correct_top1'].mean(),
            'top3_accuracy': results_df['correct_top3'].mean(),
            'top5_accuracy': results_df['correct_top5'].mean(),
            'mean_reciprocal_rank': self.calculate_mrr(results_df),
            
            # New metrics
            'hit_rate_top5': hit_rate_at_5 / num_utterances,
            'recall_any_top5': recall_any_at_5 / num_utterances,
            'recall_any_top3': recall_any_at_3 / num_utterances,
            'recall_all_top3': recall_all_at_3 / num_utterances,
            'avg_precision_at_5': np.mean(all_precision_at_5),
            'avg_precision_at_3': np.mean(all_precision_at_3),
            'avg_recall_at_5': np.mean(all_recall_at_5),
            'avg_recall_at_3': np.mean(all_recall_at_3),
            'avg_f1_at_5': np.mean(all_f1_at_5),
            'avg_f1_at_3': np.mean(all_f1_at_3),
            'avg_ndcg_at_5': np.mean(all_ndcg_at_5),
            'avg_ndcg_at_3': np.mean(all_ndcg_at_3),
            
            # Additional information
            'total_evaluated': len(results_df),
            'unique_utterances': num_utterances,
            'avg_processing_time': results_df['processing_time'].mean(),
            'api_calls_saved': len(results_df) - len(utterance_mapping_cache)
        }
        
        # If self-improvement is enabled, use the evaluation results to optimize the system
        if self.use_self_improvement:
            self._optimize_from_evaluation(results_df, metrics)
        
        # Save the results
        results_df.to_csv(save_path, index=False)
        logger.info(f"Saved evaluation results to {save_path}")
        
        # Print metrics in a nicely formatted way
        logger.info("\nEvaluation Metrics:")
        logger.info("=" * 50)
        
        logger.info("\nAccuracy Metrics:")
        for metric in ['top1_accuracy', 'top3_accuracy', 'top5_accuracy', 'mean_reciprocal_rank']:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        logger.info("\nRecall Metrics:")
        for metric in ['hit_rate_top5', 'recall_any_top5', 'recall_any_top3', 'recall_all_top3']:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        logger.info("\nPrecision, Recall, F1 Metrics:")
        for metric in ['avg_precision_at_5', 'avg_precision_at_3', 'avg_recall_at_5', 'avg_recall_at_3', 'avg_f1_at_5', 'avg_f1_at_3']:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        logger.info("\nRanking Quality Metrics:")
        for metric in ['avg_ndcg_at_5', 'avg_ndcg_at_3']:
            logger.info(f"  {metric}: {metrics[metric]:.4f}")
        
        logger.info("\nEfficiency Metrics:")
        for metric in ['total_evaluated', 'unique_utterances', 'avg_processing_time', 'api_calls_saved']:
            logger.info(f"  {metric}: {metrics[metric]}")
        
        return results_df, metrics
    
    def _optimize_from_evaluation(self, results_df, metrics):
        """
        Use evaluation results to optimize the system.
        
        Args:
            results_df: DataFrame containing evaluation results
            metrics: Dictionary containing evaluation metrics
        """
        logger.info("Optimizing system based on evaluation results...")
        
        # Update agent weights based on their performance
        agent_performances = {}
        for agent_name, _ in self.agents:
            # Get agent-specific metrics if available (from cache in the evaluation)
            # This would need to be implemented in a real system
            performance_score = 0.7  # Default score
            agent_performances[agent_name] = performance_score
        
        # Normalize agent performances
        total_performance = sum(agent_performances.values())
        for agent_name in agent_performances:
            normalized_weight = agent_performances[agent_name] / total_performance if total_performance > 0 else 1.0 / len(agent_performances)
            self.agent_weights[agent_name] = normalized_weight
        
        logger.info(f"Updated agent weights: {self.agent_weights}")
        
        # Update reranker weights based on their performance
        # This would need to be implemented in a real system
        for reranker_name, _ in self.rerankers:
            self.reranker_weights[reranker_name] = 1.0
        
        logger.info(f"Updated reranker weights: {self.reranker_weights}")
        
        # Save the updated weights
        try:
            with open("agent_weights.json", 'w') as f:
                json.dump({
                    'agent_weights': self.agent_weights,
                    'reranker_weights': self.reranker_weights
                }, f)
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
    
    def calculate_mrr(self, results_df):
        """
        Calculate the Mean Reciprocal Rank (MRR) for the evaluation results.
        """
        reciprocal_ranks = []
        
        for _, row in results_df.iterrows():
            if pd.notna(row['predicted_rank']):
                reciprocal_ranks.append(1.0 / row['predicted_rank'])
            else:
                reciprocal_ranks.append(0.0)  # FAQ not found in the top 5
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

class QueryPlanningAgent:
    """
    A specialized agent for query planning and expansion.
    """
    
    def __init__(self):
        """
        Initialize the query planning agent.
        """
        pass
    
    def plan_query(self, query):
        """
        Analyze the query and develop a plan for retrieval and agent selection.
        
        Args:
            query: The user query
            
        Returns:
            A dictionary containing the query plan
        """
        try:
            # Create a prompt for query planning
            prompt = f"""
Analyze the following user query for a banking FAQ system:

"{query}"

1. Identify the main intent of the query.
2. Determine if query expansion would be helpful.
3. If expansion would help, provide an expanded version that enhances retrieval.
4. Identify which types of retrieval agents would be most appropriate:
   - Basic_Emb: Good for simple keyword matching
   - Enhanced_Emb: Good for intent understanding
   - Enhanced_Ans: Good for queries that need detailed answers

Your response must be in JSON format:
{{
    "intent": "Main intent of the query",
    "needs_expansion": true/false,
    "expanded_query": "Expanded version of the query if needed",
    "selected_agents": ["Agent1", "Agent2", ...],
    "reasoning": "Your reasoning for the expansion and agent selection"
}}
"""
            
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing user queries and planning retrieval strategies."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            # If query expansion is needed, use the expanded query
            expanded_query = result.get('expanded_query', query) if result.get('needs_expansion', False) else query
            
            # Return the query plan
            return {
                'original_query': query,
                'expanded_query': expanded_query,
                'intent': result.get('intent', ''),
                'selected_agents': result.get('selected_agents', None),
                'reasoning': result.get('reasoning', '')
            }
        except Exception as e:
            logger.error(f"Error in query planning: {e}")
            return {
                'original_query': query,
                'expanded_query': query,
                'intent': '',
                'selected_agents': None,
                'reasoning': 'Error in query planning'
            }

class ResponseGenerationAgent:
    """
    A specialized agent for generating responses based on retrieved FAQs.
    """
    
    def __init__(self, faqs_df):
        """
        Initialize the response generation agent.
        
        Args:
            faqs_df: DataFrame containing the FAQs with 'question' and 'answer' columns
        """
        self.faqs_df = faqs_df
    
    def generate_response(self, query, faq, answer=None):
        """
        Generate a response based on the retrieved FAQ.
        
        Args:
            query: The user query
            faq: The retrieved FAQ
            answer: The FAQ answer if available
            
        Returns:
            A dictionary containing the generated response
        """
        try:
            # Get the answer from the FAQ database if not provided
            if answer is None and 'answer' in self.faqs_df.columns:
                faq_row = self.faqs_df[self.faqs_df['question'] == faq]
                if not faq_row.empty:
                    answer = faq_row.iloc[0]['answer']
            
            # Create a prompt for response generation
            prompt = f"""
User Query: "{query}"

Retrieved FAQ: "{faq}"

FAQ Answer: {answer if answer else "Not available"}

Generate a helpful and conversational response that answers the user's question based on the retrieved FAQ. The response should:
1. Be concise and focused on answering the specific query
2. Use natural, conversational language
3. Provide the correct information from the FAQ
4. Include a confidence assessment of how well the response addresses the query
5. Avoid unnecessary pleasantries or explanations

Your response must be in JSON format:
{{
    "response": "Your generated response here",
    "confidence": "HIGH/MEDIUM/LOW",
    "reasoning": "Brief explanation of why this response is appropriate"
}}
"""
            
            # Call the LLM API
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert customer service representative for a banking institution."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,  # Slightly higher temperature for more natural responses
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            result = json.loads(response.choices[0].message.content)
            
            return result
        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return {
                'response': f"Based on the '{faq}' FAQ, I can provide you with this information: {answer if answer else 'No answer available'}",
                'confidence': 'LOW',
                'reasoning': 'Error in response generation'
            }

# Example usage
def main():
    # Load the datasets
    faqs_df = pd.read_csv('aem_faqs.csv')
    training_df = pd.read_csv('training_dataset_4k.csv')
    test_df = pd.read_csv('test_set.csv')
    
    # Initialize the enhanced multi-agent FAQ mapper
    mapper = EnhancedMultiAgentFAQMapper(faqs_df, test_df)
    
    # Example: Map a single utterance
    utterance = "how can I lock my card"
    result = mapper.map_utterance(utterance, return_details=True)
    
    print("\nRanked FAQs for the utterance:")
    for i, (faq, score) in enumerate(result['ranked_faqs'], 1):
        print(f"{i}. {faq} - Relevance Score: {score:.2f}")
    
    if 'response' in result and 'response' in result['response']:
        print(f"\nGenerated Response: {result['response']['response']}")

if __name__ == "__main__":
    main()