import pandas as pd
import numpy as np
import json
import os
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import logging
import openai
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MultiAgentOrchestrator")

class AgentTask:
    """
    Represents a task to be performed by an agent.
    """
    
    def __init__(self, task_id: str, agent_name: str, function: Callable, args: Dict[str, Any], dependencies: List[str] = None):
        """
        Initialize a task.
        
        Args:
            task_id: Unique identifier for the task
            agent_name: Name of the agent that will perform the task
            function: The function to call
            args: Arguments to pass to the function
            dependencies: IDs of tasks that must complete before this one starts
        """
        self.task_id = task_id
        self.agent_name = agent_name
        self.function = function
        self.args = args
        self.dependencies = dependencies or []
        self.status = "pending"  # pending, running, completed, failed
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.latency = None
    
    def execute(self):
        """
        Execute the task.
        
        Returns:
            The result of the task
        """
        try:
            self.status = "running"
            self.start_time = time.time()
            logger.info(f"Executing task {self.task_id} by agent {self.agent_name}")
            
            self.result = self.function(**self.args)
            
            self.status = "completed"
            self.end_time = time.time()
            self.latency = self.end_time - self.start_time
            logger.info(f"Task {self.task_id} completed in {self.latency:.2f}s")
            
            return self.result
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            self.end_time = time.time()
            self.latency = self.end_time - self.start_time
            logger.error(f"Task {self.task_id} failed after {self.latency:.2f}s: {e}")
            return None

class MultiAgentOrchestrator:
    """
    Coordinates multiple agents to perform complex tasks.
    """
    
    def __init__(self, memory_system=None, max_workers=4):
        """
        Initialize the orchestrator.
        
        Args:
            memory_system: Optional memory system for learning and improvement
            max_workers: Maximum number of concurrent tasks
        """
        self.memory_system = memory_system
        self.max_workers = max_workers
        self.agent_registry = {}
        self.task_queues = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = Lock()
        self.task_results = {}
        self.workflows = {}
        
        # Metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_latency": 0,
            "agent_utilization": {}
        }
        
        logger.info(f"MultiAgentOrchestrator initialized with {max_workers} workers")
    
    def register_agent(self, agent_name: str, agent_instance: Any, agent_info: Dict[str, Any] = None):
        """
        Register an agent with the orchestrator.
        
        Args:
            agent_name: Name of the agent
            agent_instance: Instance of the agent
            agent_info: Additional information about the agent
        """
        with self.lock:
            self.agent_registry[agent_name] = {
                "instance": agent_instance,
                "info": agent_info or {},
                "status": "idle",
                "task_count": 0,
                "success_count": 0,
                "total_latency": 0
            }
            self.task_queues[agent_name] = []
            self.metrics["agent_utilization"][agent_name] = 0
        
        logger.info(f"Agent {agent_name} registered")
    
    def register_workflow(self, workflow_name: str, workflow_def: Dict[str, Any]):
        """
        Register a workflow with the orchestrator.
        
        Args:
            workflow_name: Name of the workflow
            workflow_def: Definition of the workflow
        """
        with self.lock:
            self.workflows[workflow_name] = workflow_def
        
        logger.info(f"Workflow {workflow_name} registered")
    
    def create_task(self, task_id: str, agent_name: str, function_name: str, args: Dict[str, Any], dependencies: List[str] = None):
        """
        Create a task to be executed.
        
        Args:
            task_id: Unique identifier for the task
            agent_name: Name of the agent to execute the task
            function_name: Name of the function to call
            args: Arguments to pass to the function
            dependencies: IDs of tasks that must complete before this one starts
            
        Returns:
            The created task
        """
        if agent_name not in self.agent_registry:
            logger.error(f"Agent {agent_name} not registered")
            raise ValueError(f"Agent {agent_name} not registered")
        
        # Get the agent instance
        agent = self.agent_registry[agent_name]["instance"]
        
        # Get the function
        function = getattr(agent, function_name, None)
        if function is None:
            logger.error(f"Function {function_name} not found in agent {agent_name}")
            raise ValueError(f"Function {function_name} not found in agent {agent_name}")
        
        # Create the task
        task = AgentTask(task_id, agent_name, function, args, dependencies)
        
        with self.lock:
            self.task_queues[agent_name].append(task)
            self.task_results[task_id] = {"status": "pending", "result": None}
        
        logger.info(f"Task {task_id} created for agent {agent_name}")
        
        return task
    
    def execute_workflow(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a predefined workflow.
        
        Args:
            workflow_name: Name of the workflow to execute
            input_data: Input data for the workflow
            
        Returns:
            The results of the workflow
        """
        if workflow_name not in self.workflows:
            logger.error(f"Workflow {workflow_name} not registered")
            raise ValueError(f"Workflow {workflow_name} not registered")
        
        workflow_def = self.workflows[workflow_name]
        
        # Create a task for each step in the workflow
        tasks = {}
        for step_id, step_def in workflow_def["steps"].items():
            task_id = f"{workflow_name}_{step_id}_{int(time.time())}"
            agent_name = step_def["agent"]
            function_name = step_def["function"]
            
            # Build arguments
            args = {}
            for arg_name, arg_def in step_def.get("args", {}).items():
                if isinstance(arg_def, str) and arg_def.startswith("$input."):
                    # This is a reference to input data
                    input_key = arg_def[7:]  # Remove "$input."
                    args[arg_name] = input_data.get(input_key)
                elif isinstance(arg_def, str) and arg_def.startswith("$task."):
                    # This is a reference to another task's result
                    # Will be resolved at runtime
                    args[arg_name] = arg_def
                else:
                    # This is a literal value
                    args[arg_name] = arg_def
            
            # Build dependencies
            dependencies = []
            for dep in step_def.get("dependencies", []):
                dependencies.append(f"{workflow_name}_{dep}_{int(time.time())}")
            
            # Create the task
            task = self.create_task(task_id, agent_name, function_name, args, dependencies)
            tasks[step_id] = task
        
        # Execute the tasks
        results = self.execute_tasks(list(tasks.values()))
        
        # Process and return the results based on workflow definition
        if "output" in workflow_def:
            workflow_results = {}
            for output_name, output_def in workflow_def["output"].items():
                if isinstance(output_def, str) and output_def.startswith("$task."):
                    # This is a reference to a task's result
                    task_ref = output_def.split(".")
                    task_id = task_ref[1]
                    if len(task_ref) > 2:
                        # This is a reference to a specific result key
                        result_key = task_ref[2]
                        workflow_results[output_name] = results[f"{workflow_name}_{task_id}_{int(time.time())}"]["result"].get(result_key)
                    else:
                        # This is a reference to the entire result
                        workflow_results[output_name] = results[f"{workflow_name}_{task_id}_{int(time.time())}"]["result"]
                else:
                    # This is a literal value
                    workflow_results[output_name] = output_def
            
            return workflow_results
        else:
            # Return all results
            return {task_id: info["result"] for task_id, info in results.items()}
    
    def execute_tasks(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """
        Execute a list of tasks with dependencies.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Dictionary of task_id -> task result
        """
        # Create a dictionary for easier lookup of tasks by ID
        task_dict = {task.task_id: task for task in tasks}
        
        # Track which tasks are ready to execute
        ready_tasks = [task for task in tasks if not task.dependencies]
        pending_tasks = [task for task in tasks if task.dependencies]
        
        # Execute tasks until all are complete
        while ready_tasks or pending_tasks:
            # Submit ready tasks for execution
            futures = {}
            for task in ready_tasks:
                # Resolve task arguments that depend on other tasks
                for arg_name, arg_value in task.args.items():
                    if isinstance(arg_value, str) and arg_value.startswith("$task."):
                        # This is a reference to another task's result
                        task_ref = arg_value.split(".")
                        dep_task_id = task_ref[1]
                        for dep_task in tasks:
                            if dep_task.task_id.endswith(dep_task_id):
                                if len(task_ref) > 2:
                                    # This is a reference to a specific result key
                                    result_key = task_ref[2]
                                    task.args[arg_name] = dep_task.result.get(result_key)
                                else:
                                    # This is a reference to the entire result
                                    task.args[arg_name] = dep_task.result
                
                # Mark agent as busy
                self.agent_registry[task.agent_name]["status"] = "busy"
                self.agent_registry[task.agent_name]["task_count"] += 1
                
                # Submit the task
                futures[self.executor.submit(task.execute)] = task
            
            # Wait for tasks to complete
            for future in futures:
                task = futures[future]
                task_result = future.result()
                self.task_results[task.task_id] = {"status": task.status, "result": task_result}
                
                # Update agent metrics
                with self.lock:
                    self.agent_registry[task.agent_name]["status"] = "idle"
                    self.agent_registry[task.agent_name]["total_latency"] += task.latency
                    if task.status == "completed":
                        self.agent_registry[task.agent_name]["success_count"] += 1
                        self.metrics["tasks_completed"] += 1
                    else:
                        self.metrics["tasks_failed"] += 1
                    self.metrics["total_latency"] += task.latency
                    self.metrics["agent_utilization"][task.agent_name] = (
                        self.agent_registry[task.agent_name]["total_latency"] / 
                        self.metrics["total_latency"] if self.metrics["total_latency"] > 0 else 0
                    )
            
            # Update ready and pending tasks
            completed_tasks = [task.task_id for task in ready_tasks]
            ready_tasks = []
            still_pending = []
            
            for task in pending_tasks:
                # Check if all dependencies are complete
                dependencies_met = True
                for dep_id in task.dependencies:
                    if dep_id not in self.task_results or self.task_results[dep_id]["status"] != "completed":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    ready_tasks.append(task)
                else:
                    still_pending.append(task)
            
            pending_tasks = still_pending
        
        # Return the results
        return self.task_results
    
    async def execute_task_async(self, task: AgentTask) -> Any:
        """
        Execute a task asynchronously.
        
        Args:
            task: The task to execute
            
        Returns:
            The result of the task
        """
        loop = asyncio.get_event_loop()
        
        # Execute the task in a thread pool
        result = await loop.run_in_executor(self.executor, task.execute)
        
        return result
    
    async def execute_workflow_async(self, workflow_name: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a predefined workflow asynchronously.
        
        Args:
            workflow_name: Name of the workflow to execute
            input_data: Input data for the workflow
            
        Returns:
            The results of the workflow
        """
        if workflow_name not in self.workflows:
            logger.error(f"Workflow {workflow_name} not registered")
            raise ValueError(f"Workflow {workflow_name} not registered")
        
        workflow_def = self.workflows[workflow_name]
        
        # Create a task for each step in the workflow
        tasks = {}
        for step_id, step_def in workflow_def["steps"].items():
            task_id = f"{workflow_name}_{step_id}_{int(time.time())}"
            agent_name = step_def["agent"]
            function_name = step_def["function"]
            
            # Build arguments
            args = {}
            for arg_name, arg_def in step_def.get("args", {}).items():
                if isinstance(arg_def, str) and arg_def.startswith("$input."):
                    # This is a reference to input data
                    input_key = arg_def[7:]  # Remove "$input."
                    args[arg_name] = input_data.get(input_key)
                elif isinstance(arg_def, str) and arg_def.startswith("$task."):
                    # This is a reference to another task's result
                    # Will be resolved at runtime
                    args[arg_name] = arg_def
                else:
                    # This is a literal value
                    args[arg_name] = arg_def
            
            # Build dependencies
            dependencies = []
            for dep in step_def.get("dependencies", []):
                dependencies.append(f"{workflow_name}_{dep}_{int(time.time())}")
            
            # Create the task
            task = self.create_task(task_id, agent_name, function_name, args, dependencies)
            tasks[step_id] = task
        
        # Execute the tasks
        results = await self.execute_tasks_async(list(tasks.values()))
        
        # Process and return the results based on workflow definition
        if "output" in workflow_def:
            workflow_results = {}
            for output_name, output_def in workflow_def["output"].items():
                if isinstance(output_def, str) and output_def.startswith("$task."):
                    # This is a reference to a task's result
                    task_ref = output_def.split(".")
                    task_id = task_ref[1]
                    if len(task_ref) > 2:
                        # This is a reference to a specific result key
                        result_key = task_ref[2]
                        workflow_results[output_name] = results[f"{workflow_name}_{task_id}_{int(time.time())}"]["result"].get(result_key)
                    else:
                        # This is a reference to the entire result
                        workflow_results[output_name] = results[f"{workflow_name}_{task_id}_{int(time.time())}"]["result"]
                else:
                    # This is a literal value
                    workflow_results[output_name] = output_def
            
            return workflow_results
        else:
            # Return all results
            return {task_id: info["result"] for task_id, info in results.items()}
    
    async def execute_tasks_async(self, tasks: List[AgentTask]) -> Dict[str, Any]:
        """
        Execute a list of tasks with dependencies asynchronously.
        
        Args:
            tasks: List of tasks to execute
            
        Returns:
            Dictionary of task_id -> task result
        """
        # Create a dictionary for easier lookup of tasks by ID
        task_dict = {task.task_id: task for task in tasks}
        
        # Track which tasks are ready to execute
        ready_tasks = [task for task in tasks if not task.dependencies]
        pending_tasks = [task for task in tasks if task.dependencies]
        
        # Execute tasks until all are complete
        while ready_tasks or pending_tasks:
            # Prepare async execution of ready tasks
            futures = []
            for task in ready_tasks:
                # Resolve task arguments that depend on other tasks
                for arg_name, arg_value in task.args.items():
                    if isinstance(arg_value, str) and arg_value.startswith("$task."):
                        # This is a reference to another task's result
                        task_ref = arg_value.split(".")
                        dep_task_id = task_ref[1]
                        for dep_task in tasks:
                            if dep_task.task_id.endswith(dep_task_id):
                                if len(task_ref) > 2:
                                    # This is a reference to a specific result key
                                    result_key = task_ref[2]
                                    task.args[arg_name] = dep_task.result.get(result_key)
                                else:
                                    # This is a reference to the entire result
                                    task.args[arg_name] = dep_task.result
                
                # Mark agent as busy
                self.agent_registry[task.agent_name]["status"] = "busy"
                self.agent_registry[task.agent_name]["task_count"] += 1
                
                # Create a future for the task
                futures.append(self.execute_task_async(task))
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*futures)
            
            # Update task results
            for i, task in enumerate(ready_tasks):
                task_result = results[i]
                self.task_results[task.task_id] = {"status": task.status, "result": task_result}
                
                # Update agent metrics
                with self.lock:
                    self.agent_registry[task.agent_name]["status"] = "idle"
                    self.agent_registry[task.agent_name]["total_latency"] += task.latency
                    if task.status == "completed":
                        self.agent_registry[task.agent_name]["success_count"] += 1
                        self.metrics["tasks_completed"] += 1
                    else:
                        self.metrics["tasks_failed"] += 1
                    self.metrics["total_latency"] += task.latency
                    self.metrics["agent_utilization"][task.agent_name] = (
                        self.agent_registry[task.agent_name]["total_latency"] / 
                        self.metrics["total_latency"] if self.metrics["total_latency"] > 0 else 0
                    )
            
            # Update ready and pending tasks
            completed_tasks = [task.task_id for task in ready_tasks]
            ready_tasks = []
            still_pending = []
            
            for task in pending_tasks:
                # Check if all dependencies are complete
                dependencies_met = True
                for dep_id in task.dependencies:
                    if dep_id not in self.task_results or self.task_results[dep_id]["status"] != "completed":
                        dependencies_met = False
                        break
                
                if dependencies_met:
                    ready_tasks.append(task)
                else:
                    still_pending.append(task)
            
            pending_tasks = still_pending
        
        # Return the results
        return self.task_results
    
    def generate_workflow_from_query(self, query: str) -> Dict[str, Any]:
        """
        Generate a workflow for a query.
        
        Args:
            query: The user query
            
        Returns:
            A workflow definition
        """
        # Use memory system to get recommended workflow if available
        if self.memory_system:
            recommended_workflow = self.memory_system.get_recommended_workflow(query)
            if recommended_workflow["confidence"] != "LOW":
                logger.info(f"Using recommended workflow for query: {query}")
                
                # Create a workflow based on the recommendation
                workflow = self._create_workflow_from_recommendation(query, recommended_workflow)
                return workflow
        
        # Generate a new workflow based on the query
        return self._generate_new_workflow(query)
    
    def _create_workflow_from_recommendation(self, query: str, recommendation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a workflow from a recommendation.
        
        Args:
            query: The user query
            recommendation: The workflow recommendation
            
        Returns:
            A workflow definition
        """
        # Extract recommended agents and rerankers
        agents = recommendation["recommended_agents"]
        rerankers = recommendation["recommended_rerankers"]
        
        # Create a workflow definition
        workflow = {
            "name": f"workflow_{int(time.time())}",
            "description": f"Workflow for query: {query}",
            "steps": {
                "query_planning": {
                    "agent": "QueryPlanningAgent",
                    "function": "plan_query",
                    "args": {
                        "query": "$input.query"
                    }
                }
            },
            "output": {
                "ranked_faqs": "$task.final_ranking.ranked_faqs",
                "processing_time": "$task.final_ranking.processing_time"
            }
        }
        
        # Add retrieval steps for each agent
        for i, agent in enumerate(agents):
            workflow["steps"][f"retrieval_{i}"] = {
                "agent": agent,
                "function": "map_utterance",
                "args": {
                    "utterance": "$task.query_planning.expanded_query"
                },
                "dependencies": ["query_planning"]
            }
        
        # Add reranking steps
        for i, reranker in enumerate(rerankers):
            workflow["steps"][f"reranking_{i}"] = {
                "agent": "RerankingAgent",
                "function": reranker,
                "args": {
                    "query": "$task.query_planning.expanded_query",
                    "candidates": self._combine_retrievals(agents, i)
                },
                "dependencies": [f"retrieval_{j}" for j in range(len(agents))]
            }
        
        # Add final ranking step
        workflow["steps"]["final_ranking"] = {
            "agent": "JudgeAgent",
            "function": "rerank_candidates",
            "args": {
                "utterance": "$task.query_planning.expanded_query",
                "candidates": self._combine_rerankings(rerankers),
                "agent_predictions": self._combine_agent_predictions(agents)
            },
            "dependencies": [f"reranking_{i}" for i in range(len(rerankers))]
        }
        
        # Add response generation step
        workflow["steps"]["response_generation"] = {
            "agent": "ResponseGenerationAgent",
            "function": "generate_response",
            "args": {
                "query": "$input.query",
                "faq": "$task.final_ranking.ranked_faqs[0][0]"
            },
            "dependencies": ["final_ranking"]
        }
        
        return workflow
    
    def _combine_retrievals(self, agents: List[str], reranker_index: int) -> str:
        """
        Create a reference to combine retrievals from multiple agents.
        
        Args:
            agents: List of agent names
            reranker_index: Index of the current reranker
            
        Returns:
            A string reference to the combined retrievals
        """
        references = [f"$task.retrieval_{i}.ranked_faqs" for i in range(len(agents))]
        return f"$combine({','.join(references)})"
    
    def _combine_rerankings(self, rerankers: List[str]) -> str:
        """
        Create a reference to combine rerankings from multiple rerankers.
        
        Args:
            rerankers: List of reranker names
            
        Returns:
            A string reference to the combined rerankings
        """
        references = [f"$task.reranking_{i}.reranked_faqs" for i in range(len(rerankers))]
        return f"$combine({','.join(references)})"
    
    def _combine_agent_predictions(self, agents: List[str]) -> str:
        """
        Create a reference to combine agent predictions.
        
        Args:
            agents: List of agent names
            
        Returns:
            A string reference to the combined agent predictions
        """
        references = [f"$task.retrieval_{i}.agent_predictions" for i in range(len(agents))]
        return f"$combine({','.join(references)})"
    
    def _generate_new_workflow(self, query: str) -> Dict[str, Any]:
        """
        Generate a new workflow for a query based on its characteristics.
        
        Args:
            query: The user query
            
        Returns:
            A workflow definition
        """
        # Define a default workflow
        workflow = {
            "name": f"workflow_{int(time.time())}",
            "description": f"Default workflow for query: {query}",
            "steps": {
                "query_planning": {
                    "agent": "QueryPlanningAgent",
                    "function": "plan_query",
                    "args": {
                        "query": "$input.query"
                    }
                },
                "retrieval_basic": {
                    "agent": "Basic_Emb",
                    "function": "map_utterance",
                    "args": {
                        "utterance": "$task.query_planning.expanded_query"
                    },
                    "dependencies": ["query_planning"]
                },
                "retrieval_enhanced": {
                    "agent": "Enhanced_Emb",
                    "function": "map_utterance",
                    "args": {
                        "utterance": "$task.query_planning.expanded_query"
                    },
                    "dependencies": ["query_planning"]
                },
                "retrieval_with_answers": {
                    "agent": "Enhanced_Ans",
                    "function": "map_utterance_with_answers",
                    "args": {
                        "utterance": "$task.query_planning.expanded_query"
                    },
                    "dependencies": ["query_planning"]
                },
                "reranking_semantic": {
                    "agent": "RerankingAgent",
                    "function": "semantic_reranker",
                    "args": {
                        "query": "$task.query_planning.expanded_query",
                        "candidates": "$combine($task.retrieval_basic.ranked_faqs,$task.retrieval_enhanced.ranked_faqs,$task.retrieval_with_answers.ranked_faqs)"
                    },
                    "dependencies": ["retrieval_basic", "retrieval_enhanced", "retrieval_with_answers"]
                },
                "reranking_intent": {
                    "agent": "RerankingAgent",
                    "function": "intent_reranker",
                    "args": {
                        "query": "$task.query_planning.expanded_query",
                        "candidates": "$combine($task.retrieval_basic.ranked_faqs,$task.retrieval_enhanced.ranked_faqs,$task.retrieval_with_answers.ranked_faqs)"
                    },
                    "dependencies": ["retrieval_basic", "retrieval_enhanced", "retrieval_with_answers"]
                },
                "final_ranking": {
                    "agent": "JudgeAgent",
                    "function": "rerank_candidates",
                    "args": {
                        "utterance": "$task.query_planning.expanded_query",
                        "candidates": "$combine($task.reranking_semantic.reranked_faqs,$task.reranking_intent.reranked_faqs)",
                        "agent_predictions": {
                            "Basic_Emb": "$task.retrieval_basic.ranked_faqs",
                            "Enhanced_Emb": "$task.retrieval_enhanced.ranked_faqs",
                            "Enhanced_Ans": "$task.retrieval_with_answers.ranked_faqs"
                        }
                    },
                    "dependencies": ["reranking_semantic", "reranking_intent"]
                },
                "response_generation": {
                    "agent": "ResponseGenerationAgent",
                    "function": "generate_response",
                    "args": {
                        "query": "$input.query",
                        "faq": "$task.final_ranking.ranked_faqs[0][0]"
                    },
                    "dependencies": ["final_ranking"]
                }
            },
            "output": {
                "ranked_faqs": "$task.final_ranking.ranked_faqs",
                "processing_time": "$task.final_ranking.processing_time",
                "response": "$task.response_generation.response"
            }
        }
        
        return workflow
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the orchestrator.
        
        Returns:
            Dictionary of metrics
        """
        with self.lock:
            metrics = self.metrics.copy()
            
            # Add agent-specific metrics
            metrics["agents"] = {}
            for agent_name, agent_data in self.agent_registry.items():
                metrics["agents"][agent_name] = {
                    "task_count": agent_data["task_count"],
                    "success_count": agent_data["success_count"],
                    "success_rate": agent_data["success_count"] / agent_data["task_count"] if agent_data["task_count"] > 0 else 0,
                    "total_latency": agent_data["total_latency"],
                    "avg_latency": agent_data["total_latency"] / agent_data["task_count"] if agent_data["task_count"] > 0 else 0
                }
        
        return metrics
    
    def handle_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Handle a user query by generating and executing a workflow.
        
        Args:
            query: The user query
            context: Additional context for the query
            
        Returns:
            The results of the workflow
        """
        # Generate a workflow for the query
        workflow = self.generate_workflow_from_query(query)
        
        # Register the workflow
        workflow_name = workflow["name"]
        self.register_workflow(workflow_name, workflow)
        
        # Prepare input data
        input_data = {
            "query": query
        }
        if context:
            input_data.update(context)
        
        # Execute the workflow
        results = self.execute_workflow(workflow_name, input_data)
        
        # Record the interaction in memory system if available
        if self.memory_system and "ranked_faqs" in results:
            # Extract agent and reranker performance from the task results
            agent_performance = {}
            reranker_performance = {}
            
            for task_id, task_info in self.task_results.items():
                if "retrieval" in task_id:
                    agent_name = task_id.split("_")[1]
                    agent_performance[agent_name] = {
                        "latency": task_info.get("latency", 0),
                        "success": task_info["status"] == "completed",
                        "predictions": task_info["result"] if task_info["status"] == "completed" else [],
                        "contribution": 1.0 if task_info["status"] == "completed" else 0.0
                    }
                elif "reranking" in task_id:
                    reranker_name = task_id.split("_")[1]
                    reranker_performance[reranker_name] = {
                        "latency": task_info.get("latency", 0),
                        "impact": 1.0 if task_info["status"] == "completed" else 0.0
                    }
            
            # Record the mapping
            self.memory_system.record_mapping(
                original_query=query,
                expanded_query=results.get("expanded_query", query),
                ranked_faqs=results["ranked_faqs"],
                agent_performance=agent_performance,
                reranker_performance=reranker_performance,
                workflow_info={
                    "processing_time": results.get("processing_time", 0),
                    "success_score": 1.0  # Assume success for now
                }
            )
        
        return results

# Example usage
def main():
    # Create a simple agent for testing
    class TestAgent:
        def do_something(self, arg1, arg2):
            return f"Did something with {arg1} and {arg2}"
    
    # Initialize the orchestrator
    orchestrator = MultiAgentOrchestrator()
    
    # Register an agent
    orchestrator.register_agent("TestAgent", TestAgent())
    
    # Create a task
    task = orchestrator.create_task(
        task_id="task1",
        agent_name="TestAgent",
        function_name="do_something",
        args={"arg1": "hello", "arg2": "world"}
    )
    
    # Execute the task
    results = orchestrator.execute_tasks([task])
    
    print("Task results:")
    for task_id, info in results.items():
        print(f"{task_id}: {info}")
    
    # Create a workflow
    workflow = {
        "name": "test_workflow",
        "description": "A test workflow",
        "steps": {
            "step1": {
                "agent": "TestAgent",
                "function": "do_something",
                "args": {
                    "arg1": "$input.param1",
                    "arg2": "$input.param2"
                }
            }
        },
        "output": {
            "result": "$task.step1"
        }
    }
    
    # Register the workflow
    orchestrator.register_workflow("test_workflow", workflow)
    
    # Execute the workflow
    result = orchestrator.execute_workflow("test_workflow", {"param1": "foo", "param2": "bar"})
    
    print("\nWorkflow result:")
    print(result)
    
    # Get metrics
    metrics = orchestrator.get_metrics()
    
    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
