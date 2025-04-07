import pandas as pd
import numpy as np
import os
import json
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
import argparse

# Import our components
from enhanced_faq_mapper import EnhancedMultiAgentFAQMapper
from memory_system import MemorySystem
from multi_agent_orchestrator import MultiAgentOrchestrator
from enhanced_judge_agent import EnhancedJudgeAgent
from evaluation_framework import EvaluationFramework

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("faq_mapping_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("FAQMappingSystem")

class FAQMappingSystem:
    """
    Main integration class for the enhanced multi-agent FAQ mapping system.
    """
    
    def __init__(self, 
                 faqs_file: str,
                 test_file: str = None,
                 use_memory: bool = True,
                 use_self_improvement: bool = True,
                 output_dir: str = "output",
                 max_workers: int = 4):
        """
        Initialize the FAQ mapping system.
        
        Args:
            faqs_file: Path to the CSV file containing FAQs
            test_file: Path to the CSV file containing test data
            use_memory: Whether to use memory for system improvement
            use_self_improvement: Whether to enable self-improvement
            output_dir: Directory to save output files
            max_workers: Maximum number of concurrent workers
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        try:
            self.faqs_df = pd.read_csv(faqs_file)
            logger.info(f"Loaded {len(self.faqs_df)} FAQs from {faqs_file}")
            
            if test_file:
                self.test_df = pd.read_csv(test_file)
                logger.info(f"Loaded {len(self.test_df)} test samples from {test_file}")
            else:
                self.test_df = None
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
        
        # Initialize components
        logger.info("Initializing system components...")
        
        # Initialize memory system
        self.memory_system = MemorySystem() if use_memory else None
        
        # Initialize multi-agent orchestrator
        self.orchestrator = MultiAgentOrchestrator(
            memory_system=self.memory_system,
            max_workers=max_workers
        )
        
        # Initialize judge agent
        self.judge_agent = EnhancedJudgeAgent(
            faqs_df=self.faqs_df,
            test_df=self.test_df,
            judge_consistency=use_self_improvement
        )
        
        # Initialize enhanced multi-agent system
        self.faq_mapper = EnhancedMultiAgentFAQMapper(
            faqs_df=self.faqs_df,
            test_df=self.test_df,
            use_memory=use_memory,
            use_self_improvement=use_self_improvement
        )
        
        # Initialize evaluation framework
        self.evaluation_framework = EvaluationFramework(output_dir=os.path.join(output_dir, "evaluation"))
        
        # Register components with the orchestrator
        self._register_components()
        
        logger.info("System initialization complete")
    
    def _register_components(self):
        """
        Register all components with the orchestrator.
        """
        # Register the judge agent
        self.orchestrator.register_agent(
            agent_name="JudgeAgent",
            agent_instance=self.judge_agent,
            agent_info={"type": "judge", "description": "Enhanced judge agent with consistency mechanisms"}
        )
        
        # Register basic agents
        for agent_name, agent in self.faq_mapper.agents:
            self.orchestrator.register_agent(
                agent_name=agent_name,
                agent_instance=agent,
                agent_info={"type": "retrieval", "description": f"FAQ retrieval agent: {agent_name}"}
            )
        
        # Register reranking agent
        self.orchestrator.register_agent(
            agent_name="RerankingAgent",
            agent_instance=self.faq_mapper,
            agent_info={"type": "reranking", "description": "Reranking agent with multiple reranking methods"}
        )
        
        # Register query planning agent
        self.orchestrator.register_agent(
            agent_name="QueryPlanningAgent",
            agent_instance=self.faq_mapper.query_planner,
            agent_info={"type": "planning", "description": "Query planning and expansion agent"}
        )
        
        # Register response generation agent
        self.orchestrator.register_agent(
            agent_name="ResponseGenerationAgent",
            agent_instance=self.faq_mapper.response_generator,
            agent_info={"type": "generation", "description": "Response generation agent"}
        )
        
        # Register the main system with the evaluation framework
        self.evaluation_framework.register_system(
            system_name="EnhancedMultiAgentFAQMapper",
            system_instance=self.faq_mapper,
            system_description="Enhanced multi-agent FAQ mapping system with memory and self-improvement"
        )
        
        logger.info("All components registered with orchestrator")
    
    def process_query(self, query: str, return_details: bool = False) -> Dict[str, Any]:
        """
        Process a user query and return mappings to relevant FAQs.
        
        Args:
            query: The user query
            return_details: Whether to return detailed information
            
        Returns:
            Dictionary containing mapped FAQs and optionally detailed information
        """
        logger.info(f"Processing query: {query}")
        
        # Use the orchestrator to handle the query
        result = self.orchestrator.handle_query(query)
        
        # Format the response
        response = {
            "query": query,
            "ranked_faqs": result.get("ranked_faqs", []),
            "processing_time": result.get("processing_time", 0),
            "response": result.get("response", {}).get("response", "No response generated")
        }
        
        if return_details:
            response["details"] = {
                "expanded_query": result.get("expanded_query", query),
                "agent_predictions": result.get("agent_predictions", {}),
                "workflow_steps": result
            }
        
        return response
    
    def process_batch(self, queries: List[str], save_results: bool = True) -> Dict[str, Any]:
        """
        Process a batch of queries.
        
        Args:
            queries: List of queries to process
            save_results: Whether to save the results
            
        Returns:
            Dictionary containing results for each query
        """
        results = {}
        start_time = time.time()
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            results[query] = self.process_query(query)
        
        total_time = time.time() - start_time
        
        # Create a summary
        summary = {
            "total_queries": len(queries),
            "total_time": total_time,
            "avg_time_per_query": total_time / len(queries) if queries else 0,
            "results": results
        }
        
        if save_results:
            with open(os.path.join(self.output_dir, "batch_results.json"), 'w') as f:
                json.dump(summary, f, indent=2)
        
        return summary
    
    def evaluate(self, num_samples: int = None, save_detailed_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate the system on test data.
        
        Args:
            num_samples: Number of test samples to evaluate (None for all)
            save_detailed_results: Whether to save detailed results
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.test_df is None:
            logger.error("No test data available for evaluation")
            return {"status": "error", "message": "No test data available"}
        
        logger.info(f"Evaluating system on {num_samples or len(self.test_df)} test samples")
        
        # Use the evaluation framework
        metrics = self.evaluation_framework.evaluate_system(
            system_name="EnhancedMultiAgentFAQMapper",
            test_data=self.test_df,
            num_samples=num_samples,
            save_detailed_results=save_detailed_results
        )
        
        # Generate an evaluation report
        report = self.evaluation_framework.generate_evaluation_report()
        
        return {
            "metrics": metrics,
            "report": report
        }
    
    def compare_with_baseline(self, baseline_file: str = None) -> Dict[str, Any]:
        """
        Compare the enhanced system with a baseline.
        
        Args:
            baseline_file: Path to baseline implementation file (None to use default)
            
        Returns:
            Dictionary containing comparison results
        """
        # Create a baseline system
        if baseline_file:
            # Load and use provided baseline
            try:
                from importlib.util import spec_from_file_location, module_from_spec
                spec = spec_from_file_location("baseline", baseline_file)
                baseline_module = module_from_spec(spec)
                spec.loader.exec_module(baseline_module)
                baseline_class = getattr(baseline_module, "FAQMapper")
                
                baseline = baseline_class(
                    faqs_df=self.faqs_df,
                    test_df=self.test_df,
                    use_embeddings=True
                )
            except Exception as e:
                logger.error(f"Error loading baseline: {e}")
                return {"status": "error", "message": f"Error loading baseline: {e}"}
        else:
            # Use simple baseline from Complete Implementation.txt
            from faq_mapper_implementation import FAQMapper
            baseline = FAQMapper(
                faqs_df=self.faqs_df,
                test_df=self.test_df,
                use_embeddings=True
            )
        
        # Register the baseline with the evaluation framework
        self.evaluation_framework.register_system(
            system_name="Baseline",
            system_instance=baseline,
            system_description="Baseline FAQ mapping system"
        )
        
        # Evaluate the baseline
        baseline_metrics = self.evaluation_framework.evaluate_system(
            system_name="Baseline",
            test_data=self.test_df,
            num_samples=100  # Use a subset for faster comparison
        )
        
        # Compare the systems
        comparison = self.evaluation_framework.compare_systems(
            system_names=["EnhancedMultiAgentFAQMapper", "Baseline"]
        )
        
        # Generate comparison report
        comparison_report = self.evaluation_framework.generate_evaluation_report(
            system_names=["EnhancedMultiAgentFAQMapper", "Baseline"]
        )
        
        return {
            "comparison": comparison.to_dict(orient="records"),
            "report": comparison_report
        }
    
    def get_system_insights(self) -> Dict[str, Any]:
        """
        Get insights about the system's performance and learning.
        
        Returns:
            Dictionary containing system insights
        """
        insights = {
            "memory_insights": None,
            "judge_performance": None,
            "orchestrator_metrics": None,
            "evaluation_metrics": None
        }
        
        # Get memory insights if available
        if self.memory_system:
            insights["memory_insights"] = self.memory_system.generate_insights()
        
        # Get orchestrator metrics
        insights["orchestrator_metrics"] = self.orchestrator.get_metrics()
        
        # Get evaluation metrics if available
        if "EnhancedMultiAgentFAQMapper" in self.evaluation_framework.systems:
            if self.evaluation_framework.systems["EnhancedMultiAgentFAQMapper"]["metrics"]:
                insights["evaluation_metrics"] = self.evaluation_framework.systems["EnhancedMultiAgentFAQMapper"]["metrics"]
        
        return insights

def main():
    """
    Main entry point for the FAQ mapping system.
    """
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Enhanced Multi-Agent FAQ Mapping System")
    
    # Data options
    parser.add_argument("--faqs", type=str, required=True, help="Path to FAQs CSV file")
    parser.add_argument("--test", type=str, help="Path to test data CSV file")
    
    # System options
    parser.add_argument("--no-memory", action="store_true", help="Disable memory system")
    parser.add_argument("--no-self-improve", action="store_true", help="Disable self-improvement")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum concurrent workers")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    
    # Operation mode
    parser.add_argument("--mode", type=str, choices=["interactive", "batch", "evaluate", "compare"], 
                       default="interactive", help="Operation mode")
    
    # Mode-specific options
    parser.add_argument("--batch-file", type=str, help="Path to batch queries file (one query per line)")
    parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate")
    parser.add_argument("--baseline", type=str, help="Path to baseline implementation file")
    parser.add_argument("--query", type=str, help="Single query to process")
    
    args = parser.parse_args()
    
    # Initialize the system
    system = FAQMappingSystem(
        faqs_file=args.faqs,
        test_file=args.test,
        use_memory=not args.no_memory,
        use_self_improvement=not args.no_self_improve,
        output_dir=args.output_dir,
        max_workers=args.max_workers
    )
    
    # Operate in the selected mode
    if args.mode == "interactive":
        if args.query:
            # Process a single query
            result = system.process_query(args.query, return_details=True)
            print("\nQuery: ", args.query)
            print("\nRanked FAQs:")
            for i, (faq, score) in enumerate(result["ranked_faqs"], 1):
                print(f"{i}. {faq} - Score: {score:.2f}")
            print("\nGenerated Response:", result["response"])
            print(f"\nProcessing Time: {result['processing_time']:.2f}s")
        else:
            # Interactive mode
            print("Enhanced Multi-Agent FAQ Mapping System")
            print("Type 'quit' or 'exit' to exit")
            
            while True:
                try:
                    query = input("\nEnter your query: ")
                    if query.lower() in ["quit", "exit"]:
                        break
                    
                    result = system.process_query(query)
                    
                    print("\nRanked FAQs:")
                    for i, (faq, score) in enumerate(result["ranked_faqs"], 1):
                        print(f"{i}. {faq} - Score: {score:.2f}")
                    
                    print("\nGenerated Response:", result["response"])
                    print(f"\nProcessing Time: {result['processing_time']:.2f}s")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
    
    elif args.mode == "batch":
        if not args.batch_file:
            print("Error: Batch mode requires --batch-file")
            return
        
        # Load queries from file
        try:
            with open(args.batch_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            print(f"Processing {len(queries)} queries in batch mode...")
            results = system.process_batch(queries)
            
            print(f"Batch processing complete in {results['total_time']:.2f}s")
            print(f"Average time per query: {results['avg_time_per_query']:.2f}s")
            print(f"Results saved to {os.path.join(args.output_dir, 'batch_results.json')}")
        
        except Exception as e:
            print(f"Error in batch processing: {e}")
    
    elif args.mode == "evaluate":
        print("Evaluating system performance...")
        
        try:
            eval_results = system.evaluate(num_samples=args.num_samples)
            
            print("\nEvaluation Results:")
            for metric, value in eval_results["metrics"].items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
            
            print(f"\nDetailed evaluation report saved to {os.path.join(args.output_dir, 'evaluation')}")
        
        except Exception as e:
            print(f"Error in evaluation: {e}")
    
    elif args.mode == "compare":
        print("Comparing with baseline...")
        
        try:
            comparison = system.compare_with_baseline(baseline_file=args.baseline)
            
            print("\nComparison Results:")
            for comp in comparison["comparison"]:
                print(f"\nSystem: {comp['system']}")
                for metric, value in comp.items():
                    if metric != "system" and isinstance(value, (int, float)):
                        print(f"  {metric}: {value:.4f}")
            
            print(f"\nDetailed comparison report saved to {os.path.join(args.output_dir, 'evaluation')}")
        
        except Exception as e:
            print(f"Error in comparison: {e}")

if __name__ == "__main__":
    main()
