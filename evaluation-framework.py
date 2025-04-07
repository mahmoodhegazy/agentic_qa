import pandas as pd
import numpy as np
import json
import os
import time
from datetime import datetime
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ndcg_score
import pickle
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("EvaluationFramework")

class EvaluationFramework:
    """
    A comprehensive framework for evaluating FAQ mapping systems.
    """
    
    def __init__(self, output_dir="evaluation_results"):
        """
        Initialize the evaluation framework.
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Metrics to track
        self.traditional_metrics = [
            "top1_accuracy",
            "top3_accuracy",
            "top5_accuracy",
            "mean_reciprocal_rank",
        ]
        
        self.advanced_metrics = [
            "hit_rate_top5",
            "recall_any_top5",
            "recall_any_top3",
            "recall_all_top3",
            "avg_precision_at_5",
            "avg_precision_at_3",
            "avg_recall_at_5",
            "avg_recall_at_3",
            "avg_f1_at_5",
            "avg_f1_at_3",
            "avg_ndcg_at_5",
            "avg_ndcg_at_3",
        ]
        
        self.efficiency_metrics = [
            "avg_processing_time",
            "avg_memory_usage",
            "avg_token_usage",
            "api_calls_saved"
        ]
        
        # For tracking multiple systems
        self.systems = {}
        
        # For detailed analysis
        self.per_query_results = {}
        
        logger.info("Evaluation framework initialized")
    
    def register_system(self, system_name: str, system_instance: Any, system_description: str = ""):
        """
        Register a system to be evaluated.
        
        Args:
            system_name: Name of the system
            system_instance: Instance of the system
            system_description: Description of the system
        """
        self.systems[system_name] = {
            "instance": system_instance,
            "description": system_description,
            "results": None,
            "metrics": None
        }
        
        logger.info(f"Registered system: {system_name}")
    
    def evaluate_system(self, system_name: str, test_data: pd.DataFrame, num_samples: int = None, save_detailed_results: bool = True):
        """
        Evaluate a registered system.
        
        Args:
            system_name: Name of the system to evaluate
            test_data: DataFrame containing test data
            num_samples: Number of samples to evaluate (None for all)
            save_detailed_results: Whether to save detailed results
            
        Returns:
            A dictionary of evaluation metrics
        """
        if system_name not in self.systems:
            logger.error(f"System {system_name} not registered")
            raise ValueError(f"System {system_name} not registered")
        
        system = self.systems[system_name]["instance"]
        
        # Limit the number of samples if specified
        if num_samples is not None:
            test_data = test_data.head(num_samples)
        
        logger.info(f"Evaluating system {system_name} on {len(test_data)} samples")
        
        # Call the system's evaluate method
        start_time = time.time()
        results_df, metrics = system.evaluate(
            num_samples=len(test_data),
            save_path=os.path.join(self.output_dir, f"{system_name}_evaluation_results.csv")
        )
        evaluation_time = time.time() - start_time
        
        # Store the results
        self.systems[system_name]["results"] = results_df
        self.systems[system_name]["metrics"] = metrics
        self.systems[system_name]["evaluation_time"] = evaluation_time
        
        # Add per-query results for detailed analysis
        self.per_query_results[system_name] = {}
        for utterance in results_df["utterance"].unique():
            query_results = results_df[results_df["utterance"] == utterance]
            self.per_query_results[system_name][utterance] = query_results
        
        # Save detailed results if requested
        if save_detailed_results:
            detailed_results = {
                "system_name": system_name,
                "metrics": metrics,
                "evaluation_time": evaluation_time,
                "per_query_metrics": self._calculate_per_query_metrics(results_df)
            }
            
            with open(os.path.join(self.output_dir, f"{system_name}_detailed_metrics.json"), 'w') as f:
                json.dump(detailed_results, f, indent=2)
        
        logger.info(f"Evaluation of {system_name} completed in {evaluation_time:.2f}s")
        
        return metrics
    
    def _calculate_per_query_metrics(self, results_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Calculate metrics for each unique query.
        
        Args:
            results_df: DataFrame containing evaluation results
            
        Returns:
            Dictionary of per-query metrics
        """
        per_query_metrics = {}
        
        for utterance in results_df["utterance"].unique():
            query_results = results_df[results_df["utterance"] == utterance]
            
            # Calculate metrics for this query
            query_metrics = {
                "top1_accuracy": query_results["correct_top1"].mean(),
                "top3_accuracy": query_results["correct_top3"].mean(),
                "top5_accuracy": query_results["correct_top5"].mean(),
                "precision_at_3": query_results["precision_at_3"].iloc[0] if "precision_at_3" in query_results.columns else None,
                "precision_at_5": query_results["precision_at_5"].iloc[0] if "precision_at_5" in query_results.columns else None,
                "recall_at_3": query_results["recall_at_3"].iloc[0] if "recall_at_3" in query_results.columns else None,
                "recall_at_5": query_results["recall_at_5"].iloc[0] if "recall_at_5" in query_results.columns else None,
                "f1_at_3": query_results["f1_at_3"].iloc[0] if "f1_at_3" in query_results.columns else None,
                "f1_at_5": query_results["f1_at_5"].iloc[0] if "f1_at_5" in query_results.columns else None,
                "ndcg_at_3": query_results["ndcg_at_3"].iloc[0] if "ndcg_at_3" in query_results.columns else None,
                "ndcg_at_5": query_results["ndcg_at_5"].iloc[0] if "ndcg_at_5" in query_results.columns else None,
                "processing_time": query_results["processing_time"].mean() if "processing_time" in query_results.columns else None,
                "num_faqs": len(query_results),
                "predicted_faqs": [
                    query_results[f"pred_faq_{i}"].iloc[0] if f"pred_faq_{i}" in query_results.columns and i < len(query_results) else None
                    for i in range(1, 6)
                ],
                "predicted_scores": [
                    query_results[f"pred_score_{i}"].iloc[0] if f"pred_score_{i}" in query_results.columns and i < len(query_results) else None
                    for i in range(1, 6)
                ]
            }
            
            # Add reciprocal rank if available
            if pd.notna(query_results["predicted_rank"].iloc[0]):
                query_metrics["reciprocal_rank"] = 1.0 / query_results["predicted_rank"].iloc[0]
            else:
                query_metrics["reciprocal_rank"] = 0.0
            
            per_query_metrics[utterance] = query_metrics
        
        return per_query_metrics
    
    def compare_systems(self, system_names: List[str] = None, metrics: List[str] = None, save_plot: bool = True):
        """
        Compare multiple systems based on their metrics.
        
        Args:
            system_names: List of system names to compare (None for all)
            metrics: List of metrics to compare (None for all)
            save_plot: Whether to save the comparison plot
            
        Returns:
            DataFrame containing the comparison results
        """
        # Use all registered systems if none specified
        if system_names is None:
            system_names = list(self.systems.keys())
        
        # Check if all systems have been evaluated
        for system_name in system_names:
            if system_name not in self.systems or self.systems[system_name]["metrics"] is None:
                logger.error(f"System {system_name} has not been evaluated")
                raise ValueError(f"System {system_name} has not been evaluated")
        
        # Use all metrics if none specified
        if metrics is None:
            metrics = self.traditional_metrics + self.advanced_metrics
        
        # Create a DataFrame for comparison
        comparison_data = []
        for system_name in system_names:
            system_metrics = self.systems[system_name]["metrics"]
            row = {"system": system_name}
            for metric in metrics:
                if metric in system_metrics:
                    row[metric] = system_metrics[metric]
                else:
                    row[metric] = None
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Save the comparison results
        comparison_df.to_csv(os.path.join(self.output_dir, "system_comparison.csv"), index=False)
        
        # Create a comparison plot
        if save_plot and len(metrics) > 0:
            self._plot_system_comparison(comparison_df, metrics)
        
        return comparison_df
    
    def _plot_system_comparison(self, comparison_df: pd.DataFrame, metrics: List[str]):
        """
        Create a plot comparing systems across metrics.
        
        Args:
            comparison_df: DataFrame containing comparison data
            metrics: List of metrics to plot
        """
        # Set the figure size
        plt.figure(figsize=(12, 8))
        
        # Melt the DataFrame for easier plotting
        melted_df = pd.melt(comparison_df, id_vars=["system"], value_vars=metrics, var_name="metric", value_name="value")
        
        # Create the plot
        sns.barplot(x="metric", y="value", hue="system", data=melted_df)
        
        # Customize the plot
        plt.title("System Comparison Across Metrics")
        plt.xlabel("Metric")
        plt.ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.legend(title="System")
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, "system_comparison.png"), dpi=300)
        plt.close()
    
    def analyze_errors(self, system_name: str, top_k: int = 10, save_results: bool = True):
        """
        Analyze error patterns for a system.
        
        Args:
            system_name: Name of the system to analyze
            top_k: Number of top error patterns to report
            save_results: Whether to save the analysis results
            
        Returns:
            Dictionary containing error analysis results
        """
        if system_name not in self.systems or self.systems[system_name]["results"] is None:
            logger.error(f"System {system_name} has not been evaluated")
            raise ValueError(f"System {system_name} has not been evaluated")
        
        results_df = self.systems[system_name]["results"]
        
        # Get all incorrect mappings
        incorrect_mappings = results_df[results_df["correct_top1"] == 0]
        
        # Group by actual FAQ to see which FAQs are most commonly missed
        faq_error_counts = incorrect_mappings.groupby("actual_faq").size().reset_index(name="error_count")
        faq_error_counts = faq_error_counts.sort_values(by="error_count", ascending=False)
        
        # Analyze common incorrect predictions
        incorrect_predictions = {}
        for _, row in incorrect_mappings.iterrows():
            actual_faq = row["actual_faq"]
            predicted_faq = row["pred_faq_1"] if "pred_faq_1" in row else None
            
            if actual_faq not in incorrect_predictions:
                incorrect_predictions[actual_faq] = {}
            
            if predicted_faq:
                if predicted_faq not in incorrect_predictions[actual_faq]:
                    incorrect_predictions[actual_faq][predicted_faq] = 0
                incorrect_predictions[actual_faq][predicted_faq] += 1
        
        # Find the most common incorrect prediction for each actual FAQ
        common_errors = []
        for actual_faq, predictions in incorrect_predictions.items():
            sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            for predicted_faq, count in sorted_predictions[:top_k]:
                common_errors.append({
                    "actual_faq": actual_faq,
                    "predicted_faq": predicted_faq,
                    "count": count
                })
        
        # Create a DataFrame of common errors
        common_errors_df = pd.DataFrame(common_errors)
        
        # Analyze query characteristics of errors
        query_characteristics = []
        for utterance, query_results in self.per_query_results[system_name].items():
            if query_results["correct_top1"].mean() == 0:
                # This query has errors
                query_length = len(utterance.split())
                
                characteristics = {
                    "utterance": utterance,
                    "query_length": query_length,
                    "num_faqs": len(query_results),
                    "predicted_faq": query_results["pred_faq_1"].iloc[0] if "pred_faq_1" in query_results.columns else None,
                    "actual_faq": query_results["actual_faq"].iloc[0]
                }
                
                query_characteristics.append(characteristics)
        
        query_characteristics_df = pd.DataFrame(query_characteristics)
        
        # Create correlation matrix for error characteristics
        correlation_matrix = None
        if len(query_characteristics_df) > 0:
            numeric_cols = ["query_length", "num_faqs"]
            if all(col in query_characteristics_df.columns for col in numeric_cols):
                correlation_matrix = query_characteristics_df[numeric_cols].corr()
        
        # Compile the analysis results
        analysis_results = {
            "system_name": system_name,
            "total_samples": len(results_df),
            "total_errors": len(incorrect_mappings),
            "error_rate": len(incorrect_mappings) / len(results_df) if len(results_df) > 0 else 0,
            "top_missed_faqs": faq_error_counts.head(top_k).to_dict(orient="records"),
            "common_errors": common_errors_df.head(top_k).to_dict(orient="records"),
            "error_characteristics": {
                "avg_query_length": query_characteristics_df["query_length"].mean() if len(query_characteristics_df) > 0 else None,
                "correlation": correlation_matrix.to_dict() if correlation_matrix is not None else None
            }
        }
        
        # Save the analysis results
        if save_results:
            with open(os.path.join(self.output_dir, f"{system_name}_error_analysis.json"), 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Save error-related DataFrames
            faq_error_counts.to_csv(os.path.join(self.output_dir, f"{system_name}_faq_error_counts.csv"), index=False)
            common_errors_df.to_csv(os.path.join(self.output_dir, f"{system_name}_common_errors.csv"), index=False)
            query_characteristics_df.to_csv(os.path.join(self.output_dir, f"{system_name}_error_query_characteristics.csv"), index=False)
        
        return analysis_results
    
    def analyze_performance_by_category(self, system_name: str, categorization_function = None, save_results: bool = True):
        """
        Analyze system performance across different query categories.
        
        Args:
            system_name: Name of the system to analyze
            categorization_function: Function to categorize queries (None for default)
            save_results: Whether to save the analysis results
            
        Returns:
            DataFrame containing performance by category
        """
        if system_name not in self.systems or self.systems[system_name]["results"] is None:
            logger.error(f"System {system_name} has not been evaluated")
            raise ValueError(f"System {system_name} has not been evaluated")
        
        results_df = self.systems[system_name]["results"]
        
        # Use default categorization if none provided
        if categorization_function is None:
            categorization_function = self._default_query_categorization
        
        # Categorize queries
        query_categories = {}
        for utterance in results_df["utterance"].unique():
            category = categorization_function(utterance)
            query_categories[utterance] = category
        
        # Add category to results
        categorized_results = results_df.copy()
        categorized_results["category"] = categorized_results["utterance"].map(query_categories)
        
        # Calculate metrics by category
        category_metrics = []
        for category in categorized_results["category"].unique():
            category_results = categorized_results[categorized_results["category"] == category]
            
            metrics = {
                "category": category,
                "count": len(category_results["utterance"].unique()),
                "top1_accuracy": category_results["correct_top1"].mean(),
                "top3_accuracy": category_results["correct_top3"].mean(),
                "top5_accuracy": category_results["correct_top5"].mean()
            }
            
            # Add advanced metrics if available
            for metric in ["precision_at_3", "recall_at_3", "f1_at_3", "ndcg_at_3"]:
                if metric in category_results.columns:
                    metrics[metric] = category_results[metric].mean()
            
            # Add processing time if available
            if "processing_time" in category_results.columns:
                metrics["avg_processing_time"] = category_results["processing_time"].mean()
            
            category_metrics.append(metrics)
        
        # Create a DataFrame of category metrics
        category_metrics_df = pd.DataFrame(category_metrics)
        
        # Save the results
        if save_results:
            category_metrics_df.to_csv(os.path.join(self.output_dir, f"{system_name}_performance_by_category.csv"), index=False)
            
            # Create and save a bar plot of performance by category
            self._plot_performance_by_category(category_metrics_df, system_name)
        
        return category_metrics_df
    
    def _default_query_categorization(self, utterance: str) -> str:
        """
        Default function to categorize queries.
        
        Args:
            utterance: The query to categorize
            
        Returns:
            Category name
        """
        utterance = utterance.lower()
        
        # Define category keywords
        categories = {
            "account": ["account", "login", "password", "username", "profile"],
            "payment": ["payment", "pay", "bill", "invoice", "transaction"],
            "product": ["product", "service", "plan", "package", "feature"],
            "technical": ["error", "issue", "problem", "bug", "fix", "help"],
            "security": ["security", "protect", "fraud", "lock", "freeze", "alert"]
        }
        
        # Check for category keywords
        for category, keywords in categories.items():
            if any(keyword in utterance for keyword in keywords):
                return category
        
        # Check query length for complexity
        words = utterance.split()
        if len(words) <= 3:
            return "simple"
        elif len(words) <= 6:
            return "moderate"
        else:
            return "complex"
    
    def _plot_performance_by_category(self, category_metrics_df: pd.DataFrame, system_name: str):
        """
        Create a plot of performance by category.
        
        Args:
            category_metrics_df: DataFrame containing category metrics
            system_name: Name of the system
        """
        # Set the figure size
        plt.figure(figsize=(12, 8))
        
        # Select metrics to plot
        metrics_to_plot = ["top1_accuracy", "top3_accuracy", "top5_accuracy"]
        # Add additional metrics if available
        for metric in ["precision_at_3", "recall_at_3", "f1_at_3"]:
            if metric in category_metrics_df.columns:
                metrics_to_plot.append(metric)
        
        # Melt the DataFrame for easier plotting
        melted_df = pd.melt(
            category_metrics_df,
            id_vars=["category", "count"],
            value_vars=metrics_to_plot,
            var_name="metric",
            value_name="value"
        )
        
        # Create the plot
        g = sns.catplot(
            x="category",
            y="value",
            hue="metric",
            kind="bar",
            data=melted_df,
            height=6,
            aspect=1.5
        )
        
        # Customize the plot
        plt.title(f"Performance by Category for {system_name}")
        plt.xlabel("Category")
        plt.ylabel("Value")
        plt.xticks(rotation=45, ha="right")
        
        # Add the count of queries in each category as a secondary axis
        ax2 = plt.twinx()
        sns.scatterplot(
            x=range(len(category_metrics_df)),
            y="count",
            data=category_metrics_df,
            color="red",
            s=100,
            marker="o",
            ax=ax2
        )
        ax2.set_ylabel("Query Count", color="red")
        ax2.tick_params(axis="y", labelcolor="red")
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f"{system_name}_performance_by_category.png"), dpi=300)
        plt.close()
    
    def analyze_system_components(self, system_name: str, component_usage_data: Dict[str, Any] = None, save_results: bool = True):
        """
        Analyze the contribution of different system components.
        
        Args:
            system_name: Name of the system to analyze
            component_usage_data: Dictionary of component usage data (None to extract from results)
            save_results: Whether to save the analysis results
            
        Returns:
            Dictionary containing component analysis results
        """
        if system_name not in self.systems or self.systems[system_name]["results"] is None:
            logger.error(f"System {system_name} has not been evaluated")
            raise ValueError(f"System {system_name} has not been evaluated")
        
        results_df = self.systems[system_name]["results"]
        
        # Extract component usage from results if not provided
        if component_usage_data is None:
            # Try to extract from results columns
            component_usage_data = {"agents": {}, "rerankers": {}}
            
            # Look for agent-specific columns
            agent_cols = [col for col in results_df.columns if col.startswith("agent_") and col.endswith("_used")]
            for col in agent_cols:
                agent_name = col.replace("agent_", "").replace("_used", "")
                component_usage_data["agents"][agent_name] = results_df[col].mean()
            
            # Look for reranker-specific columns
            reranker_cols = [col for col in results_df.columns if col.startswith("reranker_") and col.endswith("_used")]
            for col in reranker_cols:
                reranker_name = col.replace("reranker_", "").replace("_used", "")
                component_usage_data["rerankers"][reranker_name] = results_df[col].mean()
        
        # Analyze each component's contribution to successful mappings
        component_contribution = {}
        
        # Analyze successful vs. unsuccessful mappings
        successful_mappings = results_df[results_df["correct_top1"] == 1]
        unsuccessful_mappings = results_df[results_df["correct_top1"] == 0]
        
        # Calculate success rate when each component is used
        for component_type in ["agents", "rerankers"]:
            for component_name, usage_rate in component_usage_data.get(component_type, {}).items():
                # Check if we have component-specific usage columns
                col_name = f"{component_type[:-1]}_{component_name}_used"
                
                if col_name in results_df.columns:
                    # Calculate success rate when this component is used
                    component_used = results_df[results_df[col_name] == 1]
                    success_rate_with_component = component_used["correct_top1"].mean() if len(component_used) > 0 else 0
                    
                    # Calculate success rate when this component is not used
                    component_not_used = results_df[results_df[col_name] == 0]
                    success_rate_without_component = component_not_used["correct_top1"].mean() if len(component_not_used) > 0 else 0
                    
                    component_contribution[f"{component_type[:-1]}_{component_name}"] = {
                        "usage_rate": usage_rate,
                        "success_rate_with_component": success_rate_with_component,
                        "success_rate_without_component": success_rate_without_component,
                        "contribution_delta": success_rate_with_component - success_rate_without_component
                    }
        
        # Look for workflow patterns in successful mappings
        workflow_patterns = {}
        if "workflow_pattern" in results_df.columns:
            pattern_counts = results_df.groupby(["workflow_pattern", "correct_top1"]).size().unstack(fill_value=0).reset_index()
            pattern_counts.columns = ["workflow_pattern", "unsuccessful", "successful"]
            pattern_counts["total"] = pattern_counts["unsuccessful"] + pattern_counts["successful"]
            pattern_counts["success_rate"] = pattern_counts["successful"] / pattern_counts["total"]
            
            workflow_patterns = pattern_counts.sort_values(by="success_rate", ascending=False).to_dict(orient="records")
        
        # Analyze contribution of each step in the pipeline
        pipeline_contribution = {}
        pipeline_steps = ["query_planning", "retrieval", "reranking", "judge", "response_generation"]
        
        for step in pipeline_steps:
            # Look for step-specific metrics
            step_cols = [col for col in results_df.columns if col.startswith(f"{step}_")]
            
            if step_cols:
                step_metrics = {}
                for col in step_cols:
                    metric_name = col.replace(f"{step}_", "")
                    step_metrics[metric_name] = results_df[col].mean()
                
                pipeline_contribution[step] = step_metrics
        
        # Compile the analysis results
        analysis_results = {
            "system_name": system_name,
            "component_contribution": component_contribution,
            "workflow_patterns": workflow_patterns,
            "pipeline_contribution": pipeline_contribution,
            "overall_success_rate": results_df["correct_top1"].mean()
        }
        
        # Save the analysis results
        if save_results:
            with open(os.path.join(self.output_dir, f"{system_name}_component_analysis.json"), 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Create and save component contribution plot
            self._plot_component_contribution(component_contribution, system_name)
        
        return analysis_results
    
    def _plot_component_contribution(self, component_contribution: Dict[str, Dict[str, float]], system_name: str):
        """
        Create a plot of component contribution.
        
        Args:
            component_contribution: Dictionary of component contribution data
            system_name: Name of the system
        """
        if not component_contribution:
            return
        
        # Prepare data for plotting
        components = []
        usage_rates = []
        success_with = []
        success_without = []
        deltas = []
        
        for component_name, data in component_contribution.items():
            components.append(component_name)
            usage_rates.append(data["usage_rate"])
            success_with.append(data["success_rate_with_component"])
            success_without.append(data["success_rate_without_component"])
            deltas.append(data["contribution_delta"])
        
        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            "component": components,
            "usage_rate": usage_rates,
            "success_with_component": success_with,
            "success_without_component": success_without,
            "contribution_delta": deltas
        })
        
        # Sort by contribution delta
        plot_data = plot_data.sort_values(by="contribution_delta", ascending=False)
        
        # Set the figure size
        plt.figure(figsize=(12, 8))
        
        # Create the plot
        x = range(len(plot_data))
        plt.bar(x, plot_data["contribution_delta"], color="blue", alpha=0.6, label="Contribution Delta")
        plt.plot(x, plot_data["usage_rate"], marker="o", color="red", label="Usage Rate")
        
        # Customize the plot
        plt.title(f"Component Contribution Analysis for {system_name}")
        plt.xlabel("Component")
        plt.ylabel("Contribution Delta / Usage Rate")
        plt.xticks(x, plot_data["component"], rotation=45, ha="right")
        plt.legend()
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, f"{system_name}_component_contribution.png"), dpi=300)
        plt.close()
    
    def generate_evaluation_report(self, system_names: List[str] = None, include_error_analysis: bool = True):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            system_names: List of system names to include (None for all)
            include_error_analysis: Whether to include error analysis
            
        Returns:
            A dictionary containing the report data
        """
        # Use all evaluated systems if none specified
        if system_names is None:
            system_names = [name for name, data in self.systems.items() if data["results"] is not None]
        
        if not system_names:
            logger.error("No evaluated systems found")
            return {"status": "error", "message": "No evaluated systems found"}
        
        # Build the report
        report = {
            "title": "FAQ Mapping System Evaluation Report",
            "generated_at": datetime.now().isoformat(),
            "systems": [],
            "comparisons": {},
            "errors": {},
            "recommendations": []
        }
        
        # Add system-specific data
        for system_name in system_names:
            if system_name in self.systems and self.systems[system_name]["results"] is not None:
                system_data = {
                    "name": system_name,
                    "description": self.systems[system_name]["description"],
                    "metrics": self.systems[system_name]["metrics"],
                    "evaluation_time": self.systems[system_name].get("evaluation_time", 0)
                }
                report["systems"].append(system_data)
        
        # Add system comparison
        if len(system_names) > 1:
            comparison_df = self.compare_systems(system_names, save_plot=True)
            report["comparisons"]["overall"] = comparison_df.to_dict(orient="records")
            
            # Find the best system for each metric
            best_systems = {}
            for metric in comparison_df.columns:
                if metric != "system":
                    best_idx = comparison_df[metric].idxmax()
                    if not pd.isna(best_idx):
                        best_system = comparison_df.loc[best_idx, "system"]
                        best_value = comparison_df.loc[best_idx, metric]
                        best_systems[metric] = {"system": best_system, "value": best_value}
            
            report["comparisons"]["best_systems"] = best_systems
        
        # Add error analysis
        if include_error_analysis:
            for system_name in system_names:
                error_analysis = self.analyze_errors(system_name)
                report["errors"][system_name] = error_analysis
        
        # Generate recommendations
        recommendations = []
        
        # Recommend the best overall system
        if len(system_names) > 1:
            # Calculate an overall score based on multiple metrics
            overall_scores = {}
            for system_name in system_names:
                metrics = self.systems[system_name]["metrics"]
                # Create a weighted score (example weights)
                score = (
                    0.4 * metrics.get("top1_accuracy", 0) +
                    0.2 * metrics.get("mean_reciprocal_rank", 0) +
                    0.2 * metrics.get("avg_f1_at_3", 0) +
                    0.1 * metrics.get("recall_any_top3", 0) +
                    0.1 * metrics.get("hit_rate_top5", 0)
                )
                overall_scores[system_name] = score
            
            best_system = max(overall_scores.items(), key=lambda x: x[1])
            recommendations.append({
                "type": "best_overall_system",
                "recommendation": f"The best overall system is {best_system[0]} with a score of {best_system[1]:.4f}."
            })
        
        # Add recommendations based on error analysis
        if include_error_analysis:
            for system_name in system_names:
                if system_name in report["errors"]:
                    error_data = report["errors"][system_name]
                    if error_data["error_rate"] > 0.3:  # High error rate
                        recommendations.append({
                            "type": "high_error_rate",
                            "system": system_name,
                            "recommendation": f"System {system_name} has a high error rate ({error_data['error_rate']:.2f}). Consider improving the FAQ mapping algorithm."
                        })
                    
                    # Check for commonly missed FAQs
                    if error_data["top_missed_faqs"]:
                        top_missed = error_data["top_missed_faqs"][0]
                        recommendations.append({
                            "type": "commonly_missed_faq",
                            "system": system_name,
                            "recommendation": f"System {system_name} frequently misses the FAQ '{top_missed['actual_faq']}' ({top_missed['error_count']} times). Consider adding more training examples for this FAQ."
                        })
        
        report["recommendations"] = recommendations
        
        # Save the report
        with open(os.path.join(self.output_dir, "evaluation_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        self._generate_html_report(report)
        
        return report
    
    def _generate_html_report(self, report_data: Dict[str, Any]):
        """
        Generate an HTML report from the report data.
        
        Args:
            report_data: Dictionary containing report data
        """
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{report_data["title"]}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric-good {{ color: green; }}
                .metric-bad {{ color: red; }}
                .recommendation {{ background-color: #fffde7; padding: 10px; margin: 10px 0; border-left: 4px solid #ffd600; }}
            </style>
        </head>
        <body>
            <h1>{report_data["title"]}</h1>
            <p>Generated at: {report_data["generated_at"]}</p>
            
            <h2>Systems</h2>
        """
        
        # Add system metrics
        html_content += "<table><tr><th>System</th>"
        # Add metric headers
        if report_data["systems"]:
            for metric in report_data["systems"][0]["metrics"]:
                if metric in self.traditional_metrics + self.advanced_metrics:
                    html_content += f"<th>{metric}</th>"
            html_content += "</tr>"
            
            # Add system rows
            for system_data in report_data["systems"]:
                html_content += f"<tr><td>{system_data['name']}</td>"
                for metric in system_data["metrics"]:
                    if metric in self.traditional_metrics + self.advanced_metrics:
                        value = system_data["metrics"][metric]
                        # Format according to metric type
                        if isinstance(value, float):
                            css_class = "metric-good" if value > 0.7 else "metric-bad" if value < 0.3 else ""
                            html_content += f"<td class='{css_class}'>{value:.4f}</td>"
                        else:
                            html_content += f"<td>{value}</td>"
                html_content += "</tr>"
        
        html_content += "</table>"
        
        # Add comparison section if available
        if "comparisons" in report_data and "best_systems" in report_data["comparisons"]:
            html_content += """
            <h2>Best Systems by Metric</h2>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Best System</th>
                    <th>Value</th>
                </tr>
            """
            
            for metric, data in report_data["comparisons"]["best_systems"].items():
                html_content += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{data["system"]}</td>
                    <td>{data["value"]:.4f}</td>
                </tr>
                """
            
            html_content += "</table>"
        
        # Add recommendations
        if report_data["recommendations"]:
            html_content += "<h2>Recommendations</h2>"
            
            for rec in report_data["recommendations"]:
                html_content += f"""
                <div class="recommendation">
                    <p>{rec["recommendation"]}</p>
                </div>
                """
        
        # Add error analysis if available
        if report_data["errors"]:
            html_content += "<h2>Error Analysis</h2>"
            
            for system_name, error_data in report_data["errors"].items():
                html_content += f"""
                <h3>System: {system_name}</h3>
                <p>Total errors: {error_data["total_errors"]} out of {error_data["total_samples"]} samples (Error rate: {error_data["error_rate"]:.2f})</p>
                
                <h4>Top Missed FAQs</h4>
                <table>
                    <tr>
                        <th>FAQ</th>
                        <th>Error Count</th>
                    </tr>
                """
                
                for missed_faq in error_data["top_missed_faqs"]:
                    html_content += f"""
                    <tr>
                        <td>{missed_faq["actual_faq"]}</td>
                        <td>{missed_faq["error_count"]}</td>
                    </tr>
                    """
                
                html_content += "</table>"
                
                # Add common errors
                html_content += f"""
                <h4>Common Errors</h4>
                <table>
                    <tr>
                        <th>Actual FAQ</th>
                        <th>Predicted FAQ</th>
                        <th>Count</th>
                    </tr>
                """
                
                for error in error_data["common_errors"]:
                    html_content += f"""
                    <tr>
                        <td>{error["actual_faq"]}</td>
                        <td>{error["predicted_faq"]}</td>
                        <td>{error["count"]}</td>
                    </tr>
                    """
                
                html_content += "</table>"
        
        # Add images
        html_content += """
        <h2>Visualizations</h2>
        <div>
            <h3>System Comparison</h3>
            <img src="system_comparison.png" alt="System Comparison" style="max-width: 100%;">
        </div>
        """
        
        # Add system-specific visualizations if available
        for system_name in report_data["systems"]:
            html_content += f"""
            <div>
                <h3>{system_name} Component Contribution</h3>
                <img src="{system_name}_component_contribution.png" alt="{system_name} Component Contribution" style="max-width: 100%;">
            </div>
            <div>
                <h3>{system_name} Performance by Category</h3>
                <img src="{system_name}_performance_by_category.png" alt="{system_name} Performance by Category" style="max-width: 100%;">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save the HTML report
        with open(os.path.join(self.output_dir, "evaluation_report.html"), 'w') as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report at {os.path.join(self.output_dir, 'evaluation_report.html')}")
    
    def benchmark_systems(self, system_names: List[str] = None, test_data: pd.DataFrame = None, metrics: List[str] = None, save_results: bool = True):
        """
        Benchmark multiple systems against each other.
        
        Args:
            system_names: List of system names to benchmark (None for all)
            test_data: DataFrame containing test data (None to use what was registered)
            metrics: List of metrics to compare (None for all)
            save_results: Whether to save the benchmark results
            
        Returns:
            A dictionary containing benchmark results
        """
        # Use all registered systems if none specified
        if system_names is None:
            system_names = list(self.systems.keys())
        
        # Check if systems are registered
        for system_name in system_names:
            if system_name not in self.systems:
                logger.error(f"System {system_name} not registered")
                raise ValueError(f"System {system_name} not registered")
        
        # Use default metrics if none specified
        if metrics is None:
            metrics = self.traditional_metrics + self.advanced_metrics + self.efficiency_metrics
        
        # Run evaluation for each system
        for system_name in system_names:
            if self.systems[system_name]["results"] is None or test_data is not None:
                # Evaluate the system if it hasn't been evaluated yet or if new test data is provided
                self.evaluate_system(system_name, test_data or self.test_df)
        
        # Compare the systems
        comparison_df = self.compare_systems(system_names, metrics, save_plot=True)
        
        # Calculate benchmark scores
        benchmark_scores = {}
        for system_name in system_names:
            # Calculate a combined score across metrics
            score = 0
            count = 0
            
            for metric in metrics:
                if metric in comparison_df.columns and system_name in comparison_df["system"].values:
                    # Get the value for this system and metric
                    value = comparison_df.loc[comparison_df["system"] == system_name, metric].values[0]
                    
                    if not pd.isna(value):
                        # Weight different metrics differently
                        weight = 1.0
                        if metric in ["top1_accuracy", "mean_reciprocal_rank"]:
                            weight = 2.0  # Higher weight for important metrics
                        elif metric in self.efficiency_metrics:
                            # For efficiency metrics, lower is better
                            if value > 0:
                                value = 1.0 / value  # Invert for scoring
                            weight = 0.5  # Lower weight for efficiency metrics
                        
                        score += value * weight
                        count += weight
            
            # Calculate the final score
            if count > 0:
                benchmark_scores[system_name] = score / count
            else:
                benchmark_scores[system_name] = 0
        
        # Rank the systems
        ranked_systems = sorted(benchmark_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create the benchmark results
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "systems": system_names,
            "metrics": metrics,
            "comparison": comparison_df.to_dict(orient="records"),
            "benchmark_scores": benchmark_scores,
            "ranked_systems": [{"system": s, "score": score} for s, score in ranked_systems]
        }
        
        # Save the benchmark results
        if save_results:
            with open(os.path.join(self.output_dir, "benchmark_results.json"), 'w') as f:
                json.dump(benchmark_results, f, indent=2)
            
            # Create and save a benchmark plot
            self._plot_benchmark_results(benchmark_scores)
        
        return benchmark_results
    
    def _plot_benchmark_results(self, benchmark_scores: Dict[str, float]):
        """
        Create a plot of benchmark results.
        
        Args:
            benchmark_scores: Dictionary mapping system names to benchmark scores
        """
        # Prepare data for plotting
        systems = list(benchmark_scores.keys())
        scores = list(benchmark_scores.values())
        
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]
        sorted_systems = [systems[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]
        
        # Set the figure size
        plt.figure(figsize=(10, 6))
        
        # Create the plot
        plt.bar(sorted_systems, sorted_scores, color="skyblue")
        
        # Customize the plot
        plt.title("System Benchmark Scores")
        plt.xlabel("System")
        plt.ylabel("Benchmark Score")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, max(scores) * 1.1)  # Add some space above the highest bar
        
        # Add score values on top of bars
        for i, score in enumerate(sorted_scores):
            plt.text(i, score + 0.01, f"{score:.4f}", ha="center")
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(self.output_dir, "benchmark_results.png"), dpi=300)
        plt.close()

# Example usage
def main():
    # Load the datasets
    faqs_df = pd.read_csv('aem_faqs.csv')
    test_df = pd.read_csv('test_set.csv')
    
    # Create systems to evaluate
    from enhanced_faq_mapper import EnhancedMultiAgentFAQMapper  # Your implementation
    
    # Create a basic system
    basic_system = EnhancedMultiAgentFAQMapper(faqs_df, test_df, use_memory=False, use_self_improvement=False)
    
    # Create an enhanced system with memory and self-improvement
    enhanced_system = EnhancedMultiAgentFAQMapper(faqs_df, test_df, use_memory=True, use_self_improvement=True)
    
    # Initialize the evaluation framework
    eval_framework = EvaluationFramework(output_dir="evaluation_results")
    
    # Register the systems
    eval_framework.register_system(
        "Basic", 
        basic_system, 
        "Basic multi-agent FAQ mapping system without memory or self-improvement"
    )
    eval_framework.register_system(
        "Enhanced", 
        enhanced_system, 
        "Enhanced multi-agent FAQ mapping system with memory and self-improvement"
    )
    
    # Evaluate each system on a subset of test data
    eval_framework.evaluate_system("Basic", test_df, num_samples=50)
    eval_framework.evaluate_system("Enhanced", test_df, num_samples=50)
    
    # Compare the systems
    comparison = eval_framework.compare_systems()
    print("System Comparison:")
    print(comparison)
    
    # Analyze errors
    basic_errors = eval_framework.analyze_errors("Basic")
    print("\nBasic System Error Analysis:")
    print(json.dumps(basic_errors, indent=2))
    
    # Analyze performance by category
    basic_categories = eval_framework.analyze_performance_by_category("Basic")
    print("\nBasic System Performance by Category:")
    print(basic_categories)
    
    # Generate an evaluation report
    report = eval_framework.generate_evaluation_report()
    print("\nEvaluation Report Generated")
    
    # Benchmark the systems
    benchmark = eval_framework.benchmark_systems()
    print("\nBenchmark Results:")
    print(json.dumps(benchmark["ranked_systems"], indent=2))

if __name__ == "__main__":
    main()
