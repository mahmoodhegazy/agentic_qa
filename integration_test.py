#!/usr/bin/env python
"""
Integration test for the Agentic RAG for FAQ Mapping system.
This script tests the core functionality of the system with a small test dataset.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("integration_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IntegrationTest")

def create_test_data():
    """Create test FAQ and test query data."""
    # Create a test FAQ dataset
    faqs_data = {
        'question': [
            'How do I reset my password?',
            'Where can I view my account balance?',
            'How do I transfer money between accounts?',
            'What are the fees for international transfers?',
            'How do I report a lost card?',
            'How can I change my contact information?',
            'What is the daily withdrawal limit?',
            'How do I set up direct deposit?',
            'How do I lock my card?',
            'What should I do if I suspect fraud on my account?'
        ],
        'answer': [
            'To reset your password, click on "Forgot Password" on the login page and follow the instructions sent to your email.',
            'You can view your account balance on the dashboard after logging in, or by checking your most recent statement.',
            'To transfer money between accounts, go to the "Transfers" section after logging in and select the accounts.',
            'International transfer fees vary based on destination and amount. Basic fees start at $25 plus 1% of the transfer amount.',
            'To report a lost card, call our 24/7 customer service at 1-800-123-4567 or use the "Report Lost Card" feature in the mobile app.',
            'You can update your contact information in the "Profile" or "Settings" section after logging in.',
            'The standard daily ATM withdrawal limit is $500. This can be adjusted by contacting customer service.',
            'To set up direct deposit, provide your employer with your account number and the bank routing number found on your checks or in the app.',
            'You can lock your card temporarily through the mobile app under "Card Management" or by calling customer service.',
            'If you suspect fraud, immediately lock your card through the app and call our fraud department at 1-800-123-4567.'
        ]
    }
    
    # Create a test dataset
    test_data = {
        'utterance': [
            'forgot my password',
            'where can I see how much money I have',
            'how do I move money between my accounts',
            'how much does it cost to wire money internationally',
            'lost my card what do I do',
            'need to update my phone number',
            'what is the maximum I can take out of ATM',
            'how to get my paycheck direct deposited',
            'freeze my card',
            'I think someone used my card fraudulently'
        ],
        'FAQ': [
            'How do I reset my password?',
            'Where can I view my account balance?',
            'How do I transfer money between accounts?',
            'What are the fees for international transfers?',
            'How do I report a lost card?',
            'How can I change my contact information?',
            'What is the daily withdrawal limit?',
            'How do I set up direct deposit?',
            'How do I lock my card?',
            'What should I do if I suspect fraud on my account?'
        ],
        'Rank': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    }
    
    # Create DataFrames
    faqs_df = pd.DataFrame(faqs_data)
    test_df = pd.DataFrame(test_data)
    
    # Save to CSV
    faqs_df.to_csv('test_faqs.csv', index=False)
    test_df.to_csv('test_queries.csv', index=False)
    
    return faqs_df, test_df

def import_and_test_modules():
    """Test importing all required modules."""
    import_errors = []
    
    # Define modules to import
    modules_to_test = [
        'enhanced_faq_mapper',
        'enhanced_judge_agent',
        'memory_system',
        'multi_agent_orchestrator',
        'evaluation_framework',
        'main_integration'
    ]
    
    # Try to import each module
    for module in modules_to_test:
        try:
            __import__(module)
            logger.info(f"Successfully imported {module}")
        except ImportError as e:
            error_msg = f"Failed to import {module}: {str(e)}"
            logger.error(error_msg)
            import_errors.append(error_msg)
    
    return import_errors

def test_basic_functionality(faqs_df, test_df):
    """Test basic functionality of the system."""
    try:
        # Import the necessary components
        from enhanced_faq_mapper import EnhancedMultiAgentFAQMapper
        from enhanced_judge_agent import EnhancedJudgeAgent
        from memory_system import MemorySystem
        
        # Initialize the components
        logger.info("Initializing components...")
        memory_system = MemorySystem()
        judge_agent = EnhancedJudgeAgent(faqs_df=faqs_df, test_df=test_df)
        
        # Initialize the main FAQ mapper
        faq_mapper = EnhancedMultiAgentFAQMapper(
            faqs_df=faqs_df,
            test_df=test_df,
            use_memory=True,
            use_self_improvement=True,
            judge_agent=judge_agent,
            memory_system=memory_system
        )
        
        # Test mapping a single utterance
        test_query = "how do I lock my card"
        logger.info(f"Testing mapping for query: '{test_query}'")
        
        result = faq_mapper.map_utterance(test_query, return_details=True)
        
        # Check if mapping was successful
        if not result or not result.get('ranked_faqs'):
            logger.error("Mapping failed: No ranked FAQs returned")
            return False, "Mapping failed: No ranked FAQs returned"
        
        # Log the result
        logger.info(f"Mapping successful. Top result: {result['ranked_faqs'][0][0]} with score {result['ranked_faqs'][0][1]:.2f}")
        
        # Check if the top result is as expected
        expected_faq = "How do I lock my card?"
        if any(faq == expected_faq for faq, _ in result['ranked_faqs'][:3]):
            logger.info(f"Expected FAQ '{expected_faq}' found in top 3 results")
        else:
            logger.warning(f"Expected FAQ '{expected_faq}' not found in top 3 results")
        
        # Return success
        return True, result
    except Exception as e:
        error_msg = f"Error testing basic functionality: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def test_full_integration():
    """Test the full integration via the main integration module."""
    try:
        # Import the main integration module
        from main_integration import FAQMappingSystem
        
        # Initialize the system
        logger.info("Initializing the full system...")
        system = FAQMappingSystem(
            faqs_file='test_faqs.csv',
            test_file='test_queries.csv',
            use_memory=True,
            use_self_improvement=True
        )
        
        # Test processing a query
        test_query = "how do I lock my card"
        logger.info(f"Testing processing query: '{test_query}'")
        
        result = system.process_query(test_query, return_details=True)
        
        # Check if processing was successful
        if not result or not result.get("ranked_faqs"):
            logger.error("Processing failed: No ranked FAQs returned")
            return False, "Processing failed: No ranked FAQs returned"
        
        # Log the result
        logger.info(f"Processing successful. Top result: {result['ranked_faqs'][0][0]} with score {result['ranked_faqs'][0][1]:.2f}")
        
        # Test evaluation
        logger.info("Testing system evaluation...")
        try:
            eval_result = system.evaluate(num_samples=5)
            logger.info(f"Evaluation metrics: {json.dumps(eval_result['metrics'], indent=2)}")
        except Exception as e:
            logger.warning(f"Evaluation test failed: {str(e)}")
        
        # Return success
        return True, result
    except Exception as e:
        error_msg = f"Error testing full integration: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def check_environment():
    """Check if environment is properly set up."""
    issues = []
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        warning = "OPENAI_API_KEY environment variable not set"
        logger.warning(warning)
        issues.append(warning)
    
    # Check for required files
    required_files = [
        "enhanced_faq_mapper.py",
        "enhanced_judge_agent.py", 
        "memory_system.py",
        "multi_agent_orchestrator.py",
        "evaluation_framework.py",
        "main_integration.py"
    ]
    
    for file in required_files:
        if not os.path.exists(file):
            warning = f"Required file {file} not found"
            logger.warning(warning)
            issues.append(warning)
    
    return issues

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Integration test for the FAQ Mapping System')
    parser.add_argument('--skip-basic', action='store_true', help='Skip basic functionality test')
    parser.add_argument('--skip-full', action='store_true', help='Skip full integration test')
    parser.add_argument('--force', action='store_true', help='Force test even if environment issues are detected')
    return parser.parse_args()

def main():
    """Main function that runs all tests."""
    args = parse_arguments()
    
    print("=== FAQ Mapping System Integration Test ===")
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python version: {sys.version}")
    print("=" * 40)
    
    # Check environment
    print("\nChecking environment...")
    env_issues = check_environment()
    if env_issues and not args.force:
        print("Environment issues detected:")
        for issue in env_issues:
            print(f"- {issue}")
        print("\nFix these issues or run with --force to continue anyway")
        return
    
    # Import test
    print("\nTesting module imports...")
    import_errors = import_and_test_modules()
    if import_errors:
        print("Import errors detected:")
        for error in import_errors:
            print(f"- {error}")
        print("\nTests cannot continue due to import errors")
        return
    
    # Create test data
    print("\nCreating test data...")
    faqs_df, test_df = create_test_data()
    print(f"Created test data: {len(faqs_df)} FAQs and {len(test_df)} test queries")
    
    # Basic functionality test
    if not args.skip_basic:
        print("\nTesting basic functionality...")
        basic_success, basic_result = test_basic_functionality(faqs_df, test_df)
        
        if basic_success:
            print("✅ Basic functionality test passed")
            if isinstance(basic_result, dict) and 'ranked_faqs' in basic_result:
                print("\nTop 3 results:")
                for i, (faq, score) in enumerate(basic_result['ranked_faqs'][:3], 1):
                    print(f"{i}. {faq} - Score: {score:.2f}")
        else:
            print(f"❌ Basic functionality test failed: {basic_result}")
    
    # Full integration test
    if not args.skip_full:
        print("\nTesting full integration...")
        full_success, full_result = test_full_integration()
        
        if full_success:
            print("✅ Full integration test passed")
            if isinstance(full_result, dict) and 'ranked_faqs' in full_result:
                print("\nTop 3 results:")
                for i, (faq, score) in enumerate(full_result['ranked_faqs'][:3], 1):
                    print(f"{i}. {faq} - Score: {score:.2f}")
        else:
            print(f"❌ Full integration test failed: {full_result}")
    
    # Overall result
    print("\n=== Test Summary ===")
    if (args.skip_basic or basic_success) and (args.skip_full or full_success):
        print("✅ All tests passed successfully")
    else:
        print("❌ Some tests failed, check log for details")
    
    # Clean up
    print("\nTest completed. Check integration_test.log for detailed output.")
    # Uncomment to remove test files after testing
    # os.remove('test_faqs.csv')
    # os.remove('test_queries.csv')

if __name__ == "__main__":
    main()