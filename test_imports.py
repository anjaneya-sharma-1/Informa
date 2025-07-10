#!/usr/bin/env python3
"""Test script to check imports and identify issues"""

import sys
import traceback

def test_import(module_name, from_import=None):
    """Test a specific import and print results"""
    try:
        if from_import:
            exec(f"from {module_name} import {from_import}")
            print(f"✅ Successfully imported {from_import} from {module_name}")
        else:
            exec(f"import {module_name}")
            print(f"✅ Successfully imported {module_name}")
        return True
    except Exception as e:
        print(f"❌ Failed to import {from_import or module_name} from {module_name}: {e}")
        print(f"   Full traceback: {traceback.format_exc()}")
        return False

def main():
    """Run import tests"""
    print("🔍 Testing imports...\n")
    
    # Test basic dependencies
    test_import("pandas")
    test_import("streamlit")
    test_import("aiohttp")
    test_import("chromadb")
    
    print("\n🔍 Testing LangGraph...")
    test_import("langgraph")
    test_import("langgraph.graph", "StateGraph")
    test_import("langgraph.graph", "END")
    test_import("langchain_core.messages", "BaseMessage")
    
    print("\n🔍 Testing local modules...")
    test_import("utils.config", "AppConfig")
    test_import("utils.database", "VectorDatabase")
    test_import("agents.news_collector", "NewsCollectorAgent")
    test_import("agents.sentiment_analyzer", "SentimentAnalyzerAgent")
    test_import("agents.bias_detector", "BiasDetectorAgent")
    test_import("agents.fact_checker", "FactCheckerAgent")
    
    print("\n🔍 Testing chat agent...")
    test_import("agents.chat_agent", "ChatAgent")
    
    print("\n🔍 Testing workflow...")
    test_import("agents.workflow", "NewsWorkflow")
    test_import("agents.workflow", "ChatWorkflow")
    
    print("\n🔍 Testing main app...")
    test_import("app", "InformaApp")
    
    print("\n✅ Import testing complete!")

if __name__ == "__main__":
    main()
