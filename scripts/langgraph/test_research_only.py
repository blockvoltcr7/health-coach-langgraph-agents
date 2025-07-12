#!/usr/bin/env python3
"""Test only the research agent functionality"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.langgraph.student_homework_helper import build_homework_helper_graph, HomeworkState
from langchain_core.messages import HumanMessage
from datetime import datetime

def test_research_agent():
    """Test only the research agent"""
    
    print("\n" + "="*80)
    print("🔍 TESTING RESEARCH AGENT WITH TAVILY")
    print("="*80)
    
    # Build the graph
    homework_graph = build_homework_helper_graph()
    
    # Research question
    question = "Research what are AI agents and MCP tools"
    
    print(f"\n📝 Question: '{question}'")
    print("-"*60)
    
    # Initial state with workflow document
    state = {
        "messages": [HumanMessage(content=question)],
        "workflow": {
            "_id": f"research_test_{datetime.now().timestamp()}",
            "created_at": datetime.now().isoformat(),
            "status": "in_progress",
            "current_task": {
                "question": question,
                "type": "unknown",
                "status": "pending",
                "iterations": 0,
                "max_iterations": 3
            },
            "agents_visited": [],
            "agent_responses": {},
            "routing_history": []
        },
        "next_agent": ""
    }
    
    # Run the graph
    try:
        final_state = homework_graph.invoke(state, {"recursion_limit": 15})
        
        # Display results
        print("\n📊 RESEARCH RESULTS:")
        print("-"*40)
        
        # Show the agent's response
        for msg in reversed(final_state["messages"]):
            if hasattr(msg, 'content') and "[Supervisor:" not in msg.content and len(msg.content) > 50:
                print(msg.content)
                break
        
        # Show workflow details
        workflow = final_state.get("workflow", {})
        print(f"\n🔧 WORKFLOW DETAILS:")
        print(f"  Status: {workflow.get('status', 'unknown')}")
        print(f"  Agents Visited: {' → '.join(workflow.get('agents_visited', []))}")
        
        # Show tools used
        for agent, response in workflow.get("agent_responses", {}).items():
            if response.get("tools_used"):
                print(f"  Tools Used by {agent}: {', '.join(response['tools_used'])}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_research_agent()