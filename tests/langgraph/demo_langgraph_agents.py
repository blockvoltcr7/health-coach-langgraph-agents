#!/usr/bin/env python3
"""
LangGraph Agents Demonstration Script

This script demonstrates a simple LangGraph workflow with two agents:
1. Agent 1: Says hello and passes the conversation to Agent 2
2. Agent 2: Responds with "Hello World" and adds something creative

The agents communicate through a shared state that gets passed between them.
"""

import os
import sys
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict


# Define the state structure that will be passed between agents
class AgentState(TypedDict):
    """State that gets passed between agents in the workflow"""
    messages: list[str]
    current_message: str
    agent_1_response: str
    agent_2_response: str
    step_count: int


class LangGraphAgentsDemo:
    """Demo class for LangGraph agents workflow"""
    
    def __init__(self):
        print("🚀 Initializing LangGraph Agents Demo...")
        
        # Check if OpenAI API key is set
        if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY environment variable not set!")
            print("Please set your OpenAI API key: export OPENAI_API_KEY='your-key-here'")
            sys.exit(1)
        
        # Initialize the LLM for both agents
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        print("✅ OpenAI ChatGPT initialized successfully")
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        print("✅ LangGraph workflow built successfully")
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with two agents"""
        print("🔧 Building LangGraph workflow...")
        
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("agent_1", self._agent_1_node)
        workflow.add_node("agent_2", self._agent_2_node)
        
        # Define the flow: START -> agent_1 -> agent_2 -> END
        workflow.add_edge(START, "agent_1")
        workflow.add_edge("agent_1", "agent_2")
        workflow.add_edge("agent_2", END)
        
        print("   📊 Workflow Structure:")
        print("   START → Agent 1 → Agent 2 → END")
        
        return workflow.compile()
    
    def _agent_1_node(self, state: AgentState) -> AgentState:
        """Agent 1: Says hello and prepares message for Agent 2"""
        print("\n🤖 Agent 1 is processing...")
        print("   Task: Say hello and pass conversation to Agent 2")
        
        # Agent 1 generates a greeting
        prompt = "Say hello in a friendly way and mention that you're passing the conversation to Agent 2."
        
        print(f"   🔄 Sending prompt to ChatGPT: '{prompt}'")
        response = self.llm.invoke(prompt)
        
        # Update the state
        state["agent_1_response"] = response.content
        state["messages"].append(f"Agent 1: {response.content}")
        state["current_message"] = "Please respond with 'Hello World' and add something creative."
        state["step_count"] += 1
        
        print(f"   💬 Agent 1 says: {response.content}")
        print(f"   📈 Step count: {state['step_count']}")
        
        return state
    
    def _agent_2_node(self, state: AgentState) -> AgentState:
        """Agent 2: Responds with Hello World and something creative"""
        print("\n🤖 Agent 2 is processing...")
        print("   Task: Respond with 'Hello World' and add creativity")
        
        # Agent 2 responds to the message from Agent 1
        prompt = f"""
        Agent 1 said: {state["agent_1_response"]}
        
        Now respond with "Hello World" and add something creative and fun. 
        Keep it short and engaging.
        """
        
        print(f"   🔄 Sending prompt to ChatGPT...")
        response = self.llm.invoke(prompt)
        
        # Update the state
        state["agent_2_response"] = response.content
        state["messages"].append(f"Agent 2: {response.content}")
        state["step_count"] += 1
        
        print(f"   💬 Agent 2 says: {response.content}")
        print(f"   📈 Step count: {state['step_count']}")
        
        return state
    
    def run_demo(self):
        """Run the complete demo workflow"""
        print("\n" + "="*60)
        print("🎬 Starting LangGraph Agents Demo")
        print("="*60)
        
        # Initialize the state
        initial_state = {
            "messages": [],
            "current_message": "",
            "agent_1_response": "",
            "agent_2_response": "",
            "step_count": 0
        }
        
        print("\n📋 Initial State:")
        for key, value in initial_state.items():
            print(f"   {key}: {value}")
        
        print("\n🔄 Executing workflow...")
        
        # Execute the workflow
        result = self.workflow.invoke(initial_state)
        
        print("\n" + "="*60)
        print("📊 WORKFLOW RESULTS")
        print("="*60)
        
        print(f"\n📈 Workflow Statistics:")
        print(f"   Steps completed: {result['step_count']}")
        print(f"   Messages exchanged: {len(result['messages'])}")
        
        print(f"\n🗣️ Complete Conversation:")
        for i, message in enumerate(result["messages"], 1):
            print(f"   {i}. {message}")
        
        print(f"\n🎯 Individual Agent Responses:")
        print(f"   Agent 1: {result['agent_1_response']}")
        print(f"   Agent 2: {result['agent_2_response']}")
        
        # Verify the workflow worked correctly
        print(f"\n✅ Verification:")
        if "hello world" in result["agent_2_response"].lower():
            print("   ✅ Agent 2 correctly said 'Hello World'")
        else:
            print("   ❌ Agent 2 did not say 'Hello World'")
        
        if result["step_count"] == 2:
            print("   ✅ Workflow completed all steps correctly")
        else:
            print(f"   ❌ Expected 2 steps, got {result['step_count']}")
        
        print("\n🎉 Demo completed successfully!")
        return result


def main():
    """Main function to run the LangGraph agents demo"""
    try:
        # Create and run the demo
        demo = LangGraphAgentsDemo()
        result = demo.run_demo()
        
        print("\n" + "="*60)
        print("🏁 DEMO SUMMARY")
        print("="*60)
        print("This demo showed how LangGraph can orchestrate multiple AI agents")
        print("working together in a sequential workflow:")
        print("")
        print("1. 🤖 Agent 1 received the initial task")
        print("2. 💬 Agent 1 said hello and passed control to Agent 2")
        print("3. 🤖 Agent 2 received the state from Agent 1")
        print("4. 🌍 Agent 2 responded with 'Hello World' + creativity")
        print("5. 📊 The workflow maintained state throughout the process")
        print("")
        print("This pattern can be extended to create complex multi-agent")
        print("workflows for various AI applications!")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 