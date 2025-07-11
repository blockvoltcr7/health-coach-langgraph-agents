import allure
import pytest
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# Define the state structure that will be passed between agents
class AgentState(TypedDict):
    """State that gets passed between agents in the workflow"""
    messages: list[str]
    current_message: str
    agent_1_response: str
    agent_2_response: str
    step_count: int


class LangGraphAgentsWorkflow:
    """Simple LangGraph workflow with two agents"""
    
    def __init__(self):
        # Initialize the LLM for both agents
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with two agents"""
        # Create the state graph
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("agent_1", self._agent_1_node)
        workflow.add_node("agent_2", self._agent_2_node)
        
        # Define the flow: START -> agent_1 -> agent_2 -> END
        workflow.add_edge(START, "agent_1")
        workflow.add_edge("agent_1", "agent_2")
        workflow.add_edge("agent_2", END)
        
        return workflow.compile()
    
    def _agent_1_node(self, state: AgentState) -> AgentState:
        """Agent 1: Says hello and prepares message for Agent 2"""
        print("🤖 Agent 1 is processing...")
        
        # Agent 1 generates a greeting
        prompt = "Say hello in a friendly way and mention that you're passing the conversation to Agent 2."
        response = self.llm.invoke(prompt)
        
        # Update the state
        state["agent_1_response"] = response.content
        state["messages"].append(f"Agent 1: {response.content}")
        state["current_message"] = "Please respond with 'Hello World' and add something creative."
        state["step_count"] += 1
        
        print(f"Agent 1 says: {response.content}")
        return state
    
    def _agent_2_node(self, state: AgentState) -> AgentState:
        """Agent 2: Responds with Hello World and something creative"""
        print("🤖 Agent 2 is processing...")
        
        # Agent 2 responds to the message from Agent 1
        prompt = f"""
        Agent 1 said: {state["agent_1_response"]}
        
        Now respond with "Hello World" and add something creative and fun. 
        Keep it short and engaging.
        """
        response = self.llm.invoke(prompt)
        
        # Update the state
        state["agent_2_response"] = response.content
        state["messages"].append(f"Agent 2: {response.content}")
        state["step_count"] += 1
        
        print(f"Agent 2 says: {response.content}")
        return state
    
    def run_workflow(self, initial_state: AgentState = None) -> AgentState:
        """Run the complete workflow"""
        if initial_state is None:
            initial_state = {
                "messages": [],
                "current_message": "",
                "agent_1_response": "",
                "agent_2_response": "",
                "step_count": 0
            }
        
        # Execute the workflow
        result = self.workflow.invoke(initial_state)
        return result


@allure.epic("LangGraph Agents")
@allure.feature("Multi-Agent Workflow")
class TestLangGraphAgents:
    """Test suite for LangGraph agents workflow"""
    
    @pytest.fixture
    def workflow(self):
        """Create a LangGraph workflow instance"""
        return LangGraphAgentsWorkflow()
    
    @allure.story("Basic Two-Agent Communication")
    @allure.severity(allure.severity_level.CRITICAL)
    def test_basic_agent_workflow(self, workflow):
        """Test basic workflow where Agent 1 says hello and Agent 2 responds with Hello World"""
        
        with allure.step("Initialize the workflow"):
            initial_state = {
                "messages": [],
                "current_message": "",
                "agent_1_response": "",
                "agent_2_response": "",
                "step_count": 0
            }
            
            # Attach initial state for debugging
            allure.attach(
                str(initial_state),
                name="Initial State",
                attachment_type=allure.attachment_type.JSON
            )
        
        with allure.step("Execute the agent workflow"):
            # Run the workflow
            result = workflow.run_workflow(initial_state)
            
            # Attach the result for debugging
            allure.attach(
                str(result),
                name="Final State",
                attachment_type=allure.attachment_type.JSON
            )
        
        with allure.step("Verify Agent 1 response"):
            # Verify Agent 1 responded
            assert result["agent_1_response"] != "", "Agent 1 should have provided a response"
            assert len(result["agent_1_response"]) > 0, "Agent 1 response should not be empty"
            print(f"✅ Agent 1 Response: {result['agent_1_response']}")
        
        with allure.step("Verify Agent 2 response contains Hello World"):
            # Verify Agent 2 responded with Hello World
            assert result["agent_2_response"] != "", "Agent 2 should have provided a response"
            assert "hello world" in result["agent_2_response"].lower(), "Agent 2 should include 'Hello World' in response"
            print(f"✅ Agent 2 Response: {result['agent_2_response']}")
        
        with allure.step("Verify workflow state management"):
            # Verify the state was properly managed
            assert result["step_count"] == 2, "Should have completed 2 steps"
            assert len(result["messages"]) == 2, "Should have 2 messages in conversation"
            assert result["messages"][0].startswith("Agent 1:"), "First message should be from Agent 1"
            assert result["messages"][1].startswith("Agent 2:"), "Second message should be from Agent 2"
        
        with allure.step("Verify conversation flow"):
            # Print the full conversation for verification
            print("\n🗣️  Complete Conversation:")
            for i, message in enumerate(result["messages"], 1):
                print(f"{i}. {message}")
            
            # Attach conversation log
            conversation_log = "\n".join([f"{i}. {msg}" for i, msg in enumerate(result["messages"], 1)])
            allure.attach(
                conversation_log,
                name="Complete Conversation",
                attachment_type=allure.attachment_type.TEXT
            )
    
    @allure.story("Agent State Persistence")
    @allure.severity(allure.severity_level.NORMAL)
    def test_agent_state_persistence(self, workflow):
        """Test that agent state is properly maintained throughout the workflow"""
        
        with allure.step("Create initial state with custom data"):
            initial_state = {
                "messages": ["Initial message"],
                "current_message": "",
                "agent_1_response": "",
                "agent_2_response": "",
                "step_count": 0
            }
        
        with allure.step("Run workflow with custom initial state"):
            result = workflow.run_workflow(initial_state)
        
        with allure.step("Verify state persistence"):
            # The initial message should still be there plus the new ones
            assert len(result["messages"]) == 3, "Should have initial message plus 2 agent messages"
            assert result["messages"][0] == "Initial message", "Initial message should be preserved"
            assert result["step_count"] == 2, "Step count should be incremented correctly"
    
    @allure.story("Workflow Error Handling")
    @allure.severity(allure.severity_level.NORMAL)
    def test_workflow_with_empty_state(self, workflow):
        """Test workflow handles empty initial state gracefully"""
        
        with allure.step("Run workflow with default empty state"):
            result = workflow.run_workflow()
        
        with allure.step("Verify workflow completed successfully"):
            assert result is not None, "Workflow should return a result"
            assert result["step_count"] > 0, "Should have completed at least one step"
            assert len(result["messages"]) > 0, "Should have generated messages"


# Standalone function for quick testing
def test_simple_langgraph_workflow():
    """Simple test function that can be run independently"""
    print("🚀 Starting LangGraph Agents Test...")
    
    # Create and run the workflow
    workflow = LangGraphAgentsWorkflow()
    result = workflow.run_workflow()
    
    # Print results
    print(f"\n📊 Workflow Results:")
    print(f"   Steps completed: {result['step_count']}")
    print(f"   Messages exchanged: {len(result['messages'])}")
    print(f"   Agent 1 said: {result['agent_1_response']}")
    print(f"   Agent 2 said: {result['agent_2_response']}")
    
    # Verify basic functionality
    assert "hello world" in result["agent_2_response"].lower(), "Agent 2 should say Hello World"
    print("✅ Test passed! Agents communicated successfully.")


if __name__ == "__main__":
    # Run the simple test when script is executed directly
    test_simple_langgraph_workflow()