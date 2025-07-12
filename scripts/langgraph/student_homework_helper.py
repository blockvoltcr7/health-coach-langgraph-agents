"""
Student Homework Helper Multi-Agent System

This system demonstrates a supervisor agent that routes homework questions to specialized agents:
1. Math Solver Agent - With calculator tool for solving math problems
2. Essay Writer Agent - Specialized in helping write essays
3. Research Agent - Uses Perplexity API for research tasks
4. Study Guide Agent - Creates study guides and flashcards

The supervisor intelligently routes questions to the appropriate expert agent.
"""

import os
import json
import requests
from typing import Annotated, TypedDict, Literal, List
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, create_react_agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. Define the State
# ==========================================
class HomeworkState(TypedDict):
    """State that flows through the homework helper pipeline"""
    messages: Annotated[list[BaseMessage], add_messages]
    
    # MongoDB-like workflow document
    workflow: dict  # Contains the complete workflow state
    
    # Example workflow structure:
    # {
    #     "_id": "workflow_123",
    #     "created_at": "2024-01-11T10:00:00Z",
    #     "status": "in_progress",  # in_progress, completed, failed
    #     "current_task": {
    #         "question": "What is 156 + 789?",
    #         "type": "math",
    #         "assigned_agent": "math_solver",
    #         "status": "processing",  # pending, processing, completed
    #         "iterations": 1,
    #         "max_iterations": 3
    #     },
    #     "agents_visited": ["supervisor", "math_solver"],
    #     "agent_responses": {
    #         "math_solver": {
    #             "response": "The answer is 945",
    #             "tools_used": ["calculator"],
    #             "timestamp": "2024-01-11T10:00:05Z"
    #         }
    #     },
    #     "routing_history": [
    #         {"from": "supervisor", "to": "math_solver", "reason": "math problem detected", "timestamp": "..."}
    #     ]
    # }
    
    next_agent: Literal["math_solver", "essay_writer", "research", "study_guide", "end"]


# ==========================================
# 2. Define Tools for Agents
# ==========================================

@tool
def math_calculator(num1: float, num2: float, operation: Literal["add", "subtract", "multiply", "divide"]) -> float:
    """
    Basic math calculator tool that performs operations on two numbers.
    
    Args:
        num1: First number
        num2: Second number
        operation: Mathematical operation to perform
    
    Returns:
        float: Result of the calculation
    """
    logger.info(f"🧮 MATH TOOL CALLED: {num1} {operation} {num2}")
    
    if operation == "add":
        result = num1 + num2
    elif operation == "subtract":
        result = num1 - num2
    elif operation == "multiply":
        result = num1 * num2
    elif operation == "divide":
        if num2 != 0:
            result = num1 / num2
        else:
            return "Error: Division by zero!"
    else:
        return "Error: Unknown operation"
    
    logger.info(f"🧮 MATH TOOL RESULT: {result}")
    return result


@tool
def tavily_research(query: str) -> str:
    """
    Research tool using Tavily API to search the web for information.
    
    Args:
        query: Research query to search for
        
    Returns:
        str: Research results from Tavily
    """
    logger.info(f"🔍 TAVILY RESEARCH TOOL CALLED with query: {query}")
    
    API_KEY = os.getenv("TAVILY_API_KEY")
    if not API_KEY:
        return "Error: TAVILY_API_KEY not found in environment variables"
    
    try:
        # Import Tavily client
        from tavily import TavilyClient
        
        # Initialize client
        tavily = TavilyClient(api_key=API_KEY)
        
        # Perform search with more results for better context
        search_results = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5
        )
        
        # Format results
        formatted_results = []
        formatted_results.append(f"🔍 Search Results for: {query}\n")
        
        # Add answer if available
        if search_results.get("answer"):
            formatted_results.append(f"Summary: {search_results['answer']}\n")
        
        # Add individual results
        for i, result in enumerate(search_results.get("results", []), 1):
            formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
            formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
            formatted_results.append(f"   {result.get('content', 'No content')[:200]}...")
        
        final_result = "\n".join(formatted_results)
        logger.info(f"🔍 TAVILY RESEARCH COMPLETED: {len(final_result)} characters received")
        return final_result
        
    except ImportError:
        logger.error("Tavily package not installed. Installing...")
        # Try to use the already installed langchain_community TavilySearchResults
        try:
            # Use the working langchain Tavily integration
            from langchain_community.tools.tavily_search import TavilySearchResults
            
            search = TavilySearchResults(max_results=5)
            results = search.invoke({"query": query})
            
            # Format results
            formatted_results = [f"🔍 Search Results for: {query}\n"]
            for i, result in enumerate(results, 1):
                formatted_results.append(f"\n{i}. {result.get('title', 'No title')}")
                formatted_results.append(f"   URL: {result.get('url', 'No URL')}")
                formatted_results.append(f"   {result.get('snippet', 'No snippet')}")
            
            final_result = "\n".join(formatted_results)
            logger.info(f"🔍 TAVILY RESEARCH COMPLETED (via langchain): {len(final_result)} characters received")
            return final_result
            
        except Exception as e:
            logger.error(f"🔍 TAVILY FALLBACK ERROR: {str(e)}")
            return f"Research error: {str(e)}"
            
    except Exception as e:
        logger.error(f"🔍 TAVILY RESEARCH ERROR: {str(e)}")
        return f"Research error: {str(e)}"


# ==========================================
# 3. Create Specialized Agents
# ==========================================

def create_math_solver_agent():
    """Create the Math Solver Agent with calculator tool"""
    logger.info("🤖 Creating Math Solver Agent with calculator tool")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools([math_calculator])
    
    def math_solver_node(state: HomeworkState) -> dict:
        """Math solver agent node"""
        logger.info("🧮 MATH SOLVER AGENT ACTIVATED")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        
        # Add agent to visited list
        if "math_solver" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("math_solver")
        
        # System prompt for math agent
        system_prompt = """You are a Math Solver Agent specialized in solving mathematical problems.
        
        You have access to a calculator tool that can perform basic operations (add, subtract, multiply, divide).
        
        When given a math problem:
        1. Break down the problem into steps
        2. Use the calculator tool for calculations
        3. Show your work clearly
        4. Provide the final answer
        
        Always use the calculator tool when performing calculations to ensure accuracy."""
        
        # Create messages with system prompt
        full_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        # Get response
        response = llm_with_tools.invoke(full_messages)
        
        # Check if the model wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info("🧮 Math agent requesting tool usage")
            # The tool node will handle the actual tool execution
            return {"messages": [response], "workflow": workflow}
        else:
            logger.info(f"🧮 MATH SOLVER RESPONSE: {response.content[:100]}...")
            
            # Mark task as complete and store response
            workflow["agent_responses"]["math_solver"] = {
                "response": response.content,
                "tools_used": [],
                "timestamp": datetime.now().isoformat()
            }
            
            return {"messages": [response], "workflow": workflow}
    
    return math_solver_node


def create_essay_writer_agent():
    """Create the Essay Writer Agent"""
    logger.info("🤖 Creating Essay Writer Agent")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    
    def essay_writer_node(state: HomeworkState) -> dict:
        """Essay writer agent node"""
        logger.info("✍️ ESSAY WRITER AGENT ACTIVATED")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        
        # Add agent to visited list
        if "essay_writer" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("essay_writer")
        
        # System prompt for essay writing
        system_prompt = """You are an Essay Writer Agent specialized in helping students write essays.
        
        When helping with essays:
        1. First understand the essay topic and requirements
        2. Create an outline with introduction, body paragraphs, and conclusion
        3. Help develop a strong thesis statement
        4. Provide topic sentences for each paragraph
        5. Suggest supporting evidence and examples
        6. Help with transitions between paragraphs
        7. Ensure proper essay structure
        
        Always encourage original thinking and proper citation of sources."""
        
        # Create messages with system prompt
        full_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        # Get response
        response = llm.invoke(full_messages)
        logger.info(f"✍️ ESSAY WRITER RESPONSE: {response.content[:100]}...")
        
        # Mark task as complete and store response
        workflow["agent_responses"]["essay_writer"] = {
            "response": response.content,
            "tools_used": [],
            "timestamp": datetime.now().isoformat()
        }
        
        return {"messages": [response], "workflow": workflow}
    
    return essay_writer_node


def create_research_agent():
    """Create the Research Agent with Tavily search tool"""
    logger.info("🤖 Creating Research Agent with Tavily search tool")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Bind tools to the LLM
    llm_with_tools = llm.bind_tools([tavily_research])
    
    def research_node(state: HomeworkState) -> dict:
        """Research agent node"""
        logger.info("🔍 RESEARCH AGENT ACTIVATED")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        
        # Add agent to visited list
        if "research" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("research")
        
        # System prompt for research agent
        system_prompt = """You are a Research Agent specialized in finding reliable information.
        
        You have access to the Tavily search tool to search the web for current information.
        
        When conducting research:
        1. Identify key search terms from the user's question
        2. Use the tavily_research tool to find information
        3. Summarize findings clearly
        4. Provide sources when available
        5. Distinguish between facts and opinions
        
        Always use the research tool to find accurate, up-to-date information."""
        
        # Create messages with system prompt
        full_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        # Get response
        response = llm_with_tools.invoke(full_messages)
        
        # Check if the model wants to use tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            logger.info("🔍 Research agent requesting tool usage")
            # The tool node will handle the actual tool execution
            return {"messages": [response], "workflow": workflow}
        else:
            logger.info(f"🔍 RESEARCH RESPONSE: {response.content[:100]}...")
            
            # Mark task as complete and store response
            workflow["agent_responses"]["research"] = {
                "response": response.content,
                "tools_used": [],
                "timestamp": datetime.now().isoformat()
            }
            
            return {"messages": [response], "workflow": workflow}
    
    return research_node


def create_study_guide_agent():
    """Create the Study Guide Agent"""
    logger.info("🤖 Creating Study Guide Agent")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    def study_guide_node(state: HomeworkState) -> dict:
        """Study guide agent node"""
        logger.info("📚 STUDY GUIDE AGENT ACTIVATED")
        
        messages = state["messages"]
        workflow = state.get("workflow", {})
        
        # Add agent to visited list
        if "study_guide" not in workflow.get("agents_visited", []):
            workflow["agents_visited"].append("study_guide")
        
        # System prompt for study guide creation
        system_prompt = """You are a Study Guide Agent specialized in creating study materials.
        
        When creating study guides:
        1. Identify key concepts and topics
        2. Create clear summaries of main points
        3. Generate flashcards with questions and answers
        4. Provide memory techniques and mnemonics
        5. Create practice questions
        6. Organize information in a logical structure
        7. Use bullet points and clear formatting
        
        Make study materials engaging and easy to understand."""
        
        # Create messages with system prompt
        full_messages = [
            SystemMessage(content=system_prompt),
            *messages
        ]
        
        # Get response
        response = llm.invoke(full_messages)
        logger.info(f"📚 STUDY GUIDE RESPONSE: {response.content[:100]}...")
        
        # Mark task as complete and store response
        workflow["agent_responses"]["study_guide"] = {
            "response": response.content,
            "tools_used": [],
            "timestamp": datetime.now().isoformat()
        }
        
        return {"messages": [response], "workflow": workflow}
    
    return study_guide_node


# ==========================================
# 4. Create the Supervisor Agent
# ==========================================
def supervisor_agent(state: HomeworkState) -> dict:
    """
    Supervisor agent that analyzes the task and routes to appropriate specialist
    """
    logger.info("🎯 SUPERVISOR AGENT ACTIVATED")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    messages = state["messages"]
    workflow = state.get("workflow", {})
    
    # Initialize workflow if needed
    if not workflow:
        workflow = {
            "_id": f"workflow_{datetime.now().timestamp()}",
            "created_at": datetime.now().isoformat(),
            "status": "in_progress",
            "current_task": {},
            "agents_visited": [],
            "agent_responses": {},
            "routing_history": []
        }
    
    # Add supervisor to visited agents
    if "supervisor" not in workflow["agents_visited"]:
        workflow["agents_visited"].append("supervisor")
    
    # Get the latest user message
    latest_message = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break
    
    # Check current task status
    current_task = workflow.get("current_task", {})
    
    # If task is already completed, end the workflow
    if current_task.get("status") == "completed":
        logger.info("🎯 SUPERVISOR: Task already completed, ending workflow")
        workflow["status"] = "completed"
        return {
            "messages": [],
            "workflow": workflow,
            "next_agent": "end"
        }
    
    # Check iteration count to prevent infinite loops
    iterations = current_task.get("iterations", 0)
    max_iterations = current_task.get("max_iterations", 3)
    
    if iterations >= max_iterations:
        logger.warning(f"🎯 SUPERVISOR: Max iterations ({max_iterations}) reached, ending workflow")
        workflow["status"] = "failed"
        workflow["current_task"]["status"] = "failed"
        workflow["current_task"]["failure_reason"] = "Max iterations exceeded"
        return {
            "messages": [AIMessage(content="I apologize, but I couldn't complete this task within the allowed iterations.")],
            "workflow": workflow,
            "next_agent": "end"
        }
    
    # Check if we need to route or if task is being processed
    if current_task.get("status") == "processing" and current_task.get("assigned_agent"):
        # Check if the assigned agent has provided a response
        assigned_agent = current_task["assigned_agent"]
        if assigned_agent in workflow.get("agent_responses", {}):
            logger.info(f"🎯 SUPERVISOR: Agent {assigned_agent} completed task")
            workflow["current_task"]["status"] = "completed"
            workflow["status"] = "completed"
            return {
                "messages": [],
                "workflow": workflow,
                "next_agent": "end"
            }
    
    logger.info(f"🎯 SUPERVISOR analyzing: '{latest_message}'")
    
    # Update current task
    workflow["current_task"] = {
        "question": latest_message,
        "type": "unknown",
        "status": "pending",
        "iterations": iterations + 1,
        "max_iterations": max_iterations
    }
    
    # Create routing decision prompt
    routing_prompt = f"""You are a Homework Helper Supervisor. Analyze the student's request and route it to the appropriate specialist agent.

    Student's request: "{latest_message}"
    
    Available agents:
    1. math_solver - For mathematical problems, calculations, algebra, geometry, etc.
    2. essay_writer - For writing essays, paragraphs, creative writing, thesis statements
    3. research - For finding information, facts, current events, scientific data
    4. study_guide - For creating study guides, flashcards, summaries, test preparation
    
    Routing rules:
    - If the request involves calculations or math problems → route to 'math_solver'
    - If the request is about writing an essay or creative writing → route to 'essay_writer'
    - If the request needs factual information or research → route to 'research'
    - If the request is about studying, memorizing, or test prep → route to 'study_guide'
    - If the task is unclear → route to 'end'
    
    Agents already visited: {workflow.get('agents_visited', [])}
    
    Respond with ONLY ONE of these exact words: math_solver, essay_writer, research, study_guide, end
    """
    
    # Get routing decision
    response = llm.invoke(routing_prompt)
    next_agent = response.content.strip().lower()
    
    # Validate routing decision
    valid_agents = ["math_solver", "essay_writer", "research", "study_guide", "end"]
    if next_agent not in valid_agents:
        logger.warning(f"🎯 Invalid routing decision: {next_agent}, defaulting to 'end'")
        next_agent = "end"
    
    # Log routing decision with visual separator
    logger.info("="*60)
    logger.info(f"🎯 SUPERVISOR ROUTING DECISION: '{latest_message}' → {next_agent.upper()}")
    logger.info("="*60)
    
    # Update workflow with routing decision
    if next_agent != "end":
        workflow["current_task"]["assigned_agent"] = next_agent
        workflow["current_task"]["status"] = "processing"
        
        # Determine task type
        task_type_map = {
            "math_solver": "math",
            "essay_writer": "essay",
            "research": "research",
            "study_guide": "study_guide"
        }
        workflow["current_task"]["type"] = task_type_map.get(next_agent, "unknown")
    
    # Add routing history entry
    routing_entry = {
        "from": "supervisor",
        "to": next_agent,
        "reason": f"{workflow['current_task']['type']} task detected",
        "timestamp": datetime.now().isoformat()
    }
    workflow["routing_history"].append(routing_entry)
    
    # Create supervisor message
    supervisor_msg = AIMessage(content=f"[Supervisor: Routing to {next_agent} agent]")
    
    return {
        "messages": [supervisor_msg],
        "workflow": workflow,
        "next_agent": next_agent
    }


# ==========================================
# 5. Build the Graph
# ==========================================
def build_homework_helper_graph():
    """Build the homework helper graph with supervisor and specialized agents"""
    logger.info("🏗️ Building Homework Helper Graph")
    
    # Import tools_condition for handling tool calls
    from langgraph.prebuilt import tools_condition
    
    # Create the graph
    workflow = StateGraph(HomeworkState)
    
    # Create all agents
    math_agent = create_math_solver_agent()
    essay_agent = create_essay_writer_agent()
    research_agent = create_research_agent()
    study_agent = create_study_guide_agent()
    
    # Add all nodes
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("math_solver", math_agent)
    workflow.add_node("essay_writer", essay_agent)
    workflow.add_node("research", research_agent)
    workflow.add_node("study_guide", study_agent)
    
    # Create custom tool nodes that update workflow state
    def math_tools_with_state(state: HomeworkState) -> dict:
        """Math tools node that updates workflow state"""
        messages = state["messages"]
        workflow = state.get("workflow", {})
        
        # Execute tools
        tool_node = ToolNode(tools=[math_calculator])
        result = tool_node.invoke({"messages": messages})
        
        # Update workflow to track tool usage
        if "math_solver" not in workflow.get("agent_responses", {}):
            workflow["agent_responses"]["math_solver"] = {
                "response": "",
                "tools_used": ["calculator"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            workflow["agent_responses"]["math_solver"]["tools_used"].append("calculator")
        
        return {"messages": result["messages"], "workflow": workflow}
    
    def research_tools_with_state(state: HomeworkState) -> dict:
        """Research tools node that updates workflow state"""
        messages = state["messages"]
        workflow = state.get("workflow", {})
        
        # Execute tools
        tool_node = ToolNode(tools=[tavily_research])
        result = tool_node.invoke({"messages": messages})
        
        # Update workflow to track tool usage
        if "research" not in workflow.get("agent_responses", {}):
            workflow["agent_responses"]["research"] = {
                "response": "",
                "tools_used": ["tavily"],
                "timestamp": datetime.now().isoformat()
            }
        else:
            workflow["agent_responses"]["research"]["tools_used"].append("tavily")
        
        return {"messages": result["messages"], "workflow": workflow}
    
    # Add tool nodes for agents that use tools
    workflow.add_node("math_tools", math_tools_with_state)
    workflow.add_node("research_tools", research_tools_with_state)
    
    # Define routing function
    def route_to_agent(state: HomeworkState) -> str:
        """Route based on supervisor's decision"""
        next_agent = state.get("next_agent", "end")
        logger.info(f"🔀 Routing to: {next_agent}")
        return next_agent if next_agent != "end" else END
    
    # Add edges
    workflow.add_edge(START, "supervisor")
    
    # Supervisor routes to agents
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "math_solver": "math_solver",
            "essay_writer": "essay_writer",
            "research": "research",
            "study_guide": "study_guide",
            END: END
        }
    )
    
    # Math solver can route to tools or back to supervisor
    workflow.add_conditional_edges(
        "math_solver",
        tools_condition,
        {
            "tools": "math_tools",
            END: "supervisor"
        }
    )
    workflow.add_edge("math_tools", "math_solver")
    
    # Research agent can route to tools or back to supervisor
    workflow.add_conditional_edges(
        "research",
        tools_condition,
        {
            "tools": "research_tools",
            END: "supervisor"
        }
    )
    workflow.add_edge("research_tools", "research")
    
    # Essay and study guide agents return to supervisor
    workflow.add_edge("essay_writer", "supervisor")
    workflow.add_edge("study_guide", "supervisor")
    
    # Compile the graph
    return workflow.compile()


# ==========================================
# 6. Demo Functions
# ==========================================
def run_homework_demo():
    """Run demo scenarios to show different agent routing"""
    
    print("\n" + "="*80)
    print("🎓 STUDENT HOMEWORK HELPER - DEMO")
    print("="*80)
    
    # Build the graph
    homework_graph = build_homework_helper_graph()
    
    # Demo scenarios
    demo_questions = [
        "What is 156 + 789?",
        "Write an essay about the benefits of AI in education",
        "Research what are AI agents and MCP tools",
        "Create a study guide for machine learning basics",
        "Calculate 1000 divided by 25"
    ]
    
    for question in demo_questions:
        print(f"\n{'='*80}")
        print(f"📝 Student asks: '{question}'")
        print("="*80)
        
        # Initial state with workflow document
        state = {
            "messages": [HumanMessage(content=question)],
            "workflow": {
                "_id": f"workflow_{datetime.now().timestamp()}",
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
            final_state = homework_graph.invoke(state, {"recursion_limit": 10})
            
            # Display results
            print("\n📊 FINAL RESULT:")
            print("-"*40)
            
            # Show the last agent response
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and "[Supervisor:" not in msg.content:
                    print(f"Agent Response: {msg.content[:200]}...")
                    break
            
            # Show routing history
            if "workflow" in final_state:
                print("\n🗺️ ROUTING HISTORY:")
                for entry in final_state["workflow"].get("routing_history", []):
                    print(f"  • From: {entry['from']} → To: {entry['to']} ({entry['reason']})")
                    
                print("\n🤖 AGENTS VISITED:")
                print(f"  {' → '.join(final_state['workflow'].get('agents_visited', []))}")
                
                print("\n🔧 WORKFLOW STATUS:")
                print(f"  Status: {final_state['workflow'].get('status', 'unknown')}")
                print(f"  Task: {final_state['workflow']['current_task'].get('type', 'unknown')}")
                print(f"  Iterations: {final_state['workflow']['current_task'].get('iterations', 0)}")
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"❌ Error: {str(e)}")


def interactive_homework_helper():
    """Interactive mode where users can ask homework questions"""
    
    print("\n" + "="*80)
    print("🎓 STUDENT HOMEWORK HELPER - INTERACTIVE MODE")
    print("="*80)
    print("Ask any homework question! Type 'quit' to exit.")
    print("\nExamples:")
    print("  • 'What is 25 x 4?'")
    print("  • 'Write an essay about space exploration'")
    print("  • 'Research the water cycle'")
    print("  • 'Create a study guide for the American Revolution'")
    print("="*80)
    
    # Build the graph
    homework_graph = build_homework_helper_graph()
    
    while True:
        # Get user input
        question = input("\n📝 Your homework question: ").strip()
        
        if question.lower() == 'quit':
            print("\n👋 Thanks for using Homework Helper!")
            break
            
        if not question:
            continue
        
        print("\n" + "-"*60)
        
        # Initial state with workflow document
        state = {
            "messages": [HumanMessage(content=question)],
            "workflow": {
                "_id": f"workflow_{datetime.now().timestamp()}",
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
            final_state = homework_graph.invoke(state, {"recursion_limit": 10})
            
            # Display the agent's response
            print("\n💡 ANSWER:")
            print("-"*40)
            
            # Find and display the last non-supervisor message
            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage) and "[Supervisor:" not in msg.content:
                    print(msg.content)
                    break
            
            # Show routing path
            if "workflow" in final_state:
                print("\n🗺️ How we got your answer:")
                for entry in final_state["workflow"].get("routing_history", []):
                    print(f"  • {entry['from']} → {entry['to']} at {entry['timestamp'].split('T')[1].split('.')[0]}")
                    
                # Show which tools were used
                for agent, response in final_state["workflow"].get("agent_responses", {}).items():
                    if response.get("tools_used"):
                        print(f"\n🔧 Tools used by {agent}: {', '.join(response['tools_used'])}")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"❌ Sorry, there was an error: {str(e)}")


# ==========================================
# Main execution
# ==========================================
if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable")
        exit(1)
    
    # Run demo first
    run_homework_demo()
    
    # Then run interactive mode
    print("\n\n")
    interactive_homework_helper()