"""Limitless OS Intelligent Sales Agent.

This module provides a specialized AI agent for sales conversations focused on
selling Limitless OS services. The agent combines memory, tools, and advanced
sales techniques to qualify leads and close deals.
"""

from abc import ABC, abstractmethod
from typing import Annotated, Dict, List, Optional, Any, AsyncGenerator
from typing_extensions import TypedDict
import logging
import json
from datetime import datetime
import pytz

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Optional imports for mem0 and tools - these are gracefully handled if not available
try:
    from mem0 import AsyncMemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from .chatbot_config import ChatbotConfig

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the Limitless OS sales agent conversations.
    
    This TypedDict defines the structure of the state that flows through
    the LangGraph state machine for the sales agent.
    
    Attributes:
        messages: List of conversation messages with automatic message addition
        user_id: User identifier for mem0 memory context (required)
        metadata: Additional context data that can be passed through the graph
    """
    messages: Annotated[List[BaseMessage], add_messages]
    user_id: str
    metadata: Dict[str, Any]


@tool
def get_current_datetime() -> str:
    """Get the current date and time with timezone information.
    
    This tool provides the AI agent with current date and time context
    for all conversations, enabling time-sensitive sales interactions.
    
    Returns:
        str: Current date and time in a readable format with timezone
    """
    # Get current UTC time
    utc_now = datetime.now(pytz.UTC)
    
    # Also get US Eastern time (common business timezone)
    eastern = pytz.timezone('US/Eastern')
    eastern_now = utc_now.astimezone(eastern)
    
    return f"Current date and time: {eastern_now.strftime('%A, %B %d, %Y at %I:%M %p %Z')} (UTC: {utc_now.strftime('%Y-%m-%d %H:%M:%S UTC')})"


class LimitlessOSIntelligentAgent:
    """Advanced sales agent for Limitless OS with tools and memory.
    
    This is the primary sales agent that combines:
    - Tool usage capabilities (web search, datetime)
    - Persistent memory across conversations
    - Full LangGraph state management
    - Extensive sales-focused system prompt
    - Complete memory retrieval for context
    
    The agent serves as a supervisor sales agent focused on qualifying
    leads and closing deals for Limitless OS services.
    """
    
    def __init__(self, config: ChatbotConfig):
        """Initialize the Limitless OS sales agent with configuration.
        
        Args:
            config: ChatbotConfig object containing all necessary settings
        """
        self.config = config
        # Initialize core components
        self.llm = self._create_llm()
        self.memory = self._create_memory()
        self.tools = self._create_tools()
        self.graph = None
        # Build the LangGraph state machine
        self._build_graph()
    
    def _create_llm(self) -> ChatOpenAI:
        """Create and configure the OpenAI LLM.
        
        Returns:
            ChatOpenAI: Configured OpenAI language model instance
        """
        return ChatOpenAI(
            model=self.config.llm.model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            top_p=self.config.llm.top_p,
            frequency_penalty=self.config.llm.frequency_penalty,
            presence_penalty=self.config.llm.presence_penalty,
            api_key=self.config.llm.api_key,
        )
    
    def _create_memory(self) -> Optional[AsyncMemoryClient]:
        """Create mem0 async memory client.
        
        Returns:
            Optional[AsyncMemoryClient]: Memory client instance or None if unavailable
        """
        if not MEM0_AVAILABLE:
            logger.warning("mem0 is not available. Install with: pip install mem0ai")
            return None
        
        if not self.config.memory.api_key:
            logger.warning("MEM0_API_KEY not found in environment variables")
            return None
        
        return AsyncMemoryClient(api_key=self.config.memory.api_key)
    
    def _create_tools(self) -> List[Any]:
        """Create tools for the sales agent.
        
        Creates both web search and datetime tools for enhanced capabilities.
        
        Returns:
            List[Any]: List of initialized tool instances
        """
        tools = []
        
        # Always add the datetime tool
        tools.append(get_current_datetime)
        
        # Add web search tool if available
        if TAVILY_AVAILABLE and self.config.tools.tavily_api_key:
            tools.append(TavilySearch(
                max_results=self.config.tools.max_search_results,
                api_key=self.config.tools.tavily_api_key
            ))
        else:
            logger.warning("Tavily web search not available - check API key and installation")
        
        return tools
    
    def _build_graph(self) -> None:
        """Build the LangGraph state graph for the sales agent.
        
        Creates a graph that includes:
        1. A sales agent node that can call tools
        2. A tool node that executes the requested tools
        3. Conditional edges that route between agent and tools
        """
        graph_builder = StateGraph(AgentState)
        
        # Bind tools to the LLM so it can call them
        llm_with_tools = self.llm.bind_tools(self.tools)
        
        def sales_agent_node(state: AgentState) -> Dict[str, Any]:
            """Process messages with the sales agent LLM and tools.
            
            Args:
                state: Current conversation state
                
            Returns:
                Dict[str, Any]: Updated state with LLM response or tool calls
            """
            messages = state["messages"]
            
            # Add system prompt if configured - FIXED: Use SystemMessage instead of HumanMessage
            if self.config.system_prompt:
                # Check if we already have a system message
                has_system_message = any(isinstance(msg, SystemMessage) for msg in messages)
                
                if not has_system_message:
                    messages = [SystemMessage(content=self.config.system_prompt)] + messages
            
            # Invoke LLM with tool binding
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}
        
        # Add the main sales agent node
        graph_builder.add_node("sales_agent", sales_agent_node)
        
        # Add tool node
        if self.tools:
            tool_node = ToolNode(tools=self.tools)
            graph_builder.add_node("tools", tool_node)
            
            # Add conditional edges
            graph_builder.add_conditional_edges(
                "sales_agent",
                tools_condition,
            )
            # After tools execute, return to sales agent
            graph_builder.add_edge("tools", "sales_agent")
        
        # Start the graph at the sales agent node
        graph_builder.add_edge(START, "sales_agent")
        
        # Compile the graph
        self.graph = graph_builder.compile()
    
    async def get_all_memories(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve ALL memories for the user.
        
        Args:
            user_id: User identifier to retrieve memories for
            
        Returns:
            List[Dict[str, Any]]: List of all memory entries
        """
        if not self.memory or not user_id:
            return []
        
        try:
            # Get ALL memories for complete context
            all_memories = await self.memory.get_all(user_id=user_id)
            
            # Handle different response formats from mem0
            if isinstance(all_memories, dict):
                return all_memories.get("memories", [])
            elif isinstance(all_memories, list):
                return all_memories
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    async def chat(self, message: str, user_id: str, **kwargs) -> str:
        """Process a chat message with complete memory context.
        
        This method:
        1. Retrieves ALL memories for complete context
        2. Enhances the message with memory context
        3. Uses the sales agent to process the message (with potential tool calls)
        4. Stores the conversation in memory for future reference
        
        Args:
            message: The user's input message
            user_id: User identifier (required for memory functionality)
            **kwargs: Additional metadata
            
        Returns:
            str: The sales agent's response enhanced with tools and memory
        """
        # Get ALL memories for complete context
        memories = await self.get_all_memories(user_id)
        
        # Enhance the message with complete memory context if available
        if memories:
            # Format ALL memories for complete context
            memory_context = "\n".join([f"Memory: {mem.get('memory', '')}" for mem in memories])
            enhanced_message = f"Complete conversation history and all memories:\n{memory_context}\n\nUser: {message}"
            logger.info(f"Enhanced message with {len(memories)} memories for user {user_id}")
        else:
            enhanced_message = message
            logger.info(f"No memories found for user {user_id}")
        
        # Use user_id as thread_id for conversation continuity
        config = {"configurable": {"thread_id": user_id}}
        
        result = await self.graph.ainvoke(
            {
                "messages": [HumanMessage(content=enhanced_message)],
                "user_id": user_id,
                "metadata": kwargs
            },
            config=config
        )
        
        response = result["messages"][-1].content
        
        # Store the conversation in memory for future reference
        try:
            await self.memory.add(
                messages=[
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response}
                ],
                user_id=user_id,
                output_format=self.config.memory.output_format
            )
            logger.info(f"Added conversation to memory for user {user_id}")
        except Exception as e:
            logger.error(f"Failed to add conversation to memory: {e}")
        
        return response
    
    async def chat_stream(self, message: str, user_id: str, **kwargs) -> AsyncGenerator[str, None]:
        """Stream chat response with complete memory context.
        
        Args:
            message: The user's input message
            user_id: User identifier (required for memory functionality)
            **kwargs: Additional metadata
            
        Yields:
            str: Chunks of the response as they're generated
        """
        # Get ALL memories for complete context
        memories = await self.get_all_memories(user_id)
        
        # Enhance the message with complete memory context if available
        if memories:
            memory_context = "\n".join([f"Memory: {mem.get('memory', '')}" for mem in memories])
            enhanced_message = f"Complete conversation history and all memories:\n{memory_context}\n\nUser: {message}"
        else:
            enhanced_message = message
        
        # Use user_id as thread_id for conversation continuity
        config = {"configurable": {"thread_id": user_id}}
        
        async for chunk in self.graph.astream(
            {
                "messages": [HumanMessage(content=enhanced_message)],
                "user_id": user_id,
                "metadata": kwargs
            },
            config=config,
            stream_mode="values"
        ):
            if "messages" in chunk and chunk["messages"]:
                yield chunk["messages"][-1].content 