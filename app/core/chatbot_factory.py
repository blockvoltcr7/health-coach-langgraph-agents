"""Factory for creating the Limitless OS Intelligent Sales Agent.

This module provides a simplified factory for creating the specialized sales agent
focused on selling Limitless OS services. The factory creates properly configured
agent instances with all necessary capabilities.
"""

from typing import Optional
import logging

# Import the sales agent class
from .chatbot_base import LimitlessOSIntelligentAgent
from .chatbot_config import ChatbotConfig, get_sales_agent_config

# Set up logging for factory operations
logger = logging.getLogger(__name__)


class SalesAgentFactory:
    """Factory class for creating Limitless OS sales agent instances.
    
    This simplified factory provides a centralized way of creating the
    specialized sales agent with proper configuration and capabilities.
    """
    
    @classmethod
    def create_sales_agent(cls, config: Optional[ChatbotConfig] = None) -> LimitlessOSIntelligentAgent:
        """Create a Limitless OS sales agent instance.
        
        Args:
            config: Optional ChatbotConfig instance. If None, uses default config.
            
        Returns:
            LimitlessOSIntelligentAgent: Configured sales agent instance
        """
        if config is None:
            config = get_sales_agent_config()
        
        logger.info(f"Creating Limitless OS sales agent: {config.name}")
        return LimitlessOSIntelligentAgent(config)
    
    @classmethod
    def create_default_agent(cls) -> LimitlessOSIntelligentAgent:
        """Create a sales agent with default configuration.
        
        Returns:
            LimitlessOSIntelligentAgent: Sales agent with default settings
        """
        return cls.create_sales_agent()


# Convenience functions for creating the sales agent
def create_sales_agent(config: Optional[ChatbotConfig] = None) -> LimitlessOSIntelligentAgent:
    """Create a Limitless OS sales agent.
    
    Convenience function that wraps SalesAgentFactory.create_sales_agent()
    to provide a more functional interface.
    
    Args:
        config: Optional configuration object. If None, uses default.
        
    Returns:
        LimitlessOSIntelligentAgent: Configured sales agent instance
    """
    return SalesAgentFactory.create_sales_agent(config)


def create_default_sales_agent() -> LimitlessOSIntelligentAgent:
    """Create a sales agent with default configuration.
    
    Returns:
        LimitlessOSIntelligentAgent: Sales agent with default settings
    """
    return SalesAgentFactory.create_default_agent()


# Legacy compatibility functions (for existing code that might use these)
def create_chatbot_from_name(name: str) -> LimitlessOSIntelligentAgent:
    """Create a sales agent (legacy compatibility function).
    
    This function maintains compatibility with existing code that expects
    different chatbot types. Now it always returns the sales agent.
    
    Args:
        name: Ignored parameter (for compatibility)
        
    Returns:
        LimitlessOSIntelligentAgent: Sales agent instance
    """
    logger.info(f"Legacy function called with name: {name}, returning sales agent")
    return create_default_sales_agent()


def create_chatbot_from_config(config: ChatbotConfig) -> LimitlessOSIntelligentAgent:
    """Create a sales agent from configuration (legacy compatibility function).
    
    Args:
        config: ChatbotConfig instance
        
    Returns:
        LimitlessOSIntelligentAgent: Sales agent instance
    """
    return create_sales_agent(config) 