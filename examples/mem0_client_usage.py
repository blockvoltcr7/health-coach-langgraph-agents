"""Example Usage of Reusable Mem0 Async Client Wrapper.

This script demonstrates how to use the reusable Mem0 async client wrapper
across different APIs and services for memory management.
"""

import asyncio
import os
from typing import Dict, Any
from datetime import datetime

# Import our reusable mem0 client components
from app.mem0.mem0AsyncClient import (
    Mem0AsyncClientWrapper,
    MemoryConfig,
    get_mem0_client,
    add_conversation_memory,
    search_user_memories,
    get_user_memory_context,
    shutdown_mem0_client
)


async def example_basic_usage():
    """Example of basic memory operations using the wrapper."""
    print("\n🔧 Basic Memory Operations Example")
    print("=" * 50)
    
    try:
        # Create a memory configuration
        config = MemoryConfig(
            api_key=os.getenv("MEM0_API_KEY"),  # Set this in your environment
            output_format="v1.1",
            max_retries=3,
            timeout=30
        )
        
        # Create client wrapper instance
        client = Mem0AsyncClientWrapper(config)
        
        # Test user ID
        user_id = "demo_user_123"
        
        # 1. Add a memory
        print("\n📝 Adding memory...")
        messages = [
            {"role": "user", "content": "I'm a software developer who loves Python"},
            {"role": "assistant", "content": "Great! I'll remember that you're a Python developer"}
        ]
        
        add_result = await client.add_memory(
            messages=messages,
            user_id=user_id,
            metadata={"source": "demo", "timestamp": datetime.now().isoformat()}
        )
        print(f"✅ Memory added: {add_result}")
        
        # 2. Search memories
        print("\n🔍 Searching memories...")
        search_result = await client.search_memories(
            query="Python developer",
            user_id=user_id,
            limit=5
        )
        print(f"✅ Found {search_result.total_count} memories:")
        for memory in search_result.memories:
            print(f"   - {memory.memory}")
        
        # 3. Get all memories
        print("\n📋 Getting all memories...")
        all_memories = await client.get_all_memories(user_id)
        print(f"✅ Total memories: {len(all_memories)}")
        for memory in all_memories:
            print(f"   - ID: {memory.id}, Content: {memory.memory}")
        
        # 4. Health check
        print("\n🏥 Health check...")
        health_status = await client.health_check()
        print(f"✅ Health status: {health_status['status']}")
        
        # 5. Clean up (delete all memories for demo)
        print("\n🧹 Cleaning up demo memories...")
        await client.delete_all_memories(user_id)
        print("✅ Demo memories cleaned up")
        
    except Exception as e:
        print(f"❌ Error in basic usage example: {e}")


async def example_global_client_usage():
    """Example of using the global client instance."""
    print("\n🌐 Global Client Usage Example")
    print("=" * 50)
    
    try:
        # Get the global client instance (reusable across your app)
        client = await get_mem0_client()
        
        user_id = "global_demo_user"
        
        # Add some memories using the global client
        print("\n📝 Adding memories using global client...")
        
        # Memory 1
        await client.add_memory(
            messages=[
                {"role": "user", "content": "I prefer morning workouts"},
                {"role": "assistant", "content": "I'll remember you prefer morning workouts"}
            ],
            user_id=user_id
        )
        
        # Memory 2
        await client.add_memory(
            messages=[
                {"role": "user", "content": "I'm interested in machine learning"},
                {"role": "assistant", "content": "Great! I'll note your interest in ML"}
            ],
            user_id=user_id
        )
        
        print("✅ Memories added successfully")
        
        # Get memory history
        print("\n📚 Getting memory history...")
        history = await client.get_memory_history(user_id, limit=10)
        print(f"✅ Found {len(history)} memories in history:")
        for i, memory in enumerate(history, 1):
            print(f"   {i}. {memory.memory}")
        
        # Clean up
        await client.delete_all_memories(user_id)
        print("\n✅ Global demo memories cleaned up")
        
    except Exception as e:
        print(f"❌ Error in global client example: {e}")


async def example_convenience_functions():
    """Example of using convenience functions."""
    print("\n⚡ Convenience Functions Example")
    print("=" * 50)
    
    try:
        user_id = "convenience_demo_user"
        
        # 1. Add conversation using convenience function
        print("\n💬 Adding conversation using convenience function...")
        await add_conversation_memory(
            user_message="What's the weather like today?",
            assistant_message="I'd need to check a weather service for current conditions in your area.",
            user_id=user_id,
            metadata={"conversation_type": "weather_inquiry"}
        )
        print("✅ Conversation added")
        
        # 2. Search using convenience function
        print("\n🔍 Searching using convenience function...")
        search_results = await search_user_memories(
            query="weather",
            user_id=user_id,
            limit=5
        )
        print(f"✅ Found {search_results.total_count} memories about weather")
        
        # 3. Get formatted context using convenience function
        print("\n📝 Getting formatted context...")
        context = await get_user_memory_context(user_id, limit=10)
        print(f"✅ Memory context:\n{context}")
        
        # Clean up
        client = await get_mem0_client()
        await client.delete_all_memories(user_id)
        print("\n✅ Convenience demo memories cleaned up")
        
    except Exception as e:
        print(f"❌ Error in convenience functions example: {e}")


async def example_error_handling():
    """Example of error handling and retry mechanisms."""
    print("\n🛡️ Error Handling Example")
    print("=" * 50)
    
    try:
        # Create client with custom retry settings
        config = MemoryConfig(
            api_key=os.getenv("MEM0_API_KEY"),
            max_retries=2,  # Lower retries for demo
            timeout=10
        )
        
        client = Mem0AsyncClientWrapper(config)
        
        # Test with invalid data to see validation
        print("\n🚫 Testing validation errors...")
        try:
            await client.add_memory([], "test_user")  # Empty messages
        except ValueError as e:
            print(f"✅ Caught validation error as expected: {e}")
        
        try:
            await client.add_memory([{"invalid": "format"}], "test_user")  # Invalid format
        except ValueError as e:
            print(f"✅ Caught format error as expected: {e}")
        
        try:
            await client.search_memories("", "test_user")  # Empty query
        except ValueError as e:
            print(f"✅ Caught empty query error as expected: {e}")
        
        print("\n✅ Error handling working correctly")
        
    except Exception as e:
        print(f"❌ Error in error handling example: {e}")


async def example_sales_agent_integration():
    """Example of how to integrate with a sales agent system."""
    print("\n🤝 Sales Agent Integration Example")
    print("=" * 50)
    
    try:
        client = await get_mem0_client()
        user_id = "sales_prospect_456"
        
        # Simulate a sales conversation
        print("\n💼 Simulating sales agent conversation...")
        
        # Customer inquiry
        await add_conversation_memory(
            user_message="I'm interested in your health coaching services",
            assistant_message="Great! I'd love to help you achieve your health goals. What specific areas are you looking to improve?",
            user_id=user_id,
            metadata={"stage": "initial_inquiry", "intent": "health_coaching"}
        )
        
        # Qualification
        await add_conversation_memory(
            user_message="I want to lose weight and build muscle",
            assistant_message="Perfect! Weight loss and muscle building are excellent goals. Do you have any previous experience with fitness programs?",
            user_id=user_id,
            metadata={"stage": "qualification", "goals": ["weight_loss", "muscle_building"]}
        )
        
        # Objection handling
        await add_conversation_memory(
            user_message="I'm concerned about the cost",
            assistant_message="I understand cost is important. Let me show you our different packages and the value each provides. Many clients find the investment pays for itself in improved health and energy.",
            user_id=user_id,
            metadata={"stage": "objection_handling", "objection": "price"}
        )
        
        # Search for specific sales insights
        print("\n🔍 Searching for sales insights...")
        
        # Find customer goals
        goals_search = await search_user_memories("goals weight loss muscle", user_id, 5)
        print(f"✅ Customer goals: {[m.memory for m in goals_search.memories]}")
        
        # Find objections
        objections_search = await search_user_memories("cost price concerned", user_id, 5)
        print(f"✅ Customer objections: {[m.memory for m in objections_search.memories]}")
        
        # Get complete context for sales agent
        sales_context = await get_user_memory_context(user_id, limit=20)
        print(f"\n📋 Complete sales context for agent:\n{sales_context}")
        
        # Clean up
        await client.delete_all_memories(user_id)
        print("\n✅ Sales demo memories cleaned up")
        
    except Exception as e:
        print(f"❌ Error in sales agent integration example: {e}")


async def main():
    """Run all examples."""
    print("🚀 Mem0 Async Client Wrapper Examples")
    print("=" * 60)
    
    # Check if API key is available
    if not os.getenv("MEM0_API_KEY"):
        print("⚠️  Warning: MEM0_API_KEY not found in environment variables")
        print("   Set your API key to run the examples with actual Mem0 service")
        print("   export MEM0_API_KEY='your_api_key_here'")
        print("\n   Running examples with mock behavior...")
        return
    
    try:
        # Run all examples
        await example_basic_usage()
        await example_global_client_usage()
        await example_convenience_functions()
        await example_error_handling()
        await example_sales_agent_integration()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
    
    finally:
        # Clean up global client
        await shutdown_mem0_client()
        print("\n🧹 Global client shutdown completed")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main()) 