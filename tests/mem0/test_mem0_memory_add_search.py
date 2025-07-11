import asyncio
import os
from mem0 import AsyncMemoryClient
from dotenv import load_dotenv

load_dotenv()

async def quick_test():
    # Initialize
    client = AsyncMemoryClient(api_key=os.getenv("MEM0_API_KEY"))
    
    # Add memory
    result = await client.add(
        messages=[
            {"role": "user", "content": "i need to find a good restaurant in san francisco"},
            {"role": "assistant", "content": "Great goal! i found it for you here: https://www.yelp.com/search?find_desc=Restaurants&find_loc=San+Francisco%2C+CA&ns=1"}
        ],
        user_id="ig_user_456",
        output_format="v1.1"  # Explicitly set to v1.1
    )
    
    print(f"Memory added: {result}")

    # Search for memory
    search_result = await client.search(
        query="i need to find a good restaurant in san francisco",
        user_id="ig_user_456",
        output_format="v1.1"
    )
    print(f"Search result: {search_result}")

# Run
asyncio.run(quick_test())