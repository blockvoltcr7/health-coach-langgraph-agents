"""
Module 3.1: Customer Support Bot
Time: 20 minutes
Goal: Build a production-ready customer support bot with conversation memory
"""

import os
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import hashlib
from dataclasses import dataclass, asdict
from pymongo import MongoClient
from openai import OpenAI
import voyageai
from dotenv import load_dotenv

load_dotenv()

# Initialize clients
mongodb_client = MongoClient(os.getenv("MONGODB_URI"))
db = mongodb_client[os.getenv("MONGODB_DATABASE", "rag_course")]
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
voyage_client = voyageai.Client(api_key=os.getenv("VOYAGE_AI_API_KEY"))

@dataclass
class Message:
    """Represents a conversation message"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime
    metadata: Optional[Dict] = None

@dataclass
class Conversation:
    """Represents a support conversation"""
    conversation_id: str
    user_id: str
    messages: List[Message]
    context_used: List[Dict]
    created_at: datetime
    updated_at: datetime
    status: str = "active"  # active, resolved, escalated
    tags: List[str] = None

class CustomerSupportBot:
    """
    Production-ready customer support bot with:
    - Multi-turn conversation handling
    - Context-aware responses
    - Conversation persistence
    - Analytics tracking
    """
    
    def __init__(self, knowledge_base_collection: str = "support_knowledge_base"):
        self.kb_collection = db[knowledge_base_collection]
        self.conversations_collection = db["support_conversations"]
        self.analytics_collection = db["support_analytics"]
        
        # System configuration
        self.system_prompt = """You are a helpful customer support assistant for TechCorp.
        Use the provided context to answer customer questions accurately.
        If you cannot find the answer in the context, politely say so and offer to escalate.
        Be friendly, professional, and concise.
        Always try to resolve issues in the first response when possible."""
        
        self.max_conversation_history = 10
        self.context_chunks = 5
        
    def ingest_support_docs(self, documents: List[Dict]):
        """Ingest support documentation into knowledge base"""
        print("📚 Ingesting support documentation...")
        
        # Clear existing knowledge base
        self.kb_collection.delete_many({})
        
        processed_docs = []
        for doc in documents:
            # Generate embedding
            try:
                # Try Voyage AI first
                result = voyage_client.embed(
                    texts=[doc["content"]],
                    model="voyage-3-large",
                    input_type="document"
                )
                embedding = result.embeddings[0]
            except:
                # Fallback to OpenAI
                response = openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=doc["content"]
                )
                embedding = response.data[0].embedding
            
            # Add metadata
            processed_doc = {
                **doc,
                "embedding": embedding,
                "ingested_at": datetime.utcnow(),
                "doc_id": hashlib.md5(doc["content"].encode()).hexdigest()[:8]
            }
            processed_docs.append(processed_doc)
        
        # Insert into MongoDB
        result = self.kb_collection.insert_many(processed_docs)
        print(f"✅ Ingested {len(result.inserted_ids)} support documents")
        
        # Create index reminder
        print("\n⚠️  Remember to create vector index 'support_vector_index' on 'embedding' field")
    
    def start_conversation(self, user_id: str, initial_message: str) -> Tuple[str, str]:
        """Start a new support conversation"""
        conversation_id = hashlib.md5(
            f"{user_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create conversation
        conversation = Conversation(
            conversation_id=conversation_id,
            user_id=user_id,
            messages=[],
            context_used=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            tags=[]
        )
        
        # Process first message
        response = self.process_message(conversation, initial_message)
        
        # Save conversation
        self._save_conversation(conversation)
        
        return conversation_id, response
    
    def continue_conversation(self, conversation_id: str, message: str) -> str:
        """Continue an existing conversation"""
        # Load conversation
        conversation = self._load_conversation(conversation_id)
        if not conversation:
            return "I couldn't find that conversation. Please start a new one."
        
        # Process message
        response = self.process_message(conversation, message)
        
        # Update conversation
        conversation.updated_at = datetime.utcnow()
        self._save_conversation(conversation)
        
        return response
    
    def process_message(self, conversation: Conversation, user_message: str) -> str:
        """Process a user message and generate response"""
        # Add user message
        conversation.messages.append(Message(
            role="user",
            content=user_message,
            timestamp=datetime.utcnow()
        ))
        
        # Detect intent and extract tags
        intent_tags = self._detect_intent(user_message)
        conversation.tags.extend(intent_tags)
        
        # Retrieve relevant context
        context_chunks = self._retrieve_context(user_message, conversation)
        
        # Generate response
        response = self._generate_response(
            user_message,
            context_chunks,
            conversation.messages[-self.max_conversation_history:]
        )
        
        # Add assistant message
        conversation.messages.append(Message(
            role="assistant",
            content=response,
            timestamp=datetime.utcnow(),
            metadata={"context_used": len(context_chunks)}
        ))
        
        # Track analytics
        self._track_analytics(conversation, user_message, response, context_chunks)
        
        return response
    
    def _detect_intent(self, message: str) -> List[str]:
        """Detect intent and extract tags from message"""
        tags = []
        
        # Simple keyword-based intent detection
        # In production, use NLP models or classification
        intents = {
            "billing": ["bill", "payment", "charge", "subscription", "refund"],
            "technical": ["error", "bug", "crash", "not working", "issue"],
            "account": ["password", "login", "account", "profile", "email"],
            "feature": ["how to", "tutorial", "guide", "help", "use"],
            "urgent": ["urgent", "asap", "immediately", "critical"]
        }
        
        message_lower = message.lower()
        for intent, keywords in intents.items():
            if any(keyword in message_lower for keyword in keywords):
                tags.append(intent)
        
        return tags
    
    def _retrieve_context(self, query: str, conversation: Conversation) -> List[Dict]:
        """Retrieve relevant context using vector search"""
        # Combine current query with conversation context
        conversation_context = " ".join([
            msg.content for msg in conversation.messages[-3:]
            if msg.role == "user"
        ])
        enhanced_query = f"{query} {conversation_context}"
        
        # Generate query embedding
        try:
            result = voyage_client.embed(
                texts=[enhanced_query],
                model="voyage-3-large",
                input_type="query"
            )
            query_embedding = result.embeddings[0]
        except:
            response = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=enhanced_query
            )
            query_embedding = response.data[0].embedding
        
        # Vector search with filters based on tags
        filter_stage = {}
        if "urgent" in conversation.tags:
            filter_stage = {"priority": "high"}
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "support_vector_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": 150,
                    "limit": self.context_chunks * 3,  # Get extra for reranking
                    "filter": filter_stage if filter_stage else None
                }
            },
            {
                "$project": {
                    "title": 1,
                    "content": 1,
                    "category": 1,
                    "priority": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.kb_collection.aggregate(pipeline))
        
        # Rerank if available
        if results and os.getenv("VOYAGE_AI_API_KEY"):
            try:
                documents = [r["content"] for r in results]
                reranking = voyage_client.rerank(
                    query=query,
                    documents=documents,
                    model="rerank-2-lite",
                    top_k=self.context_chunks
                )
                
                reranked_results = []
                for result in reranking.results:
                    original = results[result.index]
                    original["rerank_score"] = result.relevance_score
                    reranked_results.append(original)
                
                results = reranked_results
            except:
                # Fallback to vector scores
                results = results[:self.context_chunks]
        else:
            results = results[:self.context_chunks]
        
        # Store context used
        conversation.context_used.extend([
            {"title": r.get("title", ""), "score": r.get("score", 0)}
            for r in results
        ])
        
        return results
    
    def _generate_response(
        self,
        user_message: str,
        context_chunks: List[Dict],
        conversation_history: List[Message]
    ) -> str:
        """Generate response using GPT with context"""
        # Build context
        context = "\n\n---\n\n".join([
            f"[{chunk.get('category', 'General')}] {chunk.get('title', '')}\n{chunk.get('content', '')}"
            for chunk in context_chunks
        ])
        
        # Build conversation history
        history_messages = []
        for msg in conversation_history[:-1]:  # Exclude current message
            history_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        
        # Create messages for GPT
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"Relevant context:\n{context}"}
        ]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})
        
        # Generate response
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    
    def _save_conversation(self, conversation: Conversation):
        """Save conversation to database"""
        conv_dict = {
            "_id": conversation.conversation_id,
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "messages": [asdict(msg) for msg in conversation.messages],
            "context_used": conversation.context_used,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "status": conversation.status,
            "tags": list(set(conversation.tags))  # Unique tags
        }
        
        self.conversations_collection.replace_one(
            {"_id": conversation.conversation_id},
            conv_dict,
            upsert=True
        )
    
    def _load_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Load conversation from database"""
        doc = self.conversations_collection.find_one({"_id": conversation_id})
        if not doc:
            return None
        
        # Reconstruct messages
        messages = [
            Message(**{k: v for k, v in msg.items() if k in Message.__annotations__})
            for msg in doc["messages"]
        ]
        
        return Conversation(
            conversation_id=doc["conversation_id"],
            user_id=doc["user_id"],
            messages=messages,
            context_used=doc["context_used"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            status=doc["status"],
            tags=doc.get("tags", [])
        )
    
    def _track_analytics(
        self,
        conversation: Conversation,
        user_message: str,
        response: str,
        context_chunks: List[Dict]
    ):
        """Track analytics for monitoring"""
        analytics_event = {
            "timestamp": datetime.utcnow(),
            "conversation_id": conversation.conversation_id,
            "user_id": conversation.user_id,
            "message_count": len(conversation.messages),
            "tags": conversation.tags,
            "context_chunks_used": len(context_chunks),
            "response_length": len(response),
            "query_length": len(user_message)
        }
        
        self.analytics_collection.insert_one(analytics_event)
    
    def escalate_conversation(self, conversation_id: str, reason: str):
        """Escalate conversation to human support"""
        conversation = self._load_conversation(conversation_id)
        if conversation:
            conversation.status = "escalated"
            conversation.messages.append(Message(
                role="system",
                content=f"Escalated to human support. Reason: {reason}",
                timestamp=datetime.utcnow()
            ))
            self._save_conversation(conversation)
            
            # In production, trigger notification to support team
            print(f"🚨 Conversation {conversation_id} escalated: {reason}")

def create_sample_knowledge_base():
    """Create sample support documentation"""
    return [
        {
            "title": "How to Reset Your Password",
            "content": "To reset your password: 1. Click 'Forgot Password' on the login page. 2. Enter your email address. 3. Check your email for a reset link. 4. Click the link and create a new password. 5. Use at least 8 characters with a mix of letters and numbers.",
            "category": "Account",
            "priority": "high"
        },
        {
            "title": "Billing and Subscription Management",
            "content": "Manage your subscription in Account Settings. You can upgrade, downgrade, or cancel anytime. Billing occurs monthly on your signup date. We accept all major credit cards and PayPal. For refunds, contact support within 30 days.",
            "category": "Billing",
            "priority": "high"
        },
        {
            "title": "API Rate Limits",
            "content": "Our API has the following rate limits: Free tier: 100 requests/hour. Pro tier: 1000 requests/hour. Enterprise: Unlimited. If you exceed limits, you'll receive a 429 error. Implement exponential backoff for best results.",
            "category": "Technical",
            "priority": "medium"
        },
        {
            "title": "Common Error Codes",
            "content": "Error 400: Bad Request - Check your input parameters. Error 401: Unauthorized - Verify your API key. Error 404: Not Found - Check the endpoint URL. Error 500: Server Error - Try again later or contact support.",
            "category": "Technical",
            "priority": "high"
        },
        {
            "title": "Getting Started Guide",
            "content": "Welcome to TechCorp! Start by creating your account and verifying your email. Then, generate an API key in your dashboard. Install our SDK using pip or npm. Check our tutorials for your first integration. Join our Discord for community support.",
            "category": "Getting Started",
            "priority": "medium"
        }
    ]

def demonstrate_support_bot():
    """Demonstrate the customer support bot"""
    print("🤖 CUSTOMER SUPPORT BOT DEMO\n")
    
    # Initialize bot
    bot = CustomerSupportBot()
    
    # Ingest knowledge base
    knowledge_base = create_sample_knowledge_base()
    bot.ingest_support_docs(knowledge_base)
    
    print("\n" + "="*60)
    print("💬 Starting Support Conversations")
    print("="*60)
    
    # Demo conversations
    conversations = [
        {
            "user_id": "user_123",
            "messages": [
                "Hi, I forgot my password and can't log in",
                "I tried that but didn't receive the email",
                "Can you help me check if my email is correct?"
            ]
        },
        {
            "user_id": "user_456",
            "messages": [
                "I'm getting error 429 when using your API",
                "How can I increase my rate limit?",
                "What's included in the Pro tier?"
            ]
        },
        {
            "user_id": "user_789",
            "messages": [
                "I need a refund for my subscription",
                "I was charged twice this month",
                "This is urgent, please help!"
            ]
        }
    ]
    
    for demo in conversations:
        print(f"\n\n{'='*60}")
        print(f"👤 User: {demo['user_id']}")
        print(f"{'='*60}")
        
        # Start conversation
        conv_id = None
        for i, message in enumerate(demo['messages']):
            print(f"\n💬 User: {message}")
            
            if i == 0:
                conv_id, response = bot.start_conversation(demo['user_id'], message)
                print(f"🆔 Conversation ID: {conv_id}")
            else:
                response = bot.continue_conversation(conv_id, message)
            
            print(f"🤖 Bot: {response}")
            
            # Show conversation tags
            conv = bot._load_conversation(conv_id)
            if conv.tags:
                print(f"🏷️  Tags: {', '.join(conv.tags)}")

def show_conversation_analytics():
    """Display conversation analytics"""
    print("\n\n📊 CONVERSATION ANALYTICS\n")
    
    analytics_collection = db["support_analytics"]
    conversations_collection = db["support_conversations"]
    
    # Basic metrics
    total_conversations = conversations_collection.count_documents({})
    total_messages = analytics_collection.count_documents({})
    
    print(f"Total Conversations: {total_conversations}")
    print(f"Total Messages: {total_messages}")
    
    # Tag distribution
    pipeline = [
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    
    tag_stats = list(conversations_collection.aggregate(pipeline))
    
    print("\n📊 Issue Categories:")
    for stat in tag_stats:
        print(f"  {stat['_id']}: {stat['count']} conversations")
    
    # Resolution metrics
    status_pipeline = [
        {"$group": {"_id": "$status", "count": {"$sum": 1}}}
    ]
    
    status_stats = list(conversations_collection.aggregate(status_pipeline))
    
    print("\n📈 Resolution Status:")
    for stat in status_stats:
        print(f"  {stat['_id']}: {stat['count']} conversations")

if __name__ == "__main__":
    print("🎓 MongoDB RAG Course - Customer Support Bot\n")
    
    try:
        # Run demonstration
        demonstrate_support_bot()
        
        # Show analytics
        show_conversation_analytics()
        
        print("\n\n🎉 Key Features Demonstrated:")
        print("✅ Multi-turn conversation handling")
        print("✅ Context-aware responses")
        print("✅ Intent detection and tagging")
        print("✅ Conversation persistence")
        print("✅ Analytics tracking")
        print("✅ Escalation handling")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("🔧 Ensure MongoDB connection and API keys are configured")