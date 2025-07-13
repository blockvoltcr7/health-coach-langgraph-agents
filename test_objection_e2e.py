#!/usr/bin/env python3
"""End-to-end test for objection lifecycle in a sales conversation."""

import asyncio
import logging
from datetime import datetime, timedelta
from app.db.mongodb.async_client import get_async_mongodb_client
from app.services.conversation_service import ConversationService
from app.db.mongodb.schemas.conversation_schema import (
    SalesStage, ObjectionType, AgentName
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_objection_lifecycle_e2e():
    """Test complete objection lifecycle in a realistic sales scenario."""
    
    # Initialize MongoDB and service
    await get_async_mongodb_client()
    service = ConversationService()
    
    # Create a conversation
    logger.info("🚀 Starting end-to-end objection lifecycle test")
    logger.info("-" * 60)
    
    conversation, event = await service.create_or_resume_conversation(
        user_id="test_objection_user",
        channel="web",
        metadata={
            "source": "website",
            "campaign": "q4_2024_promo"
        }
    )
    conversation_id = str(conversation["_id"])
    logger.info(f"✅ Created conversation: {conversation_id}")
    
    # Save initial messages
    await service.save_conversation_turn(
        conversation_id,
        "I'm interested in your AI automation platform",
        "Great! I'd love to help you learn about Limitless OS. What specific challenges are you looking to solve?",
        AgentName.SUPERVISOR.value
    )
    
    # Move to qualification stage
    await service.update_sales_stage(
        conversation_id,
        SalesStage.QUALIFICATION.value,
        "User expressed interest, moving to qualification"
    )
    logger.info(f"📊 Moved to {SalesStage.QUALIFICATION.value} stage")
    
    # Simulate qualification process
    await service.save_conversation_turn(
        conversation_id,
        "We need to automate our customer support and reduce response times",
        "That's a common challenge we solve. Our AI agents can handle 80% of support tickets. What's your current ticket volume?",
        AgentName.QUALIFIER.value
    )
    
    # User raises price objection
    logger.info("\n💰 User raises PRICE objection")
    objection1_id = await service.raise_objection(
        conversation_id,
        ObjectionType.PRICE.value,
        "The pricing seems high compared to our current solution",
        severity="high"
    )
    logger.info(f"   Objection ID: {objection1_id}")
    
    # Move to objection handling stage
    await service.update_sales_stage(
        conversation_id,
        SalesStage.OBJECTION_HANDLING.value,
        "Customer raised price objection"
    )
    
    # Agent handles objection
    await service.save_message(
        conversation_id,
        AgentName.OBJECTION_HANDLER.value,
        "I understand price is important. Let me show you the ROI - most clients see 40-60% cost reduction within 3 months"
    )
    
    # Resolve the objection
    success = await service.handle_objection(
        conversation_id,
        objection1_id,
        resolution_method="roi_demonstration",
        resolution_notes="Showed cost savings calculation and customer success stories",
        handled_by=AgentName.OBJECTION_HANDLER.value,
        confidence=0.85
    )
    logger.info(f"   ✅ Objection resolved: {success}")
    
    # User raises timing objection
    logger.info("\n⏰ User raises TIMING objection")
    objection2_id = await service.raise_objection(
        conversation_id,
        ObjectionType.TIMING.value,
        "We're in the middle of Q4 planning, can't start until next year",
        severity="medium"
    )
    
    # Defer this objection
    follow_up_date = datetime.utcnow() + timedelta(days=60)
    deferred = await service.defer_objection(
        conversation_id,
        objection2_id,
        "Customer needs to complete Q4 planning first",
        follow_up_date
    )
    logger.info(f"   📅 Objection deferred: {deferred}")
    logger.info(f"   Follow-up scheduled for: {follow_up_date.strftime('%Y-%m-%d')}")
    
    # Check active objections
    active_objections = await service.get_active_objections(conversation_id)
    logger.info(f"\n📋 Active objections: {len(active_objections)}")
    for obj in active_objections:
        logger.info(f"   - {obj['type']}: {obj['content'][:50]}...")
    
    # User raises feature objection
    logger.info("\n🔧 User raises FEATURE objection")
    objection3_id = await service.raise_objection(
        conversation_id,
        ObjectionType.FEATURE.value,
        "Does it integrate with Salesforce CRM?",
        severity="high"
    )
    
    # Atomically resolve objection and move to closing
    logger.info("\n🎯 Attempting atomic objection resolution + stage transition")
    success, error = await service.perform_atomic_objection_resolution(
        conversation_id,
        objection3_id,
        resolution_data={
            "method": "feature_confirmation",
            "notes": "Confirmed native Salesforce integration with demo",
            "handled_by": AgentName.OBJECTION_HANDLER.value,
            "confidence": 0.95
        },
        move_to_closing=True
    )
    
    if success:
        logger.info("   ✅ Atomically resolved objection and moved to closing!")
    else:
        logger.error(f"   ❌ Atomic operation failed: {error}")
    
    # Get objection analytics
    logger.info("\n📈 Objection Analytics")
    analytics = await service.get_objection_analytics()
    logger.info(f"   Total objections: {analytics['total_objections']}")
    logger.info(f"   Resolved: {analytics['resolved_count']}")
    logger.info(f"   Resolution rate: {analytics['resolution_rate']:.1%}")
    
    if analytics['by_type']:
        logger.info("   By type:")
        for type_data in analytics['by_type']:
            logger.info(f"     - {type_data['_id']}: {type_data['total']} total")
    
    # Get final conversation summary
    logger.info("\n📊 Final Conversation Summary")
    summary = await service.get_conversation_summary(conversation_id)
    logger.info(f"   Status: {summary['status']}")
    logger.info(f"   Sales stage: {summary['sales_stage']}")
    logger.info(f"   Messages: {summary['message_count']}")
    logger.info(f"   Qualified: {summary['is_qualified']}")
    
    # Show objection history
    conv = await service._repository.get_conversation_state(conversation_id)
    logger.info("\n📜 Objection History")
    for i, obj in enumerate(conv.get('objections', []), 1):
        logger.info(f"\n   Objection {i}:")
        logger.info(f"     Type: {obj['type']}")
        logger.info(f"     Status: {obj['status']}")
        logger.info(f"     Severity: {obj['severity']}")
        logger.info(f"     Content: {obj['content']}")
        if obj.get('resolution', {}).get('resolved'):
            logger.info(f"     Resolution: {obj['resolution']['method']}")
            logger.info(f"     Confidence: {obj['resolution']['confidence']}")
        elif obj['status'] == 'deferred':
            logger.info(f"     Deferred reason: {obj.get('deferred_reason', 'N/A')}")
    
    logger.info("\n✅ End-to-end objection lifecycle test completed!")
    
    # Return conversation ID for cleanup
    return conversation_id


async def cleanup_test_data(conversation_id: str):
    """Clean up test data."""
    try:
        service = ConversationService()
        await service._repository.delete_by_id(conversation_id)
        logger.info(f"🧹 Cleaned up test conversation: {conversation_id}")
    except Exception as e:
        logger.error(f"Failed to clean up: {e}")


async def main():
    """Main test runner."""
    conversation_id = None
    try:
        conversation_id = await test_objection_lifecycle_e2e()
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        if conversation_id:
            await cleanup_test_data(conversation_id)


if __name__ == "__main__":
    asyncio.run(main())