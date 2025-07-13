#!/usr/bin/env python3
"""Test script for sales stage transition validation."""

import asyncio
import logging
from datetime import datetime
from app.db.mongodb.async_client import get_async_mongodb_client
from app.db.mongodb.async_conversation_repository import AsyncConversationRepository
from app.db.mongodb.schemas.conversation_schema import SalesStage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_stage_transitions():
    """Test various stage transition scenarios."""
    
    # Initialize async MongoDB client
    await get_async_mongodb_client()
    
    # Create repository
    repo = AsyncConversationRepository()
    
    # Create a test conversation
    conversation_id = await repo.create_conversation(
        user_id="test_user_transitions",
        channel="web",
        initial_message="I'm interested in testing stage transitions"
    )
    logger.info(f"Created conversation: {conversation_id}")
    
    # Test 1: Valid transition from lead to qualification
    logger.info("\nTest 1: Lead → Qualification (Valid)")
    try:
        result = await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.QUALIFICATION.value,
            notes="User expressed interest, moving to qualification"
        )
        logger.info(f"✓ Transition successful: {result.modified_count} document(s) updated")
    except ValueError as e:
        logger.error(f"✗ Transition failed: {e}")
    
    # Test 2: Invalid transition from qualification to closed_won
    logger.info("\nTest 2: Qualification → Closed Won (Invalid - should fail)")
    try:
        result = await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.CLOSED_WON.value,
            notes="Trying to skip stages"
        )
        logger.error(f"✗ This should have failed but didn't!")
    except ValueError as e:
        logger.info(f"✓ Transition correctly rejected: {e}")
    
    # Test 3: Valid transition to qualified with context
    logger.info("\nTest 3: Qualification → Qualified (Valid with context)")
    try:
        # First update qualification data
        await repo.update_qualification(
            conversation_id,
            budget_info={"meets_criteria": True, "value": "$500/month", "confidence": 80},
            authority_info={"meets_criteria": True, "role": "CEO", "confidence": 90},
            need_info={"meets_criteria": True, "pain_points": ["manual processes"], "confidence": 85},
            timeline_info={"meets_criteria": True, "timeframe": "Q1 2025", "confidence": 75}
        )
        
        # Check if qualification is complete
        is_complete, bant_status = await repo.check_qualification_complete(conversation_id)
        logger.info(f"BANT completion status: {bant_status}")
        logger.info(f"Is qualification complete? {is_complete}")
        
        # Now transition with context
        result = await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.QUALIFIED.value,
            notes="BANT qualification complete",
            context={"qualification_complete": is_complete},
            triggered_by="qualifier_agent"
        )
        logger.info(f"✓ Transition successful: {result.modified_count} document(s) updated")
    except ValueError as e:
        logger.error(f"✗ Transition failed: {e}")
    
    # Test 4: Get next valid stages
    logger.info("\nTest 4: Get next valid stages")
    next_stages = await repo.get_next_valid_stages(conversation_id)
    logger.info(f"From 'qualified', can transition to: {next_stages}")
    
    # Test 5: Transition to objection handling
    logger.info("\nTest 5: Qualified → Objection Handling (Valid)")
    try:
        # Add an objection first
        objection_id = await repo.add_objection(
            conversation_id,
            objection_type="price",
            content="$500/month seems expensive",
            severity="high"
        )
        
        # Transition to objection handling
        result = await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.OBJECTION_HANDLING.value,
            notes="Customer raised price objection",
            triggered_by="objection_handler"
        )
        logger.info(f"✓ Transition successful: {result.modified_count} document(s) updated")
    except ValueError as e:
        logger.error(f"✗ Transition failed: {e}")
    
    # Test 6: Invalid transition without required context
    logger.info("\nTest 6: Closing → Closed Lost without reason (Invalid)")
    try:
        # First move to closing
        await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.CLOSING.value,
            notes="Moving to close",
            validate=False  # Skip validation for test setup
        )
        
        # Try to close as lost without reason
        result = await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.CLOSED_LOST.value,
            notes="Lost the deal"
        )
        logger.error(f"✗ This should have failed but didn't!")
    except ValueError as e:
        logger.info(f"✓ Transition correctly rejected: {e}")
    
    # Test 7: Valid closed_lost with reason
    logger.info("\nTest 7: Closing → Closed Lost with reason (Valid)")
    try:
        result = await repo.update_sales_stage_async(
            conversation_id,
            SalesStage.CLOSED_LOST.value,
            notes="Budget constraints",
            context={"loss_reason": "Customer budget was reduced by 50%"}
        )
        logger.info(f"✓ Transition successful: {result.modified_count} document(s) updated")
    except ValueError as e:
        logger.error(f"✗ Transition failed: {e}")
    
    # Test 8: Check terminal stage
    logger.info("\nTest 8: Check terminal stage")
    from app.db.mongodb.validators import SalesStageTransitionValidator
    is_terminal = SalesStageTransitionValidator.is_terminal_stage(SalesStage.CLOSED_LOST.value)
    logger.info(f"Is 'closed_lost' a terminal stage? {is_terminal}")
    
    # Get final conversation state
    logger.info("\nFinal conversation state:")
    final_state = await repo.get_conversation_state(conversation_id)
    logger.info(f"Current stage: {final_state.get('sales_stage')}")
    logger.info(f"Stage history count: {len(final_state.get('stage_history', []))}")
    logger.info(f"Stage transitions count: {len(final_state.get('stage_transitions', []))}")
    
    # Show stage history
    logger.info("\nStage History:")
    for entry in final_state.get('stage_history', []):
        logger.info(f"  - {entry['stage']} at {entry['timestamp']}: {entry['notes']}")
    
    # Show stage transitions
    logger.info("\nStage Transitions:")
    for transition in final_state.get('stage_transitions', []):
        logger.info(f"  - {transition['from']} → {transition['to']} by {transition['triggered_by']}: {transition['reason']}")


if __name__ == "__main__":
    asyncio.run(test_stage_transitions())