#!/usr/bin/env python3
"""Test script for sales stage transition validation using sync repository."""

import logging
from datetime import datetime
from app.db.mongodb.client import get_mongodb_client
from app.db.mongodb.schemas.conversation_schema import ConversationRepository, SalesStage
from app.db.mongodb.validators import SalesStageTransitionValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_stage_transitions():
    """Test various stage transition scenarios."""
    
    # Initialize MongoDB client
    get_mongodb_client()
    
    # Create repository
    repo = ConversationRepository()
    
    # Create a test conversation
    from app.db.mongodb.schemas.conversation_schema import ConversationSchema
    doc = ConversationSchema.create_conversation_document(
        user_id="test_user_transitions_sync",
        channel="web",
        initial_message="I'm interested in testing stage transitions"
    )
    result = repo.create_one(doc)
    conversation_id = str(result.inserted_id)
    logger.info(f"Created conversation: {conversation_id}")
    
    # Test 1: Valid transition from lead to qualification
    logger.info("\nTest 1: Lead → Qualification (Valid)")
    try:
        result = repo.update_sales_stage(
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
        result = repo.update_sales_stage(
            conversation_id,
            SalesStage.CLOSED_WON.value,
            notes="Trying to skip stages"
        )
        logger.error(f"✗ This should have failed but didn't!")
    except ValueError as e:
        logger.info(f"✓ Transition correctly rejected: {e}")
    
    # Test 3: Check valid transitions
    logger.info("\nTest 3: Check valid transitions from each stage")
    for stage in [s.value for s in SalesStage]:
        next_stages = SalesStageTransitionValidator.get_next_stages(stage)
        logger.info(f"From '{stage}' can go to: {next_stages}")
    
    # Test 4: Valid transition to qualified with context
    logger.info("\nTest 4: Qualification → Qualified (with BANT completion)")
    try:
        # First update qualification data
        repo.update_qualification(
            conversation_id,
            budget_info={"meets_criteria": True, "value": "$500/month", "confidence": 80},
            authority_info={"meets_criteria": True, "role": "CEO", "confidence": 90},
            need_info={"meets_criteria": True, "pain_points": ["manual processes"], "confidence": 85},
            timeline_info={"meets_criteria": True, "timeframe": "Q1 2025", "confidence": 75}
        )
        
        # Now transition with context
        result = repo.update_sales_stage(
            conversation_id,
            SalesStage.QUALIFIED.value,
            notes="BANT qualification complete",
            context={"qualification_complete": True},
            triggered_by="qualifier_agent"
        )
        logger.info(f"✓ Transition successful: {result.modified_count} document(s) updated")
    except ValueError as e:
        logger.error(f"✗ Transition failed: {e}")
    
    # Test 5: Test terminal stages
    logger.info("\nTest 5: Check terminal stages")
    terminal_stages = ["closed_won", "closed_lost", "closed"]
    for stage in terminal_stages:
        is_terminal = SalesStageTransitionValidator.is_terminal_stage(stage)
        logger.info(f"Is '{stage}' terminal? {is_terminal}")
    
    # Test 6: Test transition requirements
    logger.info("\nTest 6: Test transition requirements")
    # Test qualification to qualified without context
    is_valid, error = SalesStageTransitionValidator.validate_transition(
        "qualification", "qualified", context=None
    )
    logger.info(f"Qualification → Qualified without context: Valid={is_valid}, Error={error}")
    
    # Test with context
    is_valid, error = SalesStageTransitionValidator.validate_transition(
        "qualification", "qualified", context={"qualification_complete": True}
    )
    logger.info(f"Qualification → Qualified with context: Valid={is_valid}, Error={error}")
    
    # Test 7: Force invalid transition with validate=False
    logger.info("\nTest 7: Force invalid transition with validate=False")
    try:
        result = repo.update_sales_stage(
            conversation_id,
            SalesStage.CLOSED_WON.value,
            notes="Forcing transition for test",
            validate=False  # Skip validation
        )
        logger.info(f"✓ Forced transition successful: {result.modified_count} document(s) updated")
    except Exception as e:
        logger.error(f"✗ Forced transition failed: {e}")
    
    # Get final conversation state
    logger.info("\nFinal conversation state:")
    final_state = repo.find_by_id(conversation_id)
    if final_state:
        logger.info(f"Current stage: {final_state.get('sales_stage')}")
        logger.info(f"Is qualified: {final_state.get('is_qualified')}")
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
    test_stage_transitions()