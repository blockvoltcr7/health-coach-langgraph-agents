#!/usr/bin/env python3
"""Tests for sales process tracking methods."""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from bson import ObjectId

from app.db.mongodb.async_client import get_async_mongodb_client
from app.db.mongodb.async_conversation_repository import AsyncConversationRepository
from app.db.mongodb.schemas.conversation_schema import (
    SalesStage, ObjectionType, AgentName
)
from app.services.conversation_service import ConversationService
from app.db.mongodb.validators import SalesStageTransitionValidator


class TestSalesProcessTracking:
    """Test suite for sales process tracking functionality."""
    
    @pytest.fixture
    async def setup(self):
        """Setup test environment."""
        # Initialize MongoDB client
        await get_async_mongodb_client()
        
        # Create repository and service
        self.repo = AsyncConversationRepository()
        self.service = ConversationService(repository=self.repo)
        
        # Create test conversation
        self.conversation_id = await self.repo.create_conversation(
            user_id=f"test_user_{ObjectId()}",
            channel="web",
            initial_message="Testing sales process tracking"
        )
        
        yield
        
        # Cleanup - delete test conversation
        try:
            await self.repo.delete_by_id(self.conversation_id)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_stage_transition_validation(self, setup):
        """Test sales stage transition validation."""
        # Valid transition: lead → qualification
        is_valid, current_stage, error = await self.repo.validate_stage_transition(
            self.conversation_id,
            SalesStage.QUALIFICATION.value
        )
        assert is_valid is True
        assert current_stage == SalesStage.LEAD.value
        assert error is None
        
        # Perform the transition
        result = await self.repo.update_sales_stage_async(
            self.conversation_id,
            SalesStage.QUALIFICATION.value,
            notes="Moving to qualification"
        )
        assert result.modified_count == 1
        
        # Invalid transition: qualification → closed_won
        is_valid, current_stage, error = await self.repo.validate_stage_transition(
            self.conversation_id,
            SalesStage.CLOSED_WON.value
        )
        assert is_valid is False
        assert "Cannot transition from 'qualification' to 'closed_won'" in error
        
        # Test with validation enforcement
        with pytest.raises(ValueError, match="Invalid stage transition"):
            await self.repo.update_sales_stage_async(
                self.conversation_id,
                SalesStage.CLOSED_WON.value,
                validate=True
            )
    
    @pytest.mark.asyncio
    async def test_qualification_to_qualified_with_context(self, setup):
        """Test transition from qualification to qualified with BANT context."""
        # Move to qualification
        await self.repo.update_sales_stage_async(
            self.conversation_id,
            SalesStage.QUALIFICATION.value,
            validate=False
        )
        
        # Try to move to qualified without context
        with pytest.raises(ValueError, match="Cannot move to qualified without completing BANT"):
            await self.repo.update_sales_stage_async(
                self.conversation_id,
                SalesStage.QUALIFIED.value,
                validate=True
            )
        
        # Update qualification data
        await self.repo.update_qualification(
            self.conversation_id,
            budget_info={"meets_criteria": True, "value": "$1000/month", "confidence": 85},
            authority_info={"meets_criteria": True, "role": "CTO", "confidence": 90},
            need_info={"meets_criteria": True, "pain_points": ["slow processes"], "confidence": 80},
            timeline_info={"meets_criteria": True, "timeframe": "Q1 2025", "confidence": 75}
        )
        
        # Check qualification completion
        is_complete, bant_status = await self.repo.check_qualification_complete(
            self.conversation_id
        )
        assert is_complete is True
        assert all(bant_status.values())
        
        # Now transition with context should work
        result = await self.repo.update_sales_stage_async(
            self.conversation_id,
            SalesStage.QUALIFIED.value,
            context={"qualification_complete": True},
            triggered_by="qualifier"
        )
        assert result.modified_count == 1
        
        # Verify is_qualified flag was set
        conv = await self.repo.get_conversation_state(self.conversation_id)
        assert conv["is_qualified"] is True
        assert conv["qualification"]["qualified_by"] == "qualifier"
    
    @pytest.mark.asyncio
    async def test_objection_lifecycle(self, setup):
        """Test complete objection lifecycle."""
        # Add objection
        objection_id = await self.repo.add_objection(
            self.conversation_id,
            objection_type=ObjectionType.PRICE.value,
            content="The price seems too high for our budget",
            severity="high",
            raised_by="user"
        )
        assert objection_id is not None
        
        # Get active objections
        active_objections = await self.repo.get_active_objections(
            self.conversation_id
        )
        assert len(active_objections) == 1
        assert active_objections[0]["objection_id"] == objection_id
        assert active_objections[0]["status"] == "active"
        
        # Mark objection as handled
        result = await self.repo.mark_objection_handled(
            self.conversation_id,
            objection_id,
            resolution_method="value_demonstration",
            resolution_notes="Showed ROI calculation demonstrating 3x return",
            handled_by="objection_handler",
            confidence=0.85
        )
        assert result.modified_count == 1
        
        # Verify no active objections
        active_objections = await self.repo.get_active_objections(
            self.conversation_id
        )
        assert len(active_objections) == 0
        
        # Get conversation and check objection
        conv = await self.repo.get_conversation_state(self.conversation_id)
        objection = next(o for o in conv["objections"] if o["objection_id"] == objection_id)
        assert objection["status"] == "resolved"
        assert objection["resolution"]["resolved"] is True
        assert objection["resolution"]["confidence"] == 0.85
    
    @pytest.mark.asyncio
    async def test_defer_objection_with_followup(self, setup):
        """Test deferring an objection with follow-up date."""
        # Add objection
        objection_id = await self.repo.add_objection(
            self.conversation_id,
            objection_type=ObjectionType.TIMING.value,
            content="Need to wait until next quarter",
            severity="medium"
        )
        
        # Defer objection
        follow_up_date = datetime.now(timezone.utc) + timedelta(days=90)
        result = await self.repo.defer_objection(
            self.conversation_id,
            objection_id,
            reason="Customer needs board approval in Q2",
            follow_up_date=follow_up_date
        )
        assert result.modified_count == 1
        
        # Verify objection status and follow-up
        conv = await self.repo.get_conversation_state(self.conversation_id)
        objection = next(o for o in conv["objections"] if o["objection_id"] == objection_id)
        assert objection["status"] == "deferred"
        assert objection["deferred_reason"] == "Customer needs board approval in Q2"
        assert conv["follow_up"]["required"] is True
        assert conv["follow_up"]["type"] == "objection_follow_up"
    
    @pytest.mark.asyncio
    async def test_atomic_handoff_with_stage_update(self, setup):
        """Test atomic handoff with stage update."""
        # Setup: Move to qualification
        await self.repo.update_sales_stage_async(
            self.conversation_id,
            SalesStage.QUALIFICATION.value,
            validate=False
        )
        
        # Update qualification to meet requirements
        await self.repo.update_qualification(
            self.conversation_id,
            budget_info={"meets_criteria": True, "value": "$2000/month", "confidence": 90},
            authority_info={"meets_criteria": True, "role": "CEO", "confidence": 95},
            need_info={"meets_criteria": True, "pain_points": ["manual work"], "confidence": 85},
            timeline_info={"meets_criteria": True, "timeframe": "This month", "confidence": 90}
        )
        
        # Perform atomic handoff with stage update
        success = await self.repo.perform_atomic_handoff_with_stage_update(
            self.conversation_id,
            from_agent=AgentName.QUALIFIER.value,
            to_agent=AgentName.CLOSER.value,
            new_stage=SalesStage.QUALIFIED.value,
            handoff_reason="BANT qualification complete, ready for closing",
            context={"qualification_complete": True}
        )
        assert success is True
        
        # Verify both handoff and stage were updated
        conv = await self.repo.get_conversation_state(self.conversation_id)
        assert conv["current_agent"] == AgentName.CLOSER.value
        assert conv["sales_stage"] == SalesStage.QUALIFIED.value
        assert len(conv["handoffs"]) == 1
        assert conv["handoffs"][0]["from_agent"] == AgentName.QUALIFIER.value
    
    @pytest.mark.asyncio
    async def test_atomic_objection_resolution_with_stage(self, setup):
        """Test atomic objection resolution with stage progression."""
        # Setup: Move to objection handling stage
        await self.repo.update_sales_stage_async(
            self.conversation_id,
            SalesStage.OBJECTION_HANDLING.value,
            validate=False
        )
        
        # Add objection
        objection_id = await self.repo.add_objection(
            self.conversation_id,
            objection_type=ObjectionType.FEATURE.value,
            content="Missing integration with our CRM",
            severity="high"
        )
        
        # Atomically resolve and move to closing
        success, error = await self.repo.perform_atomic_objection_resolution(
            self.conversation_id,
            objection_id,
            resolution_data={
                "method": "roadmap_commitment",
                "notes": "Committed to CRM integration in next release",
                "handled_by": "objection_handler",
                "confidence": 0.9
            },
            move_to_next_stage=SalesStage.CLOSING.value
        )
        assert success is True
        assert error is None
        
        # Verify objection resolved and stage updated
        conv = await self.repo.get_conversation_state(self.conversation_id)
        assert conv["sales_stage"] == SalesStage.CLOSING.value
        objection = next(o for o in conv["objections"] if o["objection_id"] == objection_id)
        assert objection["status"] == "resolved"
    
    @pytest.mark.asyncio
    async def test_close_deal_atomic_operation(self, setup):
        """Test atomic deal closing operation."""
        # Setup: Move to closing stage
        await self.repo.update_sales_stage_async(
            self.conversation_id,
            SalesStage.CLOSING.value,
            validate=False
        )
        
        # Close deal as won
        deal_details = {
            "monthly_value": 150000,  # $1500 in cents
            "payment_method": "credit_card",
            "contract_length": "12 months",
            "discount_applied": 10,
            "close_reason": "Customer ready to proceed with annual plan"
        }
        
        success, error = await self.repo.perform_atomic_close_deal(
            self.conversation_id,
            deal_details,
            close_type="won"
        )
        assert success is True
        assert error is None
        
        # Verify deal closed properly
        conv = await self.repo.get_conversation_state(self.conversation_id)
        assert conv["sales_stage"] == SalesStage.CLOSED_WON.value
        assert conv["deal_details"]["monthly_value"] == 150000
        assert conv["deal_details"]["payment_method"] == "credit_card"
        assert "close_date" in conv["deal_details"]
    
    @pytest.mark.asyncio
    async def test_objection_analytics(self, setup):
        """Test objection analytics aggregation."""
        # Add multiple objections
        await self.repo.add_objection(
            self.conversation_id,
            ObjectionType.PRICE.value,
            "Too expensive",
            "high"
        )
        
        objection_id = await self.repo.add_objection(
            self.conversation_id,
            ObjectionType.TIMING.value,
            "Not ready yet",
            "medium"
        )
        
        # Resolve one objection
        await self.repo.mark_objection_handled(
            self.conversation_id,
            objection_id,
            "addressed_concerns",
            "Showed implementation timeline",
            "objection_handler",
            0.75
        )
        
        # Get analytics
        analytics = await self.repo.get_objection_analytics()
        assert analytics["total_objections"] >= 2
        assert analytics["resolved_count"] >= 1
        assert analytics["resolution_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_service_objection_methods(self, setup):
        """Test conversation service objection handling methods."""
        # Raise objection via service
        objection_id = await self.service.raise_objection(
            self.conversation_id,
            ObjectionType.AUTHORITY.value,
            "Need CEO approval",
            "high"
        )
        assert objection_id is not None
        
        # Get active objections
        active = await self.service.get_active_objections(self.conversation_id)
        assert len(active) == 1
        
        # Handle objection
        success = await self.service.handle_objection(
            self.conversation_id,
            objection_id,
            "escalation",
            "Scheduled meeting with CEO next week",
            "qualifier",
            0.7
        )
        assert success is True
        
        # Verify resolved
        active = await self.service.get_active_objections(self.conversation_id)
        assert len(active) == 0
    
    @pytest.mark.asyncio
    async def test_stage_transition_helper_methods(self, setup):
        """Test helper methods for stage transitions."""
        # Get next valid stages
        next_stages = await self.repo.get_next_valid_stages(self.conversation_id)
        assert SalesStage.QUALIFICATION.value in next_stages
        assert SalesStage.CLOSED_WON.value not in next_stages
        
        # Test terminal stage check
        assert SalesStageTransitionValidator.is_terminal_stage(SalesStage.CLOSED_WON.value)
        assert SalesStageTransitionValidator.is_terminal_stage(SalesStage.CLOSED_LOST.value)
        assert not SalesStageTransitionValidator.is_terminal_stage(SalesStage.LEAD.value)
    
    @pytest.mark.asyncio
    async def test_qualification_completion_check(self, setup):
        """Test BANT qualification completion checking."""
        # Initially not complete
        is_complete, bant_status = await self.repo.check_qualification_complete(
            self.conversation_id
        )
        assert is_complete is False
        assert not any(bant_status.values())
        
        # Partial qualification
        await self.repo.update_qualification(
            self.conversation_id,
            budget_info={"meets_criteria": True, "value": "$500/month", "confidence": 80},
            authority_info={"meets_criteria": True, "role": "Manager", "confidence": 70}
        )
        
        is_complete, bant_status = await self.repo.check_qualification_complete(
            self.conversation_id
        )
        assert is_complete is False
        assert bant_status["budget"] is True
        assert bant_status["authority"] is True
        assert bant_status["need"] is False
        assert bant_status["timeline"] is False
        
        # Complete qualification
        await self.repo.update_qualification(
            self.conversation_id,
            need_info={"meets_criteria": True, "pain_points": ["inefficiency"], "confidence": 85},
            timeline_info={"meets_criteria": True, "timeframe": "This quarter", "confidence": 90}
        )
        
        is_complete, bant_status = await self.repo.check_qualification_complete(
            self.conversation_id
        )
        assert is_complete is True
        assert all(bant_status.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])