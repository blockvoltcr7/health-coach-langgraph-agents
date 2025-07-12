# Qualifier Agent Prompt Design NOT FINAL!!! THIS IS NOT FINAL THIS IS A DRAFT

Here's how to structure the Qualifier Agent's prompt with clear role definition and parameterization:

## Complete Qualifier Agent Prompt Template

```python
QUALIFIER_AGENT_PROMPT = """
You are the Qualifier Agent in the Limitless OS Sales Pipeline. Your SOLE responsibility is to assess whether prospects meet our qualification criteria. You are ONE OF FOUR specialized agents in this sales process.

YOUR ROLE IN THE SALES PIPELINE:
- You are the FIRST agent prospects interact with after initial contact
- You ONLY handle qualification - you do NOT pitch, sell, or handle objections
- After you complete qualification, the Supervisor will route to:
  - Objection Handler (if qualified but has concerns)
  - Closer Agent (if qualified and ready)
  - Follow-up Agent (if not qualified)
- You must gather information and make a clear qualified/not qualified determination

CURRENT CONTEXT:
- Sales Stage: {sales_stage}
- Conversation Number: {interaction_count}
- Previous Interactions: {has_previous_interactions}

CONVERSATION HISTORY:
{conversation_history}

RELEVANT MEMORIES ABOUT THIS PROSPECT:
{memories}

QUALIFICATION CRITERIA YOU MUST ASSESS:
1. BUDGET: 
   - Minimum: $1,000/month
   - Preferred: $5,000+/month
   - Current prospect budget: {current_budget_info}

2. NEED:
   - Must have clear business pain points that Limitless OS solves
   - Examples: Process automation, AI implementation, digital transformation
   - Current identified needs: {identified_needs}

3. AUTHORITY:
   - Must be decision-maker or have direct influence
   - Can approve budget or strongly recommend
   - Current authority status: {authority_status}

4. TIMELINE:
   - Immediate: < 30 days (High priority)
   - Short-term: 1-3 months (Medium priority)
   - Long-term: 3-6 months (Low priority)
   - Current timeline: {timeline_info}

INFORMATION STILL NEEDED:
{missing_qualification_data}

YOUR CURRENT TASK:
Based on the conversation so far, you need to:
{specific_task_instruction}

RESPONSE GUIDELINES:
- Be conversational and friendly, but focused on gathering qualification data
- Ask ONE clear question at a time to fill information gaps
- If you have enough information, make your qualification determination
- Do NOT pitch features or benefits - that's the Closer's job
- Do NOT address objections - that's the Objection Handler's job
- Simply gather information and assess fit

RESPOND WITH:
1. A natural, conversational message to the prospect
2. Internal assessment in this format:
   <qualification_assessment>
   - Budget: [qualified/not qualified/unknown]
   - Need: [qualified/not qualified/unknown]
   - Authority: [qualified/not qualified/unknown]
   - Timeline: [qualified/not qualified/unknown]
   - Overall: [qualified/not qualified/needs more info]
   - Next question needed: [specific data point] or [none]
   </qualification_assessment>

Current user message: {current_message}

Your response:"""
```

## Key Parameters to Pass

### 1. **Context Parameters** (from MongoDB State)
```python
{
    "sales_stage": "qualification",  # Current stage in pipeline
    "interaction_count": 3,          # Number of interactions so far
    "has_previous_interactions": True # Boolean flag
}
```

### 2. **Conversation History** (from MongoDB)
```python
{
    "conversation_history": """
    User: I'm interested in learning about AI automation
    Assistant: Great! I'd love to understand more about your needs...
    User: We're a mid-size company looking to streamline operations
    """
}
```

### 3. **Memories** (from Mem0)
```python
{
    "memories": """
    - Prospect mentioned they're frustrated with manual processes
    - Company size: 50-100 employees
    - Industry: Manufacturing
    - Previous tools used: Basic spreadsheets
    """
}
```

### 4. **Current Qualification Status** (from State)
```python
{
    "current_budget_info": "Unknown",
    "identified_needs": "Process automation, efficiency",
    "authority_status": "Appears to be decision maker",
    "timeline_info": "Unknown",
    "missing_qualification_data": ["budget", "timeline"]
}
```

### 5. **Task-Specific Instructions**
```python
{
    "specific_task_instruction": "You need to understand their budget range and implementation timeline. You already know they have a clear need and authority.",
    "current_message": "Yes, I'm the operations director and we really need to automate our inventory management"
}
```

## Dynamic Prompt Building Function

```python
def build_qualifier_prompt(state: SalesState, memories: List[Dict]) -> str:
    """Build the qualifier agent prompt with current context."""
    
    # Extract qualification status from state
    qual_details = state.get("qualification_details", {})
    
    # Determine what information is still needed
    missing_data = []
    if not qual_details.get("budget"):
        missing_data.append("budget")
    if not qual_details.get("timeline"):
        missing_data.append("timeline")
    # ... etc
    
    # Build specific task instruction based on what's missing
    if len(missing_data) == 0:
        task_instruction = "You have all qualification data. Make your final qualified/not qualified determination."
    elif len(missing_data) == 1:
        task_instruction = f"You only need to determine their {missing_data[0]}."
    else:
        task_instruction = f"You need to understand their {', '.join(missing_data)}. Start with {missing_data[0]}."
    
    # Format the prompt
    return QUALIFIER_AGENT_PROMPT.format(
        sales_stage=state["sales_stage"],
        interaction_count=state["interaction_count"],
        has_previous_interactions=state["interaction_count"] > 1,
        conversation_history=format_conversation_history(state["conversation_history"]),
        memories=format_memories(memories),
        current_budget_info=qual_details.get("budget", "Unknown"),
        identified_needs=qual_details.get("need", "Unknown"),
        authority_status=qual_details.get("authority", "Unknown"),
        timeline_info=qual_details.get("timeline", "Unknown"),
        missing_qualification_data=", ".join(missing_data) if missing_data else "None - ready for determination",
        specific_task_instruction=task_instruction,
        current_message=state["current_message"]
    )
```

## Why This Prompt Structure Works

### 1. **Clear Role Boundaries**
- Explicitly states it's ONE of FOUR agents
- Clearly defines what it does NOT do (no pitching, no objection handling)
- Explains where prospects go next in the pipeline

### 2. **Context Awareness**
- Knows where it sits in the sales flow
- Understands the complete conversation context
- Leverages all available memories

### 3. **Specific Success Criteria**
- Clear BANT criteria with thresholds
- Structured assessment format
- Binary outcome: qualified/not qualified/needs more info

### 4. **Focused Output**
- One question at a time
- Internal assessment for the Supervisor
- Clear handoff signals

## Example Output from Qualifier

```
Thanks for that information! As the operations director, you definitely have the authority to make this decision. Since you're looking to automate inventory management, I can see how Limitless OS would be valuable for you.

To ensure we can provide the right solution for your needs, could you share what your typical monthly budget is for operational improvements or automation tools?

<qualification_assessment>
- Budget: unknown
- Need: qualified (clear automation needs)
- Authority: qualified (operations director)
- Timeline: unknown
- Overall: needs more info
- Next question needed: budget range
</qualification_assessment>
```

This structure ensures the Qualifier Agent stays in its lane while effectively gathering the necessary information for the Supervisor to make routing decisions.