"""
Travel Planning State Schema
This defines the shared state that flows through all agents in the LangGraph
"""

from typing import TypedDict, Annotated, Literal
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from datetime import date


class TravelPlanningState(TypedDict):
    """
    Central state object that flows through the entire travel planning graph.
    Each agent reads from and writes to this state.
    """
    
    # === CONVERSATION ===
    messages: Annotated[list, add_messages]
    """All conversation messages with add_messages reducer for appending"""
    
    # === TRIP REQUIREMENTS (Extracted from user) ===
    destination: str | None
    """Primary destination (e.g., 'Tokyo, Japan')"""
    
    start_date: str | None
    """Trip start date in YYYY-MM-DD format"""
    
    end_date: str | None
    """Trip end date in YYYY-MM-DD format"""
    
    num_travelers: int
    """Number of people traveling (default: 1)"""
    
    budget_total: float | None
    """Total budget in USD"""
    
    interests: list[str]
    """User interests: ['culture', 'food', 'adventure', 'nature', 'history']"""
    
    accommodation_preference: str | None
    """'hotel', 'hostel', 'airbnb', 'luxury', 'budget'"""
    
    # === EXTRACTED REQUIREMENTS (By Orchestrator) ===
    requirements_extracted: bool
    """Flag: has orchestrator extracted all requirements?"""
    
    missing_requirements: list[str]
    """List of missing required fields to ask user about"""
    
    # === AGENT OUTPUTS ===
    research_results: dict | None
    """
    Research agent output:
    {
        'attractions': [...],
        'weather': {...},
        'local_tips': [...],
        'safety_info': {...}
    }
    """
    
    flight_options: list[dict] | None
    """
    Booking agent output for flights:
    [
        {
            'airline': 'United',
            'price': 850.00,
            'departure': '2025-06-01T10:00',
            'arrival': '2025-06-01T14:00',
            'duration': '13h',
            'stops': 1
        },
        ...
    ]
    """
    
    hotel_options: list[dict] | None
    """
    Booking agent output for hotels:
    [
        {
            'name': 'Tokyo Grand Hotel',
            'price_per_night': 150.00,
            'rating': 4.5,
            'location': 'Shibuya',
            'amenities': [...]
        },
        ...
    ]
    """
    
    itinerary: dict | None
    """
    Itinerary agent output:
    {
        'day_1': {
            'morning': [...],
            'afternoon': [...],
            'evening': [...]
        },
        ...
    }
    """
    
    budget_breakdown: dict | None
    """
    Budget agent output:
    {
        'flights': 850.00,
        'accommodation': 750.00,
        'food': 400.00,
        'activities': 300.00,
        'transport': 200.00,
        'total': 2500.00,
        'remaining': 500.00
    }
    """
    
    # === WORKFLOW CONTROL ===
    current_step: Literal[
        "extract_requirements",
        "research",
        "budget_planning",
        "booking_flights",
        "booking_hotels",
        "create_itinerary",
        "review_plan",
        "complete"
    ] | None
    """Current stage in the planning workflow"""
    
    next_agent: str | None
    """Which agent should execute next"""
    
    requires_user_input: bool
    """Does the workflow need to pause for user input?"""
    
    user_feedback: str | None
    """User's response to a clarification question"""
    
    # === ERROR HANDLING ===
    errors: list[str]
    """List of errors encountered during execution"""
    
    retry_count: int
    """Number of retries for current step"""
    
    # === METADATA ===
    trip_id: str | None
    """Unique identifier for this trip planning session"""
    
    created_at: str | None
    """Timestamp when planning started"""
    
    last_updated: str | None
    """Timestamp of last state update"""


# Helper function to create initial state
def create_initial_state(user_message: str) -> TravelPlanningState:
    """Create initial state from first user message"""
    from datetime import datetime
    import uuid
    from langchain_core.messages import HumanMessage
    
    return TravelPlanningState(
        messages=[HumanMessage(content=user_message)],  # Use HumanMessage instead of tuple
        destination=None,
        start_date=None,
        end_date=None,
        num_travelers=1,
        budget_total=None,
        interests=[],
        accommodation_preference=None,
        requirements_extracted=False,
        missing_requirements=[],
        research_results=None,
        flight_options=None,
        hotel_options=None,
        itinerary=None,
        budget_breakdown=None,
        current_step="extract_requirements",
        next_agent="orchestrator",
        requires_user_input=False,
        user_feedback=None,
        errors=[],
        retry_count=0,
        trip_id=str(uuid.uuid4()),
        created_at=datetime.utcnow().isoformat(),
        last_updated=datetime.utcnow().isoformat()
    )