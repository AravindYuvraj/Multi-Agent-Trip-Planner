"""
Main Travel Planning LangGraph Workflow (Updated)

This orchestrates the complete multi-agent travel planning workflow.
Integrates with specialized sub-graph agents (research, budget, booking, itinerary).
"""

import os
import logging
from typing import Dict, Any, Literal
from datetime import datetime

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

# Import state and agents
try:
    from workflows.state import TravelPlanningState
    from agents.orchestrator_agent import orchestrator_node, route_after_orchestrator
    from agents.research_agent import research_agent_app
    from agents.budget_agent import budget_agent_app
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from workflows.state import TravelPlanningState
    from agents.orchestrator_agent import orchestrator_node, route_after_orchestrator
    from agents.research_agent import research_agent_app
    from agents.budget_agent import budget_agent_app

# --- Configuration and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def create_travel_planning_graph():
    """
    Create and compile the main travel planning multi-agent graph.
    
    Workflow:
    1. Orchestrator extracts requirements
    2. Research agent sub-graph gathers destination info (with tools)
    3. Budget agent sub-graph plans allocation (with tools)
    4. Booking agent finds flights & hotels (placeholder for now)
    5. Itinerary agent creates day-by-day plan (placeholder for now)
    6. Review and present final plan
    """
    
    logger.info("Building main travel planning graph...")
    
    # Initialize the StateGraph with our custom state
    graph = StateGraph(TravelPlanningState)
    
    # === ADD NODES ===
    
    # 1. Orchestrator - extracts requirements and validates
    graph.add_node("orchestrator", orchestrator_node)
    
    # 2. Research - sub-graph agent with tools (your implementation)
    graph.add_node("research", research_agent_app)
    
    # 3. Budget - sub-graph agent with tools (your implementation)
    graph.add_node("budget", budget_agent_app)
    
    # 4. Booking - placeholder (will be implemented next)
    graph.add_node("booking", booking_node_placeholder)
    
    # 5. Itinerary - placeholder (will be implemented next)
    graph.add_node("itinerary", itinerary_node_placeholder)
    
    # 6. Final review and synthesis
    graph.add_node("review", review_node)
    
    # === DEFINE EDGES ===
    
    # Entry point
    graph.set_entry_point("orchestrator")
    
    # Conditional routing from orchestrator
    graph.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "research": "research",      # Requirements complete -> research
            "ask_user": END,              # Missing info -> end and wait
            "error": END                  # Errors -> end
        }
    )
    
    # Linear workflow after research (agents update current_step themselves)
    graph.add_edge("research", "budget")
    graph.add_edge("budget", "booking")
    graph.add_edge("booking", "itinerary")
    graph.add_edge("itinerary", "review")
    graph.add_edge("review", END)
    
    # === COMPILE GRAPH ===
    
    # Use MemorySaver for conversation persistence
    memory = MemorySaver()
    
    # Compile the graph
    app = graph.compile(checkpointer=memory)
    
    logger.info("Main travel planning graph compiled successfully!")
    
    return app


# === PLACEHOLDER NODE FUNCTIONS ===
# These will be replaced with full sub-graph implementations

def booking_node_placeholder(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Booking agent placeholder.
    TODO: Implement as sub-graph with flight/hotel API tools
    """
    trip_id = state.get('trip_id', 'unknown')
    logger.info(f"TripID {trip_id}: Booking node (placeholder) running")
    
    destination = state.get("destination", "Unknown")
    budget = state.get("budget_breakdown", {})
    start_date = state.get("start_date")
    end_date = state.get("end_date")
    num_travelers = state.get("num_travelers", 1)
    
    # Calculate number of nights
    if start_date and end_date:
        from datetime import datetime
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        num_nights = (end - start).days
    else:
        num_nights = 5
    
    # Mock flight options (based on budget)
    flight_budget = budget.get("flights", 1000)
    flight_options = [
        {
            "id": "flight_1",
            "airline": "Premium Airlines",
            "price": round(flight_budget * 0.45, 2),  # Per person
            "departure": f"{start_date}T08:00",
            "arrival": f"{start_date}T20:00",
            "duration": "12h",
            "stops": 0,
            "route": f"Your City → {destination}",
            "booking_class": "Economy"
        },
        {
            "id": "flight_2",
            "airline": "Budget Airways",
            "price": round(flight_budget * 0.35, 2),
            "departure": f"{start_date}T14:00",
            "arrival": f"{start_date}T04:00+1",
            "duration": "14h",
            "stops": 1,
            "route": f"Your City → Hub → {destination}",
            "booking_class": "Economy"
        }
    ]
    
    # Mock hotel options (based on budget)
    accommodation_budget = budget.get("accommodation", 500)
    price_per_night = accommodation_budget / num_nights if num_nights > 0 else 100
    
    hotel_options = [
        {
            "id": "hotel_1",
            "name": f"Grand {destination.split(',')[0]} Hotel",
            "price_per_night": round(price_per_night * 1.2, 2),
            "total_price": round(price_per_night * 1.2 * num_nights, 2),
            "rating": 4.5,
            "location": "City Center",
            "amenities": ["WiFi", "Breakfast", "Gym", "Pool"],
            "room_type": "Deluxe Double"
        },
        {
            "id": "hotel_2",
            "name": f"Comfort Inn {destination.split(',')[0]}",
            "price_per_night": round(price_per_night * 0.8, 2),
            "total_price": round(price_per_night * 0.8 * num_nights, 2),
            "rating": 4.0,
            "location": "Near Transit",
            "amenities": ["WiFi", "Breakfast"],
            "room_type": "Standard Double"
        }
    ]
    
    min_flight = min(f["price"] for f in flight_options)
    min_hotel = min(h["price_per_night"] for h in hotel_options)
    
    message = (f"🛫 **Flight Options**: Found {len(flight_options)} flights to {destination}\n"
              f"   • Best price: ${min_flight:.2f} per person\n\n"
              f"🏨 **Accommodation**: Found {len(hotel_options)} hotels\n"
              f"   • Best rate: ${min_hotel:.2f} per night\n\n"
              f"Total for flights + hotels: ${(min_flight * num_travelers) + (min_hotel * num_nights):.2f}")
    
    return {
        "flight_options": flight_options,
        "hotel_options": hotel_options,
        "current_step": "create_itinerary",
        "messages": [AIMessage(content=message)],
        "last_updated": datetime.utcnow().isoformat()
    }


def itinerary_node_placeholder(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Itinerary agent placeholder.
    TODO: Implement as sub-graph with routing/optimization tools
    """
    trip_id = state.get('trip_id', 'unknown')
    logger.info(f"TripID {trip_id}: Itinerary node (placeholder) running")
    
    research = state.get("research_results", {})
    attractions = research.get("attractions", [])
    interests = state.get("interests", [])
    start_date = state.get("start_date")
    end_date = state.get("end_date")
    
    # Calculate number of days
    if start_date and end_date:
        from datetime import datetime, timedelta
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        num_days = (end - start).days
        
        # Generate day-by-day itinerary
        itinerary = {}
        for i in range(num_days):
            day_date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            day_num = i + 1
            
            if day_num == 1:
                theme = "Arrival & Orientation"
                activities = [
                    "Arrive and check into accommodation",
                    "Rest and freshen up",
                    "Explore immediate neighborhood",
                    "Welcome dinner at local restaurant"
                ]
            elif day_num == num_days:
                theme = "Departure Day"
                activities = [
                    "Final breakfast",
                    "Last-minute souvenir shopping",
                    "Check out and head to airport",
                    "Depart for home"
                ]
            else:
                theme = f"{interests[min(i-1, len(interests)-1)].title() if interests else 'Exploration'} Day"
                activities = [
                    attractions[min(i-1, len(attractions)-1)] if attractions else "Visit top attraction",
                    "Lunch at recommended restaurant",
                    attractions[min(i, len(attractions)-1)] if len(attractions) > 1 else "Explore local market",
                    "Dinner and evening entertainment"
                ]
            
            itinerary[f"day_{day_num}"] = {
                "date": day_date,
                "day_number": day_num,
                "theme": theme,
                "activities": activities
            }
    else:
        itinerary = {"error": "Missing dates"}
        num_days = 0
    
    message = (f"📅 **{num_days}-Day Itinerary Created**\n\n"
              f"Each day optimized for: {', '.join(interests) if interests else 'sightseeing'}\n"
              f"Includes {len(attractions)} researched attractions with efficient daily routes.\n\n"
              f"Your complete travel plan is ready for review!")
    
    return {
        "itinerary": itinerary,
        "current_step": "review",
        "messages": [AIMessage(content=message)],
        "last_updated": datetime.utcnow().isoformat()
    }


def review_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Final review and synthesis node.
    Presents complete travel plan to user in a beautiful format.
    """
    trip_id = state.get('trip_id', 'unknown')
    logger.info(f"TripID {trip_id}: Review node generating final summary")
    
    # Extract all the data
    destination = state.get("destination", "your destination")
    start_date = state.get("start_date", "TBD")
    end_date = state.get("end_date", "TBD")
    num_travelers = state.get("num_travelers", 1)
    budget_total = state.get("budget_total", 0)
    interests = state.get("interests", [])
    
    # Research results
    research = state.get("research_results", {})
    num_attractions = len(research.get("attractions", []))
    weather = research.get("weather", "Check forecast closer to travel")
    
    # Budget breakdown
    budget = state.get("budget_breakdown", {})
    
    # Flights and hotels
    flights = state.get("flight_options", [])
    hotels = state.get("hotel_options", [])
    
    min_flight_price = min([f["price"] for f in flights], default=0) * num_travelers
    min_hotel_price = hotels[1]["total_price"] if len(hotels) > 1 else 0
    
    # Itinerary
    itinerary = state.get("itinerary", {})
    num_days = len([k for k in itinerary.keys() if k.startswith("day_")])
    
    # Generate comprehensive summary
    summary = f"""
🎉 **Your Complete {destination} Travel Plan!**

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📅 **Trip Details**
• Destination: {destination}
• Dates: {start_date} to {end_date} ({num_days} days)
• Travelers: {num_travelers} {"person" if num_travelers == 1 else "people"}
• Budget: ${budget_total:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 **Budget Breakdown**
• Flights: ${budget.get('flights', 0):.2f} ({(budget.get('flights', 0)/budget_total*100):.0f}%)
• Accommodation: ${budget.get('accommodation', 0):.2f} ({(budget.get('accommodation', 0)/budget_total*100):.0f}%)
• Food: ${budget.get('food', 0):.2f} daily per person
• Activities: ${budget.get('activities', 0):.2f} daily per person
• Transport: ${budget.get('transport', 0):.2f} daily per person
• **Total Allocated**: ${budget.get('total', 0):.2f}
• **Remaining Buffer**: ${budget.get('remaining', 0):.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✈️ **Flights**
• {len(flights)} options found
• Best price: ${min_flight_price:.2f} (all travelers)
• Includes: {flights[0]['airline'] if flights else 'Various airlines'}

🏨 **Accommodation**
• {len(hotels)} options found
• Best rate: ${min_hotel_price:.2f} (total stay)
• Recommended: {hotels[1]['name'] if len(hotels) > 1 else 'TBD'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 **Research Highlights**
• {num_attractions} top attractions identified
• Interests covered: {', '.join(interests) if interests else 'General sightseeing'}
• Weather: {weather[:100]}...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🗓️ **Itinerary**
• {num_days}-day detailed plan created
• Daily activities optimized for your interests
• Includes must-see attractions and local experiences

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**✅ Your trip is fully planned and ready!**

**Next Steps:**
1. Review the detailed flight and hotel options
2. Confirm your preferences
3. I can help you proceed with bookings
4. Ask me about any specific day or activity

Would you like to see more details about any part of your trip?
"""
    
    return {
        "messages": [AIMessage(content=summary.strip())],
        "current_step": "complete",
        "last_updated": datetime.utcnow().isoformat()
    }


# === MAIN EXECUTION ===

if __name__ == "__main__":
    """
    Example usage of the travel planning graph with sub-graph agents.
    """
    
    logger.info("=" * 70)
    logger.info("TRAVEL PLANNING SYSTEM - MAIN WORKFLOW TEST")
    logger.info("=" * 70)
    
    # Create the graph
    app = create_travel_planning_graph()
    
    # Test message
    test_message = (
        "I want to plan a 5-day trip to Kyoto, Japan from March 15-20, 2026. "
        "We're 2 people with a budget of $4000. "
        "We're interested in temples, history, and authentic Japanese food."
    )
    
    # Create initial state
    from workflows.state import create_initial_state
    initial_state = create_initial_state(test_message)
    
    # Thread configuration for conversation persistence
    config = {"configurable": {"thread_id": "main_test_trip_001"}}
    
    # Run the complete workflow
    logger.info(f"\n💬 User: {test_message}\n")
    logger.info("🤖 Starting multi-agent workflow...\n")
    
    try:
        result = app.invoke(initial_state, config)
        
        logger.info("\n" + "=" * 70)
        logger.info("WORKFLOW COMPLETE - RESULTS")
        logger.info("=" * 70)
        
        # Print all assistant messages
        print("\n📝 Agent Responses:\n")
        for msg in result["messages"]:
            if hasattr(msg, 'type') and msg.type == "ai":
                print(f"🤖 {msg.content}\n")
                print("-" * 70 + "\n")
        
        # Print final state summary
        print("\n📊 Final State Summary:")
        print("-" * 70)
        print(f"Trip ID: {result.get('trip_id')}")
        print(f"Destination: {result.get('destination')}")
        print(f"Dates: {result.get('start_date')} to {result.get('end_date')}")
        print(f"Status: {result.get('current_step')}")
        print(f"\nData Completeness:")
        print(f"  ✓ Requirements: {result.get('requirements_extracted')}")
        print(f"  ✓ Research: {'Yes' if result.get('research_results') else 'No'}")
        print(f"  ✓ Budget: {'Yes' if result.get('budget_breakdown') else 'No'}")
        print(f"  ✓ Flights: {'Yes' if result.get('flight_options') else 'No'}")
        print(f"  ✓ Hotels: {'Yes' if result.get('hotel_options') else 'No'}")
        print(f"  ✓ Itinerary: {'Yes' if result.get('itinerary') else 'No'}")
        
        logger.info("\n✅ TEST PASSED - Full workflow executed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()