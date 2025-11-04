# itinerary_agent.py
"""
Itinerary Agent (Production Version with Real Tools)

This module defines a self-contained, ReAct-style agent for creating a day-by-day
travel itinerary. It uses real Google Maps Platform APIs to handle routing, travel times,
and point-of-interest searches.
"""

import os
import json
import logging
import googlemaps
from datetime import datetime
from itertools import permutations
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import the main state definition
try:
    # This works when the module is imported
    from ..workflows.state import TravelPlanningState
except ImportError:
    # This works when running the script directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from workflows.state import TravelPlanningState

# --- Configuration and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

ITINERARY_MODEL = os.getenv("ITINERARY_MODEL", "gpt-4o-mini")
PARSER_MODEL = os.getenv("PARSER_MODEL", "gpt-4o-mini")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY environment variable not set.")

# Initialize Google Maps client
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# --- Real Tools Definition ---

@tool
def get_travel_time(origin: str, destination: str, mode: str = "transit") -> str:
    """Gets the estimated travel time between two locations using a specified mode."""
    logger.info(f"Tool 'get_travel_time' called for {origin} -> {destination} via {mode}")
    try:
        result = gmaps.distance_matrix(origins=[origin], destinations=[destination], mode=mode)
        if result['rows'][0]['elements'][0]['status'] == 'OK':
            duration = result['rows'][0]['elements'][0]['duration']['text']
            return f"Estimated travel time from {origin} to {destination} by {mode} is {duration}."
        else:
            return f"Could not calculate travel time between {origin} and {destination}."
    except Exception as e:
        logger.error(f"Error in 'get_travel_time' tool: {e}")
        return f"Error: Could not retrieve travel time. {e}"

@tool
def find_nearby_restaurants(location_name: str, cuisine: str = "any", price_level: int = 2) -> str:
    """Finds highly-rated restaurants near a specific location."""
    logger.info(f"Tool 'find_nearby_restaurants' called for {location_name}")
    try:
        geocode_result = gmaps.geocode(location_name)
        if not geocode_result:
            return f"Error: Could not find location: {location_name}"
        
        lat_lng = geocode_result[0]['geometry']['location']
        
        keyword = f"{cuisine} restaurant" if cuisine != "any" else "restaurant"
        
        places_result = gmaps.places_nearby(
            location=lat_lng, radius=1500, type='restaurant', keyword=keyword,
            max_price=price_level
        )
        
        top_restaurants = []
        for place in places_result.get('results', [])[:3]:
            top_restaurants.append({"name": place.get('name'), "rating": place.get('rating'), "address": place.get('vicinity')})
        
        if not top_restaurants:
            return "No suitable restaurants found nearby."
            
        return json.dumps(top_restaurants)
    except Exception as e:
        logger.error(f"Error in 'find_nearby_restaurants' tool: {e}")
        return f"Error using Google Places API: {e}"

# --- Agent and Graph Node Definitions ---

def itinerary_agent_node(state: TravelPlanningState) -> Dict[str, Any]:
    """The core of the Itinerary Agent, which calls an LLM to build the plan."""
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Itinerary agent node running.")

    model = ChatOpenAI(model=ITINERARY_MODEL, temperature=0.4).with_retry(stop_after_attempt=3)
    agent_model = model.bind_tools([get_travel_time, find_nearby_restaurants])

    # Defensive checks for required data
    if not state.get('research_results') or not state['research_results'].get('attractions'):
        return {"errors": state.get("errors", []) + ["Cannot create itinerary without a list of attractions from the research agent."]}
    if not state.get('hotel_options'):
         return {"errors": state.get("errors", []) + ["Cannot create itinerary without a selected hotel location."]}

    system_prompt = """You are a world-class travel itinerary planner. Your goal is to create a practical, enjoyable, and well-paced daily schedule for a trip.

Current trip details:
- Destination: {destination}
- Dates: {start_date} to {end_date} ({num_days} days)
- Hotel: {hotel_name}
- List of Attractions to Visit: {attractions}
- User Interests: {interests}

Your task is to build a day-by-day plan:
1.  **Pacing:** Group the attractions logically. Assign 2-3 attractions per day to avoid an overly rushed schedule.
2.  **Sequencing:** For each day, determine a sensible order of visits. Start and end each day at the hotel.
3.  **Travel Time:** Use the `get_travel_time` tool to calculate the transit time *between* each stop in your daily sequence (e.g., Hotel -> Attraction A, Attraction A -> Attraction B, Attraction B -> Hotel). This is critical for a realistic plan.
4.  **Meal Planning:** For each day, suggest a lunch option by using the `find_nearby_restaurants` tool near one of the midday attractions.
5.  **Formatting:** Structure the output clearly, with headings for each day, and time slots (Morning, Afternoon, Evening). Include the calculated travel times in the plan.
6.  **Completion:** Do not finish until you have a plan for every day of the trip.
"""
    start_dt = datetime.fromisoformat(state['start_date'])
    end_dt = datetime.fromisoformat(state['end_date'])
    num_days = (end_dt - start_dt).days

    prompt = system_prompt.format(
        destination=state['destination'],
        start_date=state['start_date'],
        end_date=state['end_date'],
        num_days=num_days,
        hotel_name=state['hotel_options'][0]['name'], # Use the first hotel option as the base
        attractions=", ".join([attr['name'] for attr in state['research_results']['attractions']]),
        interests=", ".join(state['interests'])
    )

    messages = [SystemMessage(content=prompt)] + state.get("messages", [])

    try:
        response = agent_model.invoke(messages)
        logger.info(f"TripID {trip_id}: Itinerary agent LLM invoked successfully.")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"TripID {trip_id}: Error invoking itinerary agent model: {e}")
        return {"errors": state.get("errors", []) + [f"Itinerary agent LLM failed: {e}"]}

def should_continue_itinerary(state: TravelPlanningState) -> str:
    """Determines whether to continue the ReAct loop or finish."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.info(f"TripID {state.get('trip_id')}: Itinerary planning complete. Proceeding to parser.")
        return "end"
    else:
        logger.info(f"TripID {state.get('trip_id')}: Itinerary agent requested tool call. Continuing.")
        return "continue"

def parse_itinerary_output_node(state: TravelPlanningState) -> Dict[str, Any]:
    """Parses the final AI message into a structured `itinerary` dictionary."""
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Parsing final itinerary output.")

    final_message = state['messages'][-1].content

    parser_model = ChatOpenAI(model=PARSER_MODEL, temperature=0).with_retries(stop_after_attempt=3)
    structured_llm = parser_model.with_structured_output(Dict[str, Dict[str, List[str]]])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data parsing expert. Parse the provided itinerary text into a structured dictionary.
The main keys should be 'day_1', 'day_2', etc.
Each day's value should be another dictionary with keys 'morning', 'afternoon', and 'evening'.
Each time slot should contain a list of strings, where each string is an activity or a travel note.
Example: {"day_1": {"morning": ["Visit the Eiffel Tower", "Travel (20 mins) to Louvre Museum"], "afternoon": ["Explore the Louvre Museum", "Lunch near the Louvre"], "evening": ["Dinner Cruise on the Seine"]}}"""),
        ("user", "Please parse the following itinerary:\n\n{text_to_parse}")
    ])
    
    parser_chain = prompt | structured_llm

    try:
        structured_output = parser_chain.invoke({"text_to_parse": final_message})
        logger.info(f"TripID {trip_id}: Successfully parsed itinerary output into structured format.")
        
        return {
            "itinerary": structured_output,
            "current_step": "complete",
            "messages": [AIMessage(content="Your complete travel plan is ready! I've created a day-by-day itinerary for your trip.")]
        }
    except Exception as e:
        logger.error(f"TripID {trip_id}: Failed to parse itinerary output: {e}")
        return {"errors": state.get("errors", []) + ["Failed to structure itinerary output."]}

# --- Graph Assembly ---

itinerary_graph = StateGraph(TravelPlanningState)

itinerary_graph.add_node("itinerary_agent", itinerary_agent_node)
itinerary_graph.add_node("itinerary_tools", ToolNode([get_travel_time, find_nearby_restaurants]))
itinerary_graph.add_node("parser", parse_itinerary_output_node)

itinerary_graph.set_entry_point("itinerary_agent")

itinerary_graph.add_conditional_edges(
    "itinerary_agent",
    should_continue_itinerary,
    {"continue": "itinerary_tools", "end": "parser"}
)
itinerary_graph.add_edge("itinerary_tools", "itinerary_agent")
itinerary_graph.add_edge("parser", END)

# Compile the graph into a runnable application
itinerary_agent_app = itinerary_graph.compile()
logger.info("Itinerary agent graph compiled successfully.")

# --- Test Block ---
if __name__ == '__main__':
    logger.info("Running stand-alone test for Itinerary Agent...")

    # Example state after the Booking Agent has run
    initial_state_data = {
        'messages': [], 'destination': 'Kyoto, Japan', 'start_date': '2026-03-15', 'end_date': '2026-03-17', # 2 days for a concise test
        'num_travelers': 2, 'interests': ['history', 'temples', 'food'],
        'research_results': {
            'attractions': [{'name': 'Kiyomizu-dera Temple'}, {'name': 'Fushimi Inari Shrine'}, {'name': 'Arashiyama Bamboo Grove'}, {'name': 'Kinkaku-ji (Golden Pavilion)'}]
        },
        'hotel_options': [{'name': 'Hotel Gracery Kyoto Sanjo'}],
        'current_step': 'create_itinerary', 'trip_id': 'test-trip-itinerary', 'errors': [],
    }
    initial_state = TravelPlanningState(**initial_state_data)

    final_state = itinerary_agent_app.invoke(initial_state)

    print("\n--- Final State after Itinerary Planning ---")
    print(f"Current Step: {final_state.get('current_step')}")
    print(f"Errors: {final_state.get('errors')}")

    print("\n--- Structured Itinerary ---")
    if final_state.get('itinerary'):
        print(json.dumps(final_state['itinerary'], indent=2))
    else:
        print("No itinerary was generated.")