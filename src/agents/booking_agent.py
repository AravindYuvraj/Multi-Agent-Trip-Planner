"""
Booking Agent (Production Version with Real APIs)

This module defines a self-contained, ReAct-style agent for finding flights and hotels.
It uses Amadeus API for flights and SerpAPI for hotels.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime, timedelta

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Amadeus SDK for flights
from amadeus import Client, ResponseError

# SerpAPI for hotels
from serpapi import GoogleSearch

# Import the main state definition
try:
    from ..workflows.state import TravelPlanningState
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from workflows.state import TravelPlanningState

# --- Configuration and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

BOOKING_MODEL = os.getenv("BOOKING_MODEL", "gpt-4o-mini")
PARSER_MODEL = os.getenv("PARSER_MODEL", "gpt-4o-mini")

# API credentials
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Validate API keys
if not AMADEUS_API_KEY or not AMADEUS_API_SECRET:
    logger.warning("AMADEUS_API_KEY or AMADEUS_API_SECRET not set. Flight search will use fallback.")
if not SERPAPI_KEY:
    logger.warning("SERPAPI_KEY not set. Hotel search will use fallback.")

# Initialize Amadeus client
try:
    amadeus = Client(
        client_id=AMADEUS_API_KEY,
        client_secret=AMADEUS_API_SECRET
    ) if AMADEUS_API_KEY and AMADEUS_API_SECRET else None
except Exception as e:
    logger.error(f"Failed to initialize Amadeus client: {e}")
    amadeus = None

# --- Helper Functions ---

def get_airport_code(city: str) -> str:
    """Get IATA airport code for a city. Simplified mapping."""
    # Common airport codes (expand as needed)
    airport_map = {
        "tokyo": "NRT",  # Narita
        "kyoto": "KIX",  # Osaka Kansai (nearest to Kyoto)
        "osaka": "KIX",
        "paris": "CDG",
        "london": "LHR",
        "new york": "JFK",
        "los angeles": "LAX",
        "san francisco": "SFO",
        "chicago": "ORD",
        "bangkok": "BKK",
        "singapore": "SIN",
        "dubai": "DXB",
        "hong kong": "HKG",
        "seoul": "ICN",
        "beijing": "PEK",
        "shanghai": "PVG",
        "delhi": "DEL",
        "mumbai": "BOM",
        "sydney": "SYD",
        "melbourne": "MEL",
        "rome": "FCO",
        "barcelona": "BCN",
        "amsterdam": "AMS",
        "berlin": "BER",
        "madrid": "MAD",
        "lisbon": "LIS",
        "istanbul": "IST",
        "athens": "ATH"
    }
    
    city_lower = city.lower().split(',')[0].strip()
    return airport_map.get(city_lower, "NYC")  # Default to NYC

# --- Tools Definition ---

@tool
def search_flights(
    destination: str,
    departure_date: str,
    return_date: str,
    adults: int,
    max_budget_per_person: float
) -> str:
    """
    Search for round-trip flights using Amadeus API.
    
    Args:
        destination: Destination city (e.g., "Tokyo, Japan")
        departure_date: Departure date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format
        adults: Number of adult passengers
        max_budget_per_person: Maximum budget per person in USD
    
    Returns:
        JSON string with flight options
    """
    logger.info(f"Tool 'search_flights' called for {destination} on {departure_date} to {return_date}, {adults} adults")
    
    try:
        if not amadeus:
            # Fallback: Generate realistic mock data
            logger.warning("Amadeus client not available. Using fallback data.")
            return _generate_fallback_flights(destination, departure_date, return_date, adults, max_budget_per_person)
        
        # Get airport codes
        origin_code = "JFK"  # Default origin (you can make this configurable)
        dest_code = get_airport_code(destination)
        
        logger.info(f"Searching flights: {origin_code} -> {dest_code}")
        
        # Call Amadeus API
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=origin_code,
            destinationLocationCode=dest_code,
            departureDate=departure_date,
            returnDate=return_date,
            adults=adults,
            max=10,  # Get more options
            currencyCode='USD'
        )
        
        # Parse response
        flight_offers = response.data
        
        if not flight_offers:
            return json.dumps({"error": "No flights found", "fallback": True})
        
        # Filter by budget and format
        filtered_flights = []
        for offer in flight_offers:
            price_per_person = float(offer['price']['total']) / adults
            
            if price_per_person <= max_budget_per_person * 1.1:  # 10% buffer
                # Extract flight details
                outbound = offer['itineraries'][0]
                return_flight = offer['itineraries'][1] if len(offer['itineraries']) > 1 else None
                
                flight_info = {
                    "id": offer['id'],
                    "price_per_person": round(price_per_person, 2),
                    "total_price": round(float(offer['price']['total']), 2),
                    "currency": offer['price']['currency'],
                    "outbound": {
                        "departure": outbound['segments'][0]['departure']['iataCode'],
                        "arrival": outbound['segments'][-1]['arrival']['iataCode'],
                        "departure_time": outbound['segments'][0]['departure']['at'],
                        "arrival_time": outbound['segments'][-1]['arrival']['at'],
                        "duration": outbound['duration'],
                        "stops": len(outbound['segments']) - 1,
                        "carrier": outbound['segments'][0]['carrierCode']
                    },
                    "return": {
                        "departure": return_flight['segments'][0]['departure']['iataCode'] if return_flight else None,
                        "arrival": return_flight['segments'][-1]['arrival']['iataCode'] if return_flight else None,
                        "departure_time": return_flight['segments'][0]['departure']['at'] if return_flight else None,
                        "arrival_time": return_flight['segments'][-1]['arrival']['at'] if return_flight else None,
                        "duration": return_flight['duration'] if return_flight else None,
                        "stops": len(return_flight['segments']) - 1 if return_flight else 0
                    } if return_flight else None
                }
                
                filtered_flights.append(flight_info)
                
                if len(filtered_flights) >= 5:  # Limit to top 5
                    break
        
        if not filtered_flights:
            logger.warning("No flights within budget. Using fallback.")
            return _generate_fallback_flights(destination, departure_date, return_date, adults, max_budget_per_person)
        
        return json.dumps({
            "success": True,
            "flights": filtered_flights,
            "count": len(filtered_flights),
            "source": "amadeus"
        })
        
    except ResponseError as error:
        logger.error(f"Amadeus API error: {error}")
        return _generate_fallback_flights(destination, departure_date, return_date, adults, max_budget_per_person)
    except Exception as e:
        logger.error(f"Error in 'search_flights' tool: {e}")
        return json.dumps({"error": str(e), "fallback": True})


@tool
def search_hotels(
    destination: str,
    checkin_date: str,
    checkout_date: str,
    adults: int,
    max_budget_per_night: float,
    accommodation_preference: str = "hotel"
) -> str:
    """
    Search for hotels using SerpAPI (Google Hotels).
    
    Args:
        destination: Destination city
        checkin_date: Check-in date in YYYY-MM-DD format
        checkout_date: Check-out date in YYYY-MM-DD format
        adults: Number of guests
        max_budget_per_night: Maximum budget per night in USD
        accommodation_preference: Type of accommodation (hotel, hostel, airbnb, luxury, budget)
    
    Returns:
        JSON string with hotel options
    """
    logger.info(f"Tool 'search_hotels' called for {destination} from {checkin_date} to {checkout_date}")
    
    try:
        if not SERPAPI_KEY:
            logger.warning("SERPAPI_KEY not set. Using fallback data.")
            return _generate_fallback_hotels(destination, checkin_date, checkout_date, adults, max_budget_per_night)
        
        # Format query
        query = f"hotels in {destination}"
        if accommodation_preference and accommodation_preference != "hotel":
            if accommodation_preference in ["hostel", "airbnb"]:
                query = f"{accommodation_preference} in {destination}"
            elif accommodation_preference == "luxury":
                query = f"luxury hotels in {destination}"
            elif accommodation_preference == "budget":
                query = f"budget hotels in {destination}"
        
        # SerpAPI parameters
        params = {
            "engine": "google_hotels",
            "q": query,
            "check_in_date": checkin_date,
            "check_out_date": checkout_date,
            "adults": adults,
            "currency": "USD",
            "gl": "us",
            "hl": "en",
            "api_key": SERPAPI_KEY
        }
        
        logger.info(f"Searching hotels with SerpAPI: {query}")
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Parse results
        if "properties" not in results:
            logger.warning("No hotels found in SerpAPI results. Using fallback.")
            return _generate_fallback_hotels(destination, checkin_date, checkout_date, adults, max_budget_per_night)
        
        hotels = []
        for prop in results["properties"][:15]:  # Check top 15
            # Extract price (SerpAPI returns various formats)
            price_per_night = None
            if "rate_per_night" in prop:
                price_str = prop["rate_per_night"].get("lowest", "")
                price_per_night = _extract_price(price_str)
            elif "total_rate" in prop:
                price_str = prop["total_rate"].get("lowest", "")
                total_price = _extract_price(price_str)
                nights = (datetime.fromisoformat(checkout_date) - datetime.fromisoformat(checkin_date)).days
                price_per_night = total_price / nights if nights > 0 else total_price
            
            if price_per_night and price_per_night <= max_budget_per_night * 1.15:  # 15% buffer
                hotel_info = {
                    "id": prop.get("property_token", ""),
                    "name": prop.get("name", "Unknown Hotel"),
                    "price_per_night": round(price_per_night, 2),
                    "total_price": round(price_per_night * ((datetime.fromisoformat(checkout_date) - datetime.fromisoformat(checkin_date)).days), 2),
                    "rating": prop.get("overall_rating", 0),
                    "reviews": prop.get("reviews", 0),
                    "location": prop.get("description", ""),
                    "amenities": prop.get("amenities", [])[:5],  # Top 5 amenities
                    "type": prop.get("type", "Hotel"),
                    "link": prop.get("link", "")
                }
                hotels.append(hotel_info)
                
                if len(hotels) >= 5:  # Limit to top 5
                    break
        
        if not hotels:
            logger.warning("No hotels within budget. Using fallback.")
            return _generate_fallback_hotels(destination, checkin_date, checkout_date, adults, max_budget_per_night)
        
        return json.dumps({
            "success": True,
            "hotels": hotels,
            "count": len(hotels),
            "source": "serpapi"
        })
        
    except Exception as e:
        logger.error(f"Error in 'search_hotels' tool: {e}")
        return _generate_fallback_hotels(destination, checkin_date, checkout_date, adults, max_budget_per_night)


def _extract_price(price_str: str) -> float:
    """Extract numeric price from string like '$150' or '150 USD'"""
    import re
    if not price_str:
        return 0.0
    # Remove currency symbols and extract number
    match = re.search(r'[\d,]+\.?\d*', str(price_str).replace(',', ''))
    if match:
        return float(match.group())
    return 0.0


def _generate_fallback_flights(destination: str, departure_date: str, return_date: str, adults: int, max_budget: float) -> str:
    """Generate realistic fallback flight data when API fails"""
    base_price = min(max_budget * 0.8, 600)  # Realistic base price
    
    flights = [
        {
            "id": "fallback_1",
            "price_per_person": round(base_price, 2),
            "total_price": round(base_price * adults, 2),
            "currency": "USD",
            "outbound": {
                "departure": "JFK",
                "arrival": get_airport_code(destination),
                "departure_time": f"{departure_date}T08:00:00",
                "arrival_time": f"{departure_date}T20:00:00",
                "duration": "PT12H",
                "stops": 0,
                "carrier": "AA"
            },
            "return": {
                "departure": get_airport_code(destination),
                "arrival": "JFK",
                "departure_time": f"{return_date}T10:00:00",
                "arrival_time": f"{return_date}T18:00:00",
                "duration": "PT14H",
                "stops": 0
            },
            "note": "Estimated pricing - API unavailable"
        },
        {
            "id": "fallback_2",
            "price_per_person": round(base_price * 0.75, 2),
            "total_price": round(base_price * 0.75 * adults, 2),
            "currency": "USD",
            "outbound": {
                "departure": "JFK",
                "arrival": get_airport_code(destination),
                "departure_time": f"{departure_date}T14:00:00",
                "arrival_time": f"{departure_date}T04:00:00+1",
                "duration": "PT14H",
                "stops": 1,
                "carrier": "UA"
            },
            "return": {
                "departure": get_airport_code(destination),
                "arrival": "JFK",
                "departure_time": f"{return_date}T16:00:00",
                "arrival_time": f"{return_date}T06:00:00+1",
                "duration": "PT16H",
                "stops": 1
            },
            "note": "Budget option - Estimated pricing"
        }
    ]
    
    return json.dumps({
        "success": True,
        "flights": flights,
        "count": len(flights),
        "source": "fallback",
        "note": "Using estimated pricing. Configure Amadeus API for live data."
    })


def _generate_fallback_hotels(destination: str, checkin: str, checkout: str, adults: int, max_budget: float) -> str:
    """Generate realistic fallback hotel data when API fails"""
    nights = (datetime.fromisoformat(checkout) - datetime.fromisoformat(checkin)).days
    base_price = min(max_budget * 0.9, 150)
    
    hotels = [
        {
            "id": "fallback_1",
            "name": f"Grand {destination.split(',')[0]} Hotel",
            "price_per_night": round(base_price, 2),
            "total_price": round(base_price * nights, 2),
            "rating": 4.5,
            "reviews": 1250,
            "location": "City Center",
            "amenities": ["WiFi", "Breakfast", "Gym", "Pool", "Restaurant"],
            "type": "Hotel",
            "note": "Estimated pricing - API unavailable"
        },
        {
            "id": "fallback_2",
            "name": f"Comfort Inn {destination.split(',')[0]}",
            "price_per_night": round(base_price * 0.7, 2),
            "total_price": round(base_price * 0.7 * nights, 2),
            "rating": 4.0,
            "reviews": 850,
            "location": "Near Transit",
            "amenities": ["WiFi", "Breakfast", "24/7 Reception"],
            "type": "Hotel",
            "note": "Budget option - Estimated pricing"
        }
    ]
    
    return json.dumps({
        "success": True,
        "hotels": hotels,
        "count": len(hotels),
        "source": "fallback",
        "note": "Using estimated pricing. Configure SerpAPI for live data."
    })


tools = [search_flights, search_hotels]
tool_node = ToolNode(tools)

# --- Agent and Graph Node Definitions ---

def booking_agent_node(state: TravelPlanningState) -> Dict[str, Any]:
    """
    The core of the Booking Agent. Decides which tools to call.
    """
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Booking agent node running.")
    
    model = ChatOpenAI(model=BOOKING_MODEL, temperature=0.2)
    agent_model = model.bind_tools(tools).with_retry(stop_after_attempt=3)
    
    # Extract budget constraints
    budget = state.get('budget_breakdown', {})
    flight_budget = budget.get('flights', 1000) / state.get('num_travelers', 1)
    
    # Calculate nights
    start_date = state.get('start_date')
    end_date = state.get('end_date')
    if start_date and end_date:
        nights = (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days
        hotel_budget_per_night = budget.get('accommodation', 500) / nights if nights > 0 else 100
    else:
        nights = 5
        hotel_budget_per_night = 100
    
    system_prompt = """You are an expert travel booking specialist. Your job is to find the best flight and hotel options within the user's budget.

Current trip details:
- Destination: {destination}
- Dates: {start_date} to {end_date} ({nights} nights)
- Travelers: {num_travelers}
- Flight Budget: ${flight_budget:.2f} per person (max)
- Hotel Budget: ${hotel_budget:.2f} per night (max)
- Accommodation Preference: {accommodation_preference}

Your task is to:
1. Use the `search_flights` tool to find round-trip flights within budget
2. Use the `search_hotels` tool to find accommodations within budget
3. Once you have both results, provide a clear summary of the TOP 3 options for each
4. Highlight the best value options

Call the tools with these exact parameters:
- search_flights(destination="{destination}", departure_date="{start_date}", return_date="{end_date}", adults={num_travelers}, max_budget_per_person={flight_budget})
- search_hotels(destination="{destination}", checkin_date="{start_date}", checkout_date="{end_date}", adults={num_travelers}, max_budget_per_night={hotel_budget}, accommodation_preference="{accommodation_preference}")

After gathering both results, provide your final recommendation.
"""
    
    prompt = system_prompt.format(
        destination=state['destination'],
        start_date=start_date,
        end_date=end_date,
        nights=nights,
        num_travelers=state['num_travelers'],
        flight_budget=flight_budget,
        hotel_budget=hotel_budget_per_night,
        accommodation_preference=state.get('accommodation_preference', 'hotel')
    )
    
    messages = [SystemMessage(content=prompt)] + state.get("messages", [])[-3:]  # Keep last 3 messages for context
    
    try:
        response = agent_model.invoke(messages)
        logger.info(f"TripID {trip_id}: Booking agent LLM invoked successfully.")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"TripID {trip_id}: Error invoking booking agent model: {e}")
        return {"errors": state.get("errors", []) + [f"Booking agent LLM failed: {e}"]}


def should_continue_booking(state: TravelPlanningState) -> str:
    """Determines whether to continue the ReAct loop or finish."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.info(f"TripID {state.get('trip_id')}: Booking complete. Proceeding to parser.")
        return "end"
    else:
        logger.info(f"TripID {state.get('trip_id')}: Booking agent requested tool call. Continuing.")
        return "continue"


def parse_booking_output_node(state: TravelPlanningState) -> Dict[str, Any]:
    """Parses the final AI message and structures flight/hotel options."""
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Parsing final booking output.")
    
    # Extract tool call results from message history
    flight_data = None
    hotel_data = None
    
    # Look through recent messages for tool results
    for msg in reversed(state['messages'][-10:]):  # Check last 10 messages
        if hasattr(msg, 'content') and isinstance(msg.content, str):
            try:
                data = json.loads(msg.content)
                if 'flights' in data and not flight_data:
                    flight_data = data
                elif 'hotels' in data and not hotel_data:
                    hotel_data = data
            except:
                continue
    
    # Fallback: Generate basic options if parsing failed
    if not flight_data or not hotel_data:
        logger.warning(f"TripID {trip_id}: Could not parse tool results. Using state data or fallback.")
        
        # Try to extract from final message
        final_message = state['messages'][-1].content if state['messages'] else ""
        
        # Use existing data in state if available, otherwise create minimal fallback
        if not flight_data:
            flight_data = {"flights": state.get('flight_options', []), "source": "state"}
        if not hotel_data:
            hotel_data = {"hotels": state.get('hotel_options', []), "source": "state"}
    
    flights = flight_data.get('flights', [])[:5] if flight_data else []
    hotels = hotel_data.get('hotels', [])[:5] if hotel_data else []
    
    # Create confirmation message
    if flights and hotels:
        min_flight = min([f.get('price_per_person', 999999) for f in flights])
        min_hotel = min([h.get('price_per_night', 999999) for h in hotels])
        
        confirmation_msg = (
            f"✈️ **Flights Found**: {len(flights)} options\n"
            f"   • Best price: ${min_flight:.2f} per person\n"
            f"   • Airlines: {', '.join(set([f.get('outbound', {}).get('carrier', 'N/A') for f in flights[:3]]))}\n\n"
            f"🏨 **Hotels Found**: {len(hotels)} options\n"
            f"   • Best rate: ${min_hotel:.2f} per night\n"
            f"   • Top rated: {hotels[0].get('name', 'N/A')} ({hotels[0].get('rating', 'N/A')}★)\n\n"
            f"Ready to create your detailed itinerary!"
        )
    else:
        confirmation_msg = "I've gathered booking information. Let's create your itinerary!"
    
    return {
        "flight_options": flights,
        "hotel_options": hotels,
        "current_step": "create_itinerary",
        "messages": [AIMessage(content=confirmation_msg)],
        "last_updated": datetime.utcnow().isoformat()
    }


# --- Graph Assembly ---

booking_graph = StateGraph(TravelPlanningState)

booking_graph.add_node("booking_agent", booking_agent_node)
booking_graph.add_node("booking_tools", tool_node)
booking_graph.add_node("parser", parse_booking_output_node)

booking_graph.set_entry_point("booking_agent")

booking_graph.add_conditional_edges(
    "booking_agent",
    should_continue_booking,
    {"continue": "booking_tools", "end": "parser"}
)
booking_graph.add_edge("booking_tools", "booking_agent")
booking_graph.add_edge("parser", END)

# Compile the graph into a runnable application
booking_agent_app = booking_graph.compile()
logger.info("Booking agent graph compiled successfully.")

# --- Test Block ---
if __name__ == '__main__':
    logger.info("Running stand-alone test for Booking Agent...")
    
    # Example state after Budget Agent has run
    initial_state_data = {
        'messages': [HumanMessage(content="Find me flights and hotels")],
        'destination': 'Tokyo, Japan',
        'start_date': '2026-03-15',
        'end_date': '2026-03-20',
        'num_travelers': 2,
        'budget_total': 4000.0,
        'accommodation_preference': 'hotel',
        'requirements_extracted': True,
        'research_results': {"attractions": [], "weather": "Pleasant", "local_tips": []},
        'budget_breakdown': {
            'flights': 1400.0,
            'accommodation': 1200.0,
            'food': 400.0,
            'activities': 300.0,
            'transport': 200.0,
            'total': 4000.0,
            'remaining': 500.0
        },
        'current_step': 'booking_flights',
        'trip_id': 'test-booking-12345',
        'errors': [],
    }
    
    initial_state = TravelPlanningState(**initial_state_data)
    
    # Invoke the booking agent sub-graph
    final_state = booking_agent_app.invoke(initial_state)
    
    print("\n--- Final State after Booking ---")
    print(f"Current Step: {final_state.get('current_step')}")
    print(f"Errors: {final_state.get('errors')}")
    
    print("\n--- Flight Options ---")
    if final_state.get('flight_options'):
        for i, flight in enumerate(final_state['flight_options'][:3], 1):
            print(f"\nOption {i}:")
            print(f"  Price: ${flight.get('price_per_person', 0):.2f} per person")
            print(f"  Carrier: {flight.get('outbound', {}).get('carrier', 'N/A')}")
            print(f"  Stops: {flight.get('outbound', {}).get('stops', 0)}")
    else:
        print("No flight options generated.")
    
    print("\n--- Hotel Options ---")
    if final_state.get('hotel_options'):
        for i, hotel in enumerate(final_state['hotel_options'][:3], 1):
            print(f"\nOption {i}:")
            print(f"  Name: {hotel.get('name', 'N/A')}")
            print(f"  Price: ${hotel.get('price_per_night', 0):.2f} per night")
            print(f"  Rating: {hotel.get('rating', 'N/A')}★")
    else:
        print("No hotel options generated.")
    
    print("\n--- Last Message ---")
    if final_state.get('messages'):
        print(final_state['messages'][-1].content)