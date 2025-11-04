# budget_agent.py
"""
Budget Agent (Production Version)

This module defines a self-contained, ReAct-style agent for creating a travel budget.
It uses tools to estimate flight and accommodation costs to create a realistic budget breakdown.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.chat_models import init_chat_model

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

# BUDGET_MODEL = os.getenv("BUDGET_MODEL", "gpt-4o-mini")
# PARSER_MODEL = os.getenv("PARSER_MODEL", "gpt-4o-mini")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set.")

llm = init_chat_model("groq:llama-3.1-8b-instant")

# --- Tools Definition ---

@tool
def get_estimated_flight_cost(destination: str, start_date: str) -> str:
    """
    Gets the estimated round-trip flight cost for a given destination and date.
    """
    logger.info(f"Tool 'get_estimated_flight_cost' called for {destination} around {start_date}")
    try:
        query = (f"What is the average or typical round-trip flight cost per person to {destination} "
                 f"for a trip starting around {start_date}? Provide a single estimated cost in INR.")
        return TavilySearch(max_results=2).invoke({"query": query})
    except Exception as e:
        logger.error(f"Error in 'get_estimated_flight_cost' tool: {e}")
        return f"Error: Could not perform search for flight costs. {e}"

@tool
def get_estimated_accommodation_cost(destination: str, accommodation_preference: str, num_travelers: int) -> str:
    """
    Gets the estimated nightly accommodation cost based on destination, preference, and number of travelers.
    """
    logger.info(f"Tool 'get_estimated_accommodation_cost' called for {destination} ({accommodation_preference})")
    try:
        query = (f"What is the average nightly cost for {accommodation_preference} accommodation in {destination} "
                 f"suitable for {num_travelers} person(s)? Provide a single estimated cost in INR.")
        return TavilySearch(max_results=2).invoke({"query": query})
    except Exception as e:
        logger.error(f"Error in 'get_estimated_accommodation_cost' tool: {e}")
        return f"Error: Could not perform search for accommodation costs. {e}"

tools = [get_estimated_flight_cost, get_estimated_accommodation_cost]
tool_node = ToolNode(tools)

# --- Agent and Graph Node Definitions ---

def budget_agent_node(state: TravelPlanningState) -> Dict[str, Any]:
    """
    The core of the Budget Agent. It calls an LLM to decide the next action.
    """
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Budget agent node running.")

    model = llm.bind_tools(tools).with_retry(stop_after_attempt=3)

    system_prompt = """You are an expert travel budget planner. Your goal is to create a sensible and realistic budget breakdown for a trip based on the user's total budget and preferences.

Current trip details:
- Destination: {destination}
- Dates: {start_date} to {end_date} ({num_days} days)
- Travelers: {num_travelers}
- Total Budget: {total_budget}
- Accommodation Preference: {accommodation_preference}

Your task is to:
1.  Use the `get_estimated_flight_cost` tool to find the approximate cost of flights per person.
2.  Use the `get_estimated_accommodation_cost` tool to find the approximate nightly rate for lodging.
3.  Once you have these key estimates, calculate the total costs for flights and accommodation for all travelers for the entire trip duration.
4.  Allocate the REMAINING budget to other categories like 'Food', 'Activities', and 'Local Transport'. Be realistic with your allocations.
5.  Present a final, synthesized summary of this budget plan. You must provide a final answer once all tool calls are complete.
"""
    start_dt = datetime.fromisoformat(state['start_date'])
    end_dt = datetime.fromisoformat(state['end_date'])
    num_days = (end_dt - start_dt).days

    prompt = system_prompt.format(
        destination=state['destination'],
        start_date=state['start_date'],
        end_date=state['end_date'],
        num_days=num_days,
        num_travelers=state['num_travelers'],
        total_budget=state['budget_total'],
        accommodation_preference=state.get('accommodation_preference', 'hotel')
    )

    messages = [SystemMessage(content=prompt)] + state.get("messages", [])

    try:
        response = model.invoke(messages)
        logger.info(f"TripID {trip_id}: Budget agent LLM invoked successfully.")
        print(response)
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"TripID {trip_id}: Error invoking budget agent model: {e}")
        return {"errors": state.get("errors", []) + [f"Budget agent LLM failed: {e}"]}

def should_continue_budget(state: TravelPlanningState) -> str:
    """Determines whether to continue the ReAct loop or finish."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.info(f"TripID {state.get('trip_id')}: Budget planning complete. Proceeding to parser.")
        return "end"
    else:
        logger.info(f"TripID {state.get('trip_id')}: Budget agent requested tool call. Continuing.")
        return "continue"

def parse_budget_output_node(state: TravelPlanningState) -> Dict[str, Any]:
    """Parses the final AI message into a structured `budget_breakdown` dictionary."""
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Parsing final budget output.")

    final_message = state['messages'][-1].content

    # Define a Pydantic model for the structured output
    class BudgetBreakdown(BaseModel):
        flights: float = Field(..., description="Total cost of flights for all travelers")
        accommodation: float = Field(..., description="Total cost of accommodation for the entire stay")
        food: float = Field(..., description="Estimated daily food cost per person")
        activities: float = Field(..., description="Estimated daily activities cost per person")
        transport: float = Field(..., description="Estimated daily local transport cost per person")
        total: float = Field(..., description="Total budget allocated")
        remaining: float = Field(..., description="Remaining budget after all allocations")
    
    # Create parser model with structured output and retry
    parser_model = llm.with_structured_output(BudgetBreakdown).with_retry(stop_after_attempt=3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a data parsing expert. Your task is to parse the provided text into a structured dictionary.
The dictionary should contain the following numeric (float or int) keys: 'flights', 'accommodation', 'food', 'activities', 'transport', 'total', 'remaining'.
Extract the final allocated budget number for each category from the text."""),
        ("user", "Please parse the following budget plan summary:\n\n{text_to_parse}")
    ])
    
    parser_chain = prompt | parser_model

    try:
        budget_output = parser_chain.invoke({"text_to_parse": final_message})
        logger.info(f"TripID {trip_id}: Successfully parsed budget output into structured format.")

        # output validation using model_validate_json method
        budget_object = BudgetBreakdown.model_validate_json(budget_output)
        
        # Convert Pydantic model to dict for JSON serialization
        structured_output = {
            "flights": budget_object.flights,
            "accommodation": budget_object.accommodation,
            "food": budget_object.food,
            "activities": budget_object.activities,
            "transport": budget_object.transport,
            "total": budget_object.total,
            "remaining": budget_object.remaining
        }
        
        # Add a confirmation message for the user
        confirmation_msg = (f"I've created a budget plan for your trip. We've allocated "
                          f"${budget_object.flights:.2f} for flights and "
                          f"${budget_object.accommodation:.2f} for accommodation. "
                          f"Now, let's find some flights!")

        return {
            "budget_breakdown": structured_output,
            "current_step": "booking_flights",  # Transition to the next step
            "messages": [AIMessage(content=confirmation_msg)]
        }
    except Exception as e:
        logger.error(f"TripID {trip_id}: Failed to parse budget output: {e}")
        return {"errors": state.get("errors", []) + ["Failed to structure budget output."]}


# --- Graph Assembly ---

budget_graph = StateGraph(TravelPlanningState)

budget_graph.add_node("budget_agent", budget_agent_node)
budget_graph.add_node("budget_tools", tool_node)
budget_graph.add_node("parser", parse_budget_output_node)

budget_graph.set_entry_point("budget_agent")

budget_graph.add_conditional_edges(
    "budget_agent",
    should_continue_budget,
    {"continue": "budget_tools", "end": "parser"}
)
budget_graph.add_edge("budget_tools", "budget_agent")
budget_graph.add_edge("parser", END)

# Compile the graph into a runnable application
budget_agent_app = budget_graph.compile()
logger.info("Budget agent graph compiled successfully.")

# --- Test Block ---
if __name__ == '__main__':

    logger.info("Running stand-alone test for Budget Agent...")

    # Example state after the Research Agent has run
    initial_state_data = {
        'messages': [HumanMessage(content="My budget is 5000 Rs total.")],
        'destination': 'Delhi, India',
        'start_date': '2025-11-15',
        'end_date': '2025-11-20', # 5 days
        'num_travelers': 2,
        'budget_total': 5000.0,
        'accommodation_preference': 'budget hotel',
        'requirements_extracted': True,
        'research_results': {"attractions": [], "weather": "Mild and pleasant.", "local_tips": []},
        'current_step': 'budget_planning',
        'trip_id': 'test-trip-67890',
        'errors': [],
    }
    initial_state = TravelPlanningState(**initial_state_data)
    
    # Invoke the budget agent sub-graph
    final_state = budget_agent_app.invoke(initial_state)

    print("\n--- Final State after Budget Planning ---")
    print(f"Current Step: {final_state.get('current_step')}")
    print(f"Errors: {final_state.get('errors')}")

    print("\n--- Structured Budget Breakdown ---")
    if final_state.get('budget_breakdown'):
        import json
        print(json.dumps(final_state['budget_breakdown'], indent=2))
    else:
        print("No budget breakdown was generated.")
        
    print("\n--- Last Message ---")
    if final_state.get('messages'):
        print(final_state['messages'][-1].content)