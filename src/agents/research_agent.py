# research_agent.py
"""
Research Agent (Production Version)

This module defines a self-contained, ReAct-style agent for conducting travel research.
It is implemented as a LangGraph sub-graph that can be invoked by the main workflow.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
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

RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-4o-mini")
PARSER_MODEL = os.getenv("PARSER_MODEL", "gpt-4o-mini")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY environment variable not set.")

# --- Tools Definition ---
# Tools are defined with error handling to make the agent more resilient.

@tool
def search_attractions(destination: str, interests: List[str]) -> str:
    """Searches for attractions, landmarks, and activities in a destination relevant to user interests."""
    logger.info(f"Tool 'search_attractions' called for {destination} with interests: {interests}")
    try:
        query = (f"Find top-rated attractions, activities, and landmarks in {destination} "
                 f"for someone interested in {', '.join(interests)}. Include a mix of famous sites and local experiences.")
        search = TavilySearch(max_results=5)
        return search.invoke({"query": query, "search_depth": "advanced"})
    except Exception as e:
        logger.error(f"Error in 'search_attractions' tool: {e}")
        return f"Error: Could not perform search for attractions. {e}"

@tool
def get_weather_forecast(destination: str, start_date: str) -> str:
    """Gets the expected weather forecast for a destination around a specific start date."""
    logger.info(f"Tool 'get_weather_forecast' called for {destination} on {start_date}")
    try:
        query = (f"What is the typical weather forecast in {destination} around {start_date}? "
                 f"Include average temperature in Celsius, chance of rain, and general conditions.")
        search = TavilySearch(max_results=3)
        return search.invoke({"query": query})
    except Exception as e:
        logger.error(f"Error in 'get_weather_forecast' tool: {e}")
        return f"Error: Could not perform weather search. {e}"

tools = [search_attractions, get_weather_forecast]
tool_node = ToolNode(tools)

# --- Agent and Graph Node Definitions ---

def research_agent_node(state: TravelPlanningState) -> Dict[str, Any]:
    """The core of the ReAct agent. It calls the LLM to decide the next action."""
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Research agent node running.")
    
    # First create the model, then bind tools, then add retry
    model = ChatOpenAI(model=RESEARCH_MODEL, temperature=0.2)
    agent_model = model.bind_tools(tools).with_retry(stop_after_attempt=3)
    
    system_prompt = """You are an expert travel researcher, part of a multi-agent system. Your job is to use the provided tools to gather information for a travel plan.

Current trip details:
- Destination: {destination}
- Dates: Around {start_date}
- Interests: {interests}

Your task is to:
1. Use the `search_attractions` tool to find relevant activities and landmarks.
2. Use the `get_weather_forecast` tool to find weather information.
3. Once you have successfully gathered information from ALL necessary tools, synthesize your findings into a comprehensive summary.
4. Do not ask for clarification. Do not finish until all information is gathered.

Your final response should be a conversational summary of your findings.
"""
    prompt = system_prompt.format(
        destination=state['destination'],
        start_date=state['start_date'],
        interests=", ".join(state['interests'])
    )
    
    messages = [SystemMessage(content=prompt)] + state.get("messages", [])

    try:
        response = agent_model.invoke(messages)
        logger.info(f"TripID {trip_id}: Research agent LLM invoked successfully.")
        return {"messages": [response]}
    except Exception as e:
        logger.error(f"TripID {trip_id}: Error invoking research agent model: {e}")
        return {"errors": state.get("errors", []) + [f"Research agent LLM failed: {e}"]}

def should_continue_research(state: TravelPlanningState) -> str:
    """Determines whether to continue the ReAct loop or finish."""
    last_message = state["messages"][-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        logger.info(f"TripID {state.get('trip_id')}: Research complete. Proceeding to parser.")
        return "end"
    else:
        logger.info(f"TripID {state.get('trip_id')}: Research agent requested tool call. Continuing.")
        return "continue"

def parse_research_output_node(state: TravelPlanningState) -> Dict[str, Any]:
    """Parses the final AI message into a structured `research_results` dictionary."""
    trip_id = state.get('trip_id')
    logger.info(f"TripID {trip_id}: Parsing final research output.")
    
    final_message = state['messages'][-1].content
    
    # Define a Pydantic model for the structured output
    class ResearchOutput(BaseModel):
        attractions: List[str] = Field(..., description="List of top attractions")
        weather: str = Field(..., description="Weather information")
        local_tips: List[str] = Field(..., description="List of local tips")
    
    # First create the model, then add structured output, then add retry
    parser_model = ChatOpenAI(model=PARSER_MODEL, temperature=0)
    structured_llm = parser_model.with_structured_output(ResearchOutput).with_retry(stop_after_attempt=3)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a data parsing expert. Your task is to parse the provided text into a structured dictionary with keys: `attractions` (list of strings), `weather` (string), and `local_tips` (list of strings). Extract the relevant information from the text for each key."),
        ("user", "{text_to_parse}")
    ])
    
    parser_chain = prompt | structured_llm
    
    try:
        research_output = parser_chain.invoke({"text_to_parse": final_message})
        logger.info(f"TripID {trip_id}: Successfully parsed research output into structured format.")
        
        # Convert Pydantic model to dict for JSON serialization
        structured_output = {
            "attractions": research_output.attractions,
            "weather": research_output.weather,
            "local_tips": research_output.local_tips
        }
        
        return {
            "research_results": structured_output,
            "current_step": "budget_planning",  # Transition to the next step
            "messages": [AIMessage(content="I've completed the initial research for your trip! Next, let's plan your budget.")]
        }
    except Exception as e:
        logger.error(f"TripID {trip_id}: Failed to parse research output: {e}")
        return {"errors": state.get("errors", []) + ["Failed to structure research output."]}

# --- Graph Assembly ---

research_graph = StateGraph(TravelPlanningState)

research_graph.add_node("research_agent", research_agent_node)
research_graph.add_node("research_tools", tool_node)
research_graph.add_node("parser", parse_research_output_node)

research_graph.set_entry_point("research_agent")

research_graph.add_conditional_edges(
    "research_agent",
    should_continue_research,
    {"continue": "research_tools", "end": "parser"}
)
research_graph.add_edge("research_tools", "research_agent")
research_graph.add_edge("parser", END)

# Compile the graph into a runnable application
research_agent_app = research_graph.compile()
logger.info("Research agent graph compiled successfully.")

# --- Test Block ---
if __name__ == '__main__':
    logger.info("Running stand-alone test for Research Agent...")

    # Example initial state after orchestrator has run
    initial_state_data = {
        'messages': [HumanMessage(content="I want to plan a 5 day trip to Kyoto, Japan next March. I'm interested in history and food.")],
        'destination': 'Kyoto, Japan',
        'start_date': '2026-03-15',
        'end_date': '2026-03-20',
        'num_travelers': 1,
        'interests': ['history', 'food', 'temples'],
        'requirements_extracted': True,
        'current_step': 'research',
        'trip_id': 'test-trip-12345',
        'errors': [],
    }
    # Cast to the TypedDict for compatibility
    initial_state = TravelPlanningState(**initial_state_data)

    # Invoke the research agent sub-graph
    final_state = research_agent_app.invoke(initial_state)

    print("\n--- Final State after Research ---")
    print(f"Current Step: {final_state.get('current_step')}")
    print(f"Errors: {final_state.get('errors')}")

    print("\n--- Structured Research Results ---")
    if final_state.get('research_results'):
        print(json.dumps(final_state['research_results'], indent=2))
    else:
        print("No research results were generated.")



# # research_agent.py
# """
# Research Agent (Upgraded Production Version)

# This module defines a self-contained, ReAct-style agent for conducting travel research.
# This version is upgraded to use the Google Places API for finding attractions, providing
# structured, reliable data for downstream agents.
# """

# import os
# import json
# import logging
# from typing import Dict, Any, List, Optional

# import googlemaps
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.tools import tool
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
# from langchain_tavily import TavilySearch
# from langchain_core.pydantic_v1 import BaseModel, Field
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode

# # Import the main state definition
# from state import TravelPlanningState

# # --- Configuration and Logging ---
# load_dotenv()
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
# logger = logging.getLogger(__name__)

# RESEARCH_MODEL = os.getenv("RESEARCH_MODEL", "gpt-4o-mini")
# PARSER_MODEL = os.getenv("PARSER_MODEL", "gpt-4o-mini")
# TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
# GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# if not TAVILY_API_KEY or not GOOGLE_MAPS_API_KEY:
#     raise ValueError("TAVILY_API_KEY and GOOGLE_MAPS_API_KEY environment variables must be set.")

# # Initialize Google Maps client
# gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

# # --- Upgraded Tools Definition ---

# @tool
# def search_attractions(destination: str, interests: List[str]) -> str:
#     """
#     Finds top-rated attractions in a destination using Google Places API
#     based on user interests. Returns structured data including Place IDs.
#     """
#     logger.info(f"Tool 'search_attractions' (Google Places) called for {destination}")
#     try:
#         query = f"Top attractions in {destination} for someone interested in {', '.join(interests)}"
#         places_result = gmaps.places(query=query)
        
#         attractions = []
#         for place in places_result.get('results', [])[:7]: # Get top 7 results
#             attractions.append({
#                 "name": place.get('name'),
#                 "address": place.get('formatted_address'),
#                 "rating": place.get('rating'),
#                 "place_id": place.get('place_id')
#             })
        
#         if not attractions:
#             return "No attractions found for the given criteria."
            
#         return json.dumps(attractions)
#     except Exception as e:
#         logger.error(f"Error in 'search_attractions' tool: {e}")
#         return f"Error: Could not perform search for attractions using Google Places API. {e}"

# @tool
# def get_weather_forecast(destination: str, start_date: str) -> str:
#     """Gets the expected weather forecast for a destination around a specific start date."""
#     logger.info(f"Tool 'get_weather_forecast' called for {destination} on {start_date}")
#     try:
#         query = (f"What is the typical weather forecast in {destination} around {start_date}? "
#                  f"Include average temperature in Celsius, chance of rain, and what to pack.")
#         return TavilySearch (max_results=1).invoke({"query": query})
#     except Exception as e:
#         logger.error(f"Error in 'get_weather_forecast' tool: {e}")
#         return f"Error: Could not perform weather search. {e}"

# tools = [search_attractions, get_weather_forecast]
# tool_node = ToolNode(tools)

# # --- Pydantic Models for Parsing ---

# class Attraction(BaseModel):
#     name: str = Field(description="The official name of the attraction.")
#     address: Optional[str] = Field(description="The address of the attraction.")
#     rating: Optional[float] = Field(description="The user rating of the attraction.")
#     place_id: str = Field(description="The unique Google Place ID for the attraction.")

# class ResearchOutput(BaseModel):
#     attractions: List[Attraction] = Field(description="A list of recommended attractions with their details.")
#     weather: str = Field(description="A summary of the weather forecast and packing advice.")
#     local_tips: List[str] = Field(description="Actionable local tips for the traveler.")

# # --- Agent and Graph Node Definitions ---

# def research_agent_node(state: TravelPlanningState) -> Dict[str, Any]:
#     """The core of the ReAct agent. It calls the LLM to decide the next action."""
#     # This node's logic remains the same, as it's tool-agnostic.
#     # The agent will automatically learn to use the new, better `search_attractions` tool.
#     trip_id = state.get('trip_id')
#     logger.info(f"TripID {trip_id}: Research agent node running.")
    
#     model = ChatOpenAI(model=RESEARCH_MODEL, temperature=0.2).with_retry(stop_after_attempt=3)
#     agent_model = model.bind_tools(tools)
    
#     # The prompt remains largely the same, but the agent will now receive structured data from its tool.
#     system_prompt = """You are an expert travel researcher. Your job is to use the provided tools to gather information for a travel plan.

# Current trip details:
# - Destination: {destination}
# - Dates: Around {start_date}
# - Interests: {interests}

# Your task is to:
# 1.  Use the `search_attractions` tool to get a list of structured attraction data.
# 2.  Use the `get_weather_forecast` tool to find weather information.
# 3.  Synthesize your findings into a comprehensive summary, including a few local tips based on the attractions found.
# 4.  Do not finish until all information is gathered. Your final response should be a conversational summary.
# """
#     prompt = system_prompt.format(
#         destination=state['destination'],
#         start_date=state['start_date'],
#         interests=", ".join(state['interests'])
#     )
    
#     messages = [SystemMessage(content=prompt)] + state.get("messages", [])
#     response = agent_model.invoke(messages)
#     return {"messages": [response]}


# def should_continue_research(state: TravelPlanningState) -> str:
#     """Determines whether to continue the ReAct loop or finish."""
#     last_message = state["messages"][-1]
#     if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
#         return "end"
#     return "continue"

# def parse_research_output_node(state: TravelPlanningState) -> Dict[str, Any]:
#     """Parses the final AI message into the structured ResearchOutput model."""
#     trip_id = state.get('trip_id')
#     logger.info(f"TripID {trip_id}: Parsing final research output with upgraded parser.")
    
#     final_message = state['messages'][-1].content
    
#     parser_model = ChatOpenAI(model=PARSER_MODEL, temperature=0).with_retry(stop_after_attempt=3)
#     structured_llm = parser_model.with_structured_output(ResearchOutput)

#     # The prompt is updated to expect the richer Attraction object.
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "Parse the provided text into the `ResearchOutput` JSON schema. The `attractions` field should be a list of objects, each containing a name, address, rating, and place_id."),
#         ("user", "{text_to_parse}")
#     ])
    
#     parser_chain = prompt | structured_llm
    
#     try:
#         research_output = parser_chain.invoke({"text_to_parse": final_message})
#         logger.info(f"TripID {trip_id}: Successfully parsed research output into structured format.")
        
#         return {
#             "research_results": research_output.dict(), # Convert Pydantic model to dict for state
#             "current_step": "budget_planning",
#             "messages": [AIMessage(content="I've completed the initial research. Next, let's plan your budget.")]
#         }
#     except Exception as e:
#         logger.error(f"TripID {trip_id}: Failed to parse upgraded research output: {e}")
#         return {"errors": state.get("errors", []) + ["Failed to structure research output."]}

# # --- Graph Assembly ---

# research_graph = StateGraph(TravelPlanningState)

# research_graph.add_node("research_agent", research_agent_node)
# research_graph.add_node("research_tools", tool_node)
# research_graph.add_node("parser", parse_research_output_node)

# research_graph.set_entry_point("research_agent")
# research_graph.add_conditional_edges("research_agent", should_continue_research, {"continue": "research_tools", "end": "parser"})
# research_graph.add_edge("research_tools", "research_agent")
# research_graph.add_edge("parser", END)

# research_agent_app = research_graph.compile()
# logger.info("Upgraded research agent graph compiled successfully.")

# # --- Test Block ---
# if __name__ == '__main__':
#     logger.info("Running stand-alone test for UPGRADED Research Agent...")

#     initial_state_data = {
#         'messages': [HumanMessage(content="I want to plan a 5 day trip to Kyoto, Japan next March. I'm interested in history and temples.")],
#         'destination': 'Kyoto, Japan', 'start_date': '2026-03-15', 'end_date': '2026-03-20',
#         'num_travelers': 1, 'interests': ['history', 'temples', 'gardens'],
#         'requirements_extracted': True, 'current_step': 'research',
#         'trip_id': 'test-trip-upgrade-123', 'errors': [],
#     }
#     initial_state = TravelPlanningState(**initial_state_data)

#     final_state = research_agent_app.invoke(initial_state)

#     print("\n--- Final State after Upgraded Research ---")
#     print(f"Current Step: {final_state.get('current_step')}")
#     print(f"Errors: {final_state.get('errors')}")

#     print("\n--- Structured Research Results (Note the new structure) ---")
#     if final_state.get('research_results'):
#         print(json.dumps(final_state['research_results'], indent=2))
#     else:
#         print("No research results were generated.")