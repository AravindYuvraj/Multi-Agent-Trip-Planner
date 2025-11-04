"""
Orchestrator Agent - Master Planner (Updated for Sub-Graph Agents)

Responsible for:
- Extracting trip requirements from user messages
- Routing to appropriate specialized agent sub-graphs
- Managing workflow progression
- Handling user clarifications
"""

import os
import json
import re
import logging
from typing import Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# --- Configuration and Logging ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gpt-4o-mini")


class OrchestratorAgent:
    """
    Master planning agent that coordinates the entire travel planning workflow.
    Routes tasks to specialized agent sub-graphs (research, budget, booking, itinerary).
    """
    
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        self.model_name = model_name or ORCHESTRATOR_MODEL
        self.llm = ChatOpenAI(model=self.model_name, temperature=temperature)
        logger.info(f"OrchestratorAgent initialized with model: {self.model_name}")
    
    def _clean_json_response(self, content: str) -> str:
        """
        Remove markdown code blocks and extract clean JSON.
        Handles cases where LLM wraps JSON in ```json ... ```
        """
        # Remove ```json and ``` markers
        content = re.sub(r'^```json\s*', '', content, flags=re.MULTILINE)
        content = re.sub(r'\s*```$', '', content, flags=re.MULTILINE)
        return content.strip()
        
    def extract_requirements(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract trip requirements from user messages.
        Returns updated state with extracted information and missing requirements.
        
        This is the critical first step that gates the entire workflow.
        """
        trip_id = state.get('trip_id', 'unknown')
        logger.info(f"TripID {trip_id}: Orchestrator extracting requirements")
        
        # Get the latest user message
        messages = state.get("messages", [])
        
        # Handle both tuple format and Message objects
        user_messages = []
        for msg in messages:
            if isinstance(msg, tuple) and msg[0] == "user":
                user_messages.append(msg[1])
            elif hasattr(msg, 'type') and msg.type == "human":  # Changed from "user" to "human"
                user_messages.append(msg.content)
            elif hasattr(msg, 'type') and msg.type == "user":  # Keep for backwards compatibility
                user_messages.append(msg.content)
        
        if not user_messages:
            logger.warning(f"TripID {trip_id}: No user messages found in state")
            logger.warning(f"TripID {trip_id}: Messages in state: {[(type(m), getattr(m, 'type', 'tuple') if hasattr(m, 'type') else m[0] if isinstance(m, tuple) else 'unknown') for m in messages]}")
            return {
                "errors": state.get("errors", []) + ["No user message to extract requirements from"],
                "requires_user_input": True
            }
        
        latest_message = user_messages[-1]
        logger.info(f"TripID {trip_id}: Processing user message: {latest_message[:100]}...")
        
        # System prompt for requirement extraction
        system_prompt = """You are a travel planning requirements extractor. Your job is to parse user messages and extract structured trip information.

Extract the following information from the user's message:
- destination (city, country - be specific)
- start_date (YYYY-MM-DD format, infer from context like "next month", "June 2025", "in 3 weeks")
- end_date (YYYY-MM-DD format, calculate from duration if given "5 days")
- num_travelers (default to 1 if not mentioned)
- budget_total (in USD, extract if mentioned)
- interests (list from: culture, food, adventure, nature, history, shopping, nightlife, relaxation, beaches, museums, temples)
- accommodation_preference (hotel, hostel, airbnb, luxury, budget)

Current date for reference: {current_date}

Return ONLY a valid JSON object (no markdown, no code blocks, no extra text) with extracted fields. Use null for missing information.
Also include a "missing_requirements" list for critical missing fields.

CRITICAL REQUIRED fields: destination, start_date, end_date
IMPORTANT OPTIONAL fields: budget_total, num_travelers
OTHER OPTIONAL fields: interests, accommodation_preference

Example outputs:

Complete request:
{{
    "destination": "Tokyo, Japan",
    "start_date": "2025-06-01",
    "end_date": "2025-06-05",
    "num_travelers": 2,
    "budget_total": 3000,
    "interests": ["culture", "food"],
    "accommodation_preference": "hotel",
    "missing_requirements": []
}}

Incomplete request:
{{
    "destination": "Paris, France",
    "start_date": null,
    "end_date": null,
    "num_travelers": 1,
    "budget_total": null,
    "interests": null,
    "accommodation_preference": null,
    "missing_requirements": ["start_date", "end_date"]
}}

IMPORTANT: Return ONLY the JSON object, nothing else. No explanations, no markdown."""
        
        messages_to_send = [
            SystemMessage(content=system_prompt.format(
                current_date=datetime.now().strftime("%Y-%m-%d")
            )),
            HumanMessage(content=f"User message to parse:\n{latest_message}")
        ]
        
        # Call LLM to extract requirements
        try:
            response = self.llm.invoke(messages_to_send)
            logger.info(f"TripID {trip_id}: LLM responded for requirement extraction")
            
            # Clean the response
            cleaned_content = self._clean_json_response(response.content)
            extracted = json.loads(cleaned_content)
            logger.info(f"TripID {trip_id}: Successfully parsed JSON: {list(extracted.keys())}")
            
            # Handle None interests (convert to empty list)
            interests = extracted.get("interests")
            if interests is None:
                interests = []
            
            # Update state with extracted information
            updates = {
                "destination": extracted.get("destination"),
                "start_date": extracted.get("start_date"),
                "end_date": extracted.get("end_date"),
                "num_travelers": extracted.get("num_travelers", 1),
                "budget_total": extracted.get("budget_total"),
                "interests": interests,
                "accommodation_preference": extracted.get("accommodation_preference") or "hotel",
                "missing_requirements": extracted.get("missing_requirements", []),
                "requirements_extracted": len(extracted.get("missing_requirements", [])) == 0,
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Get existing messages
            existing_messages = state.get("messages", [])
            
            # Determine next step based on whether requirements are complete
            if updates["requirements_extracted"]:
                updates["current_step"] = "research"
                updates["next_agent"] = "research"
                updates["requires_user_input"] = False
                
                # Success confirmation message
                confirmation = (
                    f"Perfect! I'll help you plan your {updates['num_travelers']}-person trip to "
                    f"{updates['destination']} from {updates['start_date']} to {updates['end_date']}. "
                    f"Let me start researching the best attractions and experiences for you!"
                )
                updates["messages"] = existing_messages + [AIMessage(content=confirmation)]
                logger.info(f"TripID {trip_id}: Requirements complete. Proceeding to research.")
            else:
                updates["requires_user_input"] = True
                updates["current_step"] = "extract_requirements"
                
                # Generate clarification message
                clarification = self._generate_clarification(updates["missing_requirements"])
                updates["messages"] = existing_messages + [AIMessage(content=clarification)]
                logger.info(f"TripID {trip_id}: Requirements incomplete. Missing: {updates['missing_requirements']}")
            
            return updates
            
        except json.JSONDecodeError as e:
            logger.error(f"TripID {trip_id}: JSON parsing failed - {e}")
            logger.error(f"TripID {trip_id}: Response content: {response.content[:200]}")
            
            return {
                "errors": state.get("errors", []) + [f"Failed to parse requirements: {str(e)}"],
                "requires_user_input": True,
                "current_step": "extract_requirements",
                "messages": state.get("messages", []) + [
                    AIMessage(content="I'd love to help plan your trip! Could you tell me:\n"
                                    "- Where you'd like to go?\n"
                                    "- When (dates)?\n"
                                    "- Your approximate budget?")
                ]
            }
        except Exception as e:
            logger.error(f"TripID {trip_id}: Unexpected error in requirement extraction - {e}")
            return {
                "errors": state.get("errors", []) + [f"Orchestrator error: {str(e)}"],
                "requires_user_input": True,
                "current_step": "extract_requirements"
            }
    
    def _generate_clarification(self, missing_fields: list[str]) -> str:
        """
        Generate a natural, friendly clarification question for missing requirements.
        """
        field_questions = {
            "destination": "Where would you like to travel to?",
            "start_date": "When would you like to start your trip?",
            "end_date": "When would you like to return?",
            "budget_total": "What's your approximate budget for this trip?"
        }
        
        questions = [field_questions.get(field, f"What about {field}?") 
                     for field in missing_fields if field in field_questions]
        
        if not questions:
            return "I need a bit more information to plan your perfect trip. Could you provide more details?"
        
        if len(questions) == 1:
            return f"I'd love to help plan your trip! {questions[0]}"
        else:
            return (f"I'd love to help plan your trip! To get started, I need to know:\n\n" + 
                   "\n".join(f"• {q}" for q in questions))
    
    def route_next_step(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Determine workflow progression after an agent completes.
        
        With sub-graph agents, this is simpler - we just check current_step
        since each agent updates it upon completion.
        """
        current_step = state.get("current_step")
        trip_id = state.get('trip_id', 'unknown')
        
        logger.info(f"TripID {trip_id}: Routing from step: {current_step}")
        
        # Workflow state machine
        # Each specialized agent updates current_step when it finishes
        workflow_transitions = {
            "extract_requirements": "research",
            "research": "budget_planning",
            "budget_planning": "booking_flights",
            "booking_flights": "booking_hotels",
            "booking_hotels": "create_itinerary",
            "create_itinerary": "review",
            "review": "complete"
        }
        
        next_step = workflow_transitions.get(current_step, "complete")
        
        logger.info(f"TripID {trip_id}: Next step: {next_step}")
        
        return {
            "current_step": next_step,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def should_continue_workflow(self, state: Dict[str, Any]) -> bool:
        """
        Determine if workflow should continue or end.
        Used by conditional edges in the main graph.
        """
        is_complete = state.get("current_step") == "complete"
        needs_user_input = state.get("requires_user_input", False)
        has_errors = len(state.get("errors", [])) > 0
        retry_limit = state.get("retry_count", 0) >= 3
        
        should_continue = not (is_complete or needs_user_input or has_errors or retry_limit)
        
        logger.info(f"Workflow continue check: {should_continue} "
                   f"(complete={is_complete}, needs_input={needs_user_input}, "
                   f"errors={has_errors}, retry_limit={retry_limit})")
        
        return should_continue


# === Node function for LangGraph integration ===

def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node wrapper for orchestrator agent.
    Called by the main graph executor.
    
    This handles the initial requirement extraction phase.
    After that, specialized agents take over and update current_step themselves.
    """
    trip_id = state.get('trip_id', 'unknown')
    current_step = state.get('current_step')
    
    logger.info(f"TripID {trip_id}: Orchestrator node invoked at step: {current_step}")
    
    orchestrator = OrchestratorAgent()
    
    # If we're in the extraction phase, extract requirements
    if current_step == "extract_requirements":
        logger.info(f"TripID {trip_id}: Extracting requirements")
        updates = orchestrator.extract_requirements(state)
        return updates
    
    # Otherwise, just route to next step
    # (This shouldn't normally be called since agents update current_step directly)
    logger.info(f"TripID {trip_id}: Routing to next step")
    return orchestrator.route_next_step(state)


# === Routing helper for main graph ===

def route_after_orchestrator(state: Dict[str, Any]) -> str:
    """
    Conditional routing function for the main graph.
    Determines where to go after orchestrator extracts requirements.
    """
    trip_id = state.get('trip_id', 'unknown')
    
    # Check if we need user input (missing requirements)
    if not state.get("requirements_extracted", False):
        logger.info(f"TripID {trip_id}: Route -> END (need user input)")
        return "ask_user"
    
    # Check if there are errors
    if state.get("errors"):
        logger.info(f"TripID {trip_id}: Route -> END (errors present)")
        return "error"
    
    # Requirements are complete, proceed to research
    logger.info(f"TripID {trip_id}: Route -> research_agent")
    return "research"


if __name__ == "__main__":
    """Test orchestrator standalone"""
    
    print("Testing Orchestrator Agent...")
    
    # Test case 1: Complete request
    test_state = {
        "messages": [("user", "Plan 5 days in Kyoto, Japan from June 1-5, 2025. Budget $3000 for 2 people, interested in temples and food.")],
        "trip_id": "test-123",
        "current_step": "extract_requirements",
        "requirements_extracted": False,
        "errors": []
    }
    
    orchestrator = OrchestratorAgent()
    result = orchestrator.extract_requirements(test_state)
    
    print("\n=== Test Results ===")
    print(f"Destination: {result.get('destination')}")
    print(f"Dates: {result.get('start_date')} to {result.get('end_date')}")
    print(f"Budget: ${result.get('budget_total')}")
    print(f"Travelers: {result.get('num_travelers')}")
    print(f"Interests: {result.get('interests')}")
    print(f"Requirements Complete: {result.get('requirements_extracted')}")
    print(f"Next Step: {result.get('current_step')}")
    
    # Test case 2: Incomplete request
    test_state_2 = {
        "messages": [("user", "I want to go to Paris sometime")],
        "trip_id": "test-456",
        "current_step": "extract_requirements",
        "requirements_extracted": False,
        "errors": []
    }
    
    result_2 = orchestrator.extract_requirements(test_state_2)
    print("\n=== Incomplete Request Test ===")
    print(f"Missing: {result_2.get('missing_requirements')}")
    print(f"Needs Input: {result_2.get('requires_user_input')}")
    print(f"Clarification: {result_2.get('messages')[-1].content if result_2.get('messages') else 'None'}")