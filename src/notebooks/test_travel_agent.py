"""
Notebook: Testing Travel Planning Multi-Agent System
Run this to test the complete workflow end-to-end
"""

# Cell 1: Setup and Imports
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

print("✅ Environment configured")

# Cell 2: Import our modules
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflows.state import TravelPlanningState, create_initial_state
from workflows.travel_planning_graph import create_travel_planning_graph

print("✅ Modules imported")

# Cell 3: Create the graph
app = create_travel_planning_graph()
print("✅ Graph created and compiled")

# Cell 4: Test Case 1 - Complete trip request
print("=" * 60)
print("TEST CASE 1: Complete Trip Request")
print("=" * 60)

user_message = """
I want to plan a 5-day trip to Tokyo, Japan in June 2025. 
We're 2 people with a budget of around $3,000. 
We're interested in culture, food, and some historical sites.
"""

# Create initial state
initial_state = create_initial_state(user_message)

# Thread config for conversation persistence
config = {"configurable": {"thread_id": "test_trip_1"}}

# Run the workflow
print(f"\n💬 User: {user_message}\n")
print("🤖 Processing with multi-agent system...\n")

result = app.invoke(initial_state, config)

# Print all assistant messages
print("\n📝 Agent Responses:")
print("-" * 60)
for msg in result["messages"]:
    if hasattr(msg, 'type') and msg.type == "assistant":
        print(f"\n{msg.content}\n")
    elif isinstance(msg, tuple) and msg[0] == "assistant":
        print(f"\n{msg[1]}\n")

# Print state summary
print("\n📊 Final State Summary:")
print("-" * 60)
print(f"Trip ID: {result.get('trip_id')}")
print(f"Destination: {result.get('destination')}")
print(f"Dates: {result.get('start_date')} to {result.get('end_date')}")
print(f"Budget: ${result.get('budget_total')}")
print(f"Travelers: {result.get('num_travelers')}")
print(f"Interests: {', '.join(result.get('interests', []))}")
print(f"Current Step: {result.get('current_step')}")
print(f"Requirements Extracted: {result.get('requirements_extracted')}")

# Cell 5: Test Case 2 - Incomplete request (missing info)
print("\n" + "=" * 60)
print("TEST CASE 2: Incomplete Trip Request")
print("=" * 60)

incomplete_message = "I want to go to Paris sometime next year"

initial_state_2 = create_initial_state(incomplete_message)
config_2 = {"configurable": {"thread_id": "test_trip_2"}}

print(f"\n💬 User: {incomplete_message}\n")
print("🤖 Processing...\n")

result_2 = app.invoke(initial_state_2, config_2)

print("\n📝 Agent Response:")
print("-" * 60)
for msg in result_2["messages"]:
    if hasattr(msg, 'type') and msg.type == "assistant":
        print(f"\n{msg.content}\n")
    elif isinstance(msg, tuple) and msg[0] == "assistant":
        print(f"\n{msg[1]}\n")

print(f"\nMissing Requirements: {result_2.get('missing_requirements')}")
print(f"Requires User Input: {result_2.get('requires_user_input')}")

# Cell 6: Inspect Graph Structure
print("\n" + "=" * 60)
print("GRAPH STRUCTURE ANALYSIS")
print("=" * 60)

def safe_get_graph_info():
    """Safely get graph information without triggering visualization errors"""
    try:
        graph = app.get_graph()
        return {
            'nodes': list(graph.nodes.keys()),
            'edges': list(graph.edges)
        }
    except Exception as e:
        print(f"Error getting graph structure: {str(e)}")
        return {
            'nodes': ['orchestrator', 'research', 'budget', 'booking', 'itinerary', 'review'],
            'edges': [('orchestrator', 'research'), ('research', 'budget'), 
                     ('budget', 'booking'), ('booking', 'itinerary'), 
                     ('itinerary', 'review'), ('review', 'END')]
        }

# Get graph info safely
graph_info = safe_get_graph_info()

# Print graph structure
print("\nGraph nodes:", graph_info['nodes'])
print("Graph edges:", graph_info['edges'])

# Only attempt visualization if in a notebook environment
try:
    from IPython import get_ipython
    if 'IPKernelApp' in get_ipython().config:
        print("\nRunning in notebook environment - attempting to display graph...")
        try:
            from IPython.display import display, Markdown
            
            # Create a simple Mermaid diagram
            mermaid_code = """
            ```mermaid
            graph TD
                A[Orchestrator] -->|research| B[Research]
                B --> C[Budget]
                C --> D[Booking]
                D --> E[Itinerary]
                E --> F[Review]
                F -->|complete| G[End]
                A -->|ask_user| H[User Input]
                H -->|submit| A
            ```
            """
            display(Markdown(mermaid_code))
            
        except Exception as e:
            print(f"\nCould not display graph: {str(e)}")
            print("\nTo visualize the graph, install the required packages:")
            print("pip install ipython graphviz")
            
except Exception as e:
    # Not in a notebook, no need to visualize
    pass

# Cell 7: Test Individual Agents
print("\n" + "=" * 60)
print("TESTING INDIVIDUAL AGENTS")
print("=" * 60)

from agents.orchestrator_agent import OrchestratorAgent
from agents.research_agent import ResearchAgent

# Test Orchestrator requirement extraction
print("\n1. Testing Orchestrator - Requirement Extraction")
orchestrator = OrchestratorAgent()

test_state = {
    "messages": [("user", "5 days in Tokyo, June 2025, $3000 budget, 2 people, love food and culture")],
    "requirements_extracted": False,
    "errors": []
}

extracted = orchestrator.extract_requirements(test_state)
print(f"   Destination: {extracted.get('destination')}")
print(f"   Dates: {extracted.get('start_date')} to {extracted.get('end_date')}")
print(f"   Budget: ${extracted.get('budget_total')}")
print(f"   Interests: {extracted.get('interests')}")
print(f"   Complete: {extracted.get('requirements_extracted')}")

# Test Research Agent
print("\n2. Testing Research Agent")
research_agent = ResearchAgent()

test_research_state = {
    "destination": "Tokyo, Japan",
    "start_date": "2025-06-01",
    "end_date": "2025-06-05",
    "interests": ["culture", "food"],
    "errors": []
}

print("   Researching destination... (this may take 30-60 seconds)")
research_results = research_agent.research_destination(test_research_state)

print(f"   Attractions found: {research_results['research_results']['attractions']['count']}")
print(f"   Weather: {research_results['research_results']['weather'].get('conditions', 'N/A')}")
print(f"   Local tips available: {len(research_results['research_results']['local_tips'].get('cultural_etiquette', []))}")

# Cell 8: Performance & Cost Analysis
print("\n" + "=" * 60)
print("PERFORMANCE ANALYSIS")
print("=" * 60)

import time

start_time = time.time()

# Run a quick workflow
quick_test = create_initial_state("Quick trip to London, 3 days, $2000")
quick_config = {"configurable": {"thread_id": "perf_test"}}
quick_result = app.invoke(quick_test, quick_config)

end_time = time.time()
duration = end_time - start_time

print(f"⏱️  Total execution time: {duration:.2f} seconds")
print(f"🔢 Number of LLM calls: ~5-8 (orchestrator + research + budget + booking + itinerary)")
print(f"💰 Estimated cost per query: $0.01-0.03 (using GPT-4o-mini)")
print(f"📊 Messages in conversation: {len(quick_result.get('messages', []))}")

print("\n" + "=" * 60)
print("✅ ALL TESTS COMPLETED")
print("=" * 60)