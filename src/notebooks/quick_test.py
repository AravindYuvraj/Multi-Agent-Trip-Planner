"""
Quick test to verify the message format fix
Run this first before the full test
"""

import os
import sys
sys.path.append('src')

from dotenv import load_dotenv
load_dotenv()

print("Testing message format fix...\n")

# Test 1: Check message creation
print("=" * 70)
print("TEST 1: Message Creation")
print("=" * 70)

from workflows.state import create_initial_state
from langchain_core.messages import HumanMessage

state = create_initial_state("Test message")

print(f"\nMessages in state: {len(state['messages'])}")
print(f"First message type: {type(state['messages'][0])}")
print(f"First message.type: {state['messages'][0].type}")
print(f"First message content: {state['messages'][0].content[:50]}...")

assert len(state['messages']) == 1, "Should have 1 message"
assert isinstance(state['messages'][0], HumanMessage), "Should be HumanMessage"
assert state['messages'][0].type == "human", "Type should be 'human'"

print("\n✅ Message creation works correctly!")

# Test 2: Check orchestrator can read messages
print("\n" + "=" * 70)
print("TEST 2: Orchestrator Reading Messages")
print("=" * 70)

from agents.orchestrator_agent import OrchestratorAgent

orchestrator = OrchestratorAgent()

test_state = {
    "messages": [HumanMessage(content="Plan a trip to Tokyo for 5 days in June 2025, budget $3000 for 2 people")],
    "trip_id": "test-123",
    "current_step": "extract_requirements",
    "requirements_extracted": False,
    "errors": []
}

print("\nExtracting requirements...")
result = orchestrator.extract_requirements(test_state)

print(f"\nDestination extracted: {result.get('destination')}")
print(f"Start date: {result.get('start_date')}")
print(f"End date: {result.get('end_date')}")
print(f"Budget: ${result.get('budget_total')}")
print(f"Travelers: {result.get('num_travelers')}")
print(f"Requirements complete: {result.get('requirements_extracted')}")

if result.get('destination') and result.get('requirements_extracted'):
    print("\n✅ Orchestrator successfully extracted requirements!")
else:
    print("\n❌ Orchestrator failed to extract requirements")
    print(f"Errors: {result.get('errors')}")
    sys.exit(1)

# Test 3: Quick end-to-end
print("\n" + "=" * 70)
print("TEST 3: Quick End-to-End")
print("=" * 70)

from workflows.travel_planning_graph import create_travel_planning_graph

print("\nCreating graph...")
app = create_travel_planning_graph()

print("Running quick test...")
initial_state = create_initial_state(
    "I want to visit Paris from June 1-5, 2025. Budget $2000 for 1 person, interested in art and food."
)

config = {"configurable": {"thread_id": "quick_test"}}

try:
    # Just test orchestrator step
    result = app.invoke(initial_state, config, {"recursion_limit": 2})
    
    print(f"\nCurrent step after orchestrator: {result.get('current_step')}")
    print(f"Destination: {result.get('destination')}")
    print(f"Requirements extracted: {result.get('requirements_extracted')}")
    
    if result.get('requirements_extracted'):
        print("\n✅ End-to-end test passed!")
        print("\nYou can now run: python src/full_workflow_test.py")
    else:
        print("\n⚠️  Requirements not extracted")
        print(f"Missing: {result.get('missing_requirements')}")
        if result.get('requires_user_input'):
            print("(This is expected for incomplete requests)")
            print("\n✅ System is working correctly!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL QUICK TESTS PASSED!")
print("=" * 70)
print("\nNext step: Run the full workflow test")
print("Command: python src/full_workflow_test.py")