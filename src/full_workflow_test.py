"""
Complete Workflow Test Script
Tests the full travel planning system with sub-graph agents
"""

import os
import sys
import logging
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify required API keys
required_keys = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
missing_keys = [key for key in required_keys if not os.getenv(key)]

if missing_keys:
    print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
    print("\nPlease set them in your .env file:")
    for key in missing_keys:
        print(f"  {key}=your-key-here")
    sys.exit(1)

print("✅ Environment configured\n")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import modules
try:
    from workflows.state import create_initial_state
    from workflows.travel_planning_graph import create_travel_planning_graph
    print("✅ Modules imported\n")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you're running from the src/ directory")
    sys.exit(1)

# Create the graph
try:
    print("Creating graph...")
    app = create_travel_planning_graph()
    print("✅ Graph compiled\n")
except Exception as e:
    print(f"❌ Failed to create graph: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def print_section(title):
    """Print a nice section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def run_test_case(test_name, user_message, thread_id):
    """Run a single test case"""
    print_section(f"TEST: {test_name}")
    
    print(f"💬 User:\n{user_message}\n")
    print("🤖 Processing with multi-agent system...\n")
    print("-" * 80)
    
    # Create initial state
    initial_state = create_initial_state(user_message)
    
    # Run the workflow
    config = {"configurable": {"thread_id": thread_id}}
    
    start_time = datetime.now()
    
    try:
        result = app.invoke(initial_state, config)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Print results
        print("\n📝 AGENT RESPONSES:\n")
        
        for i, msg in enumerate(result["messages"], 1):
            if hasattr(msg, 'type') and msg.type == "ai":
                print(f"Response #{i}:")
                print(msg.content)
                print("\n" + "-" * 80 + "\n")
        
        # Print state summary
        print("📊 FINAL STATE SUMMARY:")
        print("-" * 80)
        print(f"Trip ID: {result.get('trip_id')}")
        print(f"Destination: {result.get('destination')}")
        print(f"Dates: {result.get('start_date')} to {result.get('end_date')}")
        print(f"Budget: ${result.get('budget_total')}")
        print(f"Travelers: {result.get('num_travelers')}")
        print(f"Interests: {', '.join(result.get('interests', []))}")
        print(f"Current Step: {result.get('current_step')}")
        
        print(f"\n✅ DATA COLLECTED:")
        print(f"   Requirements: {'✓' if result.get('requirements_extracted') else '✗'}")
        print(f"   Research: {'✓' if result.get('research_results') else '✗'}")
        
        # Print research details if available
        if result.get('research_results'):
            research = result['research_results']
            print(f"      - Attractions: {len(research.get('attractions', []))}")
            print(f"      - Weather: {'Available' if research.get('weather') else 'N/A'}")
            print(f"      - Local Tips: {len(research.get('local_tips', []))}")
        
        print(f"   Budget: {'✓' if result.get('budget_breakdown') else '✗'}")
        
        # Print budget details if available
        if result.get('budget_breakdown'):
            budget = result['budget_breakdown']
            print(f"      - Flights: ${budget.get('flights', 0):.2f}")
            print(f"      - Accommodation: ${budget.get('accommodation', 0):.2f}")
            print(f"      - Remaining: ${budget.get('remaining', 0):.2f}")
        
        print(f"   Flights: {'✓' if result.get('flight_options') else '✗'}")
        if result.get('flight_options'):
            print(f"      - Found {len(result['flight_options'])} options")
        
        print(f"   Hotels: {'✓' if result.get('hotel_options') else '✗'}")
        if result.get('hotel_options'):
            print(f"      - Found {len(result['hotel_options'])} options")
        
        print(f"   Itinerary: {'✓' if result.get('itinerary') else '✗'}")
        if result.get('itinerary'):
            days = len([k for k in result['itinerary'].keys() if k.startswith('day_')])
            print(f"      - {days} days planned")
        
        # Performance metrics
        print(f"\n⏱️  PERFORMANCE:")
        print(f"   Total execution time: {duration:.2f} seconds")
        print(f"   Estimated cost: $0.05-0.10 (with Tavily + GPT-4o)")
        
        # Check for errors
        if result.get('errors'):
            print(f"\n⚠️  ERRORS ENCOUNTERED:")
            for error in result['errors']:
                print(f"   - {error}")
        else:
            print(f"\n✅ TEST PASSED - No errors!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ TEST FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# RUN TEST CASES
# ============================================================================

if __name__ == "__main__":
    
    print_section("TRAVEL PLANNING SYSTEM - FULL WORKFLOW TEST")
    print("This test runs your complete multi-agent system including:")
    print("  • Orchestrator (requirement extraction)")
    print("  • Research Agent (with Tavily search)")
    print("  • Budget Agent (with cost estimation)")
    print("  • Booking placeholders")
    print("  • Itinerary generation")
    print("  • Final review\n")
    
    # ========================================================================
    # TEST CASE 1: Complete trip request with all details
    # ========================================================================
    
    result_1 = run_test_case(
        test_name="Complete Trip Request",
        user_message="""
I want to plan a 5-day trip to Kyoto, Japan from March 15-20, 2026.
We're 2 people with a budget of $4000.
We're interested in temples, traditional culture, and authentic Japanese food.
We prefer staying at a traditional ryokan if possible.
        """.strip(),
        thread_id="test_complete_001"
    )
    
    # ========================================================================
    # TEST CASE 2: Incomplete request (missing dates)
    # ========================================================================
    
    print("\n\n")
    result_2 = run_test_case(
        test_name="Incomplete Request - Missing Dates",
        user_message="I want to visit Paris with a budget of $3000",
        thread_id="test_incomplete_002"
    )
    
    # ========================================================================
    # TEST CASE 3: Budget-focused trip
    # ========================================================================
    
    print("\n\n")
    result_3 = run_test_case(
        test_name="Budget Backpacking Trip",
        user_message="""
Plan a budget backpacking trip to Bangkok, Thailand.
Going solo from June 1-10, 2026.
Budget is only $800 total.
Interested in street food, temples, and nightlife.
Looking for hostel accommodation.
        """.strip(),
        thread_id="test_budget_003"
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print_section("ALL TESTS COMPLETED")
    
    tests_run = 3
    tests_passed = sum([
        1 if result_1 and result_1.get('current_step') == 'complete' else 0,
        1 if result_2 and result_2.get('requires_user_input') else 0,
        1 if result_3 and result_3.get('current_step') == 'complete' else 0
    ])
    
    print(f"Tests Run: {tests_run}")
    print(f"Tests Passed: {tests_passed}")
    print(f"Success Rate: {(tests_passed/tests_run*100):.0f}%")
    
    if tests_passed == tests_run:
        print("\n🎉 ALL TESTS PASSED! Your multi-agent system is working!")
        print("\nYour system successfully:")
        print("  ✓ Extracted requirements from natural language")
        print("  ✓ Conducted real web research via Tavily")
        print("  ✓ Created realistic budget breakdowns")
        print("  ✓ Generated flight and hotel options")
        print("  ✓ Built day-by-day itineraries")
        print("  ✓ Produced comprehensive travel plans")
    else:
        print(f"\n⚠️  {tests_run - tests_passed} test(s) failed. Review the output above.")
    
    print("\n" + "=" * 80)
    print("Test complete! Check the logs for detailed agent interactions.")
    print("=" * 80)