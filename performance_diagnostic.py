#!/usr/bin/env python3
"""
Performance diagnostic to find the real bottleneck in the experiment
"""

import time
import asyncio
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_llm_speed():
    """Test actual LLM response time"""
    print("🧠 TESTING LLM SPEED")
    print("=" * 40)
    
    try:
        from src.tools.llm_client import PsychologicalLLMClient
        
        client = PsychologicalLLMClient(provider="openai", model="gpt-4o-mini")
        
        # Test simple decision
        start_time = time.time()
        
        response = await client.generate_psychological_response(
            system_prompt="You are a test agent. Be brief.",
            user_prompt="Should you COOPERATE or DEFECT? Respond with: DECISION: COOPERATE",
            psychological_context={}
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ LLM Response Time: {duration:.2f} seconds")
        print(f"✅ Response: {response.get('decision', 'No decision')}")
        
        if duration > 5:
            print("🚨 WARNING: LLM is unusually slow!")
        elif duration > 2:
            print("⚠️  LLM is slower than expected")
        else:
            print("✅ LLM speed looks good")
            
        return duration
        
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
        return None

async def test_decision_graph_speed():
    """Test how long the full decision graph takes"""
    print("\n📊 TESTING DECISION GRAPH SPEED")
    print("=" * 40)
    
    try:
        from src.graphs.agents.decision_graph import create_agent_decision_graph
        from src.state.agent_state import PsychologicalProfile
        
        # Create test agent
        test_agent = {
            "agent_id": "speed_test_agent",
            "psychological_profile": PsychologicalProfile(),
            "current_round": 1,
            "game_context": {"opponent_id": "test_opponent"},
            "reasoning_chain": [],
            "psychological_observations": [],
            "current_decision": None,
            "decision_confidence": 0.5,
            "expected_outcomes": {},
            "recent_memories": [],
            "trauma_triggers": [],
            "recovery_progress": 1.0,
            "agent_type": "test",
            "total_score": 0.0,
            "reference_point": 0.0
        }
        
        graph = create_agent_decision_graph()
        
        start_time = time.time()
        result = await graph.ainvoke(test_agent)
        end_time = time.time()
        
        duration = end_time - start_time
        
        print(f"✅ Decision Graph Time: {duration:.2f} seconds")
        print(f"✅ Decision Made: {result.get('current_decision', 'None')}")
        print(f"✅ Reasoning Steps: {len(result.get('reasoning_chain', []))}")
        
        if duration > 10:
            print("🚨 WARNING: Decision graph is very slow!")
        elif duration > 5:
            print("⚠️  Decision graph is slower than expected")
        else:
            print("✅ Decision graph speed looks reasonable")
            
        return duration
        
    except Exception as e:
        print(f"❌ Decision graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def test_tournament_speed():
    """Test how long a minimal tournament takes"""
    print("\n🏆 TESTING TOURNAMENT SPEED")
    print("=" * 40)
    
    try:
        from src.game.prisoner_dilemma import PrisonersDilemmaGame
        from src.state.agent_state import PsychologicalProfile
        
        # Create two simple agents
        agent1 = {
            "agent_id": "test_agent_1",
            "psychological_profile": PsychologicalProfile(),
            "current_round": 0,
            "game_context": {},
            "reasoning_chain": [],
            "psychological_observations": [],
            "current_decision": None,
            "decision_confidence": 0.5,
            "expected_outcomes": {},
            "recent_memories": [],
            "trauma_triggers": [],
            "recovery_progress": 1.0,
            "agent_type": "test",
            "total_score": 0.0,
            "reference_point": 0.0
        }
        
        agent2 = {
            "agent_id": "test_agent_2",
            "psychological_profile": PsychologicalProfile(),
            "current_round": 0,
            "game_context": {},
            "reasoning_chain": [],
            "psychological_observations": [],
            "current_decision": None,
            "decision_confidence": 0.5,
            "expected_outcomes": {},
            "recent_memories": [],
            "trauma_triggers": [],
            "recovery_progress": 1.0,
            "agent_type": "test",
            "total_score": 0.0,
            "reference_point": 0.0
        }
        
        game = PrisonersDilemmaGame()
        
        # Test 3 rounds (same as ultra_minimal)
        start_time = time.time()
        result = await game.play_tournament(agent1, agent2, 3, "speed_test")
        end_time = time.time()
        
        duration = end_time - start_time
        llm_calls = result.get("llm_calls_made", 0)
        
        print(f"✅ Tournament Time: {duration:.2f} seconds")
        print(f"✅ LLM Calls Made: {llm_calls}")
        print(f"✅ Time per LLM call: {duration/llm_calls:.2f} seconds" if llm_calls > 0 else "No LLM calls")
        
        if duration > 30:
            print("🚨 WARNING: Tournament is very slow!")
        elif duration > 15:
            print("⚠️  Tournament is slower than expected")
        else:
            print("✅ Tournament speed looks reasonable")
            
        return duration, llm_calls
        
    except Exception as e:
        print(f"❌ Tournament test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_config_complexity():
    """Analyze what the ultra_minimal config actually does"""
    print("\n⚙️  ANALYZING ULTRA_MINIMAL CONFIG")
    print("=" * 40)
    
    try:
        import yaml
        config_path = Path("config/ultra_minimal.yaml")
        
        if not config_path.exists():
            print("❌ ultra_minimal.yaml not found")
            return
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        emergent = config["experiments"]["llm_test"]["emergent_experiment"]
        baseline = config["experiments"]["llm_test"]["baseline_experiment"]
        
        print("Emergent Experiment:")
        print(f"  Population: {emergent['population_size']} agents")
        print(f"  Generations: {emergent['generations']}")
        print(f"  Interactions/gen: {emergent['interactions_per_generation']}")
        print(f"  Rounds/interaction: {emergent['rounds_per_interaction']}")
        
        print("\nBaseline Experiment:")
        print(f"  Agents: {len(baseline['agents'])}")
        print(f"  Rounds/match: {baseline['tournaments']['rounds_per_match']}")
        print(f"  Replications: {baseline['tournaments']['replications']}")
        
        # Calculate total expected operations
        baseline_rounds = baseline['tournaments']['rounds_per_match'] * baseline['tournaments']['replications']
        emergent_rounds = (emergent['population_size'] * emergent['generations'] * 
                          emergent['interactions_per_generation'] * emergent['rounds_per_interaction'])
        
        total_rounds = baseline_rounds + emergent_rounds
        
        print(f"\nCalculated totals:")
        print(f"  Baseline rounds: {baseline_rounds}")
        print(f"  Emergent rounds: {emergent_rounds}")
        print(f"  Total rounds: {total_rounds}")
        
        # Estimate LLM calls (4 per round per agent, roughly)
        estimated_llm_calls = total_rounds * 4
        print(f"  Estimated LLM calls: {estimated_llm_calls}")
        print(f"  Estimated time @ 1s/call: {estimated_llm_calls} seconds")
        
        if estimated_llm_calls > 100:
            print("🚨 This config is NOT ultra minimal!")
        
    except Exception as e:
        print(f"❌ Config analysis failed: {e}")

async def test_fallback_speed():
    """Test if fallback logic is being used instead of LLM"""
    print("\n🔄 TESTING FALLBACK LOGIC")
    print("=" * 40)
    
    try:
        from src.game.prisoner_dilemma import PrisonersDilemmaGame
        
        game = PrisonersDilemmaGame()
        
        if not game.graphs_available:
            print("✅ Using fallback decision logic (no LLM)")
            print("   This should be VERY fast")
        else:
            print("⚠️  Using LLM-based decision graphs")
            print("   This will be slower")
        
        print(f"Graph availability: {game.graphs_available}")
        print(f"LLM call count: {game.llm_call_count}")
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")

async def run_comprehensive_performance_test():
    """Run all performance tests"""
    print("🚀 COMPREHENSIVE PERFORMANCE DIAGNOSTIC")
    print("=" * 60)
    
    # Test individual components
    llm_time = await test_llm_speed()
    decision_time = await test_decision_graph_speed()
    tournament_time, llm_calls = await test_tournament_speed()
    
    # Analyze config
    analyze_config_complexity()
    
    # Check fallback usage
    await test_fallback_speed()
    
    # Summary
    print("\n📊 PERFORMANCE SUMMARY")
    print("=" * 40)
    
    if llm_time and llm_time > 3:
        print("🚨 ISSUE: LLM calls are very slow")
        print("   Check your internet connection and OpenAI API status")
    
    if decision_time and decision_time > 8:
        print("🚨 ISSUE: Decision graph is very slow")
        print("   This suggests multiple slow LLM calls per decision")
    
    if tournament_time and llm_calls:
        avg_time_per_call = tournament_time / llm_calls if llm_calls > 0 else 0
        if avg_time_per_call > 2:
            print(f"🚨 ISSUE: Average {avg_time_per_call:.2f}s per LLM call is too slow")
    
    print("\n🎯 RECOMMENDATIONS:")
    if llm_time and llm_time > 2:
        print("1. Check internet connection")
        print("2. Verify OpenAI API key and quota")
        print("3. Try running during off-peak hours")
    
    print("4. Consider using fallback mode for testing:")
    print("   - Temporarily disable LLM graphs to test pure speed")
    print("5. Monitor system resources (CPU, memory)")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_performance_test())