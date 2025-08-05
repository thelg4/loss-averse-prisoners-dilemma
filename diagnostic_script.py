#!/usr/bin/env python3
"""
Diagnostic script to identify what's broken in the loss-averse prisoner's dilemma system
"""

import asyncio
import os
import logging
from pathlib import Path
import sys
from dotenv import load_dotenv
    
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_llm_integration():
    """Test if LLM integration is working"""
    print("🧠 Testing LLM Integration...")
    
    try:
        from src.tools.llm_client import PsychologicalLLMClient
        
        # Check API key
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"API_KEY: {api_key}")
        if not api_key:
            print("❌ OPENAI_API_KEY not found in environment")
            return False
        else:
            print(f"✅ API key found: {api_key[:10]}...")
        
        # Test simple LLM call
        client = PsychologicalLLMClient()
        response = await client.generate_psychological_response(
            system_prompt="You are a test agent",
            user_prompt="Should you cooperate or defect? Respond with DECISION: COOPERATE or DEFECT",
            psychological_context={}
        )
        
        if response and response.get("decision") in ["COOPERATE", "DEFECT"]:
            print(f"✅ LLM integration working: {response['decision']}")
            return True
        else:
            print(f"❌ LLM response invalid: {response}")
            return False
            
    except Exception as e:
        print(f"❌ LLM integration failed: {e}")
        return False

def test_memory_system():
    """Test if memory system creates proper objects"""
    print("\n🧠 Testing Memory System...")
    
    try:
        from src.state.agent_state import Memory, Move
        from datetime import datetime
        
        # Create test memory
        memory = Memory(
            round_number=1,
            my_move=Move.COOPERATE,
            opponent_move=Move.DEFECT,
            my_payoff=0.0,
            opponent_payoff=5.0,
            emotional_impact=-2.0,
            timestamp=datetime.now()
        )
        
        # Test serialization
        memory_dict = memory.dict()
        
        if memory_dict and memory_dict.get("round_number") == 1:
            print("✅ Memory system working")
            return True
        else:
            print(f"❌ Memory serialization failed: {memory_dict}")
            return False
            
    except Exception as e:
        print(f"❌ Memory system failed: {e}")
        return False

def test_agent_state_system():
    """Test if agent state management works"""
    print("\n🤖 Testing Agent State System...")
    
    try:
        from src.state.agent_state import PsychologicalProfile
        
        # Create test profile
        profile = PsychologicalProfile(
            trust_level=0.7,
            loss_sensitivity=1.5,
            emotional_state="hopeful"
        )
        
        # Test trait identification
        trait = profile.get_dominant_trait()
        
        if trait and isinstance(trait, str):
            print(f"✅ Agent state system working: {trait}")
            return True
        else:
            print(f"❌ Agent state system failed: {trait}")
            return False
            
    except Exception as e:
        print(f"❌ Agent state system failed: {e}")
        return False

async def test_decision_graph():
    """Test if decision graph executes"""
    print("\n📊 Testing Decision Graph...")
    
    try:
        from src.graphs.agents.decision_graph import create_agent_decision_graph
        from src.state.agent_state import AgentState, PsychologicalProfile
        
        # Create test agent
        agent_state = {
            "agent_id": "test_agent",
            "psychological_profile": PsychologicalProfile(),
            "current_round": 1,
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
        
        # Create and test graph
        graph = create_agent_decision_graph()
        
        # This might fail if LLM is broken, but graph should at least create
        if graph:
            print("✅ Decision graph created successfully")
            return True
        else:
            print("❌ Decision graph creation failed")
            return False
            
    except Exception as e:
        print(f"❌ Decision graph failed: {e}")
        return False

def test_configuration():
    """Test if configuration is reasonable"""
    print("\n⚙️  Testing Configuration...")
    
    try:
        import yaml
        
        config_path = Path("config/longer_minimal.yaml")
        if not config_path.exists():
            print("❌ Configuration file not found")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        emergent = config["experiments"]["longer_minimal"]["emergent_experiment"]
        
        issues = []
        if emergent["population_size"] < 5:
            issues.append(f"Population too small: {emergent['population_size']} (need 10+)")
        
        if emergent["rounds_per_interaction"] < 50:
            issues.append(f"Rounds too few: {emergent['rounds_per_interaction']} (need 50+)")
        
        if emergent["generations"] < 10:
            issues.append(f"Generations too few: {emergent['generations']} (need 10+)")
        
        if issues:
            print("⚠️  Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
            return False
        else:
            print("✅ Configuration looks good")
            return True
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def suggest_fixes(results):
    """Suggest specific fixes based on test results"""
    print("\n🔧 RECOMMENDED FIXES:")
    print("=" * 50)
    
    if not results["llm"]:
        print("🚨 CRITICAL: Fix LLM Integration")
        print("   1. Check your .env file has OPENAI_API_KEY")
        print("   2. Verify API key is valid and has credits")
        print("   3. Test with: python -c 'from openai import OpenAI; print(OpenAI().models.list())'")
        print()
    
    if not results["memory"]:
        print("🚨 CRITICAL: Fix Memory System")
        print("   1. Check Pydantic version compatibility")
        print("   2. Verify Memory model serialization")
        print("   3. Test memory creation in isolation")
        print()
    
    if not results["agent_state"]:
        print("🚨 CRITICAL: Fix Agent State System")
        print("   1. Check PsychologicalProfile model")
        print("   2. Verify TypedDict vs Pydantic consistency")
        print("   3. Test agent state creation in isolation")
        print()
    
    if not results["decision_graph"]:
        print("⚠️  WARNING: Decision Graph Issues")
        print("   1. May be caused by LLM integration problems")
        print("   2. Check graph compilation and node connections")
        print("   3. Test with fallback decision logic")
        print()
    
    if not results["config"]:
        print("💡 SUGGESTION: Improve Configuration")
        print("   1. Increase population_size to 10-20")
        print("   2. Increase rounds_per_interaction to 50-100")
        print("   3. Increase generations to 10-50")
        print("   4. This will give more meaningful results")
        print()

async def main():
    """Run all diagnostic tests"""
    print("🔍 DIAGNOSING LOSS-AVERSE PRISONER'S DILEMMA SYSTEM")
    print("=" * 60)
    
    results = {
        "llm": await test_llm_integration(),
        "memory": test_memory_system(),
        "agent_state": test_agent_state_system(),
        "decision_graph": await test_decision_graph(),
        "config": test_configuration()
    }
    
    print("\n📊 DIAGNOSTIC SUMMARY:")
    print("=" * 30)
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper():15} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All systems working! Try scaling up your configuration.")
    else:
        suggest_fixes(results)
    
    return results

if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise
    
    # Run diagnostics
    asyncio.run(main())