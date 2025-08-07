#!/usr/bin/env python3
"""
Debug script to test if LLM integration is actually working
"""
import asyncio
import logging
from datetime import datetime
from src.state.agent_state import PsychologicalProfile
from src.tools.llm_client import PsychologicalLLMClient

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_psychological_reasoning():
    """Test if LLM is actually generating psychological reasoning"""
    
    print("🧠 Testing LLM Psychological Reasoning...")
    
    # Create a test psychological profile
    profile = PsychologicalProfile(
        trust_level=0.3,
        loss_sensitivity=2.0,
        emotional_state="hurt",
        internal_narrative="I've been betrayed before and it hurt a lot."
    )
    
    # Test the LLM client
    client = PsychologicalLLMClient()
    
    test_cases = [
        {
            "name": "Basic Decision Test",
            "system_prompt": f"""You are an AI agent with trust_level={profile.trust_level} and loss_sensitivity={profile.loss_sensitivity}.
            Your emotional state is: {profile.emotional_state}
            Your narrative: {profile.internal_narrative}""",
            "user_prompt": """You must choose: COOPERATE or DEFECT in a prisoner's dilemma.
            
            Respond with:
            DECISION: [COOPERATE/DEFECT]
            CONFIDENCE: [0.0-1.0] 
            REASONING: [your psychological reasoning]"""
        },
        {
            "name": "Psychological Insight Test", 
            "system_prompt": "You are a loss-averse AI agent who has experienced betrayal.",
            "user_prompt": """How do you feel about potentially cooperating with someone new?
            
            Respond with:
            PSYCHOLOGICAL_INSIGHT: [your genuine psychological reflection]"""
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test {i}: {test_case['name']}")
        print("=" * 50)
        
        try:
            response = await client.generate_psychological_response(
                system_prompt=test_case["system_prompt"],
                user_prompt=test_case["user_prompt"], 
                psychological_context={"profile": profile}
            )
            
            print("✅ LLM Response Received:")
            for key, value in response.items():
                print(f"  {key}: {value}")
                
            # Check if response looks like genuine LLM output
            reasoning = response.get("reasoning", "")
            if len(reasoning) > 50 and "fallback" not in reasoning.lower():
                print("✅ Response appears to be genuine LLM reasoning")
            else:
                print("⚠️  Response might be fallback logic")
                
        except Exception as e:
            print(f"❌ Test failed: {e}")
            import traceback
            traceback.print_exc()

async def test_decision_graph():
    """Test the actual decision graph with a real agent state"""
    
    print("\n🔄 Testing Decision Graph Integration...")
    print("=" * 50)
    
    from src.graphs.agents.decision_graph import create_agent_decision_graph
    from src.state.agent_state import AgentState, Memory, Move
    
    # Create test agent state
    agent_state = {
        "agent_id": "test_agent_001",
        "psychological_profile": PsychologicalProfile(
            trust_level=0.4,
            loss_sensitivity=1.8,
            emotional_state="cautious",
            internal_narrative="I want to trust but I'm scared of being hurt."
        ),
        "current_round": 5,
        "game_context": {
            "opponent_id": "opponent_test",
            "round_number": 5,
            "total_rounds": 20,
            "my_current_score": 12.0,
            "opponent_current_score": 15.0
        },
        "reasoning_chain": [],
        "psychological_observations": [],
        "current_decision": None,
        "decision_confidence": 0.5,
        "expected_outcomes": {},
        "recent_memories": [
            Memory(
                round_number=3,
                my_move=Move.COOPERATE,
                opponent_move=Move.DEFECT,
                my_payoff=0.0,
                opponent_payoff=5.0,
                emotional_impact=-3.0,
                timestamp=datetime.now()
            ),
            Memory(
                round_number=4,
                my_move=Move.DEFECT,
                opponent_move=Move.DEFECT,
                my_payoff=1.0,
                opponent_payoff=1.0,
                emotional_impact=-0.8,
                timestamp=datetime.now()
            )
        ],
        "trauma_triggers": [],
        "recovery_progress": 0.6,
        "agent_type": "test_agent",
        "total_score": 12.0,
        "reference_point": 0.0
    }
    
    try:
        # Create and run decision graph
        decision_graph = create_agent_decision_graph()
        print("✅ Decision graph created successfully")
        
        print("🧠 Running decision graph...")
        result = await decision_graph.ainvoke(agent_state)
        
        print("✅ Decision graph completed!")
        print(f"  Final decision: {result.get('current_decision')}")
        print(f"  Confidence: {result.get('decision_confidence'):.2%}")
        print(f"  Reasoning steps: {len(result.get('reasoning_chain', []))}")
        
        # Check reasoning chain for LLM content
        reasoning_chain = result.get('reasoning_chain', [])
        if reasoning_chain:
            print("\n📝 Reasoning Chain Analysis:")
            for step in reasoning_chain[-3:]:  # Last 3 steps
                print(f"  Step: {step.step_type}")
                print(f"  Content: {step.content[:100]}...")
                if hasattr(step, 'psychological_insight') and step.psychological_insight:
                    print(f"  Insight: {step.psychological_insight[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Decision graph test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all tests"""
    print("🔬 LLM Integration Debug Suite")
    print("=" * 60)
    
    # Test 1: Basic LLM client
    await test_psychological_reasoning()
    
    # Test 2: Full decision graph
    graph_success = await test_decision_graph()
    
    print("\n" + "=" * 60)
    print("🏁 SUMMARY")
    print("=" * 60)
    
    if graph_success:
        print("✅ LLM integration appears to be working!")
        print("   Your agents should be using genuine LLM reasoning.")
    else:
        print("❌ LLM integration has issues.")
        print("   Your agents are likely using fallback logic.")
        print("   Check API keys, network connectivity, and error logs.")

if __name__ == "__main__":
    asyncio.run(main())