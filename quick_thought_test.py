#!/usr/bin/env python3
"""
Quick test script to see thought logging in action
"""
import asyncio
from src.utils.thought_logger import ThoughtIntegratedPrisonersDilemmaGame
from src.state.agent_state import AgentState, PsychologicalProfile
from datetime import datetime

async def quick_thought_test():
    """Run a super quick 2-agent, 3-round test with thought logging"""
    
    print("🧠 QUICK THOUGHT LOGGING TEST")
    print("=" * 50)
    
    # Create thought-enabled game
    game = ThoughtIntegratedPrisonersDilemmaGame(
        experiment_id="quick_test_001",
        enable_thought_logging=True
    )
    
    # Create two agents with contrasting psychology
    optimist = {
        "agent_id": "alice_optimist",
        "psychological_profile": PsychologicalProfile(
            trust_level=0.8,
            loss_sensitivity=1.0,
            emotional_state="hopeful",
            internal_narrative="I believe people are generally good."
        ),
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
        "agent_type": "optimist",
        "total_score": 0.0,
        "reference_point": 0.0
    }
    
    skeptic = {
        "agent_id": "bob_skeptic",
        "psychological_profile": PsychologicalProfile(
            trust_level=0.3,
            loss_sensitivity=1.8,
            emotional_state="cautious",
            internal_narrative="I need to protect myself from being hurt."
        ),
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
        "agent_type": "skeptic",
        "total_score": 0.0,
        "reference_point": 0.0
    }
    
    print(f"🤖 {optimist['agent_id']}: Trust={optimist['psychological_profile'].trust_level:.2f}, Loss Sens={optimist['psychological_profile'].loss_sensitivity:.2f}")
    print(f"🤖 {skeptic['agent_id']}: Trust={skeptic['psychological_profile'].trust_level:.2f}, Loss Sens={skeptic['psychological_profile'].loss_sensitivity:.2f}")
    print()
    
    # Run just 3 rounds with full thought logging
    result = await game.play_tournament_with_thoughts(
        optimist, skeptic, rounds=3, tournament_id="quick_test"
    )
    
    print("\n🏁 QUICK TEST COMPLETE!")
    print("=" * 50)
    
    # Show final states
    final_alice = result['agent1_final_state']['psychological_profile']
    final_bob = result['agent2_final_state']['psychological_profile']
    
    print(f"Final Scores: Alice={result['agent1_final_score']}, Bob={result['agent2_final_score']}")
    print(f"Alice: Trust={final_alice.trust_level:.3f}, Loss Sens={final_alice.loss_sensitivity:.3f}")
    print(f"       Narrative: \"{final_alice.internal_narrative}\"")
    print(f"Bob:   Trust={final_bob.trust_level:.3f}, Loss Sens={final_bob.loss_sensitivity:.3f}")
    print(f"       Narrative: \"{final_bob.internal_narrative}\"")
    
    print(f"\n📁 Logs saved to: experiment_logs/quick_test_001/")
    print("   Run: cat experiment_logs/quick_test_001/agent_decisions.jsonl | head -5")

if __name__ == "__main__":
    asyncio.run(quick_thought_test())