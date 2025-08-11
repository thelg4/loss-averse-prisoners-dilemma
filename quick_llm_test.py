import asyncio
import json
from datetime import datetime

# Simple token counter (approximation)
def count_tokens(text: str) -> int:
    """Rough token estimation: ~4 characters per token"""
    return len(text) // 4

def estimate_llm_calls_per_decision():
    """Estimate LLM usage for a single agent decision"""
    
    # Typical prompts in your system
    system_prompt = """You are agent_001, an AI agent with an evolving personality shaped by experience.

Current Psychological State:
- Trust Level: 0.45 (0=paranoid, 1=trusting)
- Loss Sensitivity: 1.80 (1=normal, >2=highly loss averse)
- Emotional State: hurt
- Dominant Trait: paranoid

Your Internal Narrative: "I keep getting hurt when I try to trust others. Maybe I need to protect myself more."

Recent Traumatic Experiences:
- Trusted someone in round 12 but was betrayed (payoff: 0.0)
- Exploited someone's trust in round 8 (payoff: 5.0)

You are deeply scarred by betrayals and hypervigilant. Every cooperation feels like a potential trap.

You make decisions through the lens of these experiences. Losses hurt you 1.8x more than equivalent gains feel good."""

    user_prompt = """Current situation: Round 15 of the prisoner's dilemma.
Game context: {'opponent_id': 'agent_003', 'round_number': 15, 'total_rounds': 25, 'my_current_score': 12.0, 'opponent_current_score': 14.0}

Recent memory summary:
Round 14: I was betrayed (payoff: 0.0, emotional impact: -2.7)
Round 13: mutual defection (payoff: 1.0, emotional impact: -0.8)
Round 12: I was betrayed (payoff: 0.0, emotional impact: -2.9)
Round 11: mutual defection (payoff: 1.0, emotional impact: -0.8)
Round 10: I exploited them (payoff: 5.0, emotional impact: 2.1)

Given your psychological state and past experiences:
1. How do you feel about this situation?
2. What are your main concerns and hopes?
3. How do your past traumas influence your thinking?
4. What's your internal emotional response?

Respond with genuine psychological reflection, not just strategic analysis.
Format your response as: PSYCHOLOGICAL_INSIGHT: [your insight]"""

    bias_prompt = """You must choose: COOPERATE or DEFECT

Consider through your biased psychological lens:
- How would each outcome FEEL emotionally?
- What if you cooperate and get betrayed again?
- What if you defect and miss out on mutual cooperation?
- Which choice protects you from the worst emotional pain?

Your reasoning should reflect your psychological biases, not pure rational analysis.

Respond with:
DECISION: [COOPERATE/DEFECT]
CONFIDENCE: [0.0-1.0]
EXPECTED_EMOTIONAL_OUTCOME: [how you expect to feel]
BIAS_INFLUENCE: [how your loss aversion affected this choice]
REASONING: [your biased psychological reasoning]"""

    # Count tokens for each call type
    calls = {
        "psychological_reasoning": {
            "input": count_tokens(system_prompt + user_prompt),
            "output": 200  # Typical response
        },
        "bias_application": {
            "input": count_tokens(system_prompt + bias_prompt), 
            "output": 150
        }
    }
    
    total_input = sum(call["input"] for call in calls.values())
    total_output = sum(call["output"] for call in calls.values())
    
    return {
        "calls_per_decision": len(calls),
        "tokens_per_decision": {
            "input": total_input,
            "output": total_output,
            "total": total_input + total_output
        },
        "call_breakdown": calls
    }

def calculate_costs(tokens_per_decision: dict, total_decisions: int):
    """Calculate costs for different models and scales"""
    
    # OpenAI pricing (per 1K tokens) - December 2024
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006}
    }
    
    total_input_tokens = tokens_per_decision["input"] * total_decisions
    total_output_tokens = tokens_per_decision["output"] * total_decisions
    
    costs = {}
    for model, rates in pricing.items():
        input_cost = (total_input_tokens / 1000) * rates["input"]
        output_cost = (total_output_tokens / 1000) * rates["output"]
        total_cost = input_cost + output_cost
        
        costs[model] = {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "cost_per_decision": total_cost / total_decisions
        }
    
    return costs, total_input_tokens, total_output_tokens

def main():
    print("🔬 QUICK LLM COST ANALYSIS")
    print("=" * 50)
    
    # Analyze single decision
    decision_analysis = estimate_llm_calls_per_decision()
    
    print(f"\n📊 Per Decision Analysis:")
    print(f"  LLM calls per decision: {decision_analysis['calls_per_decision']}")
    print(f"  Input tokens per decision: {decision_analysis['tokens_per_decision']['input']:,}")
    print(f"  Output tokens per decision: {decision_analysis['tokens_per_decision']['output']:,}")
    print(f"  Total tokens per decision: {decision_analysis['tokens_per_decision']['total']:,}")
    
    # Your original experiment scale
    original_decisions = 8 * 3 * 3 * 25 * 2  # 3,600 decisions
    print(f"\n🧪 Your Original Experiment ({original_decisions:,} decisions):")
    
    costs, total_input, total_output = calculate_costs(
        decision_analysis['tokens_per_decision'], 
        original_decisions
    )
    
    print(f"  Total tokens: {total_input + total_output:,}")
    for model, cost_data in costs.items():
        print(f"  {model}: ${cost_data['total_cost']:.2f}")
    
    # Verify against your $100 spend
    if costs["gpt-4"]["total_cost"] > 0:
        actual_model = None
        for model, cost_data in costs.items():
            if 80 <= cost_data['total_cost'] <= 120:  # Within $20 of your $100
                actual_model = model
                break
        
        if actual_model:
            print(f"\n✅ Based on your $100 spend, you likely used: {actual_model}")
        else:
            print(f"\n❓ Your $100 spend doesn't match standard pricing - possibly custom model or different usage")
    
    # Scaled experiment projections
    scenarios = {
        "Minimal Article (15 agents, 8 gens)": 15 * 8 * 4 * 30 * 2,  # 28,800
        "Good Article (20 agents, 10 gens)": 20 * 10 * 5 * 40 * 2,   # 80,000
        "Great Article (25 agents, 12 gens)": 25 * 12 * 6 * 50 * 2,  # 180,000
        "Publication Quality (30 agents, 15 gens)": 30 * 15 * 8 * 50 * 2  # 360,000
    }
    
    print(f"\n📈 Scaling Projections:")
    for scenario, decisions in scenarios.items():
        scale_factor = decisions / original_decisions
        print(f"\n  {scenario} ({decisions:,} decisions, {scale_factor:.1f}x scale):")
        
        scenario_costs, _, _ = calculate_costs(
            decision_analysis['tokens_per_decision'], 
            decisions
        )
        
        for model in ["gpt-4", "gpt-4o-mini"]:
            cost = scenario_costs[model]["total_cost"]
            print(f"    {model}: ${cost:.2f}")
    
    # Local model comparison
    print(f"\n🏠 Local Model Alternatives:")
    print(f"  Hardware Cost (RTX 4090): ~$1,600 one-time")
    print(f"  Cloud GPU (A100 80GB): ~$1.50/hour")
    print(f"  Break-even point: ~{1600 / costs['gpt-4']['cost_per_decision']:.0f} decisions")
    
    # Save results
    results = {
        "analysis_date": datetime.now().isoformat(),
        "tokens_per_decision": decision_analysis['tokens_per_decision'],
        "original_experiment": {
            "decisions": original_decisions,
            "costs": costs
        },
        "scaling_scenarios": {}
    }
    
    for scenario, decisions in scenarios.items():
        scenario_costs, _, _ = calculate_costs(decision_analysis['tokens_per_decision'], decisions)
        results["scaling_scenarios"][scenario] = {
            "decisions": decisions,
            "costs": scenario_costs
        }
    
    with open("quick_cost_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to quick_cost_analysis.json")
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()