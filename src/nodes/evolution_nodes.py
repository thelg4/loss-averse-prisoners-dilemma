from typing import Dict, Any
from datetime import datetime
import logging
from ..state.agent_state import AgentState, Move, ReasoningStep, PsychologicalProfile
import random

logger = logging.getLogger(__name__)

async def process_outcome(state: AgentState) -> AgentState:
    """Process the outcome of the most recent interaction"""
    if not state["recent_memories"]:
        logger.warning(f"No memories to process for agent {state['agent_id']}")
        return state
    
    # Get the most recent memory
    latest_memory = state["recent_memories"][-1]
    profile = state["psychological_profile"]
    
    # Analyze the outcome type
    my_move = latest_memory.my_move
    opponent_move = latest_memory.opponent_move
    my_payoff = latest_memory.my_payoff
    
    outcome_type = _classify_outcome(my_move, opponent_move)
    
    # Process psychological impact
    psychological_impact = _calculate_psychological_impact(
        outcome_type, my_payoff, profile.loss_sensitivity
    )
    
    # Update emotional impact in memory
    latest_memory.emotional_impact = psychological_impact
    
    # Add reasoning step
    reasoning_step = ReasoningStep(
        step_type="outcome_processing",
        content=f"Processed {outcome_type} outcome: payoff={my_payoff}, psychological_impact={psychological_impact:.2f}",
        confidence=0.9,
        timestamp=datetime.now(),
        psychological_insight=f"Outcome type: {outcome_type}"
    )
    
    state["reasoning_chain"].append(reasoning_step)
    
    logger.debug(f"Agent {state['agent_id']} processed {outcome_type} with impact {psychological_impact:.2f}")
    
    return state

async def update_trauma_memories(state: AgentState) -> AgentState:
    """Update trauma memories based on recent experiences"""
    profile = state["psychological_profile"]
    latest_memory = state["recent_memories"][-1] if state["recent_memories"] else None
    
    if not latest_memory:
        return state
    
    # Check if this experience should become a trauma memory
    emotional_impact = latest_memory.emotional_impact
    
    if abs(emotional_impact) > 1.5:  # Significant emotional impact
        trauma_memory = {
            "trauma_type": _determine_trauma_type(latest_memory),
            "severity": abs(emotional_impact),
            "round_number": latest_memory.round_number,
            "emotional_impact": emotional_impact,
            "description": _generate_trauma_description(latest_memory),
            "timestamp": datetime.now().isoformat(),
            "decay_rate": 0.95
        }
        
        profile.trauma_memories.append(trauma_memory)
        
        # Keep only the most significant traumas (limit to 20)
        profile.trauma_memories.sort(key=lambda x: x["severity"], reverse=True)
        profile.trauma_memories = profile.trauma_memories[:20]
        
        reasoning_step = ReasoningStep(
            step_type="trauma_processing",
            content=f"Significant experience recorded as trauma: {trauma_memory['description']}",
            confidence=0.8,
            timestamp=datetime.now(),
            psychological_insight=f"Trauma type: {trauma_memory['trauma_type']}"
        )
        
        state["reasoning_chain"].append(reasoning_step)
    
    # Decay existing traumas
    for trauma in profile.trauma_memories:
        trauma["severity"] *= trauma.get("decay_rate", 0.95)
        trauma["emotional_impact"] *= trauma.get("decay_rate", 0.95)
    
    return state

async def adjust_trust_level(state: AgentState) -> AgentState:
    """Adjust trust level based on recent experiences"""
    profile = state["psychological_profile"]
    recent_memories = state["recent_memories"][-5:] if state["recent_memories"] else []
    
    if not recent_memories:
        return state
    
    # Calculate trust adjustment based on recent experiences
    trust_adjustment = 0.0
    
    for memory in recent_memories:
        if memory.my_move == Move.COOPERATE and memory.opponent_move == Move.COOPERATE:
            # Mutual cooperation increases trust
            trust_adjustment += 0.02
        elif memory.my_move == Move.COOPERATE and memory.opponent_move == Move.DEFECT:
            # Betrayal decreases trust significantly
            trust_adjustment -= 0.05 * profile.loss_sensitivity  # Amplified by loss sensitivity
        elif memory.my_move == Move.DEFECT and memory.opponent_move == Move.COOPERATE:
            # I exploited them - slight guilt reduces trust in others
            trust_adjustment -= 0.01
    
    # Apply adaptation rate
    trust_adjustment *= profile.adaptation_rate
    
    # Update trust level
    old_trust = profile.trust_level
    profile.trust_level = max(0.0, min(1.0, profile.trust_level + trust_adjustment))
    
    if abs(trust_adjustment) > 0.01:  # Only log significant changes
        reasoning_step = ReasoningStep(
            step_type="trust_adjustment",
            content=f"Trust level adjusted from {old_trust:.3f} to {profile.trust_level:.3f} (Δ{trust_adjustment:+.3f})",
            confidence=0.85,
            timestamp=datetime.now(),
            psychological_insight=f"Trust change based on recent interactions"
        )
        
        state["reasoning_chain"].append(reasoning_step)
    
    return state

async def evolve_loss_sensitivity(state: AgentState) -> AgentState:
    """Evolve loss sensitivity based on experiences"""
    profile = state["psychological_profile"]
    recent_memories = state["recent_memories"][-10:] if state["recent_memories"] else []
    
    if not recent_memories:
        return state
    
    # Count significant losses
    betrayals = sum(1 for m in recent_memories 
                   if m.my_move == Move.COOPERATE and m.opponent_move == Move.DEFECT)
    
    exploitations = sum(1 for m in recent_memories 
                       if m.my_move == Move.DEFECT and m.opponent_move == Move.COOPERATE)
    
    # Adjust loss sensitivity
    old_sensitivity = profile.loss_sensitivity
    
    if betrayals > 2:  # Multiple recent betrayals
        # Increase loss sensitivity (become more loss averse)
        increase = 0.1 * betrayals * profile.adaptation_rate
        profile.loss_sensitivity = min(3.0, profile.loss_sensitivity + increase)
    
    elif exploitations > 2:  # Multiple recent exploitations of others
        # Slight decrease in loss sensitivity (guilt/empathy effect)
        decrease = 0.05 * exploitations * profile.adaptation_rate
        profile.loss_sensitivity = max(0.5, profile.loss_sensitivity - decrease)
    
    # Natural decay toward normal (1.0) over time
    decay_rate = 0.01 * profile.adaptation_rate
    if profile.loss_sensitivity > 1.0:
        profile.loss_sensitivity -= decay_rate
    elif profile.loss_sensitivity < 1.0:
        profile.loss_sensitivity += decay_rate
    
    if abs(profile.loss_sensitivity - old_sensitivity) > 0.01:
        reasoning_step = ReasoningStep(
            step_type="loss_sensitivity_evolution",
            content=f"Loss sensitivity evolved from {old_sensitivity:.3f} to {profile.loss_sensitivity:.3f}",
            confidence=0.8,
            timestamp=datetime.now(),
            psychological_insight=f"Adaptation to recent experiences"
        )
        
        state["reasoning_chain"].append(reasoning_step)
    
    return state

async def update_internal_narrative(state: AgentState) -> AgentState:
    """Update the agent's internal narrative"""
    profile = state["psychological_profile"]
    recent_memories = state["recent_memories"][-15:] if state["recent_memories"] else []
    
    if not recent_memories:
        profile.internal_narrative = "I'm just beginning to understand this world."
        return state
    
    # Analyze recent experience patterns
    total_rounds = len(recent_memories)
    cooperations = sum(1 for m in recent_memories if m.my_move == Move.COOPERATE)
    betrayals = sum(1 for m in recent_memories 
                   if m.my_move == Move.COOPERATE and m.opponent_move == Move.DEFECT)
    mutual_cooperations = sum(1 for m in recent_memories 
                             if m.my_move == Move.COOPERATE and m.opponent_move == Move.COOPERATE)
    avg_payoff = sum(m.my_payoff for m in recent_memories) / total_rounds
    
    # Generate narrative based on experiences and psychological state
    narratives = []
    
    if betrayals >= 3:
        narratives.append("I keep getting hurt when I try to trust others.")
    
    if mutual_cooperations >= 5:
        narratives.append("Working together has been rewarding.")
    
    if profile.loss_sensitivity > 2.0:
        narratives.append("Losses feel so much more painful than gains feel good.")
    
    if profile.trust_level < 0.3:
        narratives.append("I've learned to be very careful about who I trust.")
    
    if profile.trust_level > 0.7:
        narratives.append("Despite setbacks, I still believe in cooperation.")
    
    if avg_payoff < 1.5:
        narratives.append("Things haven't been going well for me lately.")
    
    if len(profile.trauma_memories) > 5:
        narratives.append("I carry the weight of many difficult experiences.")
    
    # Combine narratives or use default
    if narratives:
        profile.internal_narrative = " ".join(narratives)
    else:
        profile.internal_narrative = "I'm learning to navigate this complex world, one interaction at a time."
    
    reasoning_step = ReasoningStep(
        step_type="narrative_update",
        content=f"Updated internal narrative: {profile.internal_narrative}",
        confidence=0.7,
        timestamp=datetime.now(),
        psychological_insight="Narrative reflects accumulated experiences"
    )
    
    state["reasoning_chain"].append(reasoning_step)
    
    return state

async def check_recovery_progress(state: AgentState) -> AgentState:
    """Check psychological recovery progress"""
    profile = state["psychological_profile"]
    
    # Calculate recovery metrics
    recent_positive_experiences = sum(1 for m in state["recent_memories"][-10:] 
                                    if m.emotional_impact > 0)
    
    trauma_severity = sum(t.get("severity", 0) for t in profile.trauma_memories)
    
    # Recovery progress based on recent positive experiences vs trauma load
    if trauma_severity > 0:
        recovery_ratio = recent_positive_experiences / (trauma_severity + 1)
        state["recovery_progress"] = min(1.0, recovery_ratio)
    else:
        state["recovery_progress"] = 1.0
    
    # Update learned heuristics based on recovery
    if state["recovery_progress"] > 0.7 and "recovery_is_possible" not in profile.learned_heuristics:
        profile.learned_heuristics.append("recovery_is_possible")
    
    reasoning_step = ReasoningStep(
        step_type="recovery_assessment",
        content=f"Recovery progress: {state['recovery_progress']:.2%}",
        confidence=0.75,
        timestamp=datetime.now(),
        psychological_insight="Tracking psychological healing over time"
    )
    
    state["reasoning_chain"].append(reasoning_step)
    
    return state

# Helper functions

def _classify_outcome(my_move: Move, opponent_move: Move) -> str:
    """Classify the outcome of an interaction"""
    if my_move == Move.COOPERATE and opponent_move == Move.COOPERATE:
        return "mutual_cooperation"
    elif my_move == Move.COOPERATE and opponent_move == Move.DEFECT:
        return "betrayal"
    elif my_move == Move.DEFECT and opponent_move == Move.COOPERATE:
        return "exploitation"
    else:  # both defect
        return "mutual_defection"

def _calculate_psychological_impact(outcome_type: str, payoff: float, loss_sensitivity: float) -> float:
    """Calculate psychological impact of an outcome"""
    base_impact = payoff - 2.0  # 2.0 is neutral expectation
    
    if outcome_type == "betrayal":
        # Betrayal hurts more than the payoff suggests
        return base_impact * loss_sensitivity * 1.5
    elif outcome_type == "exploitation":
        # Guilt reduces positive impact
        return base_impact * 0.7
    elif outcome_type == "mutual_cooperation":
        # Cooperation feels good but not amplified
        return base_impact
    else:  # mutual_defection
        # Neutral outcome, slight negative due to conflict
        return base_impact * 0.8

def _determine_trauma_type(memory) -> str:
    """Determine the type of trauma from a memory"""
    if memory.my_move == Move.COOPERATE and memory.opponent_move == Move.DEFECT:
        return "betrayal"
    elif memory.my_move == Move.DEFECT and memory.opponent_move == Move.COOPERATE:
        return "guilt"
    elif memory.emotional_impact > 1.5:
        return "success"
    else:
        return "conflict"

def _generate_trauma_description(memory) -> str:
    """Generate a description of a traumatic experience"""
    outcome_type = _classify_outcome(memory.my_move, memory.opponent_move)
    
    descriptions = {
        "betrayal": f"Trusted someone in round {memory.round_number} but was betrayed (payoff: {memory.my_payoff})",
        "exploitation": f"Exploited someone's trust in round {memory.round_number} (payoff: {memory.my_payoff})",
        "mutual_cooperation": f"Beautiful cooperation in round {memory.round_number} (payoff: {memory.my_payoff})",
        "mutual_defection": f"Mutual conflict in round {memory.round_number} (payoff: {memory.my_payoff})"
    }
    
    return descriptions.get(outcome_type, f"Significant experience in round {memory.round_number}")