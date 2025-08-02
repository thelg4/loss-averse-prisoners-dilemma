from typing import Dict, Any, List
from datetime import datetime
import logging
import random
from ..state.population_state import PopulationState
from ..state.agent_state import AgentState

logger = logging.getLogger(__name__)

async def identify_influential_agents(state: PopulationState) -> PopulationState:
    """Identify agents with high influence potential"""
    population = state["population"]
    
    # Score agents based on influence factors
    influence_scores = []
    
    for agent in population:
        score = 0.0
        
        # High total score indicates success
        score += agent.get("total_score", 0) / 1000.0
        
        # Consistent strategy (high confidence) indicates conviction
        recent_decisions = agent.get("reasoning_chain", [])[-5:]
        if recent_decisions:
            avg_confidence = sum(step.confidence for step in recent_decisions) / len(recent_decisions)
            score += avg_confidence
        
        # Extreme psychological traits are more "contagious"
        profile = agent["psychological_profile"]
        trait_extremity = abs(profile.trust_level - 0.5) + abs(profile.loss_sensitivity - 1.0)
        score += trait_extremity
        
        # Recovery progress indicates resilience/wisdom
        score += agent.get("recovery_progress", 0.5)
        
        influence_scores.append((agent["agent_id"], score))
    
    # Sort by influence score
    influence_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Top 25% are considered influential
    num_influential = max(1, len(population) // 4)
    influential_agent_ids = [agent_id for agent_id, _ in influence_scores[:num_influential]]
    
    # Store in state
    state["interaction_results"].append({
        "type": "influence_identification",
        "influential_agents": influential_agent_ids,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"Identified {len(influential_agent_ids)} influential agents: {influential_agent_ids}")
    
    return state

async def analyze_psychological_differences(state: PopulationState) -> PopulationState:
    """Analyze psychological differences between agents"""
    population = state["population"]
    
    # Calculate population averages
    trust_levels = [agent["psychological_profile"].trust_level for agent in population]
    loss_sensitivities = [agent["psychological_profile"].loss_sensitivity for agent in population]
    
    avg_trust = sum(trust_levels) / len(trust_levels)
    avg_loss_sensitivity = sum(loss_sensitivities) / len(loss_sensitivities)
    
    # Identify agents with significant deviations
    significant_deviations = []
    
    for agent in population:
        profile = agent["psychological_profile"]
        
        trust_deviation = abs(profile.trust_level - avg_trust)
        loss_deviation = abs(profile.loss_sensitivity - avg_loss_sensitivity)
        
        if trust_deviation > 0.2 or loss_deviation > 0.5:
            significant_deviations.append({
                "agent_id": agent["agent_id"],
                "trust_deviation": trust_deviation,
                "loss_deviation": loss_deviation,
                "dominant_trait": profile.get_dominant_trait()
            })
    
    # Store analysis
    state["interaction_results"].append({
        "type": "psychological_analysis",
        "population_averages": {
            "trust_level": avg_trust,
            "loss_sensitivity": avg_loss_sensitivity
        },
        "significant_deviations": significant_deviations,
        "timestamp": datetime.now().isoformat()
    })
    
    logger.info(f"Analyzed psychological differences: {len(significant_deviations)} agents with significant deviations")
    
    return state

async def simulate_trait_transmission(state: PopulationState) -> PopulationState:
    """Simulate psychological trait transmission between agents"""
    population = state["population"]
    generation = state["generation"]
    
    # Get recent influential agents
    recent_results = [r for r in state["interaction_results"] if r.get("type") == "influence_identification"]
    if not recent_results:
        logger.warning("No influential agents identified for trait transmission")
        return state
    
    influential_agent_ids = recent_results[-1]["influential_agents"]
    influential_agents = [a for a in population if a["agent_id"] in influential_agent_ids]
    
    transmission_events = []
    
    # Each non-influential agent has a chance to be influenced
    for target_agent in population:
        if target_agent["agent_id"] in influential_agent_ids:
            continue  # Skip influential agents
        
        # Select a random influential agent as potential source
        if influential_agents:
            source_agent = random.choice(influential_agents)
            
            # Calculate transmission probability
            transmission_prob = _calculate_transmission_probability(source_agent, target_agent)
            
            if random.random() < transmission_prob:
                # Select trait to transmit
                trait_to_transmit = _select_trait_for_transmission(source_agent, target_agent)
                
                if trait_to_transmit:
                    # Apply transmission
                    transmission_strength = random.uniform(0.1, 0.3)
                    _apply_trait_transmission(target_agent, source_agent, trait_to_transmit, transmission_strength)
                    
                    transmission_event = {
                        "source_agent": source_agent["agent_id"],
                        "target_agent": target_agent["agent_id"],
                        "trait": trait_to_transmit,
                        "strength": transmission_strength,
                        "generation": generation,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    transmission_events.append(transmission_event)
    
    # Store transmission events
    state["contagion_events"].extend(transmission_events)
    
    logger.info(f"Simulated {len(transmission_events)} trait transmission events in generation {generation}")
    
    return state

async def update_recipient_psychology(state: PopulationState) -> PopulationState:
    """Update psychology of agents who received trait transmissions"""
    
    # Get recent transmission events
    recent_transmissions = [e for e in state["contagion_events"] 
                          if e.get("generation") == state["generation"]]
    
    # Update learned heuristics based on transmissions
    for event in recent_transmissions:
        target_agent_id = event["target_agent"]
        trait = event["trait"]
        
        # Find the target agent
        target_agent = next((a for a in state["population"] if a["agent_id"] == target_agent_id), None)
        
        if target_agent:
            profile = target_agent["psychological_profile"]
            
            # Add learning heuristic about trait transmission
            if trait == "trust_level":
                if "social_learning_trust" not in profile.learned_heuristics:
                    profile.learned_heuristics.append("social_learning_trust")
            elif trait == "loss_sensitivity":
                if "social_learning_caution" not in profile.learned_heuristics:
                    profile.learned_heuristics.append("social_learning_caution")
    
    logger.info(f"Updated psychology for {len(recent_transmissions)} agents based on trait transmission")
    
    return state

async def log_contagion_event(state: PopulationState) -> PopulationState:
    """Log contagion events for analysis"""
    
    recent_transmissions = [e for e in state["contagion_events"] 
                          if e.get("generation") == state["generation"]]
    
    if recent_transmissions:
        # Analyze transmission patterns
        trait_counts = {}
        for event in recent_transmissions:
            trait = event["trait"]
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
        # Log summary
        state["interaction_results"].append({
            "type": "contagion_summary",
            "generation": state["generation"],
            "total_transmissions": len(recent_transmissions),
            "trait_distribution": trait_counts,
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Generation {state['generation']} contagion: {len(recent_transmissions)} transmissions, traits: {trait_counts}")
    
    return state

# Helper functions

def _calculate_transmission_probability(source_agent: AgentState, target_agent: AgentState) -> float:
    """Calculate probability of trait transmission between two agents"""
    
    # Base probability
    base_prob = 0.1
    
    # Higher probability if source agent is more successful
    source_score = source_agent.get("total_score", 0)
    target_score = target_agent.get("total_score", 0)
    
    if source_score > target_score:
        success_bonus = min(0.3, (source_score - target_score) / 1000.0)
        base_prob += success_bonus
    
    # Higher probability if source has extreme traits (more noticeable)
    source_profile = source_agent["psychological_profile"]
    trait_extremity = abs(source_profile.trust_level - 0.5) + abs(source_profile.loss_sensitivity - 1.0)
    base_prob += trait_extremity * 0.2
    
    # Lower probability if target is very confident in their own approach
    target_profile = target_agent["psychological_profile"]
    recent_confidence = 0.5  # Default
    
    recent_decisions = target_agent.get("reasoning_chain", [])[-3:]
    if recent_decisions:
        recent_confidence = sum(step.confidence for step in recent_decisions) / len(recent_decisions)
    
    confidence_penalty = recent_confidence * 0.3
    base_prob -= confidence_penalty
    
    return max(0.0, min(0.8, base_prob))

def _select_trait_for_transmission(source_agent: AgentState, target_agent: AgentState) -> str:
    """Select which psychological trait should be transmitted"""
    
    source_profile = source_agent["psychological_profile"]
    target_profile = target_agent["psychological_profile"]
    
    # Prioritize traits that are significantly different
    trait_candidates = []
    
    trust_diff = abs(source_profile.trust_level - target_profile.trust_level)
    if trust_diff > 0.2:
        trait_candidates.append("trust_level")
    
    loss_sensitivity_diff = abs(source_profile.loss_sensitivity - target_profile.loss_sensitivity)
    if loss_sensitivity_diff > 0.3:
        trait_candidates.append("loss_sensitivity")
    
    # Also consider transmitting learned heuristics
    unique_heuristics = set(source_profile.learned_heuristics) - set(target_profile.learned_heuristics)
    if unique_heuristics and len(trait_candidates) < 2:
        trait_candidates.append("heuristics")
    
    # Also consider emotional state influence
    if source_profile.emotional_state != target_profile.emotional_state:
        trait_candidates.append("emotional_state")
    
    return random.choice(trait_candidates) if trait_candidates else None

def _apply_trait_transmission(target_agent: AgentState, source_agent: AgentState, trait: str, strength: float) -> None:
    """Apply the actual trait transmission"""
    
    target_profile = target_agent["psychological_profile"]
    source_profile = source_agent["psychological_profile"]
    
    if trait == "trust_level":
        # Move target's trust level toward source's trust level
        diff = source_profile.trust_level - target_profile.trust_level
        adjustment = diff * strength
        target_profile.trust_level = max(0.0, min(1.0, target_profile.trust_level + adjustment))
    
    elif trait == "loss_sensitivity":
        # Move target's loss sensitivity toward source's
        diff = source_profile.loss_sensitivity - target_profile.loss_sensitivity
        adjustment = diff * strength
        target_profile.loss_sensitivity = max(0.5, min(3.0, target_profile.loss_sensitivity + adjustment))
    
    elif trait == "heuristics":
        # Randomly select a heuristic to transmit
        available_heuristics = list(set(source_profile.learned_heuristics) - set(target_profile.learned_heuristics))
        if available_heuristics:
            transmitted_heuristic = random.choice(available_heuristics)
            target_profile.learned_heuristics.append(transmitted_heuristic)
    
    elif trait == "emotional_state":
        # Emotional states can be "contagious" with some probability
        if random.random() < strength:
            # Only transmit certain emotional states
            if source_profile.emotional_state in ["confident", "hopeful", "traumatized", "hurt"]:
                target_profile.emotional_state = source_profile.emotional_state