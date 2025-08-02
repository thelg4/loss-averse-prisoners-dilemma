from langgraph.graph import StateGraph, END
from ...state.population_state import PopulationState
from ...nodes.experiment_nodes import (
    _take_population_snapshot,
    _run_generation_interactions
)

def create_population_management_graph() -> StateGraph:
    """Graph for managing population-level operations"""
    
    workflow = StateGraph(PopulationState)
    
    # Add basic population management nodes
    workflow.add_node("take_snapshot", _take_population_snapshot_node)
    workflow.add_node("run_interactions", _run_interactions_node)
    workflow.add_node("update_metrics", _update_population_metrics)
    
    workflow.set_entry_point("take_snapshot")
    workflow.add_edge("take_snapshot", "run_interactions")
    workflow.add_edge("run_interactions", "update_metrics")
    workflow.add_edge("update_metrics", END)
    
    return workflow.compile()

async def _take_population_snapshot_node(state: PopulationState) -> PopulationState:
    """Node wrapper for taking population snapshot"""
    # Implementation would call helper function
    return state

async def _run_interactions_node(state: PopulationState) -> PopulationState:
    """Node wrapper for running population interactions"""
    # Implementation would run agent interactions
    return state

async def _update_population_metrics(state: PopulationState) -> PopulationState:
    """Update population-level metrics"""
    population = state["population"]
    
    if population:
        # Calculate averages
        trust_levels = [agent["psychological_profile"].trust_level for agent in population]
        loss_sensitivities = [agent["psychological_profile"].loss_sensitivity for agent in population]
        
        state["avg_trust_level"] = sum(trust_levels) / len(trust_levels)
        state["avg_loss_sensitivity"] = sum(loss_sensitivities) / len(loss_sensitivities)
        
        # Update trait distribution
        trait_counts = {}
        for agent in population:
            trait = agent["psychological_profile"].get_dominant_trait()
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
        state["psychological_distribution"] = trait_counts
        state["dominant_traits"] = [trait for trait, count in trait_counts.items() if count >= len(population) * 0.1]
    
    return state