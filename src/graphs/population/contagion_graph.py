from langgraph.graph import StateGraph, END
from ...nodes.contagion_nodes import (
    identify_influential_agents,
    analyze_psychological_differences,
    simulate_trait_transmission,
    update_recipient_psychology,
    log_contagion_event
)
from ...state.population_state import PopulationState

def create_contagion_graph() -> StateGraph:
    """Graph for psychological trait contagion between agents"""
    
    workflow = StateGraph(PopulationState)
    
    workflow.add_node("identify_influencers", identify_influential_agents)
    workflow.add_node("analyze_differences", analyze_psychological_differences)
    workflow.add_node("simulate_transmission", simulate_trait_transmission)
    workflow.add_node("update_psychology", update_recipient_psychology)
    workflow.add_node("log_event", log_contagion_event)
    
    workflow.set_entry_point("identify_influencers")
    workflow.add_edge("identify_influencers", "analyze_differences")
    workflow.add_edge("analyze_differences", "simulate_transmission")
    workflow.add_edge("simulate_transmission", "update_psychology")
    workflow.add_edge("update_psychology", "log_event")
    workflow.add_edge("log_event", END)
    
    return workflow.compile()