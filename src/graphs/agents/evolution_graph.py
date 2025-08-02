from langgraph.graph import StateGraph, END
from ...nodes.evolution_nodes import (
    process_outcome,
    update_trauma_memories,
    adjust_trust_level,
    evolve_loss_sensitivity,
    update_internal_narrative,
    check_recovery_progress
)
from ...state.agent_state import AgentState

def create_psychological_evolution_graph() -> StateGraph:
    """Graph for updating agent psychology after interactions"""
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("process_outcome", process_outcome)
    workflow.add_node("update_trauma", update_trauma_memories)
    workflow.add_node("adjust_trust", adjust_trust_level)
    workflow.add_node("evolve_sensitivity", evolve_loss_sensitivity)
    workflow.add_node("update_narrative", update_internal_narrative)
    workflow.add_node("check_recovery", check_recovery_progress)
    
    workflow.set_entry_point("process_outcome")
    workflow.add_edge("process_outcome", "update_trauma")
    workflow.add_edge("update_trauma", "adjust_trust")
    workflow.add_edge("adjust_trust", "evolve_sensitivity")
    workflow.add_edge("evolve_sensitivity", "update_narrative")
    workflow.add_edge("update_narrative", "check_recovery")
    workflow.add_edge("check_recovery", END)
    
    return workflow.compile()