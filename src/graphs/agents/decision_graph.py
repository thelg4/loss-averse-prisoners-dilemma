from langgraph.graph import StateGraph, END
from ...nodes.psychological_nodes import (
    assess_psychological_state,
    retrieve_relevant_memories,
    generate_psychological_reasoning,
    apply_bias_lens,
    make_decision,
    update_confidence
)
from ...state.agent_state import AgentState

def create_agent_decision_graph() -> StateGraph:
    """Create the core agent decision-making graph"""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes for psychological decision making
    workflow.add_node("assess_psychology", assess_psychological_state)
    workflow.add_node("retrieve_memories", retrieve_relevant_memories)
    workflow.add_node("psychological_reasoning", generate_psychological_reasoning)
    workflow.add_node("apply_bias", apply_bias_lens)
    workflow.add_node("make_decision", make_decision)
    workflow.add_node("confidence_check", update_confidence)
    
    # Define edges
    workflow.set_entry_point("assess_psychology")
    workflow.add_edge("assess_psychology", "retrieve_memories")
    workflow.add_edge("retrieve_memories", "psychological_reasoning")
    workflow.add_edge("psychological_reasoning", "apply_bias")
    workflow.add_edge("apply_bias", "make_decision")
    workflow.add_edge("make_decision", "confidence_check")
    workflow.add_edge("confidence_check", END)
    
    return workflow.compile()