from langgraph.graph import StateGraph, END
from ...nodes.experiment_nodes import (
    initialize_experiment,
    run_baseline_study,
    run_emergent_bias_study,
    run_contagion_study,
    analyze_results,
    generate_report
)
from ...state.experiment_state import ExperimentState

def create_master_experiment_graph() -> StateGraph:
    """Create (but don't compile) the master experiment graph builder"""
    
    # Create the graph builder (not compiled yet)
    workflow = StateGraph(ExperimentState)
    
    workflow.add_node("initialize", initialize_experiment)
    workflow.add_node("baseline", run_baseline_study)
    workflow.add_node("emergent", run_emergent_bias_study)
    workflow.add_node("contagion", run_contagion_study)
    workflow.add_node("analyze", analyze_results)
    workflow.add_node("report", generate_report)
    
    workflow.set_entry_point("initialize")
    workflow.add_edge("initialize", "baseline")
    workflow.add_edge("baseline", "emergent")
    workflow.add_edge("emergent", "contagion")
    workflow.add_edge("contagion", "analyze")
    workflow.add_edge("analyze", "report")
    workflow.add_edge("report", END)
    
    # IMPORTANT: Return the uncompiled builder, not a compiled graph
    # The checkpointing system will compile it with the appropriate checkpointer
    return workflow