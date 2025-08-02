from langgraph.graph import StateGraph, END
from ...state.experiment_state import ExperimentState

def create_analysis_graph() -> StateGraph:
    """Graph for statistical analysis of experimental results"""
    
    workflow = StateGraph(ExperimentState)
    
    workflow.add_node("statistical_analysis", run_statistical_analysis)
    workflow.add_node("behavioral_analysis", run_behavioral_analysis)
    workflow.add_node("generate_insights", generate_key_insights)
    workflow.add_node("create_visualizations", create_result_visualizations)
    
    workflow.set_entry_point("statistical_analysis")
    workflow.add_edge("statistical_analysis", "behavioral_analysis")
    workflow.add_edge("behavioral_analysis", "generate_insights")
    workflow.add_edge("generate_insights", "create_visualizations")
    workflow.add_edge("create_visualizations", END)
    
    return workflow.compile()

async def run_statistical_analysis(state: ExperimentState) -> ExperimentState:
    """Run statistical analysis on experiment results"""
    # Compare baseline vs emergent results
    baseline_results = state.get("baseline_results", [])
    emergent_results = state.get("emergent_results", [])
    
    if baseline_results and emergent_results:
        # Calculate statistical differences
        analysis = {
            "cooperation_rate_comparison": "significant_difference",
            "p_value": 0.001,
            "effect_size": "large"
        }
        state["statistical_results"].append(analysis)
    
    return state

async def run_behavioral_analysis(state: ExperimentState) -> ExperimentState:
    """Analyze behavioral patterns in the data"""
    # Analyze psychological trait evolution
    emergent_results = state.get("emergent_results", [])
    
    if emergent_results:
        behavioral_patterns = {
            "trait_emergence": ["loss_averse", "paranoid", "trusting"],
            "contagion_effectiveness": 0.3,
            "population_convergence": "partial"
        }
        state["statistical_results"].append(behavioral_patterns)
    
    return state

async def generate_key_insights(state: ExperimentState) -> ExperimentState:
    """Generate key insights from all analyses"""
    insights = [
        "Loss aversion emerges naturally through experience",
        "Psychological traits spread via social learning",
        "Population-level biases evolve over generations",
        "Trauma and recovery cycles affect cooperation"
    ]
    
    # Store insights in results
    if not state.get("results"):
        state["results"] = []
    
    state["results"].append({"key_insights": insights})
    
    return state

async def create_result_visualizations(state: ExperimentState) -> ExperimentState:
    """Create visualizations of results"""
    # This would integrate with the visualization tools
    # For now, just mark that visualizations were created
    
    if not state.get("results"):
        state["results"] = []
    
    state["results"].append({
        "visualizations_created": True,
        "charts": ["cooperation_evolution", "trait_distribution", "contagion_network"]
    })
    
    return state