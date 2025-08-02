from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any
from langgraph.graph import add_messages
from .agent_state import AgentState

class PopulationState(TypedDict):
    """State for population-level workflows"""
    population: List[AgentState]
    generation: int
    interaction_results: Annotated[List[Dict], add_messages]
    contagion_events: Annotated[List[Dict], add_messages]
    population_metrics: Dict[str, float]
    
    # Experiment control
    current_experiment: str
    experiment_parameters: Dict[str, Any]
    should_continue: bool
    
    # Population analysis
    psychological_distribution: Dict[str, int]
    dominant_traits: List[str]
    avg_trust_level: float
    avg_loss_sensitivity: float
    avg_cooperation_rate: float
    
    # Evolution tracking
    successful_agents: List[str]
    struggling_agents: List[str]
    trait_transmission_matrix: Dict[str, Dict[str, float]]