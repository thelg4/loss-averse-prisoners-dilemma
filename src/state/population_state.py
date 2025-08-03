from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any
import operator

# Do NOT import add_messages for non-message data
from .agent_state import AgentState

class PopulationState(TypedDict):
    """State for population-level workflows"""
    population: List[AgentState]
    generation: int
    
    # Use operator.add for regular lists (NOT add_messages)
    interaction_results: Annotated[List[Dict], operator.add]
    contagion_events: Annotated[List[Dict], operator.add]
    
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