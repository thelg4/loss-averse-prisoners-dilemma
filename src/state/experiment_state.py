from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any, Optional
from langgraph.graph import add_messages
from .population_state import PopulationState
from datetime import datetime

class ExperimentState(TypedDict):
    """State for master experiment orchestration"""
    experiment_id: str
    current_phase: str
    population_state: PopulationState
    results: Annotated[List[Dict], add_messages]
    experiment_config: Dict[str, Any]
    
    # Experiment phases
    baseline_complete: bool
    emergent_complete: bool
    contagion_complete: bool
    analysis_complete: bool
    
    # Results tracking
    baseline_results: List[Dict]
    emergent_results: List[Dict]
    contagion_results: List[Dict]
    statistical_results: List[Dict]
    
    # Experiment metadata
    start_time: datetime
    current_time: datetime
    estimated_completion: Optional[datetime]
    progress_percentage: float
    
    # Configuration
    total_generations: int
    interactions_per_generation: int
    rounds_per_interaction: int
    population_size: int