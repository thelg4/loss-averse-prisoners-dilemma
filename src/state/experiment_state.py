from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any, Optional
from datetime import datetime
import operator

# Import only add_messages for actual message fields
from langgraph.graph import add_messages

from .population_state import PopulationState

class ExperimentState(TypedDict):
    """State for master experiment orchestration"""
    experiment_id: str
    current_phase: str
    population_state: PopulationState
    
    # Use operator.add for lists that should be appended to (NOT add_messages)
    results: Annotated[List[Dict], operator.add]
    experiment_config: Dict[str, Any]
    
    # Experiment phases
    baseline_complete: bool
    emergent_complete: bool
    contagion_complete: bool
    analysis_complete: bool
    
    # Results tracking - use operator.add for regular lists
    baseline_results: Annotated[List[Dict], operator.add]
    emergent_results: Annotated[List[Dict], operator.add] 
    contagion_results: Annotated[List[Dict], operator.add]
    statistical_results: Annotated[List[Dict], operator.add]
    
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