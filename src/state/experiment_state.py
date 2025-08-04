from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any, Optional
from datetime import datetime
import operator

from .population_state import PopulationState

class ExperimentState(TypedDict):
    """State for master experiment orchestration - VERIFIED CORRECT"""
    
    # Basic experiment info (these get replaced, not added to)
    experiment_id: str
    current_phase: str
    population_state: PopulationState
    experiment_config: Dict[str, Any]
    
    # Completion flags (these get replaced, not added to)
    baseline_complete: bool
    emergent_complete: bool
    contagion_complete: bool
    analysis_complete: bool
    
    # Time tracking (these get replaced, not added to)
    start_time: datetime
    current_time: datetime
    estimated_completion: Optional[datetime]
    progress_percentage: float
    
    # Configuration (these get replaced, not added to)
    total_generations: int
    interactions_per_generation: int
    rounds_per_interaction: int
    population_size: int
    
    # CRITICAL: These are the fields that use operator.add - they expect individual items to append
    # When you return {"results": [report]}, LangGraph will do: existing_results + [report]
    results: Annotated[List[Dict], operator.add]
    baseline_results: Annotated[List[Dict], operator.add]
    emergent_results: Annotated[List[Dict], operator.add] 
    contagion_results: Annotated[List[Dict], operator.add]
    statistical_results: Annotated[List[Dict], operator.add]