from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any, Optional
import operator

# Only import add_messages if you actually have message fields
# from langgraph.graph import add_messages

from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class Move(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"

class PsychologicalProfile(BaseModel):
    """Agent's evolving psychological state"""
    trust_level: float = Field(default=0.5, ge=0.0, le=1.0)
    loss_sensitivity: float = Field(default=1.0, ge=0.1, le=5.0)
    trauma_memories: List[Dict] = Field(default_factory=list)
    emotional_state: str = Field(default="neutral")
    internal_narrative: str = Field(default="")
    learned_heuristics: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    adaptation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    def get_dominant_trait(self) -> str:
        """Get the most prominent psychological trait"""
        if self.loss_sensitivity > 2.0 and self.trust_level < 0.3:
            return "traumatized_paranoid"
        elif self.loss_sensitivity > 1.8:
            return "loss_averse"
        elif self.trust_level < 0.2:
            return "paranoid"
        elif self.trust_level > 0.8:
            return "trusting"
        else:
            return "balanced"

class ReasoningStep(BaseModel):
    """Single step in reasoning chain"""
    step_type: str
    content: str
    confidence: float
    timestamp: datetime
    psychological_insight: Optional[str] = None

class Memory(BaseModel):
    """Memory of a past interaction"""
    round_number: int
    my_move: Move
    opponent_move: Move
    my_payoff: float
    opponent_payoff: float
    emotional_impact: float
    timestamp: datetime

class AgentState(TypedDict):
    """Core agent state for LangGraph workflows"""
    agent_id: str
    psychological_profile: PsychologicalProfile
    current_round: int
    game_context: Dict[str, Any]
    
    # Use operator.add for lists that should be appended to (NOT add_messages)
    reasoning_chain: Annotated[List[ReasoningStep], operator.add]
    psychological_observations: Annotated[List[Dict], operator.add]
    
    # Decision making
    current_decision: Optional[str]
    decision_confidence: float
    expected_outcomes: Dict[str, float]
    
    # Memory and learning
    recent_memories: List[Memory]
    trauma_triggers: List[str]
    recovery_progress: float
    
    # Agent metadata
    agent_type: str
    total_score: float
    reference_point: float