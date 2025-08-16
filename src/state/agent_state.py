from typing_extensions import TypedDict, Annotated
from typing import List, Dict, Any, Optional
import operator

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

class DetailedAgentHistory(BaseModel):
    """Comprehensive agent history tracking"""
    
    # Psychological evolution over time
    trust_level_history: List[Dict] = Field(default_factory=list)
    loss_sensitivity_history: List[Dict] = Field(default_factory=list)
    emotional_state_history: List[Dict] = Field(default_factory=list)
    
    # Decision tracking
    decision_history: List[Dict] = Field(default_factory=list)
    confidence_history: List[float] = Field(default_factory=list)
    
    # Interaction outcomes
    interaction_outcomes: List[Dict] = Field(default_factory=list)
    payoff_history: List[float] = Field(default_factory=list)
    
    # Learning events
    learning_events: List[Dict] = Field(default_factory=list)
    heuristic_acquisitions: List[Dict] = Field(default_factory=list)
    
    # Social influence
    influence_received: List[Dict] = Field(default_factory=list)  # Who influenced this agent
    influence_given: List[Dict] = Field(default_factory=list)    # Who this agent influenced
    
    def record_psychological_change(self, field: str, old_value: Any, new_value: Any, reason: str):
        """Record a change in psychological state"""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'field': field,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason,
            'magnitude': abs(new_value - old_value) if isinstance(new_value, (int, float)) else None
        }
        
        if field == 'trust_level':
            self.trust_level_history.append(change_record)
        elif field == 'loss_sensitivity':
            self.loss_sensitivity_history.append(change_record)
        elif field == 'emotional_state':
            self.emotional_state_history.append(change_record)
    
    def record_decision(self, decision: str, confidence: float, reasoning: str, context: Dict):
        """Record a decision with full context"""
        decision_record = {
            'timestamp': datetime.now().isoformat(),
            'decision': decision,
            'confidence': confidence,
            'reasoning': reasoning,
            'context': context,
            'psychological_state': context.get('psychological_snapshot', {})
        }
        
        self.decision_history.append(decision_record)
        self.confidence_history.append(confidence)
    
    def record_interaction_outcome(self, opponent_id: str, my_move: str, opponent_move: str, 
                                  my_payoff: float, emotional_impact: float):
        """Record interaction outcome with full details"""
        outcome_record = {
            'timestamp': datetime.now().isoformat(),
            'round_number': len(self.interaction_outcomes) + 1,
            'opponent_id': opponent_id,
            'my_move': my_move,
            'opponent_move': opponent_move,
            'my_payoff': my_payoff,
            'emotional_impact': emotional_impact,
            'outcome_type': self._classify_outcome(my_move, opponent_move)
        }
        
        self.interaction_outcomes.append(outcome_record)
        self.payoff_history.append(my_payoff)
    
    def record_learning_event(self, learning_type: str, content: str, effectiveness: float):
        """Record a learning event"""
        learning_record = {
            'timestamp': datetime.now().isoformat(),
            'learning_type': learning_type,
            'content': content,
            'effectiveness': effectiveness
        }
        
        self.learning_events.append(learning_record)
    
    def record_social_influence(self, source_agent: str, trait: str, influence_strength: float, 
                               direction: str = 'received'):
        """Record social influence events"""
        influence_record = {
            'timestamp': datetime.now().isoformat(),
            'source_agent': source_agent,
            'trait': trait,
            'influence_strength': influence_strength,
            'generation': len(self.influence_received) + len(self.influence_given)
        }
        
        if direction == 'received':
            self.influence_received.append(influence_record)
        else:
            self.influence_given.append(influence_record)
    
    def _classify_outcome(self, my_move: str, opponent_move: str) -> str:
        """Classify interaction outcome"""
        if my_move == "COOPERATE" and opponent_move == "COOPERATE":
            return "mutual_cooperation"
        elif my_move == "COOPERATE" and opponent_move == "DEFECT":
            return "betrayal"
        elif my_move == "DEFECT" and opponent_move == "COOPERATE":
            return "exploitation"
        else:
            return "mutual_defection"
    
    def get_psychological_trajectory(self) -> Dict[str, List]:
        """Get trajectory of psychological changes over time"""
        return {
            'trust_trajectory': [(h['timestamp'], h['new_value']) for h in self.trust_level_history],
            'loss_sensitivity_trajectory': [(h['timestamp'], h['new_value']) for h in self.loss_sensitivity_history],
            'emotional_trajectory': [(h['timestamp'], h['new_value']) for h in self.emotional_state_history]
        }
    
    def get_decision_patterns(self) -> Dict:
        """Analyze patterns in decision making"""
        if not self.decision_history:
            return {'pattern': 'no_decisions'}
            
        cooperations = sum(1 for d in self.decision_history if d['decision'] == 'COOPERATE')
        total_decisions = len(self.decision_history)
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Look for trends
        recent_decisions = self.decision_history[-10:]
        recent_cooperations = sum(1 for d in recent_decisions if d['decision'] == 'COOPERATE')
        recent_cooperation_rate = recent_cooperations / len(recent_decisions) if recent_decisions else 0
        
        return {
            'total_decisions': total_decisions,
            'overall_cooperation_rate': cooperations / total_decisions,
            'recent_cooperation_rate': recent_cooperation_rate,
            'average_confidence': avg_confidence,
            'decision_trend': 'increasing_cooperation' if recent_cooperation_rate > (cooperations / total_decisions) else 'decreasing_cooperation'
        }
    
    def get_learning_effectiveness(self) -> Dict:
        """Analyze learning effectiveness"""
        if not self.learning_events:
            return {'effectiveness': 'no_learning_recorded'}
        
        avg_effectiveness = sum(e['effectiveness'] for e in self.learning_events) / len(self.learning_events)
        learning_types = {}
        
        for event in self.learning_events:
            learning_type = event['learning_type']
            learning_types[learning_type] = learning_types.get(learning_type, 0) + 1
        
        return {
            'total_learning_events': len(self.learning_events),
            'average_effectiveness': avg_effectiveness,
            'learning_types': learning_types,
            'most_common_learning': max(learning_types.items(), key=lambda x: x[1])[0] if learning_types else None
        }

# Enhanced AgentState with detailed tracking
class EnhancedAgentState(AgentState):
    """Enhanced agent state with comprehensive history tracking"""
    
    detailed_history: DetailedAgentHistory = Field(default_factory=DetailedAgentHistory)
    
    def update_psychological_profile(self, field: str, new_value: Any, reason: str):
        """Update psychological profile and record the change"""
        old_value = getattr(self.psychological_profile, field, None)
        setattr(self.psychological_profile, field, new_value)
        
        # Record the change
        self.detailed_history.record_psychological_change(field, old_value, new_value, reason)
    
    def make_decision_with_tracking(self, decision: str, confidence: float, reasoning: str, context: Dict):
        """Make decision and record it with full context"""
        self.current_decision = decision
        self.decision_confidence = confidence
        
        # Create psychological snapshot
        context['psychological_snapshot'] = {
            'trust_level': self.psychological_profile.trust_level,
            'loss_sensitivity': self.psychological_profile.loss_sensitivity,
            'emotional_state': self.psychological_profile.emotional_state,
            'dominant_trait': self.psychological_profile.get_dominant_trait()
        }
        
        self.detailed_history.record_decision(decision, confidence, reasoning, context)
    
    def record_interaction_with_tracking(self, opponent_id: str, my_move: str, opponent_move: str, 
                                       my_payoff: float, emotional_impact: float):
        """Record interaction outcome with detailed tracking"""
        self.detailed_history.record_interaction_outcome(
            opponent_id, my_move, opponent_move, my_payoff, emotional_impact
        )
        
        # Also update regular memories
        from .agent_state import Memory, Move
        memory = Memory(
            round_number=len(self.detailed_history.interaction_outcomes),
            my_move=Move.COOPERATE if my_move == "COOPERATE" else Move.DEFECT,
            opponent_move=Move.COOPERATE if opponent_move == "COOPERATE" else Move.DEFECT,
            my_payoff=my_payoff,
            opponent_payoff=0.0,  # Would need to calculate
            emotional_impact=emotional_impact,
            timestamp=datetime.now()
        )
        
        self.recent_memories.append(memory)