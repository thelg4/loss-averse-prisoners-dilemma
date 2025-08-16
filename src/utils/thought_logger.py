import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
from ..state.agent_state import Move

class AgentThoughtLogger:
    """Captures and logs agent thoughts in real-time during gameplay"""
    
    def __init__(self, experiment_id: str, log_to_file: bool = True, log_to_console: bool = True):
        self.experiment_id = experiment_id
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        # Create logs directory
        self.log_dir = Path("experiment_logs") / experiment_id
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Separate log files for different types of thoughts
        self.decision_log = self.log_dir / "agent_decisions.jsonl"
        self.psychology_log = self.log_dir / "psychology_evolution.jsonl" 
        self.narrative_log = self.log_dir / "internal_narratives.jsonl"
        self.interaction_log = self.log_dir / "interactions.jsonl"
        
        # Console logger
        if log_to_console:
            self.console_logger = logging.getLogger(f"thoughts.{experiment_id}")
            self.console_logger.setLevel(logging.INFO)
            handler = logging.StreamHandler()
            formatter = logging.Formatter('🧠 %(message)s')
            handler.setFormatter(formatter)
            self.console_logger.addHandler(handler)
        
        # Track agent stories
        self.agent_stories = {}
    
    async def log_decision_process(self, agent_id: str, round_num: int, reasoning_chain: List[Dict]):
        """Log the complete decision-making process"""
        
        # Extract the key thoughts
        psychological_state = None
        memory_summary = None
        llm_reasoning = None
        bias_application = None
        final_decision = None
        
        for step in reasoning_chain:
            if step.step_type == "psychological_assessment":
                psychological_state = step.content
            elif step.step_type == "memory_retrieval":
                memory_summary = step.content
            elif step.step_type == "psychological_reasoning":
                llm_reasoning = step.content
            elif step.step_type == "bias_application":
                bias_application = step.content
            elif step.step_type == "final_decision":
                final_decision = step.content
        
        thought_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "round": round_num,
            "psychological_state": psychological_state,
            "memory_summary": memory_summary,
            "llm_reasoning": llm_reasoning,
            "bias_application": bias_application,
            "final_decision": final_decision,
            "reasoning_steps": len(reasoning_chain)
        }
        
        # Console output
        if self.log_to_console:
            self._print_decision_thoughts(agent_id, round_num, thought_entry)
        
        # File logging
        if self.log_to_file:
            await self._append_to_log(self.decision_log, thought_entry)
    
    async def log_psychology_evolution(self, agent_id: str, old_profile: Dict, new_profile: Dict, trigger_event: str):
        """Log psychological changes with reasoning"""
        
        evolution_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "trigger_event": trigger_event,
            "psychological_changes": {
                "trust_level": {
                    "old": old_profile.get("trust_level", 0.5),
                    "new": new_profile.get("trust_level", 0.5),
                    "change": new_profile.get("trust_level", 0.5) - old_profile.get("trust_level", 0.5)
                },
                "loss_sensitivity": {
                    "old": old_profile.get("loss_sensitivity", 1.0),
                    "new": new_profile.get("loss_sensitivity", 1.0),
                    "change": new_profile.get("loss_sensitivity", 1.0) - old_profile.get("loss_sensitivity", 1.0)
                },
                "emotional_state": {
                    "old": old_profile.get("emotional_state", "neutral"),
                    "new": new_profile.get("emotional_state", "neutral")
                }
            }
        }
        
        if self.log_to_console:
            self._print_psychology_change(agent_id, evolution_entry)
        
        if self.log_to_file:
            await self._append_to_log(self.psychology_log, evolution_entry)
    
    async def log_internal_narrative(self, agent_id: str, old_narrative: str, new_narrative: str, trigger: str):
        """Log changes in internal narrative"""
        
        narrative_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "trigger": trigger,
            "old_narrative": old_narrative,
            "new_narrative": new_narrative,
            "narrative_evolution": self._analyze_narrative_change(old_narrative, new_narrative)
        }
        
        if self.log_to_console:
            self._print_narrative_change(agent_id, narrative_entry)
        
        if self.log_to_file:
            await self._append_to_log(self.narrative_log, narrative_entry)
    
    async def log_interaction_outcome(self, agent_id: str, round_num: int, opponent_id: str, 
                                    my_move: str, opponent_move: str, emotional_impact: float,
                                    immediate_thoughts: str):
        """Log immediate reaction to interaction outcome"""
        
        interaction_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "round": round_num,
            "opponent_id": opponent_id,
            "my_move": my_move,
            "opponent_move": opponent_move,
            "outcome_type": self._classify_outcome(my_move, opponent_move),
            "emotional_impact": emotional_impact,
            "immediate_thoughts": immediate_thoughts
        }
        
        if self.log_to_console:
            self._print_interaction_reaction(agent_id, interaction_entry)
        
        if self.log_to_file:
            await self._append_to_log(self.interaction_log, interaction_entry)
    
    def _print_decision_thoughts(self, agent_id: str, round_num: int, thought_entry: Dict):
        """Print decision thoughts to console in a readable format"""
        print(f"\n🤖 {agent_id} - Round {round_num} Decision Process:")
        print("=" * 60)
        
        if thought_entry["psychological_state"]:
            print(f"😌 Psychology: {thought_entry['psychological_state']}")
        
        if thought_entry["memory_summary"]:
            print(f"💭 Memories: {thought_entry['memory_summary']}")
        
        if thought_entry["llm_reasoning"]:
            # Extract the most interesting part of LLM reasoning
            reasoning = thought_entry["llm_reasoning"]
            if len(reasoning) > 200:
                reasoning = reasoning[:200] + "..."
            print(f"🧠 Reasoning: {reasoning}")
        
        if thought_entry["bias_application"]:
            print(f"⚖️  Bias: {thought_entry['bias_application']}")
        
        if thought_entry["final_decision"]:
            print(f"⚡ Decision: {thought_entry['final_decision']}")
        
        print()
    
    def _print_psychology_change(self, agent_id: str, evolution_entry: Dict):
        """Print psychological evolution to console"""
        changes = evolution_entry["psychological_changes"]
        
        print(f"\n🧬 {agent_id} - Psychological Evolution")
        print("=" * 50)
        print(f"Trigger: {evolution_entry['trigger_event']}")
        
        trust_change = changes["trust_level"]["change"]
        if abs(trust_change) > 0.05:
            arrow = "📈" if trust_change > 0 else "📉"
            print(f"{arrow} Trust: {changes['trust_level']['old']:.3f} → {changes['trust_level']['new']:.3f}")
        
        loss_change = changes["loss_sensitivity"]["change"] 
        if abs(loss_change) > 0.1:
            arrow = "📈" if loss_change > 0 else "📉"
            print(f"{arrow} Loss Sensitivity: {changes['loss_sensitivity']['old']:.3f} → {changes['loss_sensitivity']['new']:.3f}")
        
        if changes["emotional_state"]["old"] != changes["emotional_state"]["new"]:
            print(f"💭 Emotion: {changes['emotional_state']['old']} → {changes['emotional_state']['new']}")
        
        print()
    
    def _print_narrative_change(self, agent_id: str, narrative_entry: Dict):
        """Print narrative evolution to console"""
        if narrative_entry["old_narrative"] != narrative_entry["new_narrative"]:
            print(f"\n📖 {agent_id} - Internal Narrative Evolution")
            print("=" * 50)
            print(f"Trigger: {narrative_entry['trigger']}")
            print(f"Old: \"{narrative_entry['old_narrative']}\"")
            print(f"New: \"{narrative_entry['new_narrative']}\"")
            print()
    
    def _print_interaction_reaction(self, agent_id: str, interaction_entry: Dict):
        """Print immediate reaction to interaction"""
        outcome = interaction_entry["outcome_type"]
        impact = interaction_entry["emotional_impact"]
        
        emoji_map = {
            "mutual_cooperation": "🤝",
            "betrayal": "💔", 
            "exploitation": "😈",
            "mutual_defection": "⚔️"
        }
        
        emoji = emoji_map.get(outcome, "🎲")
        
        print(f"\n{emoji} {agent_id} - {outcome.upper().replace('_', ' ')}")
        print(f"   Impact: {impact:.2f} | Thoughts: {interaction_entry['immediate_thoughts']}")
    
    async def _append_to_log(self, log_file: Path, entry: Dict):
        """Append entry to JSON Lines log file"""
        async with asyncio.Lock():
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
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
    
    def _analyze_narrative_change(self, old: str, new: str) -> Dict[str, Any]:
        """Analyze how narrative changed"""
        return {
            "length_change": len(new) - len(old),
            "new_themes": [theme for theme in ["hurt", "trust", "pain", "careful", "betrayed"] 
                          if theme in new.lower() and theme not in old.lower()],
            "lost_themes": [theme for theme in ["hope", "optimistic", "good", "positive"]
                           if theme in old.lower() and theme not in new.lower()]
        }
    
    async def generate_thought_summary(self, agent_id: str) -> Dict[str, Any]:
        """Generate a summary of an agent's thought evolution"""
        # This would read the log files and create a narrative summary
        # For now, return a placeholder
        return {
            "agent_id": agent_id,
            "total_decisions": 0,
            "psychological_changes": 0,
            "narrative_evolutions": 0,
            "dominant_themes": []
        }


# Integration with the existing system
class ThoughtIntegratedPrisonersDilemmaGame:
    """Extended PrisonersDilemmaGame with thought logging"""
    
    def __init__(self, experiment_id: str, enable_thought_logging: bool = True):
        # Initialize the base game
        from src.game.prisoner_dilemma import PrisonersDilemmaGame
        self.base_game = PrisonersDilemmaGame()
        
        # Add thought logging
        self.thought_logger = None
        if enable_thought_logging:
            self.thought_logger = AgentThoughtLogger(experiment_id)
    
    async def play_tournament_with_thoughts(self, agent1, agent2, rounds: int, tournament_id: str = None):
        """Play tournament with real-time thought logging"""
        
        if self.thought_logger:
            print(f"\n🎭 Starting tournament with thought logging: {agent1['agent_id']} vs {agent2['agent_id']}")
        
        # Run the original tournament but intercept the thought processes
        return await self._tournament_with_logging(agent1, agent2, rounds, tournament_id)
    
    async def _tournament_with_logging(self, agent1, agent2, rounds: int, tournament_id: str):
        """Modified tournament that captures thoughts at each step"""
        
        agent1_score = 0.0
        agent2_score = 0.0
        
        for round_num in range(1, rounds + 1):
            # Store old psychological profiles for comparison
            old_profile1 = self._extract_profile_dict(agent1["psychological_profile"])
            old_profile2 = self._extract_profile_dict(agent2["psychological_profile"])
            old_narrative1 = agent1["psychological_profile"].internal_narrative
            old_narrative2 = agent2["psychological_profile"].internal_narrative
            
            # Update game context
            agent1["current_round"] = round_num
            agent2["current_round"] = round_num
            
            # Get decisions (this will populate reasoning_chain)
            agent1_result = await self.base_game._make_agent_decision_with_fallback(agent1)
            agent2_result = await self.base_game._make_agent_decision_with_fallback(agent2)
            
            # Log decision processes
            if self.thought_logger:
                await self.thought_logger.log_decision_process(
                    agent1["agent_id"], round_num, agent1_result.get("reasoning_chain", [])
                )
                await self.thought_logger.log_decision_process(
                    agent2["agent_id"], round_num, agent2_result.get("reasoning_chain", [])
                )
            
            # Process moves and outcomes
            move1_str = agent1_result.get("current_decision", "DEFECT")
            move2_str = agent2_result.get("current_decision", "DEFECT")
            
            payoff1, payoff2 = self.base_game.payoff_matrix[(
                # self.base_game.Move.COOPERATE if move1_str == "COOPERATE" else self.base_game.Move.DEFECT,
                Move.COOPERATE if move1_str == "COOPERATE" else Move.DEFECT,
                Move.COOPERATE if move2_str == "COOPERATE" else self.base_game.Move.DEFECT
            )]
            
            # Calculate emotional impacts
            emotional_impact1 = self.base_game._calculate_emotional_impact(
                self.base_game.Move.COOPERATE if move1_str == "COOPERATE" else self.base_game.Move.DEFECT,
                self.base_game.Move.COOPERATE if move2_str == "COOPERATE" else self.base_game.Move.DEFECT,
                payoff1, agent1_result["psychological_profile"].loss_sensitivity
            )
            emotional_impact2 = self.base_game._calculate_emotional_impact(
                self.base_game.Move.COOPERATE if move2_str == "COOPERATE" else self.base_game.Move.DEFECT,
                self.base_game.Move.COOPERATE if move1_str == "COOPERATE" else self.base_game.Move.DEFECT,
                payoff2, agent2_result["psychological_profile"].loss_sensitivity
            )
            
            # Log immediate reactions
            if self.thought_logger:
                await self.thought_logger.log_interaction_outcome(
                    agent1["agent_id"], round_num, agent2["agent_id"],
                    move1_str, move2_str, emotional_impact1,
                    f"I {move1_str.lower()}ed and they {move2_str.lower()}ed. Impact: {emotional_impact1:.2f}"
                )
                await self.thought_logger.log_interaction_outcome(
                    agent2["agent_id"], round_num, agent1["agent_id"], 
                    move2_str, move1_str, emotional_impact2,
                    f"I {move2_str.lower()}ed and they {move1_str.lower()}ed. Impact: {emotional_impact2:.2f}"
                )
            
            # Apply psychological evolution
            agent1 = await self.base_game._evolve_agent_with_fallback(agent1_result)
            agent2 = await self.base_game._evolve_agent_with_fallback(agent2_result)
            
            # Log psychological changes
            if self.thought_logger:
                new_profile1 = self._extract_profile_dict(agent1["psychological_profile"])
                new_profile2 = self._extract_profile_dict(agent2["psychological_profile"])
                
                if self._profiles_changed(old_profile1, new_profile1):
                    await self.thought_logger.log_psychology_evolution(
                        agent1["agent_id"], old_profile1, new_profile1, 
                        f"Round {round_num} outcome: {move1_str} vs {move2_str}"
                    )
                
                if self._profiles_changed(old_profile2, new_profile2):
                    await self.thought_logger.log_psychology_evolution(
                        agent2["agent_id"], old_profile2, new_profile2,
                        f"Round {round_num} outcome: {move2_str} vs {move1_str}"
                    )
                
                # Check narrative changes
                if old_narrative1 != agent1["psychological_profile"].internal_narrative:
                    await self.thought_logger.log_internal_narrative(
                        agent1["agent_id"], old_narrative1, 
                        agent1["psychological_profile"].internal_narrative,
                        f"Round {round_num} experience"
                    )
                
                if old_narrative2 != agent2["psychological_profile"].internal_narrative:
                    await self.thought_logger.log_internal_narrative(
                        agent2["agent_id"], old_narrative2,
                        agent2["psychological_profile"].internal_narrative, 
                        f"Round {round_num} experience"
                    )
            
            # Update scores
            agent1_score += payoff1
            agent2_score += payoff2
        
        # Return standard tournament result
        return {
            "tournament_id": tournament_id,
            "agent1_id": agent1["agent_id"],
            "agent2_id": agent2["agent_id"], 
            "total_rounds": rounds,
            "agent1_final_score": agent1_score,
            "agent2_final_score": agent2_score,
            "agent1_final_state": agent1,
            "agent2_final_state": agent2
        }
    
    def _extract_profile_dict(self, profile) -> Dict[str, Any]:
        """Extract profile as dictionary for comparison"""
        return {
            "trust_level": profile.trust_level,
            "loss_sensitivity": profile.loss_sensitivity,
            "emotional_state": profile.emotional_state
        }
    
    def _profiles_changed(self, old: Dict, new: Dict, threshold: float = 0.01) -> bool:
        """Check if psychological profile changed significantly"""
        return (abs(old["trust_level"] - new["trust_level"]) > threshold or
                abs(old["loss_sensitivity"] - new["loss_sensitivity"]) > threshold or
                old["emotional_state"] != new["emotional_state"])