# from typing import Dict, Any, Tuple
# from datetime import datetime
# import logging
# import asyncio
# import gc
# import os
# from ..state.agent_state import AgentState, Move, Memory
# from ..graphs.agents.decision_graph import create_agent_decision_graph
# from ..graphs.agents.evolution_graph import create_psychological_evolution_graph

# # Try to import psutil for resource monitoring, but don't fail if not available
# try:
#     import psutil
#     PSUTIL_AVAILABLE = True
# except ImportError:
#     PSUTIL_AVAILABLE = False

# logger = logging.getLogger(__name__)

# class PrisonersDilemmaGame:
#     """Implements the actual prisoner's dilemma game logic with debugging and error handling"""
    
#     def __init__(self):
#         self.decision_graph = create_agent_decision_graph()
#         self.evolution_graph = create_psychological_evolution_graph()
#         self.llm_call_count = 0
        
#         # Payoff matrix
#         self.payoff_matrix = {
#             (Move.COOPERATE, Move.COOPERATE): (3, 3),
#             (Move.COOPERATE, Move.DEFECT): (0, 5),
#             (Move.DEFECT, Move.COOPERATE): (5, 0),
#             (Move.DEFECT, Move.DEFECT): (1, 1)
#         }
    
#     def _log_system_resources(self, context: str):
#         """Log current system resource usage if psutil is available"""
#         if not PSUTIL_AVAILABLE:
#             logger.debug(f"{context} - LLM calls: {self.llm_call_count}")
#             return
            
#         try:
#             process = psutil.Process(os.getpid())
#             memory_mb = process.memory_info().rss / 1024 / 1024
#             cpu_percent = process.cpu_percent()
#             logger.info(f"{context} - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, LLM calls: {self.llm_call_count}")
#         except Exception as e:
#             logger.warning(f"Could not get resource info: {e}")
    
#     async def play_tournament(
#         self, 
#         agent1: AgentState, 
#         agent2: AgentState, 
#         rounds: int,
#         tournament_id: str = None
#     ) -> Dict[str, Any]:
#         """Play a tournament with detailed debugging and error handling"""
        
#         logger.info(f"Starting {rounds}-round tournament: {agent1['agent_id']} vs {agent2['agent_id']}")
#         self._log_system_resources("Tournament start")
        
#         agent1_score = 0.0
#         agent2_score = 0.0
#         agent1_cooperations = 0
#         agent2_cooperations = 0
#         successful_rounds = 0
        
#         for round_num in range(1, rounds + 1):
#             try:
#                 # Log every few rounds
#                 if round_num % 5 == 0 or round_num <= 3:
#                     self._log_system_resources(f"Round {round_num}")
                
#                 # Update round context
#                 agent1["current_round"] = round_num
#                 agent2["current_round"] = round_num
                
#                 agent1["game_context"] = {
#                     "opponent_id": agent2["agent_id"],
#                     "round_number": round_num,
#                     "total_rounds": rounds,
#                     "my_current_score": agent1_score,
#                     "opponent_current_score": agent2_score
#                 }
                
#                 agent2["game_context"] = {
#                     "opponent_id": agent1["agent_id"],
#                     "round_number": round_num,
#                     "total_rounds": rounds,
#                     "my_current_score": agent2_score,
#                     "opponent_current_score": agent1_score
#                 }
                
#                 # Get decisions with timeout and error handling
#                 logger.info(f"Round {round_num}: Getting agent decisions...")
                
#                 try:
#                     # Add timeout to prevent hanging
#                     agent1_task = asyncio.create_task(self.decision_graph.ainvoke(agent1))
#                     agent2_task = asyncio.create_task(self.decision_graph.ainvoke(agent2))
                    
#                     # Wait for both with timeout
#                     agent1_result, agent2_result = await asyncio.wait_for(
#                         asyncio.gather(agent1_task, agent2_task),
#                         timeout=60.0  # 60 second timeout
#                     )
                    
#                     self.llm_call_count += 2
#                     logger.info(f"Round {round_num}: Got decisions (total LLM calls: {self.llm_call_count})")
                    
#                 except asyncio.TimeoutError:
#                     logger.error(f"Round {round_num}: Timeout getting decisions, using fallback")
#                     # Use fallback decisions based on psychology
#                     agent1_result = agent1.copy()
#                     agent2_result = agent2.copy()
                    
#                     agent1_result["current_decision"] = "COOPERATE" if agent1["psychological_profile"].trust_level > 0.5 else "DEFECT"
#                     agent2_result["current_decision"] = "COOPERATE" if agent2["psychological_profile"].trust_level > 0.5 else "DEFECT"
                
#                 except Exception as e:
#                     logger.error(f"Round {round_num}: Error getting decisions: {e}")
#                     # Use conservative fallback
#                     agent1_result = agent1.copy()
#                     agent2_result = agent2.copy()
#                     agent1_result["current_decision"] = "DEFECT"
#                     agent2_result["current_decision"] = "DEFECT"
                
#                 # Extract moves
#                 move1_str = agent1_result.get("current_decision", "DEFECT")
#                 move2_str = agent2_result.get("current_decision", "DEFECT")
                
#                 move1 = Move.COOPERATE if move1_str == "COOPERATE" else Move.DEFECT
#                 move2 = Move.COOPERATE if move2_str == "COOPERATE" else Move.DEFECT
                
#                 # Calculate payoffs
#                 payoff1, payoff2 = self.payoff_matrix[(move1, move2)]
                
#                 # Update scores
#                 agent1_score += payoff1
#                 agent2_score += payoff2
#                 agent1_result["total_score"] = agent1_score
#                 agent2_result["total_score"] = agent2_score
                
#                 # Count cooperations
#                 if move1 == Move.COOPERATE:
#                     agent1_cooperations += 1
#                 if move2 == Move.COOPERATE:
#                     agent2_cooperations += 1
                
#                 # Create memories
#                 memory1 = Memory(
#                     round_number=round_num,
#                     my_move=move1,
#                     opponent_move=move2,
#                     my_payoff=payoff1,
#                     opponent_payoff=payoff2,
#                     emotional_impact=self._calculate_emotional_impact(
#                         move1, move2, payoff1, agent1_result["psychological_profile"].loss_sensitivity
#                     ),
#                     timestamp=datetime.now()
#                 )
                
#                 memory2 = Memory(
#                     round_number=round_num,
#                     my_move=move2,
#                     opponent_move=move1,
#                     my_payoff=payoff2,
#                     opponent_payoff=payoff1,
#                     emotional_impact=self._calculate_emotional_impact(
#                         move2, move1, payoff2, agent2_result["psychological_profile"].loss_sensitivity
#                     ),
#                     timestamp=datetime.now()
#                 )
                
#                 # Add memories to agents
#                 agent1_result["recent_memories"].append(memory1)
#                 agent2_result["recent_memories"].append(memory2)
                
#                 # Apply psychological evolution (every 5 rounds to reduce load)
#                 if round_num % 5 == 0 or round_num == rounds:  # Every 5 rounds or final round
#                     try:
#                         logger.info(f"Round {round_num}: Applying psychological evolution...")
                        
#                         evolution_task1 = asyncio.create_task(self.evolution_graph.ainvoke(agent1_result))
#                         evolution_task2 = asyncio.create_task(self.evolution_graph.ainvoke(agent2_result))
                        
#                         agent1, agent2 = await asyncio.wait_for(
#                             asyncio.gather(evolution_task1, evolution_task2),
#                             timeout=30.0  # 30 second timeout for evolution
#                         )
                        
#                         logger.info(f"Round {round_num}: Evolution complete")
                        
#                     except asyncio.TimeoutError:
#                         logger.warning(f"Round {round_num}: Evolution timeout, skipping")
#                         agent1 = agent1_result
#                         agent2 = agent2_result
#                     except Exception as e:
#                         logger.warning(f"Round {round_num}: Evolution error: {e}")
#                         agent1 = agent1_result
#                         agent2 = agent2_result
#                 else:
#                     # Just update agent states without evolution
#                     agent1 = agent1_result
#                     agent2 = agent2_result
                
#                 successful_rounds += 1
                
#                 # Force garbage collection every 5 rounds
#                 if round_num % 5 == 0:
#                     gc.collect()
                
#                 # Log progress
#                 if round_num <= 5 or round_num % 5 == 0:
#                     logger.info(f"Round {round_num}: {agent1['agent_id']} {move1.value} vs {agent2['agent_id']} {move2.value} "
#                                f"(scores: {agent1_score:.1f} vs {agent2_score:.1f})")
                
#             except Exception as e:
#                 logger.error(f"Round {round_num} failed completely: {e}")
#                 # Continue to next round with current agent states
#                 continue
        
#         # Calculate final statistics
#         cooperation_rate1 = agent1_cooperations / successful_rounds if successful_rounds > 0 else 0
#         cooperation_rate2 = agent2_cooperations / successful_rounds if successful_rounds > 0 else 0
        
#         result = {
#             "tournament_id": tournament_id or f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#             "agent1_id": agent1["agent_id"],
#             "agent2_id": agent2["agent_id"],
#             "total_rounds": rounds,
#             "successful_rounds": successful_rounds,
#             "agent1_final_score": agent1_score,
#             "agent2_final_score": agent2_score,
#             "agent1_cooperation_rate": cooperation_rate1,
#             "agent2_cooperation_rate": cooperation_rate2,
#             "agent1_final_state": agent1,
#             "agent2_final_state": agent2,
#             "llm_calls_made": self.llm_call_count,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         self._log_system_resources("Tournament end")
        
#         logger.info(f"Tournament complete: {agent1['agent_id']} scored {agent1_score:.1f} ({cooperation_rate1:.1%} coop), "
#                    f"{agent2['agent_id']} scored {agent2_score:.1f} ({cooperation_rate2:.1%} coop). "
#                    f"Successful rounds: {successful_rounds}/{rounds}, LLM calls: {self.llm_call_count}")
        
#         return result
    
#     async def run_population_interactions(
#         self,
#         population: list,
#         interactions_per_generation: int,
#         rounds_per_interaction: int = 100
#     ) -> Dict[str, Any]:
#         """Run interactions within a population"""
        
#         logger.info(f"Running {interactions_per_generation} interactions for population of {len(population)}")
        
#         import random
#         interaction_results = []
#         total_cooperations = 0
#         total_rounds = 0
        
#         for interaction_num in range(interactions_per_generation):
#             try:
#                 # Randomly select two agents
#                 if len(population) < 2:
#                     logger.warning(f"Not enough agents for interaction {interaction_num}")
#                     continue
                    
#                 agent1, agent2 = random.sample(population, 2)
                
#                 # Run tournament
#                 tournament_result = await self.play_tournament(
#                     agent1, agent2, rounds_per_interaction, 
#                     f"pop_interaction_{interaction_num}"
#                 )
                
#                 interaction_results.append(tournament_result)
                
#                 # Update population with evolved agents
#                 for i, pop_agent in enumerate(population):
#                     if pop_agent["agent_id"] == agent1["agent_id"]:
#                         population[i] = tournament_result["agent1_final_state"]
#                     elif pop_agent["agent_id"] == agent2["agent_id"]:
#                         population[i] = tournament_result["agent2_final_state"]
                
#                 # Accumulate cooperation statistics
#                 total_cooperations += (tournament_result["agent1_cooperation_rate"] + 
#                                      tournament_result["agent2_cooperation_rate"]) * rounds_per_interaction
#                 total_rounds += 2 * rounds_per_interaction
                
#                 logger.info(f"Completed population interaction {interaction_num + 1}/{interactions_per_generation}")
                
#             except Exception as e:
#                 logger.error(f"Population interaction {interaction_num} failed: {e}")
#                 continue
        
#         avg_cooperation_rate = total_cooperations / total_rounds if total_rounds > 0 else 0
        
#         # Count trait changes
#         trait_changes = 0
#         for agent in population:
#             if len(agent.get("reasoning_chain", [])) > 0:
#                 trait_changes += len([step for step in agent["reasoning_chain"] 
#                                     if step.step_type in ["trust_adjustment", "loss_sensitivity_evolution"]])
        
#         return {
#             "total_interactions": interactions_per_generation,
#             "successful_interactions": len(interaction_results),
#             "avg_cooperation_rate": avg_cooperation_rate,
#             "trait_changes": trait_changes,
#             "interaction_results": interaction_results,
#             "timestamp": datetime.now().isoformat()
#         }
    
#     def _calculate_emotional_impact(self, my_move: Move, opponent_move: Move, 
#                                   my_payoff: float, loss_sensitivity: float) -> float:
#         """Calculate emotional impact of a game outcome"""
        
#         base_impact = my_payoff - 2.0  # 2.0 is neutral expectation
        
#         if my_move == Move.COOPERATE and opponent_move == Move.DEFECT:
#             # Betrayal - amplify negative impact
#             return base_impact * loss_sensitivity * 1.5
#         elif my_move == Move.DEFECT and opponent_move == Move.COOPERATE:
#             # Exploitation - guilt reduces positive impact
#             return base_impact * 0.7
#         elif my_move == Move.COOPERATE and opponent_move == Move.COOPERATE:
#             # Mutual cooperation - positive but not amplified
#             return base_impact
#         else:
#             # Mutual defection - neutral with slight negative
#             return base_impact * 0.8


# # Functions to replace the placeholders in experiment_nodes.py
# async def simulate_baseline_tournament_real(agent1, agent2, rounds, replication_id):
#     """Actually simulate a tournament between baseline agents"""
    
#     game = PrisonersDilemmaGame()
    
#     result = await game.play_tournament(
#         agent1, agent2, rounds, f"baseline_rep_{replication_id}"
#     )
    
#     return {
#         "replication_id": replication_id,
#         "agent1_final_score": result["agent1_final_score"],
#         "agent2_final_score": result["agent2_final_score"],
#         "agent1_cooperation_rate": result["agent1_cooperation_rate"],
#         "agent2_cooperation_rate": result["agent2_cooperation_rate"],
#         "total_rounds": result["total_rounds"],
#         "successful_rounds": result["successful_rounds"],
#         "llm_calls_made": result["llm_calls_made"],
#         "timestamp": result["timestamp"],
#         "detailed_result": result  # Include full result for analysis
#     }

# async def run_generation_interactions_real(population_state, num_interactions):
#     """Actually run interactions within a generation"""
    
#     game = PrisonersDilemmaGame()
#     population = population_state["population"]
    
#     # Get configuration from experiment parameters
#     experiment_params = population_state.get("experiment_parameters", {})
#     emergent_config = experiment_params.get("emergent_experiment", {})
#     rounds_per_interaction = emergent_config.get("rounds_per_interaction", 100)
    
#     result = await game.run_population_interactions(
#         population, num_interactions, rounds_per_interaction
#     )
    
#     # Update population state with evolved agents (already updated in-place)
#     # But we should also update population metrics
#     update_population_metrics_real(population_state)
    
#     return result

# def update_population_metrics_real(population_state):
#     """Update population-level metrics after interactions"""
    
#     population = population_state["population"]
    
#     if not population:
#         return
    
#     # Calculate averages
#     trust_levels = []
#     loss_sensitivities = []
#     total_scores = []
    
#     for agent in population:
#         profile = agent["psychological_profile"]
#         trust_levels.append(profile.trust_level)
#         loss_sensitivities.append(profile.loss_sensitivity)
#         total_scores.append(agent.get("total_score", 0))
    
#     population_state["avg_trust_level"] = sum(trust_levels) / len(trust_levels)
#     population_state["avg_loss_sensitivity"] = sum(loss_sensitivities) / len(loss_sensitivities)
    
#     # Update trait distribution
#     trait_counts = {}
#     for agent in population:
#         trait = agent["psychological_profile"].get_dominant_trait()
#         trait_counts[trait] = trait_counts.get(trait, 0) + 1
    
#     population_state["psychological_distribution"] = trait_counts
#     population_state["dominant_traits"] = [
#         trait for trait, count in trait_counts.items() 
#         if count >= len(population) * 0.1  # At least 10% of population
#     ]
    
#     # Identify successful and struggling agents
#     if total_scores:
#         avg_score = sum(total_scores) / len(total_scores)
#         population_state["successful_agents"] = [
#             agent["agent_id"] for agent in population 
#             if agent.get("total_score", 0) > avg_score * 1.2
#         ]
#         population_state["struggling_agents"] = [
#             agent["agent_id"] for agent in population 
#             if agent.get("total_score", 0) < avg_score * 0.8
#         ]

##################################################################################################
# # src/game/prisoner_dilemma.py - FIXED VERSION
# from typing import Dict, Any, Tuple
# from datetime import datetime
# import logging
# import asyncio
# import gc
# import os
# from ..state.agent_state import AgentState, Move, Memory
# from ..graphs.agents.decision_graph import create_agent_decision_graph
# from ..graphs.agents.evolution_graph import create_psychological_evolution_graph

# # Try to import psutil for resource monitoring, but don't fail if not available
# try:
#     import psutil
#     PSUTIL_AVAILABLE = True
# except ImportError:
#     PSUTIL_AVAILABLE = False

# logger = logging.getLogger(__name__)

# class PrisonersDilemmaGame:
#     """Implements the actual prisoner's dilemma game logic with debugging and error handling"""
    
#     def __init__(self):
#         # FIXED: Add try-catch for graph creation and use fallback
#         try:
#             self.decision_graph = create_agent_decision_graph()
#             self.evolution_graph = create_psychological_evolution_graph()
#             self.graphs_available = True
#             logger.info("Successfully created LLM-based decision graphs")
#         except Exception as e:
#             logger.warning(f"Failed to create LLM graphs: {e}")
#             logger.warning("Using fallback decision logic")
#             self.decision_graph = None
#             self.evolution_graph = None
#             self.graphs_available = False
        
#         self.llm_call_count = 0
        
#         # Payoff matrix
#         self.payoff_matrix = {
#             (Move.COOPERATE, Move.COOPERATE): (3, 3),
#             (Move.COOPERATE, Move.DEFECT): (0, 5),
#             (Move.DEFECT, Move.COOPERATE): (5, 0),
#             (Move.DEFECT, Move.DEFECT): (1, 1)
#         }
    
#     def _log_system_resources(self, context: str):
#         """Log current system resource usage if psutil is available"""
#         if not PSUTIL_AVAILABLE:
#             logger.debug(f"{context} - LLM calls: {self.llm_call_count}")
#             return
            
#         try:
#             process = psutil.Process(os.getpid())
#             memory_mb = process.memory_info().rss / 1024 / 1024
#             cpu_percent = process.cpu_percent()
#             logger.info(f"{context} - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, LLM calls: {self.llm_call_count}")
#         except Exception as e:
#             logger.warning(f"Could not get resource info: {e}")
    
#     async def _make_agent_decision_with_fallback(self, agent_state: AgentState) -> AgentState:
#         """Make agent decision with LLM graphs or fallback logic"""
        
#         if not self.graphs_available or not self.decision_graph:
#             logger.debug(f"Using fallback decision for {agent_state['agent_id']}")
#             return self._make_fallback_decision(agent_state)
        
#         try:
#             # Try LLM-based decision with aggressive timeout
#             result = await asyncio.wait_for(
#                 self.decision_graph.ainvoke(agent_state),
#                 timeout=15.0  # 15 second timeout
#             )
#             self.llm_call_count += 1
#             return result
            
#         except asyncio.TimeoutError:
#             logger.warning(f"LLM decision timeout for {agent_state['agent_id']}, using fallback")
#             return self._make_fallback_decision(agent_state)
#         except Exception as e:
#             logger.warning(f"LLM decision error for {agent_state['agent_id']}: {e}, using fallback")
#             return self._make_fallback_decision(agent_state)
    
#     def _make_fallback_decision(self, agent_state: AgentState) -> AgentState:
#         """Make decision using simple psychological rules (no LLM)"""
        
#         profile = agent_state["psychological_profile"]
#         recent_memories = agent_state["recent_memories"]
        
#         # Simple decision logic based on psychology
#         cooperation_probability = profile.trust_level
        
#         # Adjust based on recent experiences
#         if recent_memories:
#             recent_betrayals = sum(1 for m in recent_memories[-5:] 
#                                  if m.my_move == Move.COOPERATE and m.opponent_move == Move.DEFECT)
#             recent_successes = sum(1 for m in recent_memories[-5:] 
#                                  if m.my_move == Move.COOPERATE and m.opponent_move == Move.COOPERATE)
            
#             # Reduce cooperation after betrayals
#             cooperation_probability -= (recent_betrayals * 0.2)
            
#             # Increase cooperation after successes
#             cooperation_probability += (recent_successes * 0.1)
        
#         # Apply loss aversion bias
#         if profile.loss_sensitivity > 1.5:
#             cooperation_probability *= 0.8  # More cautious
        
#         # Make decision
#         import random
#         decision = "COOPERATE" if random.random() < cooperation_probability else "DEFECT"
#         confidence = abs(cooperation_probability - 0.5) + 0.5  # Convert to confidence
        
#         # Update agent state
#         agent_state = dict(agent_state)  # Make a copy
#         agent_state["current_decision"] = decision
#         agent_state["decision_confidence"] = min(1.0, max(0.1, confidence))
        
#         return agent_state
    
#     async def _evolve_agent_with_fallback(self, agent_state: AgentState) -> AgentState:
#         """Evolve agent psychology with LLM graphs or fallback logic"""
        
#         if not self.graphs_available or not self.evolution_graph:
#             return self._evolve_agent_fallback(agent_state)
        
#         try:
#             # Try LLM-based evolution with timeout
#             result = await asyncio.wait_for(
#                 self.evolution_graph.ainvoke(agent_state),
#                 timeout=10.0  # 10 second timeout
#             )
#             return result
            
#         except asyncio.TimeoutError:
#             logger.warning(f"LLM evolution timeout for {agent_state['agent_id']}, using fallback")
#             return self._evolve_agent_fallback(agent_state)
#         except Exception as e:
#             logger.warning(f"LLM evolution error for {agent_state['agent_id']}: {e}, using fallback")
#             return self._evolve_agent_fallback(agent_state)
    
#     def _evolve_agent_fallback(self, agent_state: AgentState) -> AgentState:
#         """Simple fallback psychology evolution"""
        
#         if not agent_state["recent_memories"]:
#             return agent_state
        
#         # Make a copy to avoid mutations
#         agent_state = dict(agent_state)
#         profile = agent_state["psychological_profile"]
        
#         # Simple evolution based on recent experiences
#         latest_memory = agent_state["recent_memories"][-1]
        
#         # Adjust trust based on latest interaction
#         if latest_memory.my_move == Move.COOPERATE and latest_memory.opponent_move == Move.DEFECT:
#             # Betrayed - reduce trust
#             profile.trust_level = max(0.0, profile.trust_level - 0.1)
#             profile.loss_sensitivity = min(3.0, profile.loss_sensitivity + 0.2)
#         elif latest_memory.my_move == Move.COOPERATE and latest_memory.opponent_move == Move.COOPERATE:
#             # Mutual cooperation - slight trust increase
#             profile.trust_level = min(1.0, profile.trust_level + 0.05)
        
#         return agent_state

#     async def play_tournament(
#         self, 
#         agent1: AgentState, 
#         agent2: AgentState, 
#         rounds: int,
#         tournament_id: str = None
#     ) -> Dict[str, Any]:
#         """Play a tournament with detailed debugging and FIXED error handling"""
        
#         logger.info(f"Starting {rounds}-round tournament: {agent1['agent_id']} vs {agent2['agent_id']}")
#         self._log_system_resources("Tournament start")
        
#         agent1_score = 0.0
#         agent2_score = 0.0
#         agent1_cooperations = 0
#         agent2_cooperations = 0
#         successful_rounds = 0
        
#         for round_num in range(1, rounds + 1):
#             try:
#                 # Log every round for debugging
#                 self._log_system_resources(f"Round {round_num}")
                
#                 # Update round context
#                 agent1["current_round"] = round_num
#                 agent2["current_round"] = round_num
                
#                 agent1["game_context"] = {
#                     "opponent_id": agent2["agent_id"],
#                     "round_number": round_num,
#                     "total_rounds": rounds,
#                     "my_current_score": agent1_score,
#                     "opponent_current_score": agent2_score
#                 }
                
#                 agent2["game_context"] = {
#                     "opponent_id": agent1["agent_id"],
#                     "round_number": round_num,
#                     "total_rounds": rounds,
#                     "my_current_score": agent2_score,
#                     "opponent_current_score": agent1_score
#                 }
                
#                 # FIXED: Get decisions with better error handling and fallback
#                 logger.info(f"Round {round_num}: Getting agent decisions...")
                
#                 try:
#                     # Use the new fallback decision method
#                     agent1_result = await self._make_agent_decision_with_fallback(agent1)
#                     agent2_result = await self._make_agent_decision_with_fallback(agent2)
                    
#                     logger.info(f"Round {round_num}: Got decisions (total LLM calls: {self.llm_call_count})")
                    
#                 except Exception as e:
#                     logger.error(f"Round {round_num}: Critical decision error: {e}")
#                     # Use most basic fallback
#                     agent1_result = dict(agent1)
#                     agent2_result = dict(agent2)
                    
#                     agent1_result["current_decision"] = "DEFECT"  # Conservative
#                     agent2_result["current_decision"] = "DEFECT"
#                     agent1_result["decision_confidence"] = 0.5
#                     agent2_result["decision_confidence"] = 0.5
                
#                 # Extract moves with safety checks
#                 move1_str = agent1_result.get("current_decision", "DEFECT")
#                 move2_str = agent2_result.get("current_decision", "DEFECT")
                
#                 move1 = Move.COOPERATE if move1_str == "COOPERATE" else Move.DEFECT
#                 move2 = Move.COOPERATE if move2_str == "COOPERATE" else Move.DEFECT
                
#                 # Calculate payoffs
#                 payoff1, payoff2 = self.payoff_matrix[(move1, move2)]
                
#                 # Update scores
#                 agent1_score += payoff1
#                 agent2_score += payoff2
#                 agent1_result["total_score"] = agent1_score
#                 agent2_result["total_score"] = agent2_score
                
#                 # Count cooperations
#                 if move1 == Move.COOPERATE:
#                     agent1_cooperations += 1
#                 if move2 == Move.COOPERATE:
#                     agent2_cooperations += 1
                
#                 # Create memories
#                 memory1 = Memory(
#                     round_number=round_num,
#                     my_move=move1,
#                     opponent_move=move2,
#                     my_payoff=payoff1,
#                     opponent_payoff=payoff2,
#                     emotional_impact=self._calculate_emotional_impact(
#                         move1, move2, payoff1, agent1_result["psychological_profile"].loss_sensitivity
#                     ),
#                     timestamp=datetime.now()
#                 )
                
#                 memory2 = Memory(
#                     round_number=round_num,
#                     my_move=move2,
#                     opponent_move=move1,
#                     my_payoff=payoff2,
#                     opponent_payoff=payoff1,
#                     emotional_impact=self._calculate_emotional_impact(
#                         move2, move1, payoff2, agent2_result["psychological_profile"].loss_sensitivity
#                     ),
#                     timestamp=datetime.now()
#                 )
                
#                 # Add memories to agents
#                 agent1_result["recent_memories"].append(memory1)
#                 agent2_result["recent_memories"].append(memory2)
                
#                 # FIXED: Apply psychological evolution with fallback (every round for testing)
#                 try:
#                     logger.info(f"Round {round_num}: Applying psychological evolution...")
                    
#                     agent1 = await self._evolve_agent_with_fallback(agent1_result)
#                     agent2 = await self._evolve_agent_with_fallback(agent2_result)
                    
#                     logger.info(f"Round {round_num}: Evolution complete")
                    
#                 except Exception as e:
#                     logger.warning(f"Round {round_num}: Evolution error: {e}")
#                     agent1 = agent1_result
#                     agent2 = agent2_result
                
#                 successful_rounds += 1
                
#                 # Log progress
#                 logger.info(f"Round {round_num}: {agent1['agent_id']} {move1.value} vs {agent2['agent_id']} {move2.value} "
#                            f"(scores: {agent1_score:.1f} vs {agent2_score:.1f})")
                
#             except Exception as e:
#                 logger.error(f"Round {round_num} failed completely: {e}")
#                 # Continue to next round with current agent states
#                 continue
        
#         # Calculate final statistics
#         cooperation_rate1 = agent1_cooperations / successful_rounds if successful_rounds > 0 else 0
#         cooperation_rate2 = agent2_cooperations / successful_rounds if successful_rounds > 0 else 0
        
#         result = {
#             "tournament_id": tournament_id or f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#             "agent1_id": agent1["agent_id"],
#             "agent2_id": agent2["agent_id"],
#             "total_rounds": rounds,
#             "successful_rounds": successful_rounds,
#             "agent1_final_score": agent1_score,
#             "agent2_final_score": agent2_score,
#             "agent1_cooperation_rate": cooperation_rate1,
#             "agent2_cooperation_rate": cooperation_rate2,
#             "agent1_final_state": agent1,
#             "agent2_final_state": agent2,
#             "llm_calls_made": self.llm_call_count,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         self._log_system_resources("Tournament end")
        
#         logger.info(f"Tournament complete: {agent1['agent_id']} scored {agent1_score:.1f} ({cooperation_rate1:.1%} coop), "
#                    f"{agent2['agent_id']} scored {agent2_score:.1f} ({cooperation_rate2:.1%} coop). "
#                    f"Successful rounds: {successful_rounds}/{rounds}, LLM calls: {self.llm_call_count}")
        
#         return result
    
#     async def run_population_interactions(
#         self,
#         population: list,
#         interactions_per_generation: int,
#         rounds_per_interaction: int = 100
#     ) -> Dict[str, Any]:
#         """Run interactions within a population with better error handling"""
        
#         logger.info(f"Running {interactions_per_generation} interactions for population of {len(population)}")
        
#         import random
#         interaction_results = []
#         total_cooperations = 0
#         total_rounds = 0
        
#         for interaction_num in range(interactions_per_generation):
#             try:
#                 # Randomly select two agents
#                 if len(population) < 2:
#                     logger.warning(f"Not enough agents for interaction {interaction_num}")
#                     continue
                    
#                 agent1, agent2 = random.sample(population, 2)
                
#                 # Run tournament with error handling
#                 tournament_result = await self.play_tournament(
#                     agent1, agent2, rounds_per_interaction, 
#                     f"pop_interaction_{interaction_num}"
#                 )
                
#                 interaction_results.append(tournament_result)
                
#                 # Update population with evolved agents
#                 for i, pop_agent in enumerate(population):
#                     if pop_agent["agent_id"] == agent1["agent_id"]:
#                         population[i] = tournament_result["agent1_final_state"]
#                     elif pop_agent["agent_id"] == agent2["agent_id"]:
#                         population[i] = tournament_result["agent2_final_state"]
                
#                 # Accumulate cooperation statistics
#                 total_cooperations += (tournament_result["agent1_cooperation_rate"] + 
#                                      tournament_result["agent2_cooperation_rate"]) * rounds_per_interaction
#                 total_rounds += 2 * rounds_per_interaction
                
#                 logger.info(f"Completed population interaction {interaction_num + 1}/{interactions_per_generation}")
                
#             except Exception as e:
#                 logger.error(f"Population interaction {interaction_num} failed: {e}")
#                 continue
        
#         avg_cooperation_rate = total_cooperations / total_rounds if total_rounds > 0 else 0
        
#         # Count trait changes
#         trait_changes = 0
#         for agent in population:
#             if len(agent.get("reasoning_chain", [])) > 0:
#                 trait_changes += len([step for step in agent["reasoning_chain"] 
#                                     if step.step_type in ["trust_adjustment", "loss_sensitivity_evolution"]])
        
#         return {
#             "total_interactions": interactions_per_generation,
#             "successful_interactions": len(interaction_results),
#             "avg_cooperation_rate": avg_cooperation_rate,
#             "trait_changes": trait_changes,
#             "interaction_results": interaction_results,
#             "timestamp": datetime.now().isoformat()
#         }
    
#     def _calculate_emotional_impact(self, my_move: Move, opponent_move: Move, 
#                                   my_payoff: float, loss_sensitivity: float) -> float:
#         """Calculate emotional impact of a game outcome"""
        
#         base_impact = my_payoff - 2.0  # 2.0 is neutral expectation
        
#         if my_move == Move.COOPERATE and opponent_move == Move.DEFECT:
#             # Betrayal - amplify negative impact
#             return base_impact * loss_sensitivity * 1.5
#         elif my_move == Move.DEFECT and opponent_move == Move.COOPERATE:
#             # Exploitation - guilt reduces positive impact
#             return base_impact * 0.7
#         elif my_move == Move.COOPERATE and opponent_move == Move.COOPERATE:
#             # Mutual cooperation - positive but not amplified
#             return base_impact
#         else:
#             # Mutual defection - neutral with slight negative
#             return base_impact * 0.8


# # Updated functions for experiment_nodes.py
# async def simulate_baseline_tournament_real(agent1, agent2, rounds, replication_id):
#     """Actually simulate a tournament between baseline agents with error handling"""
    
#     game = PrisonersDilemmaGame()
    
#     try:
#         result = await game.play_tournament(
#             agent1, agent2, rounds, f"baseline_rep_{replication_id}"
#         )
        
#         return {
#             "replication_id": replication_id,
#             "agent1_final_score": result["agent1_final_score"],
#             "agent2_final_score": result["agent2_final_score"],
#             "agent1_cooperation_rate": result["agent1_cooperation_rate"],
#             "agent2_cooperation_rate": result["agent2_cooperation_rate"],
#             "total_rounds": result["total_rounds"],
#             "successful_rounds": result["successful_rounds"],
#             "llm_calls_made": result["llm_calls_made"],
#             "timestamp": result["timestamp"],
#             "detailed_result": result  # Include full result for analysis
#         }
#     except Exception as e:
#         logger.error(f"Baseline tournament {replication_id} failed: {e}")
#         # Return a fallback result
#         return {
#             "replication_id": replication_id,
#             "agent1_final_score": 0,
#             "agent2_final_score": 0,
#             "agent1_cooperation_rate": 0,
#             "agent2_cooperation_rate": 0,
#             "total_rounds": rounds,
#             "successful_rounds": 0,
#             "llm_calls_made": 0,
#             "timestamp": datetime.now().isoformat(),
#             "error": str(e)
#         }

# async def run_generation_interactions_real(population_state, num_interactions):
#     """Actually run interactions within a generation with error handling"""
    
#     game = PrisonersDilemmaGame()
#     population = population_state["population"]
    
#     # Get configuration from experiment parameters
#     experiment_params = population_state.get("experiment_parameters", {})
#     emergent_config = experiment_params.get("emergent_experiment", {})
#     rounds_per_interaction = emergent_config.get("rounds_per_interaction", 100)
    
#     try:
#         result = await game.run_population_interactions(
#             population, num_interactions, rounds_per_interaction
#         )
        
#         # Update population state with evolved agents (already updated in-place)
#         # But we should also update population metrics
#         update_population_metrics_real(population_state)
        
#         return result
#     except Exception as e:
#         logger.error(f"Generation interactions failed: {e}")
#         # Return fallback result
#         return {
#             "total_interactions": num_interactions,
#             "successful_interactions": 0,
#             "avg_cooperation_rate": 0,
#             "trait_changes": 0,
#             "interaction_results": [],
#             "timestamp": datetime.now().isoformat(),
#             "error": str(e)
#         }

# def update_population_metrics_real(population_state):
#     """Update population-level metrics after interactions"""
    
#     population = population_state["population"]
    
#     if not population:
#         return
    
#     try:
#         # Calculate averages with error handling
#         trust_levels = []
#         loss_sensitivities = []
#         total_scores = []
        
#         for agent in population:
#             profile = agent["psychological_profile"]
#             trust_levels.append(profile.trust_level)
#             loss_sensitivities.append(profile.loss_sensitivity)
#             total_scores.append(agent.get("total_score", 0))
        
#         population_state["avg_trust_level"] = sum(trust_levels) / len(trust_levels)
#         population_state["avg_loss_sensitivity"] = sum(loss_sensitivities) / len(loss_sensitivities)
        
#         # Update trait distribution
#         trait_counts = {}
#         for agent in population:
#             trait = agent["psychological_profile"].get_dominant_trait()
#             trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
#         population_state["psychological_distribution"] = trait_counts
#         population_state["dominant_traits"] = [
#             trait for trait, count in trait_counts.items() 
#             if count >= len(population) * 0.1  # At least 10% of population
#         ]
        
#         # Identify successful and struggling agents
#         if total_scores:
#             avg_score = sum(total_scores) / len(total_scores)
#             population_state["successful_agents"] = [
#                 agent["agent_id"] for agent in population 
#                 if agent.get("total_score", 0) > avg_score * 1.2
#             ]
#             population_state["struggling_agents"] = [
#                 agent["agent_id"] for agent in population 
#                 if agent.get("total_score", 0) < avg_score * 0.8
#             ]
#     except Exception as e:
#         logger.error(f"Error updating population metrics: {e}")

# src/game/prisoner_dilemma.py - MEMORY OPTIMIZED VERSION
from typing import Dict, Any, Tuple
from datetime import datetime
import logging
import asyncio
import gc
import os
from ..state.agent_state import AgentState, Move, Memory
from ..graphs.agents.decision_graph import create_agent_decision_graph
from ..graphs.agents.evolution_graph import create_psychological_evolution_graph

# Try to import psutil for resource monitoring, but don't fail if not available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)

class PrisonersDilemmaGame:
    """Implements the actual prisoner's dilemma game logic with memory optimization"""
    
    def __init__(self):
        # FIXED: Add try-catch for graph creation and use fallback
        try:
            self.decision_graph = create_agent_decision_graph()
            self.evolution_graph = create_psychological_evolution_graph()
            self.graphs_available = True
            logger.info("Successfully created LLM-based decision graphs")
        except Exception as e:
            logger.warning(f"Failed to create LLM graphs: {e}")
            logger.warning("Using fallback decision logic")
            self.decision_graph = None
            self.evolution_graph = None
            self.graphs_available = False
        
        self.llm_call_count = 0
        
        # Payoff matrix
        self.payoff_matrix = {
            (Move.COOPERATE, Move.COOPERATE): (3, 3),
            (Move.COOPERATE, Move.DEFECT): (0, 5),
            (Move.DEFECT, Move.COOPERATE): (5, 0),
            (Move.DEFECT, Move.DEFECT): (1, 1)
        }
    
    def _log_system_resources(self, context: str):
        """Log current system resource usage if psutil is available"""
        if not PSUTIL_AVAILABLE:
            logger.debug(f"{context} - LLM calls: {self.llm_call_count}")
            return
            
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
            logger.info(f"{context} - Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%, LLM calls: {self.llm_call_count}")
            
            # MEMORY LEAK WARNING
            if memory_mb > 1000:  # More than 1GB
                logger.warning(f"HIGH MEMORY USAGE: {memory_mb:.1f}MB - potential memory leak")
                
        except Exception as e:
            logger.warning(f"Could not get resource info: {e}")
    
    def _cleanup_agent_memory(self, agent_state: AgentState):
        """Clean up agent memory to prevent leaks"""
        # Limit reasoning chain to last 10 steps
        if len(agent_state["reasoning_chain"]) > 10:
            agent_state["reasoning_chain"] = agent_state["reasoning_chain"][-10:]
        
        # Limit recent memories to last 20
        if len(agent_state["recent_memories"]) > 20:
            agent_state["recent_memories"] = agent_state["recent_memories"][-20:]
        
        # Limit psychological observations to last 10
        if len(agent_state["psychological_observations"]) > 10:
            agent_state["psychological_observations"] = agent_state["psychological_observations"][-10:]
        
        # Limit trauma memories to most significant 10
        trauma_memories = agent_state["psychological_profile"].trauma_memories
        if len(trauma_memories) > 10:
            # Sort by severity and keep top 10
            trauma_memories.sort(key=lambda x: x.get("severity", 0), reverse=True)
            agent_state["psychological_profile"].trauma_memories = trauma_memories[:10]
    
    async def _make_agent_decision_with_fallback(self, agent_state: AgentState) -> AgentState:
        """Make agent decision with LLM graphs or fallback logic - MEMORY OPTIMIZED"""
        
        # Clean up memory before making decision
        self._cleanup_agent_memory(agent_state)
        
        if not self.graphs_available or not self.decision_graph:
            logger.debug(f"Using fallback decision for {agent_state['agent_id']}")
            return self._make_fallback_decision(agent_state)
        
        try:
            # Try LLM-based decision with aggressive timeout
            result = await asyncio.wait_for(
                self.decision_graph.ainvoke(agent_state),
                timeout=15.0  # 15 second timeout
            )
            self.llm_call_count += 1
            
            # Clean up result memory
            self._cleanup_agent_memory(result)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"LLM decision timeout for {agent_state['agent_id']}, using fallback")
            return self._make_fallback_decision(agent_state)
        except Exception as e:
            logger.warning(f"LLM decision error for {agent_state['agent_id']}: {e}, using fallback")
            return self._make_fallback_decision(agent_state)
    
    def _make_fallback_decision(self, agent_state: AgentState) -> AgentState:
        """Make decision using simple psychological rules (no LLM)"""
        
        profile = agent_state["psychological_profile"]
        recent_memories = agent_state["recent_memories"]
        
        # Simple decision logic based on psychology
        cooperation_probability = profile.trust_level
        
        # Adjust based on recent experiences
        if recent_memories:
            recent_betrayals = sum(1 for m in recent_memories[-5:] 
                                 if m.my_move == Move.COOPERATE and m.opponent_move == Move.DEFECT)
            recent_successes = sum(1 for m in recent_memories[-5:] 
                                 if m.my_move == Move.COOPERATE and m.opponent_move == Move.COOPERATE)
            
            # Reduce cooperation after betrayals
            cooperation_probability -= (recent_betrayals * 0.2)
            
            # Increase cooperation after successes
            cooperation_probability += (recent_successes * 0.1)
        
        # Apply loss aversion bias
        if profile.loss_sensitivity > 1.5:
            cooperation_probability *= 0.8  # More cautious
        
        # Make decision
        import random
        decision = "COOPERATE" if random.random() < cooperation_probability else "DEFECT"
        confidence = abs(cooperation_probability - 0.5) + 0.5  # Convert to confidence
        
        # Update agent state (create new copy to avoid mutation)
        result_state = dict(agent_state)
        result_state["current_decision"] = decision
        result_state["decision_confidence"] = min(1.0, max(0.1, confidence))
        
        return result_state
    
    async def _evolve_agent_with_fallback(self, agent_state: AgentState) -> AgentState:
        """Evolve agent psychology with LLM graphs or fallback logic - MEMORY OPTIMIZED"""
        
        # Clean up memory before evolution
        self._cleanup_agent_memory(agent_state)
        
        if not self.graphs_available or not self.evolution_graph:
            return self._evolve_agent_fallback(agent_state)
        
        try:
            # Try LLM-based evolution with timeout
            result = await asyncio.wait_for(
                self.evolution_graph.ainvoke(agent_state),
                timeout=10.0  # 10 second timeout
            )
            
            # Clean up result memory
            self._cleanup_agent_memory(result)
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"LLM evolution timeout for {agent_state['agent_id']}, using fallback")
            return self._evolve_agent_fallback(agent_state)
        except Exception as e:
            logger.warning(f"LLM evolution error for {agent_state['agent_id']}: {e}, using fallback")
            return self._evolve_agent_fallback(agent_state)
    
    def _evolve_agent_fallback(self, agent_state: AgentState) -> AgentState:
        """Simple fallback psychology evolution"""
        
        if not agent_state["recent_memories"]:
            return agent_state
        
        # Make a copy to avoid mutations
        result_state = dict(agent_state)
        profile = result_state["psychological_profile"]
        
        # Simple evolution based on recent experiences
        latest_memory = agent_state["recent_memories"][-1]
        
        # Adjust trust based on latest interaction
        if latest_memory.my_move == Move.COOPERATE and latest_memory.opponent_move == Move.DEFECT:
            # Betrayed - reduce trust
            profile.trust_level = max(0.0, profile.trust_level - 0.1)
            profile.loss_sensitivity = min(3.0, profile.loss_sensitivity + 0.2)
        elif latest_memory.my_move == Move.COOPERATE and latest_memory.opponent_move == Move.COOPERATE:
            # Mutual cooperation - slight trust increase
            profile.trust_level = min(1.0, profile.trust_level + 0.05)
        
        return result_state

    async def play_tournament(
        self, 
        agent1: AgentState, 
        agent2: AgentState, 
        rounds: int,
        tournament_id: str = None
    ) -> Dict[str, Any]:
        """Play a tournament with memory optimization and error handling"""
        
        logger.info(f"Starting {rounds}-round tournament: {agent1['agent_id']} vs {agent2['agent_id']}")
        self._log_system_resources("Tournament start")
        
        agent1_score = 0.0
        agent2_score = 0.0
        agent1_cooperations = 0
        agent2_cooperations = 0
        successful_rounds = 0
        
        for round_num in range(1, rounds + 1):
            try:
                # Log every round for debugging
                self._log_system_resources(f"Round {round_num}")
                
                # MEMORY OPTIMIZATION: Force garbage collection every round
                if round_num % 1 == 0:
                    gc.collect()
                
                # Update round context
                agent1["current_round"] = round_num
                agent2["current_round"] = round_num
                
                agent1["game_context"] = {
                    "opponent_id": agent2["agent_id"],
                    "round_number": round_num,
                    "total_rounds": rounds,
                    "my_current_score": agent1_score,
                    "opponent_current_score": agent2_score
                }
                
                agent2["game_context"] = {
                    "opponent_id": agent1["agent_id"],
                    "round_number": round_num,
                    "total_rounds": rounds,
                    "my_current_score": agent2_score,
                    "opponent_current_score": agent1_score
                }
                
                # FIXED: Get decisions with better error handling and fallback
                logger.info(f"Round {round_num}: Getting agent decisions...")
                
                try:
                    # Use the new fallback decision method
                    agent1_result = await self._make_agent_decision_with_fallback(agent1)
                    agent2_result = await self._make_agent_decision_with_fallback(agent2)
                    
                    logger.info(f"Round {round_num}: Got decisions (total LLM calls: {self.llm_call_count})")
                    
                except Exception as e:
                    logger.error(f"Round {round_num}: Critical decision error: {e}")
                    # Use most basic fallback
                    agent1_result = dict(agent1)
                    agent2_result = dict(agent2)
                    
                    agent1_result["current_decision"] = "DEFECT"  # Conservative
                    agent2_result["current_decision"] = "DEFECT"
                    agent1_result["decision_confidence"] = 0.5
                    agent2_result["decision_confidence"] = 0.5
                
                # Extract moves with safety checks
                move1_str = agent1_result.get("current_decision", "DEFECT")
                move2_str = agent2_result.get("current_decision", "DEFECT")
                
                move1 = Move.COOPERATE if move1_str == "COOPERATE" else Move.DEFECT
                move2 = Move.COOPERATE if move2_str == "COOPERATE" else Move.DEFECT
                
                # Calculate payoffs
                payoff1, payoff2 = self.payoff_matrix[(move1, move2)]
                
                # Update scores
                agent1_score += payoff1
                agent2_score += payoff2
                agent1_result["total_score"] = agent1_score
                agent2_result["total_score"] = agent2_score
                
                # Count cooperations
                if move1 == Move.COOPERATE:
                    agent1_cooperations += 1
                if move2 == Move.COOPERATE:
                    agent2_cooperations += 1
                
                # Create memories
                memory1 = Memory(
                    round_number=round_num,
                    my_move=move1,
                    opponent_move=move2,
                    my_payoff=payoff1,
                    opponent_payoff=payoff2,
                    emotional_impact=self._calculate_emotional_impact(
                        move1, move2, payoff1, agent1_result["psychological_profile"].loss_sensitivity
                    ),
                    timestamp=datetime.now()
                )
                
                memory2 = Memory(
                    round_number=round_num,
                    my_move=move2,
                    opponent_move=move1,
                    my_payoff=payoff2,
                    opponent_payoff=payoff1,
                    emotional_impact=self._calculate_emotional_impact(
                        move2, move1, payoff2, agent2_result["psychological_profile"].loss_sensitivity
                    ),
                    timestamp=datetime.now()
                )
                
                # Add memories to agents (but limit memory size)
                agent1_result["recent_memories"].append(memory1)
                agent2_result["recent_memories"].append(memory2)
                
                # MEMORY OPTIMIZATION: Limit memory size
                if len(agent1_result["recent_memories"]) > 20:
                    agent1_result["recent_memories"] = agent1_result["recent_memories"][-20:]
                if len(agent2_result["recent_memories"]) > 20:
                    agent2_result["recent_memories"] = agent2_result["recent_memories"][-20:]
                
                # FIXED: Apply psychological evolution with fallback (every round for testing)
                try:
                    logger.info(f"Round {round_num}: Applying psychological evolution...")
                    
                    agent1 = await self._evolve_agent_with_fallback(agent1_result)
                    agent2 = await self._evolve_agent_with_fallback(agent2_result)
                    
                    logger.info(f"Round {round_num}: Evolution complete")
                    
                except Exception as e:
                    logger.warning(f"Round {round_num}: Evolution error: {e}")
                    agent1 = agent1_result
                    agent2 = agent2_result
                
                successful_rounds += 1
                
                # Log progress
                logger.info(f"Round {round_num}: {agent1['agent_id']} {move1.value} vs {agent2['agent_id']} {move2.value} "
                           f"(scores: {agent1_score:.1f} vs {agent2_score:.1f})")
                
            except Exception as e:
                logger.error(f"Round {round_num} failed completely: {e}")
                # Continue to next round with current agent states
                continue
        
        # MEMORY OPTIMIZATION: Final cleanup before returning
        self._cleanup_agent_memory(agent1)
        self._cleanup_agent_memory(agent2)
        gc.collect()
        
        # Calculate final statistics
        cooperation_rate1 = agent1_cooperations / successful_rounds if successful_rounds > 0 else 0
        cooperation_rate2 = agent2_cooperations / successful_rounds if successful_rounds > 0 else 0
        
        result = {
            "tournament_id": tournament_id or f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "agent1_id": agent1["agent_id"],
            "agent2_id": agent2["agent_id"],
            "total_rounds": rounds,
            "successful_rounds": successful_rounds,
            "agent1_final_score": agent1_score,
            "agent2_final_score": agent2_score,
            "agent1_cooperation_rate": cooperation_rate1,
            "agent2_cooperation_rate": cooperation_rate2,
            "agent1_final_state": agent1,
            "agent2_final_state": agent2,
            "llm_calls_made": self.llm_call_count,
            "timestamp": datetime.now().isoformat()
        }
        
        self._log_system_resources("Tournament end")
        
        logger.info(f"Tournament complete: {agent1['agent_id']} scored {agent1_score:.1f} ({cooperation_rate1:.1%} coop), "
                   f"{agent2['agent_id']} scored {agent2_score:.1f} ({cooperation_rate2:.1%} coop). "
                   f"Successful rounds: {successful_rounds}/{rounds}, LLM calls: {self.llm_call_count}")
        
        return result
    
    async def run_population_interactions(
        self,
        population: list,
        interactions_per_generation: int,
        rounds_per_interaction: int = 100
    ) -> Dict[str, Any]:
        """Run interactions within a population with memory optimization"""
        
        logger.info(f"Running {interactions_per_generation} interactions for population of {len(population)}")
        
        import random
        interaction_results = []
        total_cooperations = 0
        total_rounds = 0
        
        for interaction_num in range(interactions_per_generation):
            try:
                # MEMORY OPTIMIZATION: Garbage collect between interactions
                if interaction_num % 1 == 0:
                    gc.collect()
                
                # Randomly select two agents
                if len(population) < 2:
                    logger.warning(f"Not enough agents for interaction {interaction_num}")
                    continue
                    
                agent1, agent2 = random.sample(population, 2)
                
                # Clean up agent memory before interaction
                self._cleanup_agent_memory(agent1)
                self._cleanup_agent_memory(agent2)
                
                # Run tournament with error handling
                tournament_result = await self.play_tournament(
                    agent1, agent2, rounds_per_interaction, 
                    f"pop_interaction_{interaction_num}"
                )
                
                interaction_results.append(tournament_result)
                
                # Update population with evolved agents
                for i, pop_agent in enumerate(population):
                    if pop_agent["agent_id"] == agent1["agent_id"]:
                        population[i] = tournament_result["agent1_final_state"]
                    elif pop_agent["agent_id"] == agent2["agent_id"]:
                        population[i] = tournament_result["agent2_final_state"]
                
                # Accumulate cooperation statistics
                total_cooperations += (tournament_result["agent1_cooperation_rate"] + 
                                     tournament_result["agent2_cooperation_rate"]) * rounds_per_interaction
                total_rounds += 2 * rounds_per_interaction
                
                logger.info(f"Completed population interaction {interaction_num + 1}/{interactions_per_generation}")
                
            except Exception as e:
                logger.error(f"Population interaction {interaction_num} failed: {e}")
                continue
        
        avg_cooperation_rate = total_cooperations / total_rounds if total_rounds > 0 else 0
        
        # Count trait changes
        trait_changes = 0
        for agent in population:
            if len(agent.get("reasoning_chain", [])) > 0:
                trait_changes += len([step for step in agent["reasoning_chain"] 
                                    if step.step_type in ["trust_adjustment", "loss_sensitivity_evolution"]])
        
        # MEMORY OPTIMIZATION: Final cleanup
        for agent in population:
            self._cleanup_agent_memory(agent)
        gc.collect()
        
        return {
            "total_interactions": interactions_per_generation,
            "successful_interactions": len(interaction_results),
            "avg_cooperation_rate": avg_cooperation_rate,
            "trait_changes": trait_changes,
            "interaction_results": interaction_results,
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_emotional_impact(self, my_move: Move, opponent_move: Move, 
                                  my_payoff: float, loss_sensitivity: float) -> float:
        """Calculate emotional impact of a game outcome"""
        
        base_impact = my_payoff - 2.0  # 2.0 is neutral expectation
        
        if my_move == Move.COOPERATE and opponent_move == Move.DEFECT:
            # Betrayal - amplify negative impact
            return base_impact * loss_sensitivity * 1.5
        elif my_move == Move.DEFECT and opponent_move == Move.COOPERATE:
            # Exploitation - guilt reduces positive impact
            return base_impact * 0.7
        elif my_move == Move.COOPERATE and opponent_move == Move.COOPERATE:
            # Mutual cooperation - positive but not amplified
            return base_impact
        else:
            # Mutual defection - neutral with slight negative
            return base_impact * 0.8


# Updated functions for experiment_nodes.py remain the same as previous version
async def simulate_baseline_tournament_real(agent1, agent2, rounds, replication_id):
    """Actually simulate a tournament between baseline agents with error handling"""
    
    game = PrisonersDilemmaGame()
    
    try:
        result = await game.play_tournament(
            agent1, agent2, rounds, f"baseline_rep_{replication_id}"
        )
        
        return {
            "replication_id": replication_id,
            "agent1_final_score": result["agent1_final_score"],
            "agent2_final_score": result["agent2_final_score"],
            "agent1_cooperation_rate": result["agent1_cooperation_rate"],
            "agent2_cooperation_rate": result["agent2_cooperation_rate"],
            "total_rounds": result["total_rounds"],
            "successful_rounds": result["successful_rounds"],
            "llm_calls_made": result["llm_calls_made"],
            "timestamp": result["timestamp"],
            "detailed_result": result  # Include full result for analysis
        }
    except Exception as e:
        logger.error(f"Baseline tournament {replication_id} failed: {e}")
        # Return a fallback result
        return {
            "replication_id": replication_id,
            "agent1_final_score": 0,
            "agent2_final_score": 0,
            "agent1_cooperation_rate": 0,
            "agent2_cooperation_rate": 0,
            "total_rounds": rounds,
            "successful_rounds": 0,
            "llm_calls_made": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

async def run_generation_interactions_real(population_state, num_interactions):
    """Actually run interactions within a generation with error handling"""
    
    game = PrisonersDilemmaGame()
    population = population_state["population"]
    
    # Get configuration from experiment parameters
    experiment_params = population_state.get("experiment_parameters", {})
    emergent_config = experiment_params.get("emergent_experiment", {})
    rounds_per_interaction = emergent_config.get("rounds_per_interaction", 100)
    
    try:
        result = await game.run_population_interactions(
            population, num_interactions, rounds_per_interaction
        )
        
        # Update population state with evolved agents (already updated in-place)
        # But we should also update population metrics
        update_population_metrics_real(population_state)
        
        return result
    except Exception as e:
        logger.error(f"Generation interactions failed: {e}")
        # Return fallback result
        return {
            "total_interactions": num_interactions,
            "successful_interactions": 0,
            "avg_cooperation_rate": 0,
            "trait_changes": 0,
            "interaction_results": [],
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

def update_population_metrics_real(population_state):
    """Update population-level metrics after interactions"""
    
    population = population_state["population"]
    
    if not population:
        return
    
    try:
        # Calculate averages with error handling
        trust_levels = []
        loss_sensitivities = []
        total_scores = []
        
        for agent in population:
            profile = agent["psychological_profile"]
            trust_levels.append(profile.trust_level)
            loss_sensitivities.append(profile.loss_sensitivity)
            total_scores.append(agent.get("total_score", 0))
        
        population_state["avg_trust_level"] = sum(trust_levels) / len(trust_levels)
        population_state["avg_loss_sensitivity"] = sum(loss_sensitivities) / len(loss_sensitivities)
        
        # Update trait distribution
        trait_counts = {}
        for agent in population:
            trait = agent["psychological_profile"].get_dominant_trait()
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
        
        population_state["psychological_distribution"] = trait_counts
        population_state["dominant_traits"] = [
            trait for trait, count in trait_counts.items() 
            if count >= len(population) * 0.1  # At least 10% of population
        ]
        
        # Identify successful and struggling agents
        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
            population_state["successful_agents"] = [
                agent["agent_id"] for agent in population 
                if agent.get("total_score", 0) > avg_score * 1.2
            ]
            population_state["struggling_agents"] = [
                agent["agent_id"] for agent in population 
                if agent.get("total_score", 0) < avg_score * 0.8
            ]
    except Exception as e:
        logger.error(f"Error updating population metrics: {e}")