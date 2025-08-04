# from typing import Dict, Any, Tuple
# from datetime import datetime
# import logging
# from ..state.agent_state import AgentState, Move, Memory
# from ..graphs.agents.decision_graph import create_agent_decision_graph
# from ..graphs.agents.evolution_graph import create_psychological_evolution_graph

# logger = logging.getLogger(__name__)

# class PrisonersDilemmaGame:
#     """Implements the actual prisoner's dilemma game logic"""
    
#     def __init__(self):
#         self.decision_graph = create_agent_decision_graph()
#         self.evolution_graph = create_psychological_evolution_graph()
        
#         # Payoff matrix
#         self.payoff_matrix = {
#             (Move.COOPERATE, Move.COOPERATE): (3, 3),
#             (Move.COOPERATE, Move.DEFECT): (0, 5),
#             (Move.DEFECT, Move.COOPERATE): (5, 0),
#             (Move.DEFECT, Move.DEFECT): (1, 1)
#         }
    
#     async def play_tournament(
#         self, 
#         agent1: AgentState, 
#         agent2: AgentState, 
#         rounds: int,
#         tournament_id: str = None
#     ) -> Dict[str, Any]:
#         """Play a full tournament between two agents"""
        
#         logger.info(f"Starting {rounds}-round tournament: {agent1['agent_id']} vs {agent2['agent_id']}")
        
#         agent1_score = 0.0
#         agent2_score = 0.0
#         agent1_cooperations = 0
#         agent2_cooperations = 0
        
#         for round_num in range(1, rounds + 1):
#             # Update round context
#             agent1["current_round"] = round_num
#             agent2["current_round"] = round_num
            
#             agent1["game_context"] = {
#                 "opponent_id": agent2["agent_id"],
#                 "round_number": round_num,
#                 "total_rounds": rounds,
#                 "my_current_score": agent1_score,
#                 "opponent_current_score": agent2_score
#             }
            
#             agent2["game_context"] = {
#                 "opponent_id": agent1["agent_id"],
#                 "round_number": round_num,
#                 "total_rounds": rounds,
#                 "my_current_score": agent2_score,
#                 "opponent_current_score": agent1_score
#             }
            
#             try:
#                 # Get decisions from both agents using their decision graphs
#                 agent1_result = await self.decision_graph.ainvoke(agent1)
#                 agent2_result = await self.decision_graph.ainvoke(agent2)
                
#                 # Extract moves
#                 move1_str = agent1_result["current_decision"]
#                 move2_str = agent2_result["current_decision"]
                
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
                
#                 # Apply psychological evolution after each round
#                 agent1 = await self.evolution_graph.ainvoke(agent1_result)
#                 agent2 = await self.evolution_graph.ainvoke(agent2_result)
                
#                 if round_num % 50 == 0:  # Log progress every 50 rounds
#                     logger.info(f"Round {round_num}: {agent1['agent_id']} {move1.value} vs {agent2['agent_id']} {move2.value}")
                
#             except Exception as e:
#                 logger.error(f"Error in round {round_num}: {e}")
#                 # Continue with next round
#                 continue
        
#         # Calculate final statistics
#         cooperation_rate1 = agent1_cooperations / rounds if rounds > 0 else 0
#         cooperation_rate2 = agent2_cooperations / rounds if rounds > 0 else 0
        
#         result = {
#             "tournament_id": tournament_id or f"tournament_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#             "agent1_id": agent1["agent_id"],
#             "agent2_id": agent2["agent_id"],
#             "total_rounds": rounds,
#             "agent1_final_score": agent1_score,
#             "agent2_final_score": agent2_score,
#             "agent1_cooperation_rate": cooperation_rate1,
#             "agent2_cooperation_rate": cooperation_rate2,
#             "agent1_final_state": agent1,
#             "agent2_final_state": agent2,
#             "timestamp": datetime.now().isoformat()
#         }
        
#         logger.info(f"Tournament complete: {agent1['agent_id']} scored {agent1_score:.1f} ({cooperation_rate1:.1%} coop), "
#                    f"{agent2['agent_id']} scored {agent2_score:.1f} ({cooperation_rate2:.1%} coop)")
        
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
#             # Randomly select two agents
#             agent1, agent2 = random.sample(population, 2)
            
#             # Run tournament
#             tournament_result = await self.play_tournament(
#                 agent1, agent2, rounds_per_interaction, 
#                 f"pop_interaction_{interaction_num}"
#             )
            
#             interaction_results.append(tournament_result)
            
#             # Update population with evolved agents
#             for i, pop_agent in enumerate(population):
#                 if pop_agent["agent_id"] == agent1["agent_id"]:
#                     population[i] = tournament_result["agent1_final_state"]
#                 elif pop_agent["agent_id"] == agent2["agent_id"]:
#                     population[i] = tournament_result["agent2_final_state"]
            
#             # Accumulate cooperation statistics
#             total_cooperations += (tournament_result["agent1_cooperation_rate"] + 
#                                  tournament_result["agent2_cooperation_rate"]) * rounds_per_interaction
#             total_rounds += 2 * rounds_per_interaction
        
#         avg_cooperation_rate = total_cooperations / total_rounds if total_rounds > 0 else 0
        
#         # Count trait changes
#         trait_changes = 0
#         for agent in population:
#             if len(agent.get("reasoning_chain", [])) > 0:
#                 trait_changes += len([step for step in agent["reasoning_chain"] 
#                                     if step.step_type in ["trust_adjustment", "loss_sensitivity_evolution"]])
        
#         return {
#             "total_interactions": interactions_per_generation,
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

# src/game/prisoner_dilemma.py
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
    """Implements the actual prisoner's dilemma game logic with debugging and error handling"""
    
    def __init__(self):
        self.decision_graph = create_agent_decision_graph()
        self.evolution_graph = create_psychological_evolution_graph()
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
        except Exception as e:
            logger.warning(f"Could not get resource info: {e}")
    
    async def play_tournament(
        self, 
        agent1: AgentState, 
        agent2: AgentState, 
        rounds: int,
        tournament_id: str = None
    ) -> Dict[str, Any]:
        """Play a tournament with detailed debugging and error handling"""
        
        logger.info(f"Starting {rounds}-round tournament: {agent1['agent_id']} vs {agent2['agent_id']}")
        self._log_system_resources("Tournament start")
        
        agent1_score = 0.0
        agent2_score = 0.0
        agent1_cooperations = 0
        agent2_cooperations = 0
        successful_rounds = 0
        
        for round_num in range(1, rounds + 1):
            try:
                # Log every few rounds
                if round_num % 5 == 0 or round_num <= 3:
                    self._log_system_resources(f"Round {round_num}")
                
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
                
                # Get decisions with timeout and error handling
                logger.info(f"Round {round_num}: Getting agent decisions...")
                
                try:
                    # Add timeout to prevent hanging
                    agent1_task = asyncio.create_task(self.decision_graph.ainvoke(agent1))
                    agent2_task = asyncio.create_task(self.decision_graph.ainvoke(agent2))
                    
                    # Wait for both with timeout
                    agent1_result, agent2_result = await asyncio.wait_for(
                        asyncio.gather(agent1_task, agent2_task),
                        timeout=60.0  # 60 second timeout
                    )
                    
                    self.llm_call_count += 2
                    logger.info(f"Round {round_num}: Got decisions (total LLM calls: {self.llm_call_count})")
                    
                except asyncio.TimeoutError:
                    logger.error(f"Round {round_num}: Timeout getting decisions, using fallback")
                    # Use fallback decisions based on psychology
                    agent1_result = agent1.copy()
                    agent2_result = agent2.copy()
                    
                    agent1_result["current_decision"] = "COOPERATE" if agent1["psychological_profile"].trust_level > 0.5 else "DEFECT"
                    agent2_result["current_decision"] = "COOPERATE" if agent2["psychological_profile"].trust_level > 0.5 else "DEFECT"
                
                except Exception as e:
                    logger.error(f"Round {round_num}: Error getting decisions: {e}")
                    # Use conservative fallback
                    agent1_result = agent1.copy()
                    agent2_result = agent2.copy()
                    agent1_result["current_decision"] = "DEFECT"
                    agent2_result["current_decision"] = "DEFECT"
                
                # Extract moves
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
                
                # Add memories to agents
                agent1_result["recent_memories"].append(memory1)
                agent2_result["recent_memories"].append(memory2)
                
                # Apply psychological evolution (every 5 rounds to reduce load)
                if round_num % 5 == 0 or round_num == rounds:  # Every 5 rounds or final round
                    try:
                        logger.info(f"Round {round_num}: Applying psychological evolution...")
                        
                        evolution_task1 = asyncio.create_task(self.evolution_graph.ainvoke(agent1_result))
                        evolution_task2 = asyncio.create_task(self.evolution_graph.ainvoke(agent2_result))
                        
                        agent1, agent2 = await asyncio.wait_for(
                            asyncio.gather(evolution_task1, evolution_task2),
                            timeout=30.0  # 30 second timeout for evolution
                        )
                        
                        logger.info(f"Round {round_num}: Evolution complete")
                        
                    except asyncio.TimeoutError:
                        logger.warning(f"Round {round_num}: Evolution timeout, skipping")
                        agent1 = agent1_result
                        agent2 = agent2_result
                    except Exception as e:
                        logger.warning(f"Round {round_num}: Evolution error: {e}")
                        agent1 = agent1_result
                        agent2 = agent2_result
                else:
                    # Just update agent states without evolution
                    agent1 = agent1_result
                    agent2 = agent2_result
                
                successful_rounds += 1
                
                # Force garbage collection every 5 rounds
                if round_num % 5 == 0:
                    gc.collect()
                
                # Log progress
                if round_num <= 5 or round_num % 5 == 0:
                    logger.info(f"Round {round_num}: {agent1['agent_id']} {move1.value} vs {agent2['agent_id']} {move2.value} "
                               f"(scores: {agent1_score:.1f} vs {agent2_score:.1f})")
                
            except Exception as e:
                logger.error(f"Round {round_num} failed completely: {e}")
                # Continue to next round with current agent states
                continue
        
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
        """Run interactions within a population"""
        
        logger.info(f"Running {interactions_per_generation} interactions for population of {len(population)}")
        
        import random
        interaction_results = []
        total_cooperations = 0
        total_rounds = 0
        
        for interaction_num in range(interactions_per_generation):
            try:
                # Randomly select two agents
                if len(population) < 2:
                    logger.warning(f"Not enough agents for interaction {interaction_num}")
                    continue
                    
                agent1, agent2 = random.sample(population, 2)
                
                # Run tournament
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


# Functions to replace the placeholders in experiment_nodes.py
async def simulate_baseline_tournament_real(agent1, agent2, rounds, replication_id):
    """Actually simulate a tournament between baseline agents"""
    
    game = PrisonersDilemmaGame()
    
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

async def run_generation_interactions_real(population_state, num_interactions):
    """Actually run interactions within a generation"""
    
    game = PrisonersDilemmaGame()
    population = population_state["population"]
    
    # Get configuration from experiment parameters
    experiment_params = population_state.get("experiment_parameters", {})
    emergent_config = experiment_params.get("emergent_experiment", {})
    rounds_per_interaction = emergent_config.get("rounds_per_interaction", 100)
    
    result = await game.run_population_interactions(
        population, num_interactions, rounds_per_interaction
    )
    
    # Update population state with evolved agents (already updated in-place)
    # But we should also update population metrics
    update_population_metrics_real(population_state)
    
    return result

def update_population_metrics_real(population_state):
    """Update population-level metrics after interactions"""
    
    population = population_state["population"]
    
    if not population:
        return
    
    # Calculate averages
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