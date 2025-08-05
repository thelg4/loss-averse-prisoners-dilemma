# #!/usr/bin/env python3
# """
# Detailed Agent History Inspector - Extract everything that happened to each agent
# """

# import sqlite3
# import pickle
# import json
# import pandas as pd
# from pathlib import Path
# from datetime import datetime
# import sys
# import zlib
# import msgpack
# from typing import Dict, List, Any

# def try_multiple_unpickling_methods(data):
#     """Try different methods to unpickle the data"""
    
#     methods = [
#         ("Direct pickle.loads", lambda d: pickle.loads(d)),
#         ("Pickle with protocol 0", lambda d: pickle.loads(d, encoding='latin-1')),
#         ("Pickle with errors='ignore'", lambda d: pickle.loads(d, errors='ignore')),
#         ("Zlib decompress then pickle", lambda d: pickle.loads(zlib.decompress(d))),
#         ("Msgpack unpack", lambda d: msgpack.unpackb(d, raw=False, strict_map_key=False)),
#     ]
    
#     for method_name, method_func in methods:
#         try:
#             result = method_func(data)
#             print(f"✅ SUCCESS with {method_name}")
#             return result
#         except Exception as e:
#             print(f"❌ {method_name} failed: {e}")
    
#     return None

# def extract_all_agent_data(db_path: str) -> Dict[str, Any]:
#     """Extract comprehensive agent data from experiment database with robust error handling"""
    
#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()
        
#         # First, let's see what tables we have
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#         tables = cursor.fetchall()
#         print(f"Available tables: {[table[0] for table in tables]}")
        
#         # Try to get data from different possible sources
#         experiment_state = None
        
#         # Method 1: Try checkpoints table
#         if any('checkpoints' in table[0] for table in tables):
#             print("Trying checkpoints table...")
#             experiment_state = try_extract_from_checkpoints(cursor)
        
#         # Method 2: Try writes table (LangGraph's new format)
#         if not experiment_state and any('writes' in table[0] for table in tables):
#             print("Trying writes table...")
#             experiment_state = try_extract_from_writes(cursor)
        
#         # Method 3: Try other tables
#         if not experiment_state:
#             print("Trying alternative extraction methods...")
#             experiment_state = try_alternative_extraction(cursor, tables)
        
#         conn.close()
        
#         if experiment_state:
#             print("✅ Successfully extracted experiment data")
#             return experiment_state
#         else:
#             print("❌ Could not extract experiment data from any source")
#             return {}
        
#     except Exception as e:
#         print(f"Error reading database: {e}")
#         import traceback
#         traceback.print_exc()
#         return {}

# def try_extract_from_checkpoints(cursor):
#     """Try to extract data from checkpoints table"""
    
#     try:
#         # Get the latest checkpoint with all data
#         cursor.execute("""
#             SELECT checkpoint 
#             FROM checkpoints 
#             ORDER BY checkpoint_id DESC 
#             LIMIT 1
#         """)
        
#         row = cursor.fetchone()
#         if not row:
#             print("No checkpoint data found")
#             return None
        
#         print(f"Found checkpoint data ({len(row[0])} bytes)")
        
#         # Try multiple unpickling methods
#         checkpoint_data = try_multiple_unpickling_methods(row[0])
        
#         if not checkpoint_data:
#             return None
        
#         if isinstance(checkpoint_data, dict) and 'channel_values' in checkpoint_data:
#             return checkpoint_data['channel_values']
#         else:
#             return checkpoint_data
            
#     except Exception as e:
#         print(f"Error extracting from checkpoints: {e}")
#         return None

# def try_extract_from_writes(cursor):
#     """Try to extract data from writes table (LangGraph's new format)"""
    
#     try:
#         # Get all writes grouped by checkpoint
#         cursor.execute("""
#             SELECT checkpoint_id, channel, value, type 
#             FROM writes 
#             ORDER BY checkpoint_id DESC, idx
#         """)
        
#         writes = cursor.fetchall()
        
#         if not writes:
#             print("No writes data found")
#             return None
        
#         print(f"Found {len(writes)} write entries")
        
#         # Group by checkpoint and get the latest one
#         checkpoint_writes = {}
#         for checkpoint_id, channel, value, data_type in writes:
#             if checkpoint_id not in checkpoint_writes:
#                 checkpoint_writes[checkpoint_id] = {}
            
#             # Decode the value based on type
#             decoded_value = None
#             if data_type == 'msgpack':
#                 decoded_value = try_multiple_unpickling_methods(value)
#             elif data_type == 'pickle':
#                 decoded_value = try_multiple_unpickling_methods(value)
#             else:
#                 try:
#                     decoded_value = json.loads(value.decode('utf-8') if isinstance(value, bytes) else value)
#                 except:
#                     decoded_value = str(value)
            
#             if decoded_value is not None:
#                 checkpoint_writes[checkpoint_id][channel] = decoded_value
        
#         # Get the latest checkpoint
#         if checkpoint_writes:
#             latest_checkpoint = max(checkpoint_writes.keys())
#             print(f"Using latest checkpoint: {latest_checkpoint}")
#             return checkpoint_writes[latest_checkpoint]
        
#         return None
        
#     except Exception as e:
#         print(f"Error extracting from writes: {e}")
#         return None

# def try_alternative_extraction(cursor, tables):
#     """Try alternative extraction methods"""
    
#     try:
#         # Look for any table that might contain experiment data
#         for table_info in tables:
#             table_name = table_info[0]
#             print(f"Examining table: {table_name}")
            
#             # Get table schema
#             cursor.execute(f"PRAGMA table_info({table_name})")
#             columns = cursor.fetchall()
#             print(f"  Columns: {[col[1] for col in columns]}")
            
#             # Look for blob columns that might contain pickled data
#             blob_columns = [col[1] for col in columns if col[2].upper() in ['BLOB', 'TEXT']]
            
#             if blob_columns:
#                 for blob_col in blob_columns:
#                     try:
#                         cursor.execute(f"SELECT {blob_col} FROM {table_name} LIMIT 1")
#                         row = cursor.fetchone()
                        
#                         if row and row[0]:
#                             print(f"  Trying to decode {blob_col} column...")
#                             data = try_multiple_unpickling_methods(row[0])
#                             if data:
#                                 print(f"  ✅ Successfully decoded {blob_col} from {table_name}")
#                                 return data
#                     except Exception as e:
#                         print(f"  ❌ Failed to decode {blob_col}: {e}")
        
#         return None
        
#     except Exception as e:
#         print(f"Error in alternative extraction: {e}")
#         return None

# def extract_agent_histories(experiment_state: Dict[str, Any]) -> Dict[str, Dict]:
#     """Extract detailed history for each agent with robust error handling"""
    
#     agent_histories = {}
    
#     # Try different ways to get population data
#     population = None
    
#     # Method 1: Direct population_state access
#     if 'population_state' in experiment_state:
#         population_state = experiment_state['population_state']
#         if isinstance(population_state, dict):
#             population = population_state.get('population', [])
    
#     # Method 2: Direct population access
#     if not population and 'population' in experiment_state:
#         population = experiment_state['population']
    
#     # Method 3: Look for population in any nested structure
#     if not population:
#         population = find_population_in_nested_dict(experiment_state)
    
#     if not population:
#         print("Could not find population data in experiment state")
#         print(f"Available keys: {list(experiment_state.keys()) if isinstance(experiment_state, dict) else 'Not a dict'}")
#         return {}
    
#     print(f"Found {len(population)} agents in final population")
    
#     for i, agent in enumerate(population):
#         try:
#             # Handle different agent data formats
#             agent_data = agent
#             if hasattr(agent, '__dict__'):
#                 agent_data = agent.__dict__
#             elif hasattr(agent, 'dict'):
#                 agent_data = agent.dict()
            
#             agent_id = agent_data.get('agent_id', f'agent_{i:03d}')
            
#             # Extract complete agent history with error handling
#             agent_history = {
#                 'agent_id': agent_id,
#                 'agent_type': safe_get(agent_data, 'agent_type', 'unknown'),
#                 'final_psychological_profile': extract_psychological_profile(agent_data),
#                 'total_score': safe_get(agent_data, 'total_score', 0),
#                 'recovery_progress': safe_get(agent_data, 'recovery_progress', 1.0),
#                 'complete_reasoning_chain': safe_get(agent_data, 'reasoning_chain', []),
#                 'all_memories': safe_get(agent_data, 'recent_memories', []),
#                 'trauma_triggers': safe_get(agent_data, 'trauma_triggers', []),
#                 'psychological_observations': safe_get(agent_data, 'psychological_observations', []),
#                 'decision_history': extract_decision_history(agent_data),
#                 'psychological_evolution': extract_psychological_evolution(agent_data),
#                 'interaction_summary': summarize_interactions(agent_data),
#                 'trauma_analysis': analyze_trauma_patterns(agent_data),
#                 'learning_progression': extract_learning_progression(agent_data)
#             }
            
#             agent_histories[agent_id] = agent_history
            
#         except Exception as e:
#             print(f"Error processing agent {i}: {e}")
#             continue
    
#     return agent_histories

# def find_population_in_nested_dict(data, path=""):
#     """Recursively search for population data in nested structures"""
    
#     if isinstance(data, dict):
#         # Direct check for population
#         if 'population' in data and isinstance(data['population'], list):
#             print(f"Found population at path: {path}/population")
#             return data['population']
        
#         # Recursive search
#         for key, value in data.items():
#             result = find_population_in_nested_dict(value, f"{path}/{key}")
#             if result:
#                 return result
    
#     elif isinstance(data, list) and len(data) > 0:
#         # Check if this looks like a population (list of agent-like objects)
#         first_item = data[0]
#         if isinstance(first_item, (dict, object)) and (
#             (isinstance(first_item, dict) and 'agent_id' in first_item) or
#             (hasattr(first_item, 'agent_id'))
#         ):
#             print(f"Found population-like list at path: {path}")
#             return data
    
#     return None

# def safe_get(data, key, default=None):
#     """Safely get a value from data that might be dict or object"""
    
#     if isinstance(data, dict):
#         return data.get(key, default)
#     elif hasattr(data, key):
#         return getattr(data, key, default)
#     else:
#         return default

# def extract_psychological_profile(agent_data):
#     """Extract psychological profile with error handling"""
    
#     profile = safe_get(agent_data, 'psychological_profile', {})
    
#     if hasattr(profile, '__dict__'):
#         profile = profile.__dict__
#     elif hasattr(profile, 'dict'):
#         profile = profile.dict()
    
#     return profile

# def extract_decision_history(agent_data: Dict) -> List[Dict]:
#     """Extract all decisions made by this agent"""
#     decisions = []
    
#     reasoning_chain = safe_get(agent_data, 'reasoning_chain', [])
    
#     for step in reasoning_chain:
#         try:
#             step_data = step
#             if hasattr(step, '__dict__'):
#                 step_data = step.__dict__
#             elif hasattr(step, 'dict'):
#                 step_data = step.dict()
            
#             step_type = safe_get(step_data, 'step_type', '')
            
#             if step_type in ['final_decision', 'bias_application']:
#                 decision_info = {
#                     'timestamp': safe_get(step_data, 'timestamp', 'unknown'),
#                     'step_type': step_type,
#                     'content': safe_get(step_data, 'content', ''),
#                     'confidence': safe_get(step_data, 'confidence', 0.5),
#                     'psychological_insight': safe_get(step_data, 'psychological_insight', None)
#                 }
                
#                 # Convert timestamp to string if it's a datetime object
#                 if hasattr(decision_info['timestamp'], 'isoformat'):
#                     decision_info['timestamp'] = decision_info['timestamp'].isoformat()
                
#                 decisions.append(decision_info)
#         except Exception as e:
#             print(f"Error processing reasoning step: {e}")
#             continue
    
#     return decisions

# def extract_psychological_evolution(agent_data: Dict) -> Dict:
#     """Track how agent's psychology evolved over time"""
    
#     profile = extract_psychological_profile(agent_data)
#     reasoning_chain = safe_get(agent_data, 'reasoning_chain', [])
    
#     evolution_events = []
    
#     for step in reasoning_chain:
#         try:
#             step_data = step
#             if hasattr(step, '__dict__'):
#                 step_data = step.__dict__
#             elif hasattr(step, 'dict'):
#                 step_data = step.dict()
            
#             step_type = safe_get(step_data, 'step_type', '')
            
#             if step_type in ['trust_adjustment', 'loss_sensitivity_evolution', 'trauma_processing', 'recovery_assessment']:
#                 event = {
#                     'timestamp': safe_get(step_data, 'timestamp', 'unknown'),
#                     'event_type': step_type,
#                     'description': safe_get(step_data, 'content', ''),
#                     'psychological_insight': safe_get(step_data, 'psychological_insight', None)
#                 }
                
#                 # Convert timestamp to string if it's a datetime object
#                 if hasattr(event['timestamp'], 'isoformat'):
#                     event['timestamp'] = event['timestamp'].isoformat()
                
#                 evolution_events.append(event)
#         except Exception as e:
#             print(f"Error processing evolution step: {e}")
#             continue
    
#     # Get dominant trait safely
#     dominant_trait = 'unknown'
#     if hasattr(profile, 'get_dominant_trait'):
#         try:
#             dominant_trait = profile.get_dominant_trait()
#         except:
#             pass
#     elif isinstance(profile, dict):
#         # Try to determine dominant trait from available data
#         trust_level = profile.get('trust_level', 0.5)
#         loss_sensitivity = profile.get('loss_sensitivity', 1.0)
        
#         if loss_sensitivity > 2.0 and trust_level < 0.3:
#             dominant_trait = "traumatized_paranoid"
#         elif loss_sensitivity > 1.8:
#             dominant_trait = "loss_averse"
#         elif trust_level < 0.2:
#             dominant_trait = "paranoid"
#         elif trust_level > 0.8:
#             dominant_trait = "trusting"
#         else:
#             dominant_trait = "balanced"
    
#     return {
#         'final_trust_level': safe_get(profile, 'trust_level', 0.5),
#         'final_loss_sensitivity': safe_get(profile, 'loss_sensitivity', 1.0),
#         'final_emotional_state': safe_get(profile, 'emotional_state', 'unknown'),
#         'final_internal_narrative': safe_get(profile, 'internal_narrative', ''),
#         'learned_heuristics': safe_get(profile, 'learned_heuristics', []),
#         'dominant_trait': dominant_trait,
#         'evolution_events': evolution_events,
#         'trauma_count': len(safe_get(profile, 'trauma_memories', []))
#     }

# def summarize_interactions(agent_data: Dict) -> Dict:
#     """Summarize all interactions this agent had"""
    
#     memories = safe_get(agent_data, 'recent_memories', [])
    
#     if not memories:
#         return {'total_interactions': 0}
    
#     cooperations = 0
#     defections = 0
#     betrayals = 0
#     mutual_cooperations = 0
#     total_payoff = 0
#     emotional_impacts = []
    
#     for memory in memories:
#         try:
#             memory_data = memory
#             if hasattr(memory, '__dict__'):
#                 memory_data = memory.__dict__
#             elif hasattr(memory, 'dict'):
#                 memory_data = memory.dict()
            
#             # Get move data safely
#             my_move = safe_get(memory_data, 'my_move', None)
#             opponent_move = safe_get(memory_data, 'opponent_move', None)
            
#             # Handle enum values
#             if hasattr(my_move, 'value'):
#                 my_move = my_move.value
#             if hasattr(opponent_move, 'value'):
#                 opponent_move = opponent_move.value
            
#             # Count moves
#             if my_move == 'cooperate':
#                 cooperations += 1
#                 if opponent_move == 'defect':
#                     betrayals += 1
#                 elif opponent_move == 'cooperate':
#                     mutual_cooperations += 1
#             elif my_move == 'defect':
#                 defections += 1
            
#             # Sum payoffs
#             payoff = safe_get(memory_data, 'my_payoff', 0)
#             total_payoff += payoff
            
#             # Collect emotional impacts
#             emotional_impact = safe_get(memory_data, 'emotional_impact', 0)
#             emotional_impacts.append(emotional_impact)
            
#         except Exception as e:
#             print(f"Error processing memory: {e}")
#             continue
    
#     avg_emotional_impact = sum(emotional_impacts) / len(emotional_impacts) if emotional_impacts else 0
    
#     return {
#         'total_interactions': len(memories),
#         'cooperations': cooperations,
#         'defections': defections,
#         'cooperation_rate': cooperations / len(memories) if memories else 0,
#         'betrayals_experienced': betrayals,
#         'mutual_cooperations': mutual_cooperations,
#         'total_payoff': total_payoff,
#         'average_payoff': total_payoff / len(memories) if memories else 0,
#         'average_emotional_impact': avg_emotional_impact,
#         'most_painful_experience': min(emotional_impacts) if emotional_impacts else 0,
#         'most_positive_experience': max(emotional_impacts) if emotional_impacts else 0
#     }

# def analyze_trauma_patterns(agent_data: Dict) -> Dict:
#     """Analyze trauma patterns and recovery"""
    
#     profile = extract_psychological_profile(agent_data)
#     trauma_memories = safe_get(profile, 'trauma_memories', [])
    
#     if not trauma_memories:
#         return {'trauma_count': 0, 'analysis': 'No significant traumas recorded'}
    
#     trauma_types = {}
#     total_severity = 0
    
#     for trauma in trauma_memories:
#         try:
#             trauma_data = trauma
#             if hasattr(trauma, '__dict__'):
#                 trauma_data = trauma.__dict__
#             elif hasattr(trauma, 'dict'):
#                 trauma_data = trauma.dict()
            
#             trauma_type = safe_get(trauma_data, 'trauma_type', 'unknown')
#             severity = safe_get(trauma_data, 'severity', 0)
            
#             trauma_types[trauma_type] = trauma_types.get(trauma_type, 0) + 1
#             total_severity += severity
#         except Exception as e:
#             print(f"Error processing trauma: {e}")
#             continue
    
#     return {
#         'trauma_count': len(trauma_memories),
#         'trauma_types': trauma_types,
#         'most_common_trauma': max(trauma_types.items(), key=lambda x: x[1])[0] if trauma_types else None,
#         'total_trauma_severity': total_severity,
#         'average_trauma_severity': total_severity / len(trauma_memories) if trauma_memories else 0,
#         'recovery_progress': safe_get(agent_data, 'recovery_progress', 1.0),
#         'current_trauma_load': sum(safe_get(t, 'severity', 0) for t in trauma_memories[-5:])  # Recent traumas
#     }

# def extract_learning_progression(agent_data: Dict) -> Dict:
#     """Track how agent learned and adapted over time"""
    
#     reasoning_chain = safe_get(agent_data, 'reasoning_chain', [])
    
#     learning_events = []
    
#     for step in reasoning_chain:
#         try:
#             step_data = step
#             if hasattr(step, '__dict__'):
#                 step_data = step.__dict__
#             elif hasattr(step, 'dict'):
#                 step_data = step.dict()
            
#             step_type = safe_get(step_data, 'step_type', '')
            
#             if 'learning' in step_type.lower() or 'heuristic' in step_type.lower():
#                 learning_events.append({
#                     'timestamp': safe_get(step_data, 'timestamp', 'unknown'),
#                     'learning_type': step_type,
#                     'content': safe_get(step_data, 'content', '')
#                 })
#         except Exception as e:
#             print(f"Error processing learning step: {e}")
#             continue
    
#     profile = extract_psychological_profile(agent_data)
#     learned_heuristics = safe_get(profile, 'learned_heuristics', [])
    
#     return {
#         'total_learning_events': len(learning_events),
#         'learned_heuristics': learned_heuristics,
#         'learning_timeline': learning_events,
#         'adaptation_rate': safe_get(profile, 'adaptation_rate', 0.1),
#         'learning_effectiveness': len(learned_heuristics) / max(1, len(learning_events))
#     }

# def save_agent_histories(agent_histories: Dict, output_dir: str = "agent_analysis"):
#     """Save detailed agent histories to files"""
    
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # Save individual agent files
#     for agent_id, history in agent_histories.items():
#         agent_file = output_path / f"{agent_id}_complete_history.json"
        
#         # Make serializable
#         serializable_history = make_serializable(history)
        
#         with open(agent_file, 'w') as f:
#             json.dump(serializable_history, f, indent=2)
        
#         print(f"Saved detailed history for {agent_id}")
    
#     # Save summary analysis
#     summary = create_population_summary(agent_histories)
#     summary_file = output_path / "population_summary.json"
    
#     with open(summary_file, 'w') as f:
#         json.dump(make_serializable(summary), f, indent=2)
    
#     # Create CSV for easy analysis
#     create_agent_csv(agent_histories, output_path / "agents_summary.csv")
    
#     print(f"\nAll analysis saved to {output_path}/")
#     print(f"- Individual agent histories: {len(agent_histories)} files")
#     print(f"- Population summary: population_summary.json")
#     print(f"- CSV for analysis: agents_summary.csv")

# def create_population_summary(agent_histories: Dict) -> Dict:
#     """Create population-level summary"""
    
#     total_agents = len(agent_histories)
    
#     # Aggregate statistics with safe extraction
#     trust_levels = []
#     loss_sensitivities = []
#     cooperation_rates = []
#     trauma_counts = []
    
#     for h in agent_histories.values():
#         # Safely extract psychological evolution data
#         psych_evo = h.get('psychological_evolution', {})
#         trust_levels.append(psych_evo.get('final_trust_level', 0.5))
#         loss_sensitivities.append(psych_evo.get('final_loss_sensitivity', 1.0))
        
#         # Safely extract interaction summary data
#         interaction_summary = h.get('interaction_summary', {})
#         cooperation_rates.append(interaction_summary.get('cooperation_rate', 0.0))
        
#         # Safely extract trauma analysis data
#         trauma_analysis = h.get('trauma_analysis', {})
#         trauma_counts.append(trauma_analysis.get('trauma_count', 0))
    
#     return {
#         'experiment_summary': {
#             'total_agents': total_agents,
#             'analysis_timestamp': datetime.now().isoformat()
#         },
#         'population_psychology': {
#             'average_trust_level': sum(trust_levels) / len(trust_levels) if trust_levels else 0,
#             'average_loss_sensitivity': sum(loss_sensitivities) / len(loss_sensitivities) if loss_sensitivities else 0,
#             'trust_level_range': [min(trust_levels), max(trust_levels)] if trust_levels else [0, 0],
#             'loss_sensitivity_range': [min(loss_sensitivities), max(loss_sensitivities)] if loss_sensitivities else [0, 0]
#         },
#         'behavioral_patterns': {
#             'average_cooperation_rate': sum(cooperation_rates) / len(cooperation_rates) if cooperation_rates else 0,
#             'cooperation_rate_range': [min(cooperation_rates), max(cooperation_rates)] if cooperation_rates else [0, 0],
#             'agents_with_trauma': sum(1 for t in trauma_counts if t > 0),
#             'average_trauma_count': sum(trauma_counts) / len(trauma_counts) if trauma_counts else 0
#         },
#         'agent_types': {
#             agent_type: sum(1 for h in agent_histories.values() if h['agent_type'] == agent_type)
#             for agent_type in set(h['agent_type'] for h in agent_histories.values())
#         }
#     }

# def create_agent_csv(agent_histories: Dict, csv_path: Path):
#     """Create CSV file for easy analysis in Excel/pandas"""
    
#     rows = []
    
#     for agent_id, history in agent_histories.items():
#         # Safely extract data with defaults
#         interaction = history.get('interaction_summary', {})
#         trauma = history.get('trauma_analysis', {})
#         evolution = history.get('psychological_evolution', {})
        
#         row = {
#             'agent_id': agent_id,
#             'agent_type': history.get('agent_type', 'unknown'),
#             'final_trust_level': evolution.get('final_trust_level', 0.5),
#             'final_loss_sensitivity': evolution.get('final_loss_sensitivity', 1.0),
#             'dominant_trait': evolution.get('dominant_trait', 'unknown'),
#             'emotional_state': evolution.get('final_emotional_state', 'unknown'),
#             'total_score': history.get('total_score', 0),
#             'total_interactions': interaction.get('total_interactions', 0),
#             'cooperation_rate': interaction.get('cooperation_rate', 0.0),
#             'betrayals_experienced': interaction.get('betrayals_experienced', 0),
#             'mutual_cooperations': interaction.get('mutual_cooperations', 0),
#             'average_payoff': interaction.get('average_payoff', 0.0),
#             'average_emotional_impact': interaction.get('average_emotional_impact', 0.0),
#             'trauma_count': trauma.get('trauma_count', 0),
#             'recovery_progress': history.get('recovery_progress', 1.0),
#             'learned_heuristics_count': len(evolution.get('learned_heuristics', [])),
#             'total_reasoning_steps': len(history.get('complete_reasoning_chain', [])),
#             'psychological_evolution_events': len(evolution.get('evolution_events', []))
#         }
        
#         rows.append(row)
    
#     df = pd.DataFrame(rows)
#     df.to_csv(csv_path, index=False)

# def make_serializable(obj):
#     """Convert objects to JSON-serializable format"""
#     if isinstance(obj, datetime):
#         return obj.isoformat()
#     elif isinstance(obj, dict):
#         return {k: make_serializable(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [make_serializable(item) for item in obj]
#     elif hasattr(obj, '__dict__'):
#         if hasattr(obj, 'dict'):  # Pydantic model
#             return make_serializable(obj.dict())
#         else:
#             return make_serializable(obj.__dict__)
#     elif hasattr(obj, 'value'):  # Enum
#         return obj.value
#     elif isinstance(obj, (str, int, float, bool, type(None))):
#         return obj
#     else:
#         return str(obj)

# def main():
#     """Main function to extract and analyze all agent data"""
    
#     # Find most recent experiment database
#     current_dir = Path(".")
#     db_files = []
    
#     for db_file in current_dir.glob("experiment_*.db"):
#         mod_time = db_file.stat().st_mtime
#         db_files.append((db_file, mod_time))
    
#     if not db_files:
#         print("No experiment databases found")
#         return
    
#     # Use most recent database
#     db_files.sort(key=lambda x: x[1], reverse=True)
#     most_recent_db = db_files[0][0]
    
#     print(f"Analyzing experiment database: {most_recent_db}")
#     print("=" * 60)
    
#     # Extract all experiment data
#     print("Extracting experiment data...")
#     experiment_state = extract_all_agent_data(str(most_recent_db))
    
#     if not experiment_state:
#         print("Failed to extract experiment data")
#         return
    
#     # Extract detailed agent histories
#     print("Extracting agent histories...")
#     agent_histories = extract_agent_histories(experiment_state)
    
#     if not agent_histories:
#         print("No agent histories found")
#         return
    
#     print(f"Extracted detailed histories for {len(agent_histories)} agents")
    
#     # Save all analysis
#     print("Saving analysis files...")
#     save_agent_histories(agent_histories)
    
#     # Print quick summary
#     print("\n" + "=" * 60)
#     print("QUICK SUMMARY")
#     print("=" * 60)
    
#     for agent_id, history in list(agent_histories.items())[:3]:  # Show first 3 agents
#         print(f"\n{agent_id}:")
#         print(f"  Type: {history.get('agent_type', 'unknown')}")
        
#         # Safely get psychological evolution data
#         psych_evo = history.get('psychological_evolution', {})
#         print(f"  Final Trust: {psych_evo.get('final_trust_level', 0.5):.3f}")
#         print(f"  Final Loss Sensitivity: {psych_evo.get('final_loss_sensitivity', 1.0):.3f}")
        
#         # Safely get interaction summary data
#         interaction = history.get('interaction_summary', {})
#         print(f"  Cooperation Rate: {interaction.get('cooperation_rate', 0.0):.1%}")
        
#         # Safely get trauma analysis data
#         trauma = history.get('trauma_analysis', {})
#         print(f"  Traumas: {trauma.get('trauma_count', 0)}")
        
#         print(f"  Total Score: {history.get('total_score', 0)}")
#         print(f"  Reasoning Steps: {len(history.get('complete_reasoning_chain', []))}")
    
#     if len(agent_histories) > 3:
#         print(f"\n... and {len(agent_histories) - 3} more agents")
    
#     print(f"\nDetailed analysis saved to ./agent_analysis/")

# if __name__ == "__main__":
#     try:
#         import pandas as pd
#     except ImportError:
#         print("pandas not installed. Installing...")
#         import subprocess
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
#         import pandas as pd
    
#     try:
#         import msgpack
#     except ImportError:
#         print("msgpack not installed. Installing...")
#         import subprocess
#         subprocess.check_call([sys.executable, "-m", "pip", "install", "msgpack"])
#         import msgpack
    
#     main()

#!/usr/bin/env python3
"""
Fixed Detailed Agent Inspector - Using Enhanced Decoding
"""

import sqlite3
import json
import pandas as pd
import msgpack
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict, List, Any

def decode_msgpack_with_recursive_exttype(data):
    """Decode msgpack with recursive ExtType handling - SAME AS ENHANCED EXTRACTOR"""
    
    def decode_hook(code, data):
        """Custom decoder for all ExtType objects"""
        if code == 5:  # Pydantic model
            try:
                decoded = msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
                if isinstance(decoded, list) and len(decoded) >= 3:
                    return decoded[2] if isinstance(decoded[2], dict) else decoded
                return decoded
            except:
                return {"pydantic_decode_error": True}
        
        elif code == 0:  # Enum (Move)
            try:
                decoded = msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
                if isinstance(decoded, list) and len(decoded) >= 3:
                    return decoded[2]  # Return just 'cooperate' or 'defect'
                return decoded
            except:
                return {"enum_decode_error": True}
        
        return msgpack.ExtType(code, data)
    
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
    except Exception as e:
        print(f"Failed to decode msgpack: {e}")
        return None

def recursively_decode_exttype(obj):
    """Recursively decode any remaining ExtType objects - SAME AS ENHANCED EXTRACTOR"""
    
    if isinstance(obj, msgpack.ExtType):
        if obj.code == 0:  # Enum
            try:
                inner_data = msgpack.unpackb(obj.data, raw=False, strict_map_key=False)
                if isinstance(inner_data, list) and len(inner_data) >= 3:
                    return inner_data[2]  # Return the enum value
            except:
                pass
        elif obj.code == 5:  # Pydantic model
            try:
                inner_data = msgpack.unpackb(obj.data, raw=False, strict_map_key=False)
                if isinstance(inner_data, list) and len(inner_data) >= 3:
                    decoded_data = inner_data[2]
                    return recursively_decode_exttype(decoded_data)
            except:
                pass
        return f"ExtType(code={obj.code})"
    
    elif isinstance(obj, dict):
        return {k: recursively_decode_exttype(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [recursively_decode_exttype(item) for item in obj]
    
    else:
        return obj

def extract_all_agent_data(db_path: str) -> Dict[str, Any]:
    """Extract experiment data using enhanced msgpack decoding"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Database: {db_path}")
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Available tables: {[table[0] for table in tables]}")
        
        # Get latest checkpoint from writes table (same as enhanced extractor)
        if any('writes' in table[0] for table in tables):
            cursor.execute("""
                SELECT checkpoint_id, channel, value, type 
                FROM writes 
                ORDER BY checkpoint_id DESC, idx
                LIMIT 50
            """)
            
            writes = cursor.fetchall()
            
            # Group by checkpoint and get the latest
            checkpoint_data = {}
            for checkpoint_id, channel, value, data_type in writes:
                if checkpoint_id not in checkpoint_data:
                    checkpoint_data[checkpoint_id] = {}
                
                if data_type == 'msgpack':
                    # Use our enhanced decoding
                    decoded = decode_msgpack_with_recursive_exttype(value)
                    fully_decoded = recursively_decode_exttype(decoded)
                    checkpoint_data[checkpoint_id][channel] = fully_decoded
            
            # Get the latest checkpoint
            if checkpoint_data:
                latest_checkpoint_id = max(checkpoint_data.keys())
                latest_data = checkpoint_data[latest_checkpoint_id]
                print(f"Using enhanced decoding with checkpoint: {latest_checkpoint_id}")
                
                conn.close()
                return latest_data
        
        conn.close()
        return {}
        
    except Exception as e:
        print(f"Error reading database: {e}")
        import traceback
        traceback.print_exc()
        return {}

def extract_agent_histories(experiment_state: Dict[str, Any]) -> Dict[str, Dict]:
    """Extract detailed history for each agent with enhanced decoding"""
    
    agent_histories = {}
    
    # Get population data
    population_state = experiment_state.get('population_state')
    if not population_state:
        print("No population state found")
        return {}
    
    population = population_state.get('population', [])
    print(f"Found {len(population)} agents in final population")
    
    for agent in population:
        try:
            agent_id = agent.get('agent_id', 'unknown')
            
            # Extract memories with proper decoding
            memories = agent.get('recent_memories', [])
            decoded_memories = recursively_decode_exttype(memories)
            
            # Analyze the decoded memories
            cooperations = 0
            defections = 0
            betrayals = 0
            mutual_cooperations = 0
            total_payoff = 0
            
            for memory in decoded_memories:
                if isinstance(memory, dict):
                    my_move = memory.get('my_move')
                    opponent_move = memory.get('opponent_move')
                    my_payoff = memory.get('my_payoff', 0)
                    
                    total_payoff += my_payoff
                    
                    if my_move == 'cooperate':
                        cooperations += 1
                        if opponent_move == 'defect':
                            betrayals += 1
                        elif opponent_move == 'cooperate':
                            mutual_cooperations += 1
                    elif my_move == 'defect':
                        defections += 1
            
            # Extract psychological profile
            profile = agent.get('psychological_profile', {})
            trust_level = profile.get('trust_level', 0.5)
            loss_sensitivity = profile.get('loss_sensitivity', 1.0)
            emotional_state = profile.get('emotional_state', 'unknown')
            
            # Calculate stats
            total_interactions = len(decoded_memories)
            cooperation_rate = cooperations / total_interactions if total_interactions > 0 else 0
            
            # Create comprehensive agent history
            agent_history = {
                'agent_id': agent_id,
                'agent_type': agent.get('agent_type', 'unknown'),
                'total_score': agent.get('total_score', 0),
                'recovery_progress': agent.get('recovery_progress', 1.0),
                'complete_reasoning_chain': agent.get('reasoning_chain', []),
                'all_memories': decoded_memories,
                'trauma_triggers': agent.get('trauma_triggers', []),
                'psychological_observations': agent.get('psychological_observations', []),
                
                # FIXED: Calculate interaction summary from ACTUAL decoded data
                'interaction_summary': {
                    'total_interactions': total_interactions,
                    'cooperations': cooperations,
                    'defections': defections,
                    'cooperation_rate': cooperation_rate,
                    'betrayals_experienced': betrayals,
                    'mutual_cooperations': mutual_cooperations,
                    'total_payoff': total_payoff,
                    'average_payoff': total_payoff / total_interactions if total_interactions > 0 else 0
                },
                
                # FIXED: Use ACTUAL psychological evolution data
                'psychological_evolution': {
                    'final_trust_level': trust_level,
                    'final_loss_sensitivity': loss_sensitivity,
                    'final_emotional_state': emotional_state,
                    'final_internal_narrative': profile.get('internal_narrative', ''),
                    'learned_heuristics': profile.get('learned_heuristics', []),
                    'dominant_trait': _get_dominant_trait(trust_level, loss_sensitivity),
                    'evolution_events': [],  # Would need reasoning chain analysis
                    'trauma_count': len(profile.get('trauma_memories', []))
                },
                
                # FIXED: Trauma analysis from actual data
                'trauma_analysis': {
                    'trauma_count': len(profile.get('trauma_memories', [])),
                    'trauma_types': {},
                    'analysis': f'Experienced {betrayals} betrayals' if betrayals > 0 else 'No significant traumas'
                }
            }
            
            agent_histories[agent_id] = agent_history
            
        except Exception as e:
            print(f"Error processing agent: {e}")
            continue
    
    return agent_histories

def _get_dominant_trait(trust_level: float, loss_sensitivity: float) -> str:
    """Determine dominant psychological trait"""
    if loss_sensitivity > 2.0 and trust_level < 0.3:
        return "traumatized_paranoid"
    elif loss_sensitivity > 1.8:
        return "loss_averse"
    elif trust_level < 0.2:
        return "paranoid"
    elif trust_level > 0.8:
        return "trusting"
    else:
        return "balanced"

def create_agent_csv(agent_histories: Dict, csv_path: Path):
    """Create CSV file for easy analysis"""
    
    rows = []
    
    for agent_id, history in agent_histories.items():
        # Safely extract data with defaults
        interaction = history.get('interaction_summary', {})
        trauma = history.get('trauma_analysis', {})
        evolution = history.get('psychological_evolution', {})
        
        row = {
            'agent_id': agent_id,
            'agent_type': history.get('agent_type', 'unknown'),
            'final_trust_level': evolution.get('final_trust_level', 0.5),
            'final_loss_sensitivity': evolution.get('final_loss_sensitivity', 1.0),
            'dominant_trait': evolution.get('dominant_trait', 'unknown'),
            'emotional_state': evolution.get('final_emotional_state', 'unknown'),
            'total_score': history.get('total_score', 0),
            'total_interactions': interaction.get('total_interactions', 0),
            'cooperation_rate': interaction.get('cooperation_rate', 0.0),
            'betrayals_experienced': interaction.get('betrayals_experienced', 0),
            'mutual_cooperations': interaction.get('mutual_cooperations', 0),
            'average_payoff': interaction.get('average_payoff', 0.0),
            'trauma_count': trauma.get('trauma_count', 0),
            'recovery_progress': history.get('recovery_progress', 1.0),
            'learned_heuristics_count': len(evolution.get('learned_heuristics', [])),
            'total_reasoning_steps': len(history.get('complete_reasoning_chain', [])),
            'psychological_evolution_events': len(evolution.get('evolution_events', []))
        }
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

def save_agent_histories(agent_histories: Dict, output_dir: str = "agent_analysis"):
    """Save detailed agent histories to files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save individual agent files
    for agent_id, history in agent_histories.items():
        agent_file = output_path / f"{agent_id}_complete_history.json"
        
        # Make serializable
        serializable_history = make_serializable(history)
        
        with open(agent_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"Saved detailed history for {agent_id}")
    
    # Save summary analysis
    summary = create_population_summary(agent_histories)
    summary_file = output_path / "population_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(make_serializable(summary), f, indent=2)
    
    # Create CSV for easy analysis
    create_agent_csv(agent_histories, output_path / "agents_summary.csv")
    
    print(f"\nAll analysis saved to {output_path}/")
    print(f"- Individual agent histories: {len(agent_histories)} files")
    print(f"- Population summary: population_summary.json")
    print(f"- CSV for analysis: agents_summary.csv")

def create_population_summary(agent_histories: Dict) -> Dict:
    """Create population-level summary"""
    
    total_agents = len(agent_histories)
    
    # Aggregate statistics with safe extraction
    trust_levels = []
    loss_sensitivities = []
    cooperation_rates = []
    trauma_counts = []
    
    for h in agent_histories.values():
        # Safely extract psychological evolution data
        psych_evo = h.get('psychological_evolution', {})
        trust_levels.append(psych_evo.get('final_trust_level', 0.5))
        loss_sensitivities.append(psych_evo.get('final_loss_sensitivity', 1.0))
        
        # Safely extract interaction summary data
        interaction_summary = h.get('interaction_summary', {})
        cooperation_rates.append(interaction_summary.get('cooperation_rate', 0.0))
        
        # Safely extract trauma analysis data
        trauma_analysis = h.get('trauma_analysis', {})
        trauma_counts.append(trauma_analysis.get('trauma_count', 0))
    
    return {
        'experiment_summary': {
            'total_agents': total_agents,
            'analysis_timestamp': datetime.now().isoformat()
        },
        'population_psychology': {
            'average_trust_level': sum(trust_levels) / len(trust_levels) if trust_levels else 0,
            'average_loss_sensitivity': sum(loss_sensitivities) / len(loss_sensitivities) if loss_sensitivities else 0,
            'trust_level_range': [min(trust_levels), max(trust_levels)] if trust_levels else [0, 0],
            'loss_sensitivity_range': [min(loss_sensitivities), max(loss_sensitivities)] if loss_sensitivities else [0, 0]
        },
        'behavioral_patterns': {
            'average_cooperation_rate': sum(cooperation_rates) / len(cooperation_rates) if cooperation_rates else 0,
            'cooperation_rate_range': [min(cooperation_rates), max(cooperation_rates)] if cooperation_rates else [0, 0],
            'agents_with_trauma': sum(1 for t in trauma_counts if t > 0),
            'average_trauma_count': sum(trauma_counts) / len(trauma_counts) if trauma_counts else 0
        },
        'agent_types': {
            agent_type: sum(1 for h in agent_histories.values() if h['agent_type'] == agent_type)
            for agent_type in set(h['agent_type'] for h in agent_histories.values())
        }
    }

def make_serializable(obj):
    """Convert objects to JSON-serializable format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        if hasattr(obj, 'dict'):  # Pydantic model
            return make_serializable(obj.dict())
        else:
            return make_serializable(obj.__dict__)
    elif hasattr(obj, 'value'):  # Enum
        return obj.value
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)

def main():
    """Main function to extract and analyze all agent data"""
    
    # Find most recent experiment database
    current_dir = Path(".")
    db_files = []
    
    for db_file in current_dir.glob("experiment_*.db"):
        mod_time = db_file.stat().st_mtime
        db_files.append((db_file, mod_time))
    
    if not db_files:
        print("No experiment databases found")
        return
    
    # Use most recent database
    db_files.sort(key=lambda x: x[1], reverse=True)
    most_recent_db = db_files[0][0]
    
    print(f"Analyzing experiment database: {most_recent_db}")
    print("=" * 60)
    
    # Extract all experiment data using ENHANCED decoding
    print("Extracting experiment data...")
    experiment_state = extract_all_agent_data(str(most_recent_db))
    
    if not experiment_state:
        print("Failed to extract experiment data")
        return
    
    # Extract detailed agent histories
    print("Extracting agent histories...")
    agent_histories = extract_agent_histories(experiment_state)
    
    if not agent_histories:
        print("No agent histories found")
        return
    
    print(f"Extracted detailed histories for {len(agent_histories)} agents")
    
    # Save all analysis
    print("Saving analysis files...")
    save_agent_histories(agent_histories)
    
    # Print CORRECTED summary
    print("\n" + "=" * 60)
    print("CORRECTED QUICK SUMMARY")
    print("=" * 60)
    
    for agent_id, history in list(agent_histories.items()):
        print(f"\n{agent_id}:")
        print(f"  Type: {history.get('agent_type', 'unknown')}")
        
        # Use ACTUAL psychological evolution data
        psych_evo = history.get('psychological_evolution', {})
        print(f"  Final Trust: {psych_evo.get('final_trust_level', 0.5):.3f}")
        print(f"  Final Loss Sensitivity: {psych_evo.get('final_loss_sensitivity', 1.0):.3f}")
        
        # Use ACTUAL interaction summary data
        interaction = history.get('interaction_summary', {})
        print(f"  Cooperation Rate: {interaction.get('cooperation_rate', 0.0):.1%}")
        print(f"  Betrayals: {interaction.get('betrayals_experienced', 0)}")
        
        print(f"  Total Score: {history.get('total_score', 0)}")
        print(f"  Reasoning Steps: {len(history.get('complete_reasoning_chain', []))}")
    
    print(f"\nDetailed analysis saved to ./agent_analysis/")

if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd
    
    main()