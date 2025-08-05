#!/usr/bin/env python3
"""
Comprehensive diagnostic to understand what's happening in the ultra_minimal experiment
"""

import sqlite3
import msgpack
import json
from pathlib import Path
from datetime import datetime
import sys

def analyze_experiment_lifecycle(db_path):
    """Analyze the complete lifecycle of an experiment"""
    
    print(f"🔍 ANALYZING EXPERIMENT LIFECYCLE")
    print(f"Database: {db_path}")
    print("=" * 60)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all checkpoints in chronological order
        cursor.execute("""
            SELECT checkpoint_id, channel, value, type 
            FROM writes 
            ORDER BY checkpoint_id, idx
        """)
        
        writes = cursor.fetchall()
        
        # Group by checkpoint
        checkpoint_progression = {}
        for checkpoint_id, channel, value, data_type in writes:
            if checkpoint_id not in checkpoint_progression:
                checkpoint_progression[checkpoint_id] = {}
            
            # Decode msgpack data
            if data_type == 'msgpack':
                try:
                    decoded = decode_msgpack_with_hooks(value)
                    checkpoint_progression[checkpoint_id][channel] = decoded
                except Exception as e:
                    checkpoint_progression[checkpoint_id][channel] = f"<decode_error: {e}>"
        
        # Analyze progression
        print(f"📊 EXPERIMENT PROGRESSION")
        print(f"Total checkpoints: {len(checkpoint_progression)}")
        
        for i, (checkpoint_id, state) in enumerate(checkpoint_progression.items()):
            print(f"\n--- CHECKPOINT {i+1}: {checkpoint_id[:8]}... ---")
            
            # Key experiment fields
            phase = state.get('current_phase', 'unknown')
            progress = state.get('progress_percentage', 0)
            
            print(f"Phase: {phase}")
            print(f"Progress: {progress}%")
            
            # Population state analysis
            population_state = state.get('population_state')
            if population_state:
                if isinstance(population_state, dict):
                    population = population_state.get('population', [])
                    generation = population_state.get('generation', 'unknown')
                    print(f"Population: {len(population)} agents")
                    print(f"Generation: {generation}")
                else:
                    print(f"Population state: {type(population_state)}")
            else:
                print("Population state: MISSING")
            
            # Configuration analysis
            config = state.get('experiment_config')
            if config:
                emergent_config = config.get('emergent_experiment', {})
                pop_size = emergent_config.get('population_size', 'unknown')
                generations = emergent_config.get('generations', 'unknown')
                print(f"Config - Population size: {pop_size}, Generations: {generations}")
            
            # Results analysis
            baseline_results = state.get('baseline_results', [])
            emergent_results = state.get('emergent_results', [])
            print(f"Results - Baseline: {len(baseline_results)}, Emergent: {len(emergent_results)}")
            
            # Completion flags
            flags = {
                'baseline_complete': state.get('baseline_complete', False),
                'emergent_complete': state.get('emergent_complete', False),
                'contagion_complete': state.get('contagion_complete', False),
                'analysis_complete': state.get('analysis_complete', False)
            }
            completed_phases = [k for k, v in flags.items() if v]
            print(f"Completed phases: {completed_phases}")
        
        # Final state analysis
        if checkpoint_progression:
            final_checkpoint = list(checkpoint_progression.values())[-1]
            print(f"\n🎯 FINAL STATE ANALYSIS")
            print("=" * 40)
            
            final_population_state = final_checkpoint.get('population_state')
            if final_population_state and isinstance(final_population_state, dict):
                final_population = final_population_state.get('population', [])
                print(f"✅ Final population: {len(final_population)} agents")
                
                if final_population:
                    # Analyze first agent
                    first_agent = final_population[0]
                    print(f"Sample agent ID: {first_agent.get('agent_id', 'unknown')}")
                    print(f"Sample agent type: {first_agent.get('agent_type', 'unknown')}")
                    print(f"Sample agent score: {first_agent.get('total_score', 0)}")
                    
                    # Check memories
                    memories = first_agent.get('recent_memories', [])
                    print(f"Sample agent memories: {len(memories)}")
                    
                    if memories:
                        # Try to decode first memory
                        try:
                            first_memory = memories[0]
                            if hasattr(first_memory, '__dict__'):
                                print(f"Memory structure: {first_memory.__dict__}")
                            else:
                                print(f"Memory data: {first_memory}")
                        except Exception as e:
                            print(f"Memory decode error: {e}")
                else:
                    print("❌ Population is empty!")
            else:
                print("❌ No valid population state in final checkpoint")
        
        conn.close()
        
        # Summary and recommendations
        print(f"\n🚨 DIAGNOSIS & RECOMMENDATIONS")
        print("=" * 40)
        
        if not checkpoint_progression:
            print("❌ CRITICAL: No checkpoints found - experiment may not have started")
            print("   Recommendation: Check if main.py is running properly")
        
        elif len(checkpoint_progression) == 1:
            print("⚠️  WARNING: Only one checkpoint - experiment may have crashed early")
            print("   Recommendation: Check logs for errors during initialization")
        
        else:
            final_state = list(checkpoint_progression.values())[-1]
            final_phase = final_state.get('current_phase', 'unknown')
            
            if final_phase == 'initialized':
                print("⚠️  ISSUE: Experiment stuck at initialization phase")
                print("   Recommendation: Check baseline study execution")
            elif 'population_state' not in final_state:
                print("❌ CRITICAL: Population state never created")
                print("   Recommendation: Check initialize_experiment function")
            elif not final_state.get('population_state', {}).get('population'):
                print("❌ CRITICAL: Population is empty")
                print("   Recommendation: Check agent creation in initialize_experiment")
            else:
                print("✅ Population created successfully")
                print("   The data extraction should work - check the extractor logic")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def decode_msgpack_with_hooks(data):
    """Decode msgpack with proper hooks for Pydantic models"""
    
    def decode_hook(code, data):
        if code == 5:  # Pydantic model
            try:
                decoded = msgpack.unpackb(data, raw=False, strict_map_key=False)
                if isinstance(decoded, list) and len(decoded) >= 3:
                    # Extract the actual data dict
                    return decoded[2] if isinstance(decoded[2], dict) else decoded
                return decoded
            except:
                return {"__pydantic_decode_error__": True}
        return msgpack.ExtType(code, data)
    
    return msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)

def run_quick_test_experiment():
    """Run a minimal test to see what's being created"""
    
    print(f"\n🧪 RUNNING QUICK TEST EXPERIMENT")
    print("=" * 40)
    
    try:
        # Import and run a single experiment node
        import sys
        sys.path.append('.')
        
        from src.nodes.experiment_nodes import initialize_experiment
        from datetime import datetime
        
        # Create minimal experiment state
        test_state = {
            "experiment_id": "diagnostic_test",
            "experiment_config": {
                "emergent_experiment": {
                    "population_size": 2,
                    "generations": 1,
                    "interactions_per_generation": 1,
                    "rounds_per_interaction": 3
                }
            }
        }
        
        print("Testing initialize_experiment function...")
        
        # Test initialization
        import asyncio
        result = asyncio.run(initialize_experiment(test_state))
        
        print(f"✅ Initialization result keys: {list(result.keys())}")
        
        population_state = result.get('population_state')
        if population_state:
            population = population_state.get('population', [])
            print(f"✅ Created {len(population)} agents")
            
            if population:
                agent = population[0]
                print(f"✅ Sample agent: {agent.get('agent_id')} ({agent.get('agent_type')})")
            else:
                print("❌ Population is empty")
        else:
            print("❌ No population_state created")
    
    except Exception as e:
        print(f"❌ Test experiment failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main diagnostic function"""
    
    # Find the most recent database
    db_files = list(Path(".").glob("experiment_*.db"))
    if not db_files:
        print("❌ No experiment databases found")
        return
    
    latest_db = max(db_files, key=lambda f: f.stat().st_mtime)
    
    # Run comprehensive analysis
    analyze_experiment_lifecycle(latest_db)
    
    # Run test experiment
    run_quick_test_experiment()

if __name__ == "__main__":
    main()