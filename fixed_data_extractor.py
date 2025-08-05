#!/usr/bin/env python3
"""
Fixed data extractor that properly handles msgpack ExtType objects
"""

import sqlite3
import msgpack
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import re

def decode_msgpack_with_pydantic(data):
    """Decode msgpack data with proper Pydantic model handling"""
    
    def decode_hook(code, data):
        """Custom decoder for Pydantic models stored as ExtType"""
        if code == 5:  # Pydantic model code
            try:
                # This is a Pydantic model - try to extract the data
                decoded = msgpack.unpackb(data, raw=False, strict_map_key=False)
                
                if isinstance(decoded, list) and len(decoded) >= 3:
                    model_class = decoded[0]  # e.g., 'src.state.agent_state'
                    model_name = decoded[1]   # e.g., 'Memory'
                    model_data = decoded[2]   # The actual data dict
                    
                    if isinstance(model_data, dict):
                        return model_data
                
                return decoded
            except Exception as e:
                # If we can't decode it, return raw data
                return {"raw_data": data.hex()[:100]}
        
        return msgpack.ExtType(code, data)
    
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
    except Exception as e:
        print(f"Failed to decode msgpack: {e}")
        return None

def extract_latest_experiment_data():
    """Extract data from the most recent experiment"""
    
    # Find most recent database
    db_files = list(Path(".").glob("experiment_*.db"))
    if not db_files:
        print("No experiment databases found")
        return None
    
    latest_db = max(db_files, key=lambda f: f.stat().st_mtime)
    print(f"Analyzing: {latest_db}")
    
    try:
        conn = sqlite3.connect(latest_db)
        cursor = conn.cursor()
        
        # Get latest checkpoint from writes table
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
                decoded = decode_msgpack_with_pydantic(value)
                checkpoint_data[checkpoint_id][channel] = decoded
        
        # Get the latest checkpoint
        if checkpoint_data:
            latest_checkpoint_id = max(checkpoint_data.keys())
            latest_data = checkpoint_data[latest_checkpoint_id]
            print(f"Using checkpoint: {latest_checkpoint_id}")
            
            return latest_data
        
        conn.close()
        
    except Exception as e:
        print(f"Error reading database: {e}")
        return None

def analyze_agent_data(experiment_data):
    """Analyze agent data with proper msgpack decoding"""
    
    if not experiment_data or 'population_state' not in experiment_data:
        print("No population state found")
        return
    
    population_state = experiment_data['population_state']
    population = population_state.get('population', [])
    
    print(f"\n=== ANALYZING {len(population)} AGENTS ===")
    
    agents_data = []
    
    for agent in population:
        agent_id = agent.get('agent_id', 'unknown')
        print(f"\n--- Agent: {agent_id} ---")
        
        # Extract basic info
        total_score = agent.get('total_score', 0)
        agent_type = agent.get('agent_type', 'unknown')
        
        # Extract psychological profile
        profile = agent.get('psychological_profile', {})
        trust_level = profile.get('trust_level', 0.5)
        loss_sensitivity = profile.get('loss_sensitivity', 1.0)
        emotional_state = profile.get('emotional_state', 'unknown')
        
        # Extract memories with proper decoding
        memories = agent.get('recent_memories', [])
        print(f"  Raw memories count: {len(memories)}")
        
        decoded_memories = []
        cooperations = 0
        defections = 0
        betrayals = 0
        total_payoff = 0
        
        for memory in memories:
            if isinstance(memory, dict):
                # Already decoded
                decoded_memories.append(memory)
            else:
                # Try to decode if it's an ExtType or other format
                try:
                    if hasattr(memory, 'dict'):
                        decoded = memory.dict()
                        decoded_memories.append(decoded)
                    else:
                        decoded_memories.append(memory)
                except:
                    pass
        
        print(f"  Decoded memories count: {len(decoded_memories)}")
        
        # Analyze decoded memories
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
                elif my_move == 'defect':
                    defections += 1
                
                print(f"    Round {memory.get('round_number', '?')}: {my_move} vs {opponent_move} (payoff: {my_payoff})")
        
        # Calculate stats
        total_interactions = len(decoded_memories)
        cooperation_rate = cooperations / total_interactions if total_interactions > 0 else 0
        average_payoff = total_payoff / total_interactions if total_interactions > 0 else 0
        
        # Extract reasoning chain
        reasoning_chain = agent.get('reasoning_chain', [])
        reasoning_steps = len(reasoning_chain)
        
        agent_data = {
            'agent_id': agent_id,
            'agent_type': agent_type,
            'total_score': total_score,
            'trust_level': trust_level,
            'loss_sensitivity': loss_sensitivity,
            'emotional_state': emotional_state,
            'total_interactions': total_interactions,
            'cooperations': cooperations,
            'defections': defections,
            'cooperation_rate': cooperation_rate,
            'betrayals_experienced': betrayals,
            'total_payoff': total_payoff,
            'average_payoff': average_payoff,
            'reasoning_steps': reasoning_steps
        }
        
        agents_data.append(agent_data)
        
        print(f"  Final Analysis:")
        print(f"    Score: {total_score}")
        print(f"    Interactions: {total_interactions}")
        print(f"    Cooperation rate: {cooperation_rate:.1%}")
        print(f"    Betrayals: {betrayals}")
        print(f"    Trust level: {trust_level:.2f}")
        print(f"    Loss sensitivity: {loss_sensitivity:.2f}")
    
    # Create corrected CSV
    if agents_data:
        df = pd.DataFrame(agents_data)
        output_file = "corrected_agents_summary.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Corrected analysis saved to: {output_file}")
        
        # Show summary
        print(f"\n=== CORRECTED SUMMARY ===")
        for _, agent in df.iterrows():
            print(f"{agent['agent_id']}: {agent['cooperation_rate']:.1%} cooperation, "
                  f"{agent['betrayals_experienced']} betrayals, score {agent['total_score']}")

def main():
    """Run the fixed data extraction"""
    
    print("🔧 FIXED DATA EXTRACTOR")
    print("=" * 50)
    
    # Extract experiment data
    experiment_data = extract_latest_experiment_data()
    
    if experiment_data:
        analyze_agent_data(experiment_data)
    else:
        print("❌ Could not extract experiment data")

if __name__ == "__main__":
    # Install pandas if needed
    try:
        import pandas as pd
    except ImportError:
        print("Installing pandas...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
        import pandas as pd
    
    main()