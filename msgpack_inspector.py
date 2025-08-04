#!/usr/bin/env python3
"""
Inspector for LangGraph msgpack checkpoints.
"""

import sqlite3
import msgpack
import json
from pathlib import Path
from datetime import datetime
import sys

def decode_msgpack_data(data):
    """Decode msgpack data with error handling"""
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception as e:
        print(f"❌ Msgpack decode failed: {e}")
        return None

def inspect_writes_table(db_path):
    """Inspect the writes table to see individual state updates"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== WRITES TABLE ANALYSIS ===")
        
        # Get all writes grouped by checkpoint
        cursor.execute("""
            SELECT checkpoint_id, channel, value, type 
            FROM writes 
            ORDER BY checkpoint_id, idx
        """)
        
        writes = cursor.fetchall()
        
        # Group by checkpoint
        checkpoint_writes = {}
        for checkpoint_id, channel, value, data_type in writes:
            if checkpoint_id not in checkpoint_writes:
                checkpoint_writes[checkpoint_id] = {}
            
            # Decode the value
            if data_type == 'msgpack':
                decoded_value = decode_msgpack_data(value)
                checkpoint_writes[checkpoint_id][channel] = decoded_value
            else:
                checkpoint_writes[checkpoint_id][channel] = f"<{data_type} data>"
        
        # Show the final checkpoint state
        if checkpoint_writes:
            # Get the last checkpoint ID
            last_checkpoint_id = list(checkpoint_writes.keys())[-1]
            final_state = checkpoint_writes[last_checkpoint_id]
            
            print(f"=== FINAL STATE (Checkpoint: {last_checkpoint_id[:8]}...) ===")
            
            # Show all channels in the final state
            for channel, value in final_state.items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {channel}: {value}")
                elif isinstance(value, list):
                    print(f"  {channel}: List with {len(value)} items")
                    if value and len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if len(value) > 1:
                            print(f"    Last item type: {type(value[-1])}")
                elif isinstance(value, dict):
                    print(f"  {channel}: Dict with keys: {list(value.keys())}")
                else:
                    print(f"  {channel}: {type(value)}")
            
            # Show experiment progress through all checkpoints
            print(f"\n=== EXPERIMENT PROGRESS ===")
            for checkpoint_id, state in checkpoint_writes.items():
                phase = state.get('current_phase', 'unknown')
                progress = state.get('progress_percentage', 0)
                print(f"  {checkpoint_id[:8]}... {phase} ({progress}%)")
            
            # Detailed analysis of final state
            print(f"\n=== DETAILED FINAL STATE ANALYSIS ===")
            
            # Basic experiment info
            exp_id = final_state.get('experiment_id', 'Unknown')
            phase = final_state.get('current_phase', 'Unknown')
            progress = final_state.get('progress_percentage', 0)
            
            print(f"Experiment ID: {exp_id}")
            print(f"Current Phase: {phase}")
            print(f"Progress: {progress}%")
            
            # Completion flags
            completion_flags = {
                'baseline_complete': final_state.get('baseline_complete', False),
                'emergent_complete': final_state.get('emergent_complete', False),
                'contagion_complete': final_state.get('contagion_complete', False),
                'analysis_complete': final_state.get('analysis_complete', False)
            }
            print(f"Completion flags: {completion_flags}")
            
            # Results analysis
            result_fields = ['baseline_results', 'emergent_results', 'contagion_results', 'statistical_results', 'results']
            print(f"\nResults summary:")
            for field in result_fields:
                if field in final_state:
                    value = final_state[field]
                    if isinstance(value, list):
                        print(f"  {field}: {len(value)} items")
                        if value:
                            # Show structure of first item
                            first_item = value[0]
                            if isinstance(first_item, dict):
                                print(f"    Sample item keys: {list(first_item.keys())}")
                    else:
                        print(f"  {field}: {type(value)} (not a list)")
            
            # Show final results in detail
            final_results = final_state.get('results', [])
            if final_results:
                print(f"\n=== FINAL REPORTS ({len(final_results)} reports) ===")
                
                for i, report in enumerate(final_results):
                    print(f"\nReport {i+1}:")
                    if isinstance(report, dict):
                        for key, value in report.items():
                            if key == 'key_insights' and isinstance(value, list):
                                print(f"  {key}:")
                                for insight in value:
                                    print(f"    • {insight}")
                            elif key == 'experiment_metadata' and isinstance(value, dict):
                                print(f"  {key}:")
                                for meta_key, meta_value in value.items():
                                    print(f"    {meta_key}: {meta_value}")
                            elif isinstance(value, dict):
                                print(f"  {key}: {json.dumps(value, indent=4)}")
                            elif isinstance(value, list):
                                print(f"  {key}: List with {len(value)} items")
                            else:
                                print(f"  {key}: {value}")
                    else:
                        print(f"  Report is {type(report)}: {report}")
            
            # Population state analysis
            population_state = final_state.get('population_state')
            if population_state and isinstance(population_state, dict):
                print(f"\n=== POPULATION STATE ===")
                population = population_state.get('population', [])
                print(f"Population size: {len(population)}")
                print(f"Generation: {population_state.get('generation', 'Unknown')}")
                print(f"Average trust level: {population_state.get('avg_trust_level', 'Unknown')}")
                print(f"Average loss sensitivity: {population_state.get('avg_loss_sensitivity', 'Unknown')}")
                print(f"Contagion events: {len(population_state.get('contagion_events', []))}")
                print(f"Interaction results: {len(population_state.get('interaction_results', []))}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error inspecting writes table: {e}")
        import traceback
        traceback.print_exc()

def inspect_msgpack_checkpoints(db_path):
    """Inspect msgpack checkpoints"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== MSGPACK CHECKPOINTS ===")
        
        # Get the latest checkpoint
        cursor.execute("""
            SELECT checkpoint_id, checkpoint, metadata 
            FROM checkpoints 
            ORDER BY checkpoint_id DESC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if row:
            checkpoint_id, checkpoint_data, metadata_data = row
            
            print(f"Latest checkpoint: {checkpoint_id}")
            
            # Try to decode the checkpoint
            decoded_checkpoint = decode_msgpack_data(checkpoint_data)
            if decoded_checkpoint:
                print(f"Checkpoint structure: {type(decoded_checkpoint)}")
                if isinstance(decoded_checkpoint, dict):
                    print(f"Checkpoint keys: {list(decoded_checkpoint.keys())}")
                    
                    # Look for channel values
                    if 'channel_values' in decoded_checkpoint:
                        channel_values = decoded_checkpoint['channel_values']
                        print(f"Channel values keys: {list(channel_values.keys()) if isinstance(channel_values, dict) else type(channel_values)}")
            
            # Try to decode metadata
            decoded_metadata = decode_msgpack_data(metadata_data)
            if decoded_metadata:
                print(f"Metadata: {decoded_metadata}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error inspecting msgpack checkpoints: {e}")

def main():
    """Main inspection function"""
    
    # Find most recent experiment database
    current_dir = Path(".")
    db_files = []
    
    for db_file in current_dir.glob("experiment_*.db"):
        mod_time = db_file.stat().st_mtime
        db_files.append((db_file, mod_time))
    
    if not db_files:
        print("No experiment databases found")
        return
    
    # Sort by modification time and get the most recent
    db_files.sort(key=lambda x: x[1], reverse=True)
    most_recent_db = db_files[0][0]
    
    print(f"Analyzing msgpack data in: {most_recent_db}")
    print("=" * 80)
    
    # Inspect the writes table (this contains the actual state data)
    inspect_writes_table(most_recent_db)
    
    # Also try the checkpoints table
    inspect_msgpack_checkpoints(most_recent_db)

if __name__ == "__main__":
    # Check if msgpack is available
    try:
        import msgpack
    except ImportError:
        print("❌ msgpack is not installed. Install it with: pip install msgpack")
        sys.exit(1)
    
    main()