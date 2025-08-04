#!/usr/bin/env python3
"""
Robust database inspector that handles various checkpoint formats and pickle issues.
"""

import sqlite3
import pickle
import json
import zlib
from pathlib import Path
from datetime import datetime
import sys

def explore_database_structure(db_path):
    """Explore the database structure in detail"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"=== DATABASE STRUCTURE ===")
        print(f"Database: {db_path}")
        print(f"Size: {db_path.stat().st_size / 1024:.1f} KB")
        print(f"Modified: {datetime.fromtimestamp(db_path.stat().st_mtime)}")
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {[table[0] for table in tables]}")
        
        # Explore each table
        for table in tables:
            table_name = table[0]
            print(f"\n--- TABLE: {table_name} ---")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("Columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]
            print(f"Row count: {row_count}")
            
            # Sample some data (first few rows)
            if row_count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                rows = cursor.fetchall()
                print("Sample rows:")
                for i, row in enumerate(rows):
                    print(f"  Row {i+1}: {len(row)} columns")
                    for j, col_data in enumerate(row):
                        col_name = columns[j][1] if j < len(columns) else f"col_{j}"
                        if isinstance(col_data, bytes):
                            print(f"    {col_name}: <bytes, length={len(col_data)}>")
                        else:
                            data_str = str(col_data)
                            if len(data_str) > 100:
                                data_str = data_str[:100] + "..."
                            print(f"    {col_name}: {data_str}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error exploring database structure: {e}")

def try_different_unpickling_methods(data):
    """Try different methods to unpickle the data"""
    
    methods = [
        ("Direct pickle.loads", lambda d: pickle.loads(d)),
        ("Pickle with protocol 0", lambda d: pickle.loads(d, encoding='latin-1')),
        ("Pickle with errors='ignore'", lambda d: pickle.loads(d, errors='ignore')),
        ("Zlib decompress then pickle", lambda d: pickle.loads(zlib.decompress(d))),
    ]
    
    for method_name, method_func in methods:
        try:
            result = method_func(data)
            print(f"✅ SUCCESS with {method_name}")
            return result
        except Exception as e:
            print(f"❌ {method_name} failed: {e}")
    
    return None

def inspect_checkpoints_table(db_path):
    """Inspect the checkpoints table specifically"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"\n=== CHECKPOINTS TABLE ANALYSIS ===")
        
        # Get checkpoints table info
        cursor.execute("PRAGMA table_info(checkpoints)")
        columns = cursor.fetchall()
        print("Checkpoint table columns:")
        for col in columns:
            print(f"  {col[1]} ({col[2]})")
        
        # Get all checkpoint IDs and some metadata
        cursor.execute("SELECT checkpoint_id FROM checkpoints ORDER BY checkpoint_id")
        checkpoint_ids = cursor.fetchall()
        print(f"Checkpoint IDs: {[cid[0] for cid in checkpoint_ids]}")
        
        # Try to examine each checkpoint
        for checkpoint_id_row in checkpoint_ids:
            checkpoint_id = checkpoint_id_row[0]
            print(f"\n--- CHECKPOINT {checkpoint_id} ---")
            
            cursor.execute("SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,))
            row = cursor.fetchone()
            
            if row:
                print(f"Row has {len(row)} columns")
                
                # Find the checkpoint data column (usually the largest blob)
                checkpoint_data = None
                for i, col_data in enumerate(row):
                    col_name = columns[i][1] if i < len(columns) else f"col_{i}"
                    
                    if isinstance(col_data, bytes):
                        print(f"  {col_name}: <bytes, length={len(col_data)}>")
                        if len(col_data) > 100:  # Assume the largest bytes column is the checkpoint
                            checkpoint_data = col_data
                    else:
                        data_str = str(col_data)
                        if len(data_str) > 50:
                            data_str = data_str[:50] + "..."
                        print(f"  {col_name}: {data_str}")
                
                # Try to unpickle the checkpoint data
                if checkpoint_data:
                    print(f"  Attempting to unpickle checkpoint data ({len(checkpoint_data)} bytes)...")
                    
                    # Try different unpickling methods
                    unpickled_data = try_different_unpickling_methods(checkpoint_data)
                    
                    if unpickled_data:
                        print(f"  Unpickled data type: {type(unpickled_data)}")
                        
                        if isinstance(unpickled_data, dict):
                            print(f"  Dictionary keys: {list(unpickled_data.keys())}")
                            
                            # Look for channel_values or similar
                            if 'channel_values' in unpickled_data:
                                channel_values = unpickled_data['channel_values']
                                print(f"  Channel values type: {type(channel_values)}")
                                
                                if isinstance(channel_values, dict):
                                    print(f"  Channel values keys: {list(channel_values.keys())}")
                                    
                                    # Extract key experiment info
                                    exp_id = channel_values.get('experiment_id', 'Not found')
                                    phase = channel_values.get('current_phase', 'Not found')
                                    progress = channel_values.get('progress_percentage', 'Not found')
                                    
                                    print(f"  🔍 Experiment ID: {exp_id}")
                                    print(f"  🔍 Current Phase: {phase}")
                                    print(f"  🔍 Progress: {progress}")
                                    
                                    # Count results
                                    results_summary = {}
                                    result_keys = ['baseline_results', 'emergent_results', 'contagion_results', 'statistical_results', 'results']
                                    for key in result_keys:
                                        if key in channel_values:
                                            value = channel_values[key]
                                            if isinstance(value, list):
                                                results_summary[key] = len(value)
                                            else:
                                                results_summary[key] = f"Not a list: {type(value)}"
                                    
                                    print(f"  🔍 Results summary: {results_summary}")
                                    
                                    # If this is the last checkpoint, show more detail
                                    if checkpoint_id == checkpoint_ids[-1][0]:
                                        print(f"\n  === FINAL CHECKPOINT DETAILS ===")
                                        
                                        # Show completion flags
                                        completion_flags = {
                                            'baseline_complete': channel_values.get('baseline_complete'),
                                            'emergent_complete': channel_values.get('emergent_complete'),
                                            'contagion_complete': channel_values.get('contagion_complete'),
                                            'analysis_complete': channel_values.get('analysis_complete')
                                        }
                                        print(f"  Completion flags: {completion_flags}")
                                        
                                        # Show final results if available
                                        final_results = channel_values.get('results', [])
                                        if final_results:
                                            print(f"  Final results count: {len(final_results)}")
                                            
                                            last_result = final_results[-1]
                                            if isinstance(last_result, dict):
                                                print(f"  Last result keys: {list(last_result.keys())}")
                                                
                                                if 'key_insights' in last_result:
                                                    print(f"  Key insights found:")
                                                    for insight in last_result['key_insights']:
                                                        print(f"    • {insight}")
                        
                        # If we successfully unpickled the final checkpoint, we're done
                        if checkpoint_id == checkpoint_ids[-1][0] and unpickled_data:
                            break
        
        conn.close()
        
    except Exception as e:
        print(f"Error inspecting checkpoints table: {e}")
        import traceback
        traceback.print_exc()

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
    
    print(f"Analyzing most recent experiment database: {most_recent_db}")
    print("=" * 80)
    
    # First, explore the overall database structure
    explore_database_structure(most_recent_db)
    
    # Then, focus on the checkpoints table
    inspect_checkpoints_table(most_recent_db)

if __name__ == "__main__":
    main()