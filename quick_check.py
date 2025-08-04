#!/usr/bin/env python3
"""
Quick check of the most recent experiment results.
"""

import sqlite3
import pickle
import json
from pathlib import Path
from datetime import datetime

def find_most_recent_experiment():
    """Find the most recent experiment database"""
    current_dir = Path(".")
    db_files = []
    
    for db_file in current_dir.glob("experiment_*.db"):
        mod_time = db_file.stat().st_mtime
        db_files.append((db_file, mod_time))
    
    if not db_files:
        return None
    
    # Sort by modification time and return the most recent
    db_files.sort(key=lambda x: x[1], reverse=True)
    return db_files[0][0]

def quick_inspect_sqlite(db_path):
    """Quick inspection of SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Database: {db_path}")
        print(f"Size: {db_path.stat().st_size / 1024:.1f} KB")
        print(f"Modified: {datetime.fromtimestamp(db_path.stat().st_mtime)}")
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Tables: {[table[0] for table in tables]}")
        
        # Try to get checkpoint data
        if any('checkpoints' in table[0] for table in tables):
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            checkpoint_count = cursor.fetchone()[0]
            print(f"Checkpoints: {checkpoint_count}")
            
            if checkpoint_count > 0:
                # Get the latest checkpoint
                cursor.execute("""
                    SELECT checkpoint 
                    FROM checkpoints 
                    ORDER BY checkpoint_id DESC 
                    LIMIT 1
                """)
                
                row = cursor.fetchone()
                if row:
                    try:
                        checkpoint_data = pickle.loads(row[0])
                        print("\nLatest checkpoint structure:")
                        
                        if isinstance(checkpoint_data, dict):
                            print("Checkpoint keys:", list(checkpoint_data.keys()))
                            
                            if 'channel_values' in checkpoint_data:
                                values = checkpoint_data['channel_values']
                                print("Channel values keys:", list(values.keys()) if isinstance(values, dict) else type(values))
                                
                                if isinstance(values, dict):
                                    # Show some key information
                                    print(f"  experiment_id: {values.get('experiment_id', 'Not found')}")
                                    print(f"  current_phase: {values.get('current_phase', 'Not found')}")
                                    print(f"  progress_percentage: {values.get('progress_percentage', 'Not found')}")
                                    print(f"  baseline_complete: {values.get('baseline_complete', 'Not found')}")
                                    print(f"  emergent_complete: {values.get('emergent_complete', 'Not found')}")
                                    print(f"  contagion_complete: {values.get('contagion_complete', 'Not found')}")
                                    print(f"  analysis_complete: {values.get('analysis_complete', 'Not found')}")
                                    
                                    # Count results
                                    baseline_results = values.get('baseline_results', [])
                                    emergent_results = values.get('emergent_results', [])
                                    contagion_results = values.get('contagion_results', [])
                                    statistical_results = values.get('statistical_results', [])
                                    final_results = values.get('results', [])
                                    
                                    print(f"  baseline_results: {len(baseline_results)} items")
                                    print(f"  emergent_results: {len(emergent_results)} items")
                                    print(f"  contagion_results: {len(contagion_results)} items")
                                    print(f"  statistical_results: {len(statistical_results)} items")
                                    print(f"  final_results: {len(final_results)} items")
                                    
                                    # Show final report if available
                                    if final_results and len(final_results) > 0:
                                        final_report = final_results[-1]
                                        print(f"\nFinal report keys: {list(final_report.keys()) if isinstance(final_report, dict) else type(final_report)}")
                                        
                                        if isinstance(final_report, dict) and 'key_insights' in final_report:
                                            print("\nKey insights:")
                                            for insight in final_report['key_insights']:
                                                print(f"  • {insight}")
                    
                    except Exception as e:
                        print(f"Error unpickling checkpoint: {e}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error reading database: {e}")

def main():
    print("Quick check of most recent experiment...")
    
    recent_db = find_most_recent_experiment()
    
    if recent_db:
        quick_inspect_sqlite(recent_db)
    else:
        print("No experiment databases found")
        
        # Check for fallback JSON files
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            json_files = list(checkpoints_dir.glob("*_checkpoint.json"))
            if json_files:
                print(f"Found {len(json_files)} fallback checkpoint files:")
                for json_file in json_files:
                    print(f"  {json_file.name}")
            else:
                print("No fallback checkpoint files found either")
        else:
            print("No checkpoints directory found")

if __name__ == "__main__":
    main()