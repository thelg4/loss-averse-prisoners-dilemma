#!/usr/bin/env python3
"""
Script to inspect experiment results in detail.
Usage: python inspect_results.py [experiment_id]
"""

import asyncio
import sys
import json
import sqlite3
import pickle
from pathlib import Path
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.checkpointing import ExperimentCheckpointManager

def print_section(title, content=None, level=1):
    """Print a formatted section header"""
    if level == 1:
        print("\n" + "="*80)
        print(f"{title}")
        print("="*80)
    elif level == 2:
        print("\n" + "-"*60)
        print(f"{title}")
        print("-"*60)
    else:
        print(f"\n{title}:")
    
    if content is not None:
        if isinstance(content, (dict, list)):
            print(json.dumps(content, indent=2, default=str))
        else:
            print(content)

def load_experiment_from_sqlite(db_path):
    """Load experiment data directly from SQLite database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Try to get the latest checkpoint from LangGraph's checkpoint table
        cursor.execute("""
            SELECT checkpoint 
            FROM checkpoints 
            ORDER BY checkpoint_id DESC 
            LIMIT 1
        """)
        
        row = cursor.fetchone()
        if row:
            # LangGraph stores checkpoints as pickle data
            checkpoint_data = pickle.loads(row[0])
            if 'channel_values' in checkpoint_data:
                return checkpoint_data['channel_values']
            return checkpoint_data
        
        conn.close()
        return None
        
    except Exception as e:
        print(f"Error reading SQLite database {db_path}: {e}")
        return None

def find_experiment_databases():
    """Find all experiment database files"""
    current_dir = Path(".")
    db_files = []
    
    # Look for experiment database files
    for db_file in current_dir.glob("experiment_*.db"):
        # Extract experiment ID from filename
        filename = db_file.stem
        if filename.startswith("experiment_"):
            experiment_id = filename.replace("experiment_", "")
            
            # Get file modification time
            mod_time = datetime.fromtimestamp(db_file.stat().st_mtime)
            
            db_files.append({
                'experiment_id': experiment_id,
                'db_path': str(db_file),
                'last_modified': mod_time,
                'file_size': db_file.stat().st_size
            })
    
    # Sort by modification time (newest first)
    db_files.sort(key=lambda x: x['last_modified'], reverse=True)
    
    return db_files
    """Inspect and display experiment results in detail"""
    
    print_section("EXPERIMENT OVERVIEW", level=1)
    
    # Basic info
    print(f"Experiment ID: {experiment_data.get('experiment_id', 'Unknown')}")
    print(f"Current Phase: {experiment_data.get('current_phase', 'Unknown')}")
    print(f"Progress: {experiment_data.get('progress_percentage', 0):.1f}%")
    print(f"Start Time: {experiment_data.get('start_time', 'Unknown')}")
    print(f"Current Time: {experiment_data.get('current_time', 'Unknown')}")
    
    # Completion status
    print_section("COMPLETION STATUS", level=2)
    print(f"Baseline Complete: {experiment_data.get('baseline_complete', False)}")
    print(f"Emergent Complete: {experiment_data.get('emergent_complete', False)}")
    print(f"Contagion Complete: {experiment_data.get('contagion_complete', False)}")
    print(f"Analysis Complete: {experiment_data.get('analysis_complete', False)}")
    
    # Configuration
    config = experiment_data.get('experiment_config', {})
    if config:
        print_section("EXPERIMENT CONFIGURATION", level=2)
        print(f"Total Generations: {experiment_data.get('total_generations', 'Unknown')}")
        print(f"Population Size: {experiment_data.get('population_size', 'Unknown')}")
        print(f"Interactions per Generation: {experiment_data.get('interactions_per_generation', 'Unknown')}")
        print(f"Rounds per Interaction: {experiment_data.get('rounds_per_interaction', 'Unknown')}")
    
    # Results counts
    print_section("RESULTS SUMMARY", level=2)
    baseline_results = experiment_data.get('baseline_results', [])
    emergent_results = experiment_data.get('emergent_results', [])
    contagion_results = experiment_data.get('contagion_results', [])
    statistical_results = experiment_data.get('statistical_results', [])
    final_results = experiment_data.get('results', [])
    
    print(f"Baseline Results: {len(baseline_results)} entries")
    print(f"Emergent Results: {len(emergent_results)} entries")
    print(f"Contagion Results: {len(contagion_results)} entries")
    print(f"Statistical Results: {len(statistical_results)} entries")
    print(f"Final Reports: {len(final_results)} entries")
    
    # Population state
    population_state = experiment_data.get('population_state', {})
    if population_state:
        print_section("POPULATION STATE", level=2)
        population = population_state.get('population', [])
        print(f"Population Size: {len(population)} agents")
        print(f"Current Generation: {population_state.get('generation', 'Unknown')}")
        print(f"Average Trust Level: {population_state.get('avg_trust_level', 'Unknown')}")
        print(f"Average Loss Sensitivity: {population_state.get('avg_loss_sensitivity', 'Unknown')}")
        print(f"Contagion Events: {len(population_state.get('contagion_events', []))}")
        print(f"Interaction Results: {len(population_state.get('interaction_results', []))}")
    
    # Detailed results
    if baseline_results:
        print_section("BASELINE RESULTS DETAILS", level=2)
        for i, result in enumerate(baseline_results[:3]):  # Show first 3
            print(f"Result {i+1}: {result}")
    
    if emergent_results:
        print_section("EMERGENT RESULTS DETAILS", level=2)
        print(f"First result: {emergent_results[0] if emergent_results else 'None'}")
        if len(emergent_results) > 1:
            print(f"Last result: {emergent_results[-1]}")
    
    if contagion_results:
        print_section("CONTAGION RESULTS DETAILS", level=2)
        for i, result in enumerate(contagion_results):
            print(f"Contagion Study {i+1}:")
            print(f"  Events: {len(result.get('contagion_events', []))}")
            print(f"  Analysis: {result.get('analysis', {})}")
    
    if statistical_results:
        print_section("STATISTICAL RESULTS DETAILS", level=2)
        for i, result in enumerate(statistical_results):
            print(f"Statistical Analysis {i+1}: {result}")
    
    # Final reports
    if final_results:
        print_section("FINAL REPORTS", level=2)
        for i, report in enumerate(final_results):
            print(f"\nReport {i+1}:")
            
            # Experiment metadata
            if 'experiment_metadata' in report:
                metadata = report['experiment_metadata']
                print(f"  Experiment ID: {metadata.get('experiment_id', 'Unknown')}")
                print(f"  Duration: {metadata.get('total_duration', 'Unknown')}")
                print(f"  Completion Time: {metadata.get('completion_time', 'Unknown')}")
            
            # Key insights
            if 'key_insights' in report:
                print("  Key Insights:")
                for insight in report['key_insights']:
                    print(f"    • {insight}")
            
            # Findings
            if 'baseline_findings' in report:
                print(f"  Baseline Findings: {report['baseline_findings']}")
            
            if 'emergent_findings' in report:
                print(f"  Emergent Findings: {report['emergent_findings']}")
            
            if 'contagion_findings' in report:
                print(f"  Contagion Findings: {report['contagion_findings']}")
            
            if 'statistical_analysis' in report:
                print(f"  Statistical Analysis: {report['statistical_analysis']}")
            
            if 'publication_ready_summary' in report:
                summary = report['publication_ready_summary']
                print(f"  Publication Summary:")
                print(f"    Title: {summary.get('title', 'Unknown')}")
                print(f"    Methodology: {summary.get('methodology', 'Unknown')}")
                print(f"    Significance: {summary.get('significance', 'Unknown')}")

def main():
    """Main function to inspect experiment results"""
    
    if len(sys.argv) > 1:
        experiment_id = sys.argv[1]
        
        # Try to find the database file for this experiment
        db_path = f"experiment_{experiment_id}.db"
        if Path(db_path).exists():
            print(f"Loading experiment from database: {db_path}")
            experiment_data = load_experiment_from_sqlite(db_path)
        else:
            # Fallback to checkpoint manager
            checkpoint_manager = ExperimentCheckpointManager()
            experiment_data = checkpoint_manager.load_experiment_checkpoint(experiment_id)
        
        if experiment_data:
            inspect_experiment_results(experiment_data)
        else:
            print(f"❌ Experiment '{experiment_id}' not found")
            return
    else:
        # List all experiments from both sources
        print("Scanning for experiment databases...")
        
        # Check SQLite databases
        db_experiments = find_experiment_databases()
        
        # Check fallback checkpoints
        checkpoint_manager = ExperimentCheckpointManager()
        fallback_experiments = checkpoint_manager.list_experiments()
        
        all_experiments = []
        
        # Add database experiments
        for db_exp in db_experiments:
            all_experiments.append({
                'experiment_id': db_exp['experiment_id'],
                'source': 'database',
                'last_modified': db_exp['last_modified'],
                'db_path': db_exp['db_path'],
                'file_size': db_exp['file_size']
            })
        
        # Add fallback experiments
        for fb_exp in fallback_experiments:
            all_experiments.append({
                'experiment_id': fb_exp['experiment_id'],
                'source': 'fallback',
                'current_phase': fb_exp.get('current_phase', 'unknown'),
                'progress': fb_exp.get('progress_percentage', 0)
            })
        
        if not all_experiments:
            print("No experiments found")
            return
        
        print(f"\nFound {len(all_experiments)} experiments:")
        print("-" * 80)
        
        for i, exp in enumerate(all_experiments):
            exp_id = exp['experiment_id']
            source = exp['source']
            
            if source == 'database':
                size_mb = exp['file_size'] / (1024 * 1024)
                print(f"{i+1:2d}. {exp_id} [DB] - {size_mb:.1f}MB - {exp['last_modified']}")
            else:
                phase = exp.get('current_phase', 'unknown')
                progress = exp.get('progress', 0)
                print(f"{i+1:2d}. {exp_id} [FB] - {phase} ({progress:.1f}%)")
        
        try:
            choice = input(f"\nEnter experiment number (1-{len(all_experiments)}) or press Enter for most recent: ").strip()
            
            if not choice:
                # Use most recent experiment
                selected_exp = all_experiments[0]
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(all_experiments):
                    selected_exp = all_experiments[idx]
                else:
                    print("Invalid choice")
                    return
            
            experiment_id = selected_exp['experiment_id']
            
            # Load the experiment data
            if selected_exp['source'] == 'database':
                db_path = selected_exp.get('db_path', f"experiment_{experiment_id}.db")
                print(f"Loading from database: {db_path}")
                experiment_data = load_experiment_from_sqlite(db_path)
            else:
                print(f"Loading from fallback checkpoint: {experiment_id}")
                experiment_data = checkpoint_manager.load_experiment_checkpoint(experiment_id)
            
            if experiment_data:
                print(f"\n{'='*80}")
                print(f"EXPERIMENT: {experiment_id}")
                print(f"SOURCE: {selected_exp['source'].upper()}")
                inspect_experiment_results(experiment_data)
            else:
                print("❌ Could not load experiment data")
        
        except (ValueError, KeyboardInterrupt):
            print("\nCancelled")
            return

if __name__ == "__main__":
    main()