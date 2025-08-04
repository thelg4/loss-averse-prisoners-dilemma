#!/usr/bin/env python3
"""
Generation-by-Generation Evolution Analysis
Track how agents evolved across all generations
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from datetime import datetime
import numpy as np
import sys

def analyze_generation_evolution(experiment_state: dict) -> dict:
    """Analyze how the population evolved generation by generation"""
    
    emergent_results = experiment_state.get('emergent_results', [])
    
    if not emergent_results:
        print("No emergent results found for generation analysis")
        return {}
    
    generation_data = []
    
    for result in emergent_results:
        generation = result.get('generation', 0)
        snapshot = result.get('snapshot', {})
        
        generation_record = {
            'generation': generation,
            'population_size': snapshot.get('population_size', 0),
            'avg_trust_level': snapshot.get('avg_trust_level', 0.5),
            'avg_loss_sensitivity': snapshot.get('avg_loss_sensitivity', 1.0),
            'dominant_traits': snapshot.get('dominant_traits', []),
            'timestamp': result.get('timestamp', 'unknown')
        }
        
        generation_data.append(generation_record)
    
    # Sort by generation
    generation_data.sort(key=lambda x: x['generation'])
    
    return {
        'generation_timeline': generation_data,
        'evolution_metrics': calculate_evolution_metrics(generation_data),
        'trait_evolution': analyze_trait_evolution(generation_data),
        'psychological_trends': analyze_psychological_trends(generation_data)
    }

def calculate_evolution_metrics(generation_data: list) -> dict:
    """Calculate key evolution metrics across generations"""
    
    if not generation_data:
        return {}
    
    # Trust level evolution
    trust_levels = [g['avg_trust_level'] for g in generation_data]
    trust_trend = np.polyfit(range(len(trust_levels)), trust_levels, 1)[0]  # Linear trend
    
    # Loss sensitivity evolution
    loss_sensitivities = [g['avg_loss_sensitivity'] for g in generation_data]
    loss_trend = np.polyfit(range(len(loss_sensitivities)), loss_sensitivities, 1)[0]
    
    # Calculate volatility (how much traits changed)
    trust_volatility = np.std(trust_levels) if len(trust_levels) > 1 else 0
    loss_volatility = np.std(loss_sensitivities) if len(loss_sensitivities) > 1 else 0
    
    return {
        'trust_evolution': {
            'initial_level': trust_levels[0] if trust_levels else 0.5,
            'final_level': trust_levels[-1] if trust_levels else 0.5,
            'total_change': trust_levels[-1] - trust_levels[0] if len(trust_levels) > 1 else 0,
            'trend_slope': trust_trend,
            'volatility': trust_volatility,
            'direction': 'increasing' if trust_trend > 0.01 else 'decreasing' if trust_trend < -0.01 else 'stable'
        },
        'loss_sensitivity_evolution': {
            'initial_level': loss_sensitivities[0] if loss_sensitivities else 1.0,
            'final_level': loss_sensitivities[-1] if loss_sensitivities else 1.0,
            'total_change': loss_sensitivities[-1] - loss_sensitivities[0] if len(loss_sensitivities) > 1 else 0,
            'trend_slope': loss_trend,
            'volatility': loss_volatility,
            'direction': 'increasing' if loss_trend > 0.01 else 'decreasing' if loss_trend < -0.01 else 'stable'
        },
        'evolution_summary': {
            'total_generations': len(generation_data),
            'major_trust_shifts': sum(1 for i in range(1, len(trust_levels)) 
                                    if abs(trust_levels[i] - trust_levels[i-1]) > 0.1),
            'major_loss_shifts': sum(1 for i in range(1, len(loss_sensitivities)) 
                                   if abs(loss_sensitivities[i] - loss_sensitivities[i-1]) > 0.2)
        }
    }

def analyze_trait_evolution(generation_data: list) -> dict:
    """Analyze how dominant traits evolved over generations"""
    
    trait_timeline = {}
    all_traits = set()
    
    for gen_data in generation_data:
        generation = gen_data['generation']
        traits = gen_data.get('dominant_traits', [])
        
        # Count occurrences of each trait
        trait_counts = {}
        for trait in traits:
            trait_counts[trait] = trait_counts.get(trait, 0) + 1
            all_traits.add(trait)
        
        trait_timeline[generation] = trait_counts
    
    # Analyze trait emergence and disappearance
    trait_emergence = {}
    
    for trait in all_traits:
        appearances = []
        for generation, trait_counts in trait_timeline.items():
            count = trait_counts.get(trait, 0)
            appearances.append((generation, count))
        
        # Find first appearance
        first_appearance = next((gen for gen, count in appearances if count > 0), None)
        
        # Find peak
        peak_generation, peak_count = max(appearances, key=lambda x: x[1])
        
        trait_emergence[trait] = {
            'first_appearance': first_appearance,
            'peak_generation': peak_generation,
            'peak_count': peak_count,
            'trajectory': appearances
        }
    
    return {
        'trait_timeline': trait_timeline,
        'trait_emergence_patterns': trait_emergence,
        'dominant_trait_evolution': _analyze_dominant_trait_shifts(trait_timeline),
        'trait_diversity': _calculate_trait_diversity(trait_timeline)
    }

def _analyze_dominant_trait_shifts(trait_timeline: dict) -> dict:
    """Analyze shifts in population's dominant traits"""
    
    shifts = []
    previous_dominant = None
    
    for generation in sorted(trait_timeline.keys()):
        trait_counts = trait_timeline[generation]
        
        if trait_counts:
            current_dominant = max(trait_counts.items(), key=lambda x: x[1])[0]
            
            if previous_dominant and current_dominant != previous_dominant:
                shifts.append({
                    'generation': generation,
                    'from_trait': previous_dominant,
                    'to_trait': current_dominant,
                    'shift_type': _classify_trait_shift(previous_dominant, current_dominant)
                })
            
            previous_dominant = current_dominant
    
    return {
        'total_shifts': len(shifts),
        'shift_events': shifts,
        'shift_frequency': len(shifts) / len(trait_timeline) if trait_timeline else 0
    }

def _classify_trait_shift(from_trait: str, to_trait: str) -> str:
    """Classify the type of trait shift"""
    
    # Define trait relationships
    cooperative_traits = ['trusting', 'balanced', 'cooperative_optimist']
    defensive_traits = ['paranoid', 'loss_averse', 'traumatized_paranoid']
    
    from_cooperative = from_trait in cooperative_traits
    to_cooperative = to_trait in cooperative_traits
    
    if from_cooperative and not to_cooperative:
        return 'cooperation_to_defense'
    elif not from_cooperative and to_cooperative:
        return 'defense_to_cooperation'
    elif from_cooperative and to_cooperative:
        return 'cooperative_shift'
    else:
        return 'defensive_shift'

def _calculate_trait_diversity(trait_timeline: dict) -> dict:
    """Calculate trait diversity metrics over time"""
    
    diversity_timeline = []
    
    for generation in sorted(trait_timeline.keys()):
        trait_counts = trait_timeline[generation]
        
        if trait_counts:
            # Shannon diversity index
            total_count = sum(trait_counts.values())
            shannon_diversity = -sum((count/total_count) * np.log(count/total_count) 
                                   for count in trait_counts.values() if count > 0)
            
            # Simple diversity (number of different traits)
            simple_diversity = len(trait_counts)
            
            diversity_timeline.append({
                'generation': generation,
                'shannon_diversity': shannon_diversity,
                'simple_diversity': simple_diversity,
                'dominant_trait_proportion': max(trait_counts.values()) / total_count
            })
    
    return {
        'diversity_timeline': diversity_timeline,
        'avg_shannon_diversity': np.mean([d['shannon_diversity'] for d in diversity_timeline]) if diversity_timeline else 0,
        'diversity_trend': 'increasing' if len(diversity_timeline) > 1 and diversity_timeline[-1]['shannon_diversity'] > diversity_timeline[0]['shannon_diversity'] else 'decreasing'
    }

def analyze_psychological_trends(generation_data: list) -> dict:
    """Analyze psychological trends and patterns"""
    
    # Trust level patterns
    trust_levels = [g['avg_trust_level'] for g in generation_data]
    loss_sensitivities = [g['avg_loss_sensitivity'] for g in generation_data]
    
    # Identify phases of evolution
    phases = identify_evolution_phases(trust_levels, loss_sensitivities)
    
    # Look for cyclical patterns
    trust_cycles = detect_cycles(trust_levels)
    loss_cycles = detect_cycles(loss_sensitivities)
    
    return {
        'evolution_phases': phases,
        'cyclical_patterns': {
            'trust_cycles': trust_cycles,
            'loss_sensitivity_cycles': loss_cycles
        },
        'stability_analysis': analyze_stability(trust_levels, loss_sensitivities),
        'convergence_analysis': analyze_convergence(trust_levels, loss_sensitivities)
    }

def identify_evolution_phases(trust_levels: list, loss_sensitivities: list) -> list:
    """Identify distinct phases in psychological evolution"""
    
    phases = []
    current_phase = None
    phase_start = 0
    
    for i in range(1, len(trust_levels)):
        trust_change = trust_levels[i] - trust_levels[i-1]
        loss_change = loss_sensitivities[i] - loss_sensitivities[i-1]
        
        # Determine phase type
        if trust_change > 0.05 and loss_change < -0.1:
            phase_type = 'recovery'
        elif trust_change < -0.05 and loss_change > 0.1:
            phase_type = 'trauma_response'
        elif abs(trust_change) < 0.02 and abs(loss_change) < 0.05:
            phase_type = 'stabilization'
        else:
            phase_type = 'transition'
        
        if current_phase != phase_type:
            if current_phase is not None:
                phases.append({
                    'phase_type': current_phase,
                    'start_generation': phase_start,
                    'end_generation': i-1,
                    'duration': i - phase_start
                })
            
            current_phase = phase_type
            phase_start = i
    
    # Add final phase
    if current_phase is not None:
        phases.append({
            'phase_type': current_phase,
            'start_generation': phase_start,
            'end_generation': len(trust_levels) - 1,
            'duration': len(trust_levels) - phase_start
        })
    
    return phases

def detect_cycles(values: list, min_cycle_length: int = 5) -> dict:
    """Detect cyclical patterns in psychological metrics"""
    
    if len(values) < min_cycle_length * 2:
        return {'cycles_detected': False, 'reason': 'insufficient_data'}
    
    # Simple cycle detection using autocorrelation
    autocorr_values = []
    
    for lag in range(1, len(values) // 2):
        correlation = np.corrcoef(values[:-lag], values[lag:])[0, 1]
        if not np.isnan(correlation):
            autocorr_values.append((lag, correlation))
    
    # Find peaks in autocorrelation (potential cycle lengths)
    potential_cycles = []
    for i in range(1, len(autocorr_values) - 1):
        lag, corr = autocorr_values[i]
        prev_corr = autocorr_values[i-1][1]
        next_corr = autocorr_values[i+1][1]
        
        if corr > prev_corr and corr > next_corr and corr > 0.3:
            potential_cycles.append({'cycle_length': lag, 'strength': corr})
    
    return {
        'cycles_detected': len(potential_cycles) > 0,
        'potential_cycles': potential_cycles,
        'strongest_cycle': max(potential_cycles, key=lambda x: x['strength']) if potential_cycles else None
    }

def analyze_stability(trust_levels: list, loss_sensitivities: list) -> dict:
    """Analyze psychological stability over time"""
    
    # Calculate rolling volatility
    window_size = min(10, len(trust_levels) // 3)
    
    trust_volatilities = []
    loss_volatilities = []
    
    for i in range(window_size, len(trust_levels)):
        trust_window = trust_levels[i-window_size:i]
        loss_window = loss_sensitivities[i-window_size:i]
        
        trust_volatilities.append(np.std(trust_window))
        loss_volatilities.append(np.std(loss_window))
    
    return {
        'trust_stability': {
            'average_volatility': np.mean(trust_volatilities) if trust_volatilities else 0,
            'volatility_trend': 'increasing' if len(trust_volatilities) > 1 and trust_volatilities[-1] > trust_volatilities[0] else 'decreasing' if len(trust_volatilities) > 1 else 'stable',
            'most_stable_period': np.argmin(trust_volatilities) if trust_volatilities else None
        },
        'loss_sensitivity_stability': {
            'average_volatility': np.mean(loss_volatilities) if loss_volatilities else 0,
            'volatility_trend': 'increasing' if len(loss_volatilities) > 1 and loss_volatilities[-1] > loss_volatilities[0] else 'decreasing' if len(loss_volatilities) > 1 else 'stable',
            'most_stable_period': np.argmin(loss_volatilities) if loss_volatilities else None
        }
    }

def analyze_convergence(trust_levels: list, loss_sensitivities: list) -> dict:
    """Analyze whether population is converging to stable values"""
    
    if len(trust_levels) < 10:
        return {'convergence_detected': False, 'reason': 'insufficient_data'}
    
    # Look at the last 25% of generations
    recent_portion = max(5, len(trust_levels) // 4)
    recent_trust = trust_levels[-recent_portion:]
    recent_loss = loss_sensitivities[-recent_portion:]
    
    # Check if values are stabilizing
    trust_range = max(recent_trust) - min(recent_trust)
    loss_range = max(recent_loss) - min(recent_loss)
    
    trust_converging = trust_range < 0.1  # Within 10% range
    loss_converging = loss_range < 0.2   # Within 20% range
    
    return {
        'trust_convergence': {
            'converging': trust_converging,
            'recent_range': trust_range,
            'convergence_value': np.mean(recent_trust) if trust_converging else None
        },
        'loss_sensitivity_convergence': {
            'converging': loss_converging,
            'recent_range': loss_range,
            'convergence_value': np.mean(recent_loss) if loss_converging else None
        },
        'overall_convergence': trust_converging and loss_converging
    }

def create_evolution_visualizations(evolution_analysis: dict, output_dir: str = "evolution_analysis"):
    """Create visualizations of psychological evolution"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    generation_timeline = evolution_analysis.get('generation_timeline', [])
    
    if not generation_timeline:
        print("No generation data available for visualization")
        return
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame(generation_timeline)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Population Psychological Evolution Over Generations', fontsize=16, fontweight='bold')
    
    # Plot 1: Trust Level Evolution
    axes[0, 0].plot(df['generation'], df['avg_trust_level'], 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].set_title('Average Trust Level Evolution')
    axes[0, 0].set_xlabel('Generation')
    axes[0, 0].set_ylabel('Trust Level')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)
    
    # Plot 2: Loss Sensitivity Evolution
    axes[0, 1].plot(df['generation'], df['avg_loss_sensitivity'], 'r-', linewidth=2, marker='s', markersize=4)
    axes[0, 1].set_title('Average Loss Sensitivity Evolution')
    axes[0, 1].set_xlabel('Generation')
    axes[0, 1].set_ylabel('Loss Sensitivity')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Combined Trust vs Loss Sensitivity
    scatter = axes[1, 0].scatter(df['avg_trust_level'], df['avg_loss_sensitivity'], 
                                c=df['generation'], cmap='viridis', s=50, alpha=0.7)
    axes[1, 0].set_title('Trust Level vs Loss Sensitivity Trajectory')
    axes[1, 0].set_xlabel('Trust Level')
    axes[1, 0].set_ylabel('Loss Sensitivity')
    axes[1, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[1, 0], label='Generation')
    
    # Plot 4: Trait Diversity Over Time
    trait_evolution = evolution_analysis.get('trait_evolution', {})
    diversity_data = trait_evolution.get('trait_diversity', {}).get('diversity_timeline', [])
    
    if diversity_data:
        diversity_df = pd.DataFrame(diversity_data)
        axes[1, 1].plot(diversity_df['generation'], diversity_df['shannon_diversity'], 
                       'g-', linewidth=2, marker='^', markersize=4, label='Shannon Diversity')
        axes[1, 1].plot(diversity_df['generation'], diversity_df['simple_diversity'], 
                       'm--', linewidth=2, marker='v', markersize=4, label='Simple Diversity')
        axes[1, 1].set_title('Trait Diversity Evolution')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Diversity Index')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'No trait diversity data available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Trait Diversity Evolution')
    
    plt.tight_layout()
    plt.savefig(output_path / 'psychological_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create trait evolution heatmap if data is available
    trait_timeline = trait_evolution.get('trait_timeline', {})
    if trait_timeline:
        create_trait_evolution_heatmap(trait_timeline, output_path)
    
    print(f"Evolution visualizations saved to {output_path}/")

def create_trait_evolution_heatmap(trait_timeline: dict, output_path: Path):
    """Create heatmap showing trait evolution over generations"""
    
    # Get all unique traits
    all_traits = set()
    for trait_counts in trait_timeline.values():
        all_traits.update(trait_counts.keys())
    
    # Create matrix for heatmap
    generations = sorted(trait_timeline.keys())
    traits = sorted(all_traits)
    
    heatmap_data = []
    for generation in generations:
        trait_counts = trait_timeline[generation]
        row = [trait_counts.get(trait, 0) for trait in traits]
        heatmap_data.append(row)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(np.array(heatmap_data).T, 
                xticklabels=generations, 
                yticklabels=traits,
                cmap='YlOrRd', 
                annot=True, 
                fmt='d',
                cbar_kws={'label': 'Agent Count'})
    
    plt.title('Trait Distribution Across Generations', fontsize=14, fontweight='bold')
    plt.xlabel('Generation')
    plt.ylabel('Psychological Traits')
    plt.tight_layout()
    plt.savefig(output_path / 'trait_evolution_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

def print_evolution_summary(evolution_analysis: dict):
    """Print a summary of the evolution analysis"""
    
    print("\n" + "="*60)
    print("GENERATION EVOLUTION SUMMARY")
    print("="*60)
    
    # Basic metrics
    timeline = evolution_analysis.get('generation_timeline', [])
    if timeline:
        print(f"Total Generations Analyzed: {len(timeline)}")
        print(f"Generation Range: {timeline[0]['generation']} to {timeline[-1]['generation']}")
    
    # Evolution metrics
    evolution_metrics = evolution_analysis.get('evolution_metrics', {})
    
    trust_evolution = evolution_metrics.get('trust_evolution', {})
    if trust_evolution:
        print(f"\nTrust Level Evolution:")
        print(f"  Initial: {trust_evolution.get('initial_level', 0):.3f}")
        print(f"  Final: {trust_evolution.get('final_level', 0):.3f}")
        print(f"  Total Change: {trust_evolution.get('total_change', 0):+.3f}")
        print(f"  Direction: {trust_evolution.get('direction', 'unknown')}")
    
    loss_evolution = evolution_metrics.get('loss_sensitivity_evolution', {})
    if loss_evolution:
        print(f"\nLoss Sensitivity Evolution:")
        print(f"  Initial: {loss_evolution.get('initial_level', 0):.3f}")
        print(f"  Final: {loss_evolution.get('final_level', 0):.3f}")
        print(f"  Total Change: {loss_evolution.get('total_change', 0):+.3f}")
        print(f"  Direction: {loss_evolution.get('direction', 'unknown')}")
    
    # Trait evolution
    trait_evolution = evolution_analysis.get('trait_evolution', {})
    dominant_shifts = trait_evolution.get('dominant_trait_evolution', {})
    
    if dominant_shifts:
        shifts = dominant_shifts.get('shift_events', [])
        print(f"\nTrait Evolution:")
        print(f"  Total Dominant Trait Shifts: {len(shifts)}")
        
        if shifts:
            print("  Major Shifts:")
            for shift in shifts[-3:]:  # Show last 3 shifts
                print(f"    Gen {shift['generation']}: {shift['from_trait']} → {shift['to_trait']} ({shift['shift_type']})")
    
    # Convergence analysis
    psychological_trends = evolution_analysis.get('psychological_trends', {})
    convergence = psychological_trends.get('convergence_analysis', {})
    
    if convergence:
        print(f"\nConvergence Analysis:")
        trust_conv = convergence.get('trust_convergence', {})
        loss_conv = convergence.get('loss_sensitivity_convergence', {})
        
        print(f"  Trust Converging: {trust_conv.get('converging', False)}")
        if trust_conv.get('converging'):
            print(f"    Convergence Value: {trust_conv.get('convergence_value', 0):.3f}")
        
        print(f"  Loss Sensitivity Converging: {loss_conv.get('converging', False)}")
        if loss_conv.get('converging'):
            print(f"    Convergence Value: {loss_conv.get('convergence_value', 0):.3f}")
        
        print(f"  Overall Convergence: {convergence.get('overall_convergence', False)}")

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

def main():
    """Main function to run generation analysis"""
    
    # Import the detailed agent inspector
    sys.path.append('.')
    
    try:
        # Use the database inspector to get experiment data
        from detailed_agent_inspector import extract_all_agent_data
        
        # Find most recent experiment
        most_recent_db = find_most_recent_experiment()
        if not most_recent_db:
            print("No experiment databases found")
            return
        
        print(f"Analyzing generations from: {most_recent_db}")
        
        # Extract experiment data
        experiment_state = extract_all_agent_data(str(most_recent_db))
        if not experiment_state:
            print("Failed to extract experiment data")
            return
        
        # Analyze generation evolution
        print("Analyzing generation-by-generation evolution...")
        evolution_analysis = analyze_generation_evolution(experiment_state)
        
        if not evolution_analysis:
            print("No generation data found for analysis")
            return
        
        # Save analysis results
        output_dir = Path("generation_analysis")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "generation_evolution_analysis.json", 'w') as f:
            json.dump(evolution_analysis, f, indent=2, default=str)
        
        # Create visualizations
        create_evolution_visualizations(evolution_analysis, str(output_dir))
        
        # Print summary
        print_evolution_summary(evolution_analysis)
        
        print(f"\nComplete generation analysis saved to {output_dir}/")
        
    except Exception as e:
        print(f"Error during generation analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
    except ImportError as e:
        print(f"Missing required packages: {e}")
        print("Install with: pip install matplotlib seaborn pandas numpy")
        sys.exit(1)
    
    main()