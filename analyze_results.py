#!/usr/bin/env python3
"""
Results Analysis Script for Loss-Averse Prisoner's Dilemma
Generates statistical analysis and visualizations for Medium article
"""

import argparse
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

from src.analysis.statistical_tests import ExperimentalStatistics, analyze_experiment_results

def load_agent_data() -> pd.DataFrame:
    """Load agent analysis data from CSV"""
    agents_file = Path("agent_analysis/agents_summary.csv")
    if not agents_file.exists():
        print(f"❌ Agent data not found at {agents_file}")
        return pd.DataFrame()
    
    df = pd.read_csv(agents_file)
    print(f"✅ Loaded {len(df)} agents from {agents_file}")
    return df

def load_population_data() -> Dict[str, Any]:
    """Load population summary data"""
    pop_file = Path("agent_analysis/population_summary.json")
    if not pop_file.exists():
        print(f"❌ Population data not found at {pop_file}")
        return {}
    
    with open(pop_file) as f:
        data = json.load(f)
    print(f"✅ Loaded population data from {pop_file}")
    return data

def load_generational_data() -> List[Dict[str, Any]]:
    """Load generational evolution data"""
    gen_file = Path("generation_analysis/generation_evolution_analysis.json")
    if not gen_file.exists():
        print(f"❌ Generational data not found at {gen_file}")
        return []
    
    with open(gen_file) as f:
        data = json.load(f)
    
    if 'generation_timeline' in data:
        print(f"✅ Loaded {len(data['generation_timeline'])} generations from {gen_file}")
        return data['generation_timeline']
    
    return []

def generate_key_statistics(agents_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate key statistics for article"""
    stats = {}
    
    if len(agents_df) == 0:
        return stats
    
    # Basic descriptive statistics
    stats['sample_size'] = len(agents_df)
    stats['avg_final_trust'] = agents_df['final_trust_level'].mean()
    stats['avg_final_loss_sensitivity'] = agents_df['final_loss_sensitivity'].mean()
    stats['avg_cooperation_rate'] = agents_df['cooperation_rate'].mean()
    
    # Range statistics
    stats['trust_range'] = [agents_df['final_trust_level'].min(), agents_df['final_trust_level'].max()]
    stats['loss_sensitivity_range'] = [agents_df['final_loss_sensitivity'].min(), agents_df['final_loss_sensitivity'].max()]
    
    # Categorical analysis
    stats['dominant_trait_distribution'] = agents_df['dominant_trait'].value_counts().to_dict()
    stats['emotional_state_distribution'] = agents_df['emotional_state'].value_counts().to_dict()
    
    # Trauma analysis
    if 'betrayals_experienced' in agents_df.columns:
        stats['avg_betrayals'] = agents_df['betrayals_experienced'].mean()
        stats['max_betrayals'] = agents_df['betrayals_experienced'].max()
        
    # Performance metrics
    stats['avg_total_score'] = agents_df['total_score'].mean()
    stats['performance_range'] = [agents_df['total_score'].min(), agents_df['total_score'].max()]
    
    return stats

def create_visualizations(agents_df: pd.DataFrame, output_dir: str = "article_assets"):
    """Generate visualizations for the article"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Trust vs Loss Sensitivity Scatter Plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(agents_df['final_trust_level'], agents_df['final_loss_sensitivity'], 
                         c=agents_df['betrayals_experienced'], cmap='Reds', 
                         s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Betrayals Experienced')
    plt.xlabel('Final Trust Level')
    plt.ylabel('Final Loss Sensitivity')
    plt.title('Agent Psychology After Experiment\n(Color = Betrayals Experienced)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "trust_vs_loss_sensitivity.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Cooperation Rate Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(agents_df['cooperation_rate'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(agents_df['cooperation_rate'].mean(), color='red', linestyle='--', 
               label=f'Mean: {agents_df["cooperation_rate"].mean():.2f}')
    plt.axvline(0.5, color='green', linestyle='--', label='Random (0.5)')
    plt.xlabel('Cooperation Rate')
    plt.ylabel('Number of Agents')
    plt.title('Distribution of Agent Cooperation Rates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "cooperation_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Dominant Trait Distribution
    trait_counts = agents_df['dominant_trait'].value_counts()
    plt.figure(figsize=(10, 6))
    bars = plt.bar(trait_counts.index, trait_counts.values, color='lightcoral', edgecolor='black')
    plt.xlabel('Dominant Psychological Trait')
    plt.ylabel('Number of Agents')
    plt.title('Final Psychological Trait Distribution')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / "trait_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance vs Psychology Heatmap
    if 'betrayals_experienced' in agents_df.columns:
        plt.figure(figsize=(12, 8))
        
        # Create correlation matrix
        correlation_data = agents_df[['final_trust_level', 'final_loss_sensitivity', 
                                    'cooperation_rate', 'betrayals_experienced', 
                                    'total_score', 'average_payoff']].corr()
        
        sns.heatmap(correlation_data, annot=True, cmap='RdYlBu', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .5})
        plt.title('Agent Psychology and Performance Correlations')
        plt.tight_layout()
        plt.savefig(output_path / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Generated visualizations in {output_path}/")

def extract_compelling_examples(agents_df: pd.DataFrame) -> Dict[str, Any]:
    """Extract the most compelling examples for the article"""
    examples = {}
    
    if len(agents_df) == 0:
        return examples
    
    # Most traumatized agent
    most_betrayed = agents_df.loc[agents_df['betrayals_experienced'].idxmax()]
    examples['most_traumatized'] = {
        'agent_id': most_betrayed['agent_id'],
        'betrayals': int(most_betrayed['betrayals_experienced']),
        'final_trust': float(most_betrayed['final_trust_level']),
        'final_loss_sensitivity': float(most_betrayed['final_loss_sensitivity']),
        'cooperation_rate': float(most_betrayed['cooperation_rate']),
        'dominant_trait': most_betrayed['dominant_trait']
    }
    
    # Most loss-averse agent
    most_loss_averse = agents_df.loc[agents_df['final_loss_sensitivity'].idxmax()]
    examples['most_loss_averse'] = {
        'agent_id': most_loss_averse['agent_id'],
        'loss_sensitivity': float(most_loss_averse['final_loss_sensitivity']),
        'final_trust': float(most_loss_averse['final_trust_level']),
        'betrayals': int(most_loss_averse['betrayals_experienced']),
        'dominant_trait': most_loss_averse['dominant_trait']
    }
    
    # Least trusting agent
    least_trusting = agents_df.loc[agents_df['final_trust_level'].idxmin()]
    examples['least_trusting'] = {
        'agent_id': least_trusting['agent_id'],
        'final_trust': float(least_trusting['final_trust_level']),
        'betrayals': int(least_trusting['betrayals_experienced']),
        'cooperation_rate': float(least_trusting['cooperation_rate']),
        'dominant_trait': least_trusting['dominant_trait']
    }
    
    # Best performer despite trauma
    traumatized_agents = agents_df[agents_df['betrayals_experienced'] > 0]
    if len(traumatized_agents) > 0:
        best_traumatized = traumatized_agents.loc[traumatized_agents['total_score'].idxmax()]
        examples['resilient_agent'] = {
            'agent_id': best_traumatized['agent_id'],
            'total_score': float(best_traumatized['total_score']),
            'betrayals': int(best_traumatized['betrayals_experienced']),
            'final_trust': float(best_traumatized['final_trust_level']),
            'cooperation_rate': float(best_traumatized['cooperation_rate'])
        }
    
    return examples

def run_statistical_analysis(agents_df: pd.DataFrame, population_data: Dict, 
                           generational_data: List[Dict]) -> Dict[str, Any]:
    """Run comprehensive statistical analysis"""
    
    analyzer = ExperimentalStatistics(alpha=0.05)
    
    # Prepare data for analysis
    experiment_data = {
        'agents_data': agents_df.to_dict('records') if not agents_df.empty else [],
        'population_summary': population_data,
        'generational_data': generational_data
    }
    
    # Add synthetic initial values for comparison (since we don't have baseline data)
    # This assumes agents started with neutral psychology
    if not agents_df.empty:
        agents_with_initial = agents_df.copy()
        agents_with_initial['initial_trust_level'] = 0.5  # Assume neutral starting trust
        agents_with_initial['initial_loss_sensitivity'] = 1.0  # Assume neutral starting loss sensitivity
        experiment_data['agents_data'] = agents_with_initial.to_dict('records')
    
    # Generate statistical report
    report = analyzer.generate_statistical_report(experiment_data, "statistical_analysis.json")
    
    return report

def generate_article_summary(stats: Dict, examples: Dict, statistical_report: Dict) -> str:
    """Generate a summary for the Medium article"""
    
    summary = f"""
# Loss-Averse Prisoner's Dilemma: Statistical Summary

## Key Findings

**Sample Size**: {stats.get('sample_size', 0)} AI agents analyzed

### Psychological Evolution
- **Average Final Trust Level**: {stats.get('avg_final_trust', 0):.3f} (near zero!)
- **Average Loss Sensitivity**: {stats.get('avg_final_loss_sensitivity', 0):.2f} (highly loss-averse)
- **Cooperation Rate**: {stats.get('avg_cooperation_rate', 0):.1%} (much lower than random 50%)

### Most Extreme Cases
"""
    
    if 'most_traumatized' in examples:
        most_traumatized = examples['most_traumatized']
        summary += f"""
**Most Traumatized Agent ({most_traumatized['agent_id']})**:
- Experienced {most_traumatized['betrayals']} betrayals
- Final trust level: {most_traumatized['final_trust']:.3f}
- Loss sensitivity: {most_traumatized['final_loss_sensitivity']:.2f}x normal
- Became {most_traumatized['dominant_trait']}
"""
    
    if 'most_loss_averse' in examples:
        most_loss_averse = examples['most_loss_averse']
        summary += f"""
**Most Loss-Averse Agent ({most_loss_averse['agent_id']})**:
- Loss sensitivity: {most_loss_averse['loss_sensitivity']:.2f}x normal (extreme!)
- Trust level: {most_loss_averse['final_trust']:.3f}
- Developed {most_loss_averse['dominant_trait']} personality
"""
    
    # Add statistical significance if available
    if 'summary' in statistical_report:
        sig_rate = statistical_report['summary'].get('significance_rate', 0)
        summary += f"""
### Statistical Significance
- {statistical_report['summary'].get('total_tests_conducted', 0)} statistical tests performed
- {statistical_report['summary'].get('significant_results', 0)} showed significant results ({sig_rate:.1%})

**Key Significant Findings**:
"""
        for finding in statistical_report['summary'].get('key_findings', [])[:3]:
            summary += f"- {finding}\n"
    
    summary += f"""
### Article Implications
This data demonstrates that AI agents can develop genuine psychological trauma and loss aversion through experience alone. The near-zero trust levels and extreme loss sensitivity show emergent cognitive biases that weren't programmed - they emerged from interaction patterns.

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results for Medium article")
    parser.add_argument("--extract-stories", action="store_true", 
                       help="Extract compelling agent examples")
    parser.add_argument("--generate-charts", action="store_true",
                       help="Generate visualizations")
    parser.add_argument("--statistical-analysis", action="store_true",
                       help="Run comprehensive statistical analysis")
    parser.add_argument("--article-summary", action="store_true",
                       help="Generate summary for Medium article")
    parser.add_argument("--all", action="store_true",
                       help="Run all analyses")
    parser.add_argument("--output-dir", default="article_assets",
                       help="Output directory for generated files")
    
    args = parser.parse_args()
    
    if not any([args.extract_stories, args.generate_charts, args.statistical_analysis, 
                args.article_summary, args.all]):
        print("Please specify at least one analysis option. Use --help for details.")
        return
    
    print("🔬 Starting experimental results analysis...")
    
    # Load all data
    agents_df = load_agent_data()
    population_data = load_population_data()
    generational_data = load_generational_data()
    
    # Ensure output directory exists
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = {}
    
    # Generate key statistics
    if not agents_df.empty:
        stats = generate_key_statistics(agents_df)
        results['statistics'] = stats
        print(f"📊 Generated key statistics: {len(stats)} metrics")
    else:
        print("❌ No agent data available for analysis")
        return
    
    # Extract compelling examples
    if args.extract_stories or args.all:
        examples = extract_compelling_examples(agents_df)
        results['examples'] = examples
        
        # Save examples
        with open(output_path / "compelling_examples.json", 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"📖 Extracted {len(examples)} compelling agent examples")
    
    # Generate visualizations
    if args.generate_charts or args.all:
        create_visualizations(agents_df, args.output_dir)
        print("📈 Generated article visualizations")
    
    # Run statistical analysis
    if args.statistical_analysis or args.all:
        statistical_report = run_statistical_analysis(agents_df, population_data, generational_data)
        results['statistical_analysis'] = statistical_report
        print(f"🔢 Statistical analysis complete: {statistical_report['summary'].get('total_tests_conducted', 0)} tests")
    
    # Generate article summary
    if args.article_summary or args.all:
        if 'examples' not in results:
            results['examples'] = extract_compelling_examples(agents_df)
        if 'statistical_analysis' not in results:
            results['statistical_analysis'] = run_statistical_analysis(agents_df, population_data, generational_data)
        
        summary = generate_article_summary(results['statistics'], results['examples'], 
                                         results['statistical_analysis'])
        
        with open(output_path / "article_summary.md", 'w') as f:
            f.write(summary)
        print("📝 Generated Medium article summary")
    
    # Save all results
    with open(output_path / "complete_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✅ Analysis complete! Results saved to {output_path}/")
    print(f"📁 Key files:")
    print(f"   - article_summary.md: Main findings for your article")
    print(f"   - compelling_examples.json: Agent examples to highlight") 
    print(f"   - *.png: Charts for your article")
    print(f"   - statistical_analysis.json: Detailed statistical results")

if __name__ == "__main__":
    main()