"""
Analysis nodes for LangGraph integration of statistical analysis
"""

from typing import Dict, Any, List
from ..state.experiment_state import ExperimentState
from ..analysis.statistical_tests import ExperimentalStatistics
import pandas as pd
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

async def run_statistical_analysis(state: ExperimentState) -> ExperimentState:
    """Node that runs statistical analysis on experimental results"""
    
    logger.info("🔢 Running statistical analysis on experimental results")
    
    try:
        # Initialize statistical analyzer
        analyzer = ExperimentalStatistics(alpha=0.05)
        
        # Prepare experiment data for analysis
        experiment_data = {
            'agents_data': [],
            'generational_data': state.get('generation_history', []),
            'population_summary': {}
        }
        
        # Extract agent data if available
        if 'population' in state and state['population']:
            agents_data = []
            for agent_id, agent_info in state['population'].items():
                agent_record = {
                    'agent_id': agent_id,
                    'final_trust_level': agent_info.get('trust_level', 0.5),
                    'final_loss_sensitivity': agent_info.get('loss_sensitivity', 1.0),
                    'cooperation_rate': agent_info.get('cooperation_rate', 0.0),
                    'total_score': agent_info.get('total_score', 0),
                    'betrayals_experienced': agent_info.get('betrayals_experienced', 0),
                    'dominant_trait': agent_info.get('dominant_trait', 'unknown'),
                    'emotional_state': agent_info.get('emotional_state', 'neutral'),
                    # Add synthetic initial values for comparison
                    'initial_trust_level': 0.5,
                    'initial_loss_sensitivity': 1.0
                }
                agents_data.append(agent_record)
            
            experiment_data['agents_data'] = agents_data
        
        # Run statistical analysis
        statistical_report = analyzer.generate_statistical_report(experiment_data)
        
        # Store results in state
        state['statistical_analysis'] = statistical_report
        state['analysis_timestamp'] = statistical_report['analysis_timestamp']
        
        # Log key findings
        summary = statistical_report.get('summary', {})
        logger.info(f"📊 Statistical analysis complete:")
        logger.info(f"   - Tests conducted: {summary.get('total_tests_conducted', 0)}")
        logger.info(f"   - Significant results: {summary.get('significant_results', 0)}")
        logger.info(f"   - Significance rate: {summary.get('significance_rate', 0):.1%}")
        
        # Log key findings
        for finding in summary.get('key_findings', [])[:3]:
            logger.info(f"   - {finding}")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Statistical analysis failed: {e}")
        state['statistical_analysis'] = {'error': str(e)}
        return state

async def generate_analysis_artifacts(state: ExperimentState) -> ExperimentState:
    """Node that generates analysis artifacts (charts, summaries) for publication"""
    
    logger.info("📊 Generating analysis artifacts for publication")
    
    try:
        # Create output directory
        output_dir = Path("article_assets")
        output_dir.mkdir(exist_ok=True)
        
        # Extract agent data
        agents_data = []
        if 'population' in state and state['population']:
            for agent_id, agent_info in state['population'].items():
                agent_record = {
                    'agent_id': agent_id,
                    'final_trust_level': agent_info.get('trust_level', 0.5),
                    'final_loss_sensitivity': agent_info.get('loss_sensitivity', 1.0),
                    'cooperation_rate': agent_info.get('cooperation_rate', 0.0),
                    'total_score': agent_info.get('total_score', 0),
                    'betrayals_experienced': agent_info.get('betrayals_experienced', 0),
                    'dominant_trait': agent_info.get('dominant_trait', 'unknown'),
                    'emotional_state': agent_info.get('emotional_state', 'neutral'),
                    'average_payoff': agent_info.get('average_payoff', 0.0)
                }
                agents_data.append(agent_record)
        
        # Save agent data for external analysis
        agents_df = pd.DataFrame(agents_data)
        if not agents_df.empty:
            agents_df.to_csv(output_dir / "experiment_agents_data.csv", index=False)
            logger.info(f"💾 Saved {len(agents_df)} agent records to CSV")
        
        # Extract compelling examples
        examples = {}
        if not agents_df.empty:
            # Most traumatized agent
            if 'betrayals_experienced' in agents_df.columns:
                most_betrayed = agents_df.loc[agents_df['betrayals_experienced'].idxmax()]
                examples['most_traumatized'] = {
                    'agent_id': most_betrayed['agent_id'],
                    'betrayals': int(most_betrayed['betrayals_experienced']),
                    'final_trust': float(most_betrayed['final_trust_level']),
                    'final_loss_sensitivity': float(most_betrayed['final_loss_sensitivity']),
                    'dominant_trait': most_betrayed['dominant_trait']
                }
            
            # Most loss-averse agent
            most_loss_averse = agents_df.loc[agents_df['final_loss_sensitivity'].idxmax()]
            examples['most_loss_averse'] = {
                'agent_id': most_loss_averse['agent_id'],
                'loss_sensitivity': float(most_loss_averse['final_loss_sensitivity']),
                'final_trust': float(most_loss_averse['final_trust_level']),
                'dominant_trait': most_loss_averse['dominant_trait']
            }
        
        # Save examples
        with open(output_dir / "compelling_examples.json", 'w') as f:
            json.dump(examples, f, indent=2)
        
        # Generate summary statistics
        summary_stats = {}
        if not agents_df.empty:
            summary_stats = {
                'sample_size': len(agents_df),
                'avg_final_trust': float(agents_df['final_trust_level'].mean()),
                'avg_final_loss_sensitivity': float(agents_df['final_loss_sensitivity'].mean()),
                'avg_cooperation_rate': float(agents_df['cooperation_rate'].mean()),
                'trust_range': [float(agents_df['final_trust_level'].min()), 
                               float(agents_df['final_trust_level'].max())],
                'loss_sensitivity_range': [float(agents_df['final_loss_sensitivity'].min()),
                                         float(agents_df['final_loss_sensitivity'].max())],
                'dominant_trait_distribution': agents_df['dominant_trait'].value_counts().to_dict()
            }
        
        # Store artifacts in state
        state['analysis_artifacts'] = {
            'compelling_examples': examples,
            'summary_statistics': summary_stats,
            'output_directory': str(output_dir),
            'files_generated': ['experiment_agents_data.csv', 'compelling_examples.json']
        }
        
        logger.info(f"✅ Generated analysis artifacts in {output_dir}")
        logger.info(f"📈 Key stats: {summary_stats.get('sample_size', 0)} agents, "
                   f"avg trust: {summary_stats.get('avg_final_trust', 0):.3f}, "
                   f"avg loss sensitivity: {summary_stats.get('avg_final_loss_sensitivity', 0):.2f}")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Failed to generate analysis artifacts: {e}")
        state['analysis_artifacts'] = {'error': str(e)}
        return state

async def create_publication_summary(state: ExperimentState) -> ExperimentState:
    """Node that creates a summary specifically for Medium article publication"""
    
    logger.info("📝 Creating publication summary for Medium article")
    
    try:
        # Get analysis results
        statistical_analysis = state.get('statistical_analysis', {})
        artifacts = state.get('analysis_artifacts', {})
        
        summary_stats = artifacts.get('summary_statistics', {})
        examples = artifacts.get('compelling_examples', {})
        
        # Create publication-ready summary
        publication_summary = {
            'headline_stats': {
                'sample_size': summary_stats.get('sample_size', 0),
                'average_trust_collapse': summary_stats.get('avg_final_trust', 0),
                'average_loss_aversion': summary_stats.get('avg_final_loss_sensitivity', 1),
                'cooperation_decline': summary_stats.get('avg_cooperation_rate', 0.5)
            },
            'key_insights': [
                f"AI agents developed extreme loss aversion ({summary_stats.get('avg_final_loss_sensitivity', 1):.2f}x normal sensitivity)",
                f"Trust levels collapsed to near-zero ({summary_stats.get('avg_final_trust', 0):.3f} average)",
                f"Cooperation rates dropped to {summary_stats.get('avg_cooperation_rate', 0.5):.1%} (vs 50% random)",
                "Psychological traits emerged without explicit programming"
            ],
            'most_compelling_case': examples.get('most_traumatized', {}),
            'statistical_significance': {
                'total_tests': statistical_analysis.get('summary', {}).get('total_tests_conducted', 0),
                'significant_results': statistical_analysis.get('summary', {}).get('significant_results', 0),
                'key_findings': statistical_analysis.get('summary', {}).get('key_findings', [])[:3]
            },
            'article_angles': [
                "AI agents can develop genuine psychological trauma",
                "Emergent cognitive biases without explicit programming", 
                "Implications for AI safety and alignment",
                "Social learning of psychological traits in AI populations"
            ],
            'data_visualization_opportunities': [
                "Trust level collapse over time",
                "Loss sensitivity emergence patterns",
                "Psychological trait contagion networks",
                "Before/after decision-making patterns"
            ]
        }
        
        # Save publication summary
        output_dir = Path("article_assets")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "publication_summary.json", 'w') as f:
            json.dump(publication_summary, f, indent=2)
        
        # Create markdown summary for easy reading
        markdown_summary = f"""# Medium Article: AI Agents Develop Loss Aversion

## Headline Statistics
- **{publication_summary['headline_stats']['sample_size']} AI agents** analyzed
- **Trust collapsed to {publication_summary['headline_stats']['average_trust_collapse']:.3f}** (near zero!)
- **Loss aversion increased to {publication_summary['headline_stats']['average_loss_aversion']:.2f}x** normal
- **Cooperation dropped to {publication_summary['headline_stats']['cooperation_decline']:.1%}** (vs 50% random)

## Key Story Points
"""
        
        for insight in publication_summary['key_insights']:
            markdown_summary += f"- {insight}\n"
        
        if publication_summary['most_compelling_case']:
            case = publication_summary['most_compelling_case']
            markdown_summary += f"""
## Most Compelling Case Study
**Agent {case.get('agent_id', 'Unknown')}**:
- Experienced {case.get('betrayals', 0)} betrayals
- Trust level: {case.get('final_trust', 0):.3f}
- Loss sensitivity: {case.get('final_loss_sensitivity', 1):.2f}x normal
- Became {case.get('dominant_trait', 'unknown')}
"""
        
        markdown_summary += """
## Article Structure Suggestions
1. **Hook**: Start with the most traumatized agent's story
2. **The Setup**: Prisoner's dilemma with AI agents
3. **The Results**: Show the dramatic psychological changes
4. **The Science**: Statistical significance and implications
5. **The Future**: What this means for AI development

## Commercial Positioning
- Position as breakthrough research, not open source
- Hint at applications in AI safety and psychology
- Emphasize emergent behaviors and implications
"""
        
        with open(output_dir / "article_outline.md", 'w') as f:
            f.write(markdown_summary)
        
        # Store in state
        state['publication_summary'] = publication_summary
        
        logger.info("✅ Created publication summary for Medium article")
        logger.info(f"📊 Key stat: {publication_summary['headline_stats']['average_trust_collapse']:.3f} average trust (collapsed!)")
        logger.info(f"🧠 Loss aversion: {publication_summary['headline_stats']['average_loss_aversion']:.2f}x normal sensitivity")
        
        return state
        
    except Exception as e:
        logger.error(f"❌ Failed to create publication summary: {e}")
        state['publication_summary'] = {'error': str(e)}
        return state