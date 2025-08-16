"""
Statistical analysis module for prisoner's dilemma experiments.
Provides comprehensive statistical testing for agent behavioral changes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, chi2_contingency
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'statistic': self.statistic,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'confidence_interval': list(self.confidence_interval),
            'interpretation': self.interpretation,
            'significant': self.significant
        }

class ExperimentalStatistics:
    """Comprehensive statistical analysis for agent psychology experiments"""
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = {}
        
    def cohen_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def confidence_interval(self, data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        n = len(data)
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
        return (mean - h, mean + h)
    
    def test_trait_emergence(self, 
                           before_values: List[float], 
                           after_values: List[float],
                           trait_name: str) -> StatisticalResult:
        """Test if psychological traits significantly emerged/changed"""
        
        # Use paired t-test if same agents, independent t-test otherwise
        if len(before_values) == len(after_values):
            # Paired t-test for within-subjects design
            statistic, p_value = wilcoxon(before_values, after_values, alternative='two-sided')
            test_name = f"Wilcoxon Signed-Rank Test - {trait_name} Change"
        else:
            # Independent samples t-test
            statistic, p_value = ttest_ind(before_values, after_values)
            test_name = f"Independent T-Test - {trait_name} Difference"
        
        effect_size = self.cohen_d(before_values, after_values)
        ci = self.confidence_interval(after_values)
        
        # Interpret results
        significant = p_value < self.alpha
        if significant:
            if effect_size > 0.8:
                magnitude = "large"
            elif effect_size > 0.5:
                magnitude = "medium" 
            else:
                magnitude = "small"
            interpretation = f"Significant {magnitude} change in {trait_name} (p={p_value:.4f})"
        else:
            interpretation = f"No significant change in {trait_name} (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            significant=significant
        )
    
    def test_population_differences(self,
                                  control_group: List[float],
                                  experimental_group: List[float],
                                  metric_name: str) -> StatisticalResult:
        """Test differences between control and experimental populations"""
        
        # Use Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(control_group, experimental_group, alternative='two-sided')
        
        effect_size = self.cohen_d(control_group, experimental_group)
        ci = self.confidence_interval(experimental_group)
        
        significant = p_value < self.alpha
        if significant:
            direction = "higher" if np.mean(experimental_group) > np.mean(control_group) else "lower"
            interpretation = f"Experimental group shows significantly {direction} {metric_name} (p={p_value:.4f})"
        else:
            interpretation = f"No significant difference in {metric_name} between groups (p={p_value:.4f})"
        
        return StatisticalResult(
            test_name=f"Mann-Whitney U Test - {metric_name}",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=interpretation,
            significant=significant
        )
    
    def analyze_behavioral_patterns(self, agents_data: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Comprehensive analysis of agent behavioral patterns"""
        results = {}
        
        # Test 1: Trust level changes
        if 'initial_trust_level' in agents_data.columns and 'final_trust_level' in agents_data.columns:
            initial_trust = agents_data['initial_trust_level'].dropna().tolist()
            final_trust = agents_data['final_trust_level'].dropna().tolist()
            results['trust_change'] = self.test_trait_emergence(initial_trust, final_trust, "Trust Level")
        
        # Test 2: Loss sensitivity emergence
        if 'initial_loss_sensitivity' in agents_data.columns and 'final_loss_sensitivity' in agents_data.columns:
            initial_loss = agents_data['initial_loss_sensitivity'].dropna().tolist()
            final_loss = agents_data['final_loss_sensitivity'].dropna().tolist()
            results['loss_sensitivity_change'] = self.test_trait_emergence(initial_loss, final_loss, "Loss Sensitivity")
        
        # Test 3: Cooperation rate analysis
        if 'cooperation_rate' in agents_data.columns:
            coop_rates = agents_data['cooperation_rate'].dropna().tolist()
            # Test if cooperation rate significantly differs from 0.5 (random)
            statistic, p_value = ttest_ind(coop_rates, [0.5] * len(coop_rates))
            results['cooperation_deviation'] = StatisticalResult(
                test_name="One-Sample T-Test - Cooperation vs Random",
                statistic=statistic,
                p_value=p_value,
                effect_size=abs(np.mean(coop_rates) - 0.5) / np.std(coop_rates),
                confidence_interval=self.confidence_interval(coop_rates),
                interpretation=f"Cooperation rate significantly differs from random (p={p_value:.4f})" if p_value < self.alpha else "No significant deviation from random cooperation",
                significant=p_value < self.alpha
            )
        
        # Test 4: Trauma impact analysis
        if 'betrayals_experienced' in agents_data.columns and 'final_loss_sensitivity' in agents_data.columns:
            betrayals = agents_data['betrayals_experienced'].tolist()
            loss_sens = agents_data['final_loss_sensitivity'].tolist()
            
            # Correlation between betrayals and loss sensitivity
            correlation, p_value = stats.pearsonr(betrayals, loss_sens)
            results['betrayal_loss_correlation'] = StatisticalResult(
                test_name="Pearson Correlation - Betrayals vs Loss Sensitivity",
                statistic=correlation,
                p_value=p_value,
                effect_size=correlation,  # Correlation is its own effect size
                confidence_interval=(correlation - 0.1, correlation + 0.1),  # Rough CI
                interpretation=f"{'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.5 else 'Weak'} correlation between betrayals and loss sensitivity (r={correlation:.3f}, p={p_value:.4f})",
                significant=p_value < self.alpha
            )
        
        return results
    
    def compare_agent_types(self, 
                           rational_agents: pd.DataFrame, 
                           emergent_agents: pd.DataFrame) -> Dict[str, StatisticalResult]:
        """Compare rational vs emergent-bias agent populations"""
        results = {}
        
        metrics_to_compare = [
            ('final_trust_level', 'Trust Level'),
            ('final_loss_sensitivity', 'Loss Sensitivity'),
            ('cooperation_rate', 'Cooperation Rate'),
            ('total_score', 'Total Score'),
            ('average_payoff', 'Average Payoff')
        ]
        
        for column, metric_name in metrics_to_compare:
            if column in rational_agents.columns and column in emergent_agents.columns:
                rational_values = rational_agents[column].dropna().tolist()
                emergent_values = emergent_agents[column].dropna().tolist()
                
                if len(rational_values) > 0 and len(emergent_values) > 0:
                    results[f'{column}_comparison'] = self.test_population_differences(
                        rational_values, emergent_values, metric_name
                    )
        
        return results
    
    def analyze_contagion_effects(self, 
                                generational_data: List[Dict]) -> Dict[str, StatisticalResult]:
        """Analyze how psychological traits spread through populations over generations"""
        results = {}
        
        if len(generational_data) < 2:
            return results
        
        # Extract trait distributions over generations
        generations = []
        trust_means = []
        loss_means = []
        
        for gen_data in generational_data:
            if 'avg_trust_level' in gen_data and 'avg_loss_sensitivity' in gen_data:
                generations.append(gen_data['generation'])
                trust_means.append(gen_data['avg_trust_level'])
                loss_means.append(gen_data['avg_loss_sensitivity'])
        
        if len(generations) > 2:
            # Test for trend over generations
            trust_corr, trust_p = stats.pearsonr(generations, trust_means)
            loss_corr, loss_p = stats.pearsonr(generations, loss_means)
            
            results['trust_evolution'] = StatisticalResult(
                test_name="Trust Level Evolution Over Generations",
                statistic=trust_corr,
                p_value=trust_p,
                effect_size=trust_corr,
                confidence_interval=self.confidence_interval(trust_means),
                interpretation=f"Trust shows {'significant' if trust_p < self.alpha else 'non-significant'} {'decline' if trust_corr < 0 else 'increase'} over generations (r={trust_corr:.3f})",
                significant=trust_p < self.alpha
            )
            
            results['loss_sensitivity_evolution'] = StatisticalResult(
                test_name="Loss Sensitivity Evolution Over Generations",
                statistic=loss_corr,
                p_value=loss_p,
                effect_size=loss_corr,
                confidence_interval=self.confidence_interval(loss_means),
                interpretation=f"Loss sensitivity shows {'significant' if loss_p < self.alpha else 'non-significant'} {'increase' if loss_corr > 0 else 'decrease'} over generations (r={loss_corr:.3f})",
                significant=loss_p < self.alpha
            )
        
        return results
    
    def generate_statistical_report(self, 
                                   experiment_data: Dict,
                                   output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive statistical analysis report"""
        
        report = {
            'analysis_timestamp': datetime.now().isoformat(),
            'alpha_level': self.alpha,
            'summary': {},
            'detailed_results': {}
        }
        
        # Load and analyze agent data
        if 'agents_data' in experiment_data:
            agents_df = pd.DataFrame(experiment_data['agents_data'])
            behavioral_results = self.analyze_behavioral_patterns(agents_df)
            report['detailed_results']['behavioral_patterns'] = {
                k: v.to_dict() for k, v in behavioral_results.items()
            }
        
        # Compare agent types if available
        if 'rational_agents' in experiment_data and 'emergent_agents' in experiment_data:
            rational_df = pd.DataFrame(experiment_data['rational_agents'])
            emergent_df = pd.DataFrame(experiment_data['emergent_agents'])
            comparison_results = self.compare_agent_types(rational_df, emergent_df)
            report['detailed_results']['agent_type_comparisons'] = {
                k: v.to_dict() for k, v in comparison_results.items()
            }
        
        # Analyze generational evolution
        if 'generational_data' in experiment_data:
            contagion_results = self.analyze_contagion_effects(experiment_data['generational_data'])
            report['detailed_results']['contagion_analysis'] = {
                k: v.to_dict() for k, v in contagion_results.items()
            }
        
        # Generate summary
        all_results = []
        for category in report['detailed_results'].values():
            all_results.extend(category.values())
        
        significant_tests = [r for r in all_results if r['significant']]
        report['summary'] = {
            'total_tests_conducted': len(all_results),
            'significant_results': len(significant_tests),
            'significance_rate': len(significant_tests) / len(all_results) if all_results else 0,
            'key_findings': [r['interpretation'] for r in significant_tests[:5]]  # Top 5 findings
        }
        
        # Save report if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report

def analyze_experiment_results(results_path: str, output_path: str = "statistical_analysis.json") -> Dict[str, Any]:
    """Main function to analyze experimental results"""
    
    # Initialize statistical analyzer
    analyzer = ExperimentalStatistics(alpha=0.05)
    
    # Load experimental data
    # This would need to be adapted based on your actual data format
    experiment_data = {}
    
    # Example of loading different data sources
    if Path(results_path).exists():
        with open(results_path, 'r') as f:
            experiment_data = json.load(f)
    
    # Generate comprehensive report
    report = analyzer.generate_statistical_report(experiment_data, output_path)
    
    return report