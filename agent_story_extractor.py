# #!/usr/bin/env python3
# """
# Agent Story Extractor - Create narrative stories of agent psychological journeys
# """

# import json
# from pathlib import Path
# from datetime import datetime
# from typing import Dict, List, Any
# import re

# def extract_agent_story(agent_history: Dict) -> Dict[str, Any]:
#     """Extract the narrative story of an agent's psychological journey"""
    
#     agent_id = agent_history['agent_id']
    
#     # Build the story structure
#     story = {
#         'agent_id': agent_id,
#         'story_title': generate_story_title(agent_history),
#         'psychological_arc': analyze_psychological_arc(agent_history),
#         'key_moments': identify_key_moments(agent_history),
#         'character_development': track_character_development(agent_history),
#         'relationships': analyze_relationships(agent_history),
#         'internal_monologue': extract_internal_thoughts(agent_history),
#         'story_summary': generate_story_summary(agent_history),
#         'narrative_themes': identify_narrative_themes(agent_history)
#     }
    
#     return story

# def generate_story_title(agent_history: Dict) -> str:
#     """Generate an evocative title for the agent's story"""
    
#     final_trait = agent_history.get('psychological_evolution', {}).get('dominant_trait', 'balanced')
#     trauma_count = agent_history.get('trauma_analysis', {}).get('trauma_count', 0)
#     interaction_summary = agent_history.get('interaction_summary', {})
#     cooperation_rate = interaction_summary.get('cooperation_rate', 0.0)
#     recovery_progress = agent_history.get('recovery_progress', 1.0)
    
#     # Generate title based on psychological journey
#     if trauma_count > 5 and recovery_progress > 0.7:
#         return f"{agent_history['agent_id']}: From Trauma to Trust"
#     elif final_trait == 'traumatized_paranoid':
#         return f"{agent_history['agent_id']}: The Wounded Guardian"
#     elif final_trait == 'trusting' and cooperation_rate > 0.8:
#         return f"{agent_history['agent_id']}: The Eternal Optimist"
#     elif final_trait == 'loss_averse':
#         return f"{agent_history['agent_id']}: Learning to Fear Loss"
#     elif cooperation_rate < 0.3 and interaction_summary.get('total_interactions', 0) > 0:
#         return f"{agent_history['agent_id']}: The Lone Wolf"
#     elif interaction_summary.get('total_interactions', 0) == 0:
#         return f"{agent_history['agent_id']}: The Potential Unawakened"
#     else:
#         return f"{agent_history['agent_id']}: A Journey of Adaptation"

# def analyze_psychological_arc(agent_history: Dict) -> Dict[str, Any]:
#     """Analyze the agent's psychological character arc"""
    
#     evolution = agent_history.get('psychological_evolution', {})
#     trauma_analysis = agent_history.get('trauma_analysis', {})
    
#     # Identify the arc type
#     initial_trust = 0.5  # Assume neutral start
#     final_trust = evolution.get('final_trust_level', 0.5)
    
#     initial_loss_sensitivity = 1.0  # Assume neutral start
#     final_loss_sensitivity = evolution.get('final_loss_sensitivity', 1.0)
    
#     trust_change = final_trust - initial_trust
#     loss_change = final_loss_sensitivity - initial_loss_sensitivity
    
#     # Check if agent had any interactions
#     interaction_summary = agent_history.get('interaction_summary', {})
#     total_interactions = interaction_summary.get('total_interactions', 0)
    
#     # Classify the arc based on available data
#     if total_interactions == 0:
#         arc_type = "untested_potential"
#         arc_description = "An agent with untested potential, initialized with balanced psychology but never challenged by experience."
#     elif trust_change > 0.2 and loss_change < -0.3:
#         arc_type = "redemption"
#         arc_description = "A journey from cynicism to hope, learning to trust again despite past hurts."
#     elif trust_change < -0.2 and loss_change > 0.5:
#         arc_type = "tragic_fall"
#         arc_description = "A descent from innocence to paranoia, shaped by betrayal and loss."
#     elif abs(trust_change) < 0.1 and abs(loss_change) < 0.2:
#         arc_type = "steady_character"
#         arc_description = "A stable personality that adapts without fundamental change."
#     elif trauma_analysis.get('trauma_count', 0) > 3 and agent_history.get('recovery_progress', 1.0) > 0.6:
#         arc_type = "survivor"
#         arc_description = "Enduring significant trauma but finding ways to heal and grow."
#     else:
#         arc_type = "emerging_complexity"
#         arc_description = "A mind in the early stages of psychological development, with patterns yet to emerge."
    
#     return {
#         'arc_type': arc_type,
#         'arc_description': arc_description,
#         'trust_journey': {
#             'starting_point': initial_trust,
#             'ending_point': final_trust,
#             'total_change': trust_change,
#             'change_magnitude': abs(trust_change),
#             'direction': 'increasing' if trust_change > 0.05 else 'decreasing' if trust_change < -0.05 else 'stable'
#         },
#         'loss_sensitivity_journey': {
#             'starting_point': initial_loss_sensitivity,
#             'ending_point': final_loss_sensitivity,
#             'total_change': loss_change,
#             'change_magnitude': abs(loss_change),
#             'direction': 'increasing' if loss_change > 0.1 else 'decreasing' if loss_change < -0.1 else 'stable'
#         },
#         'character_growth_metrics': {
#             'psychological_complexity': len(evolution['learned_heuristics']),
#             'trauma_resilience': agent_history['recovery_progress'],
#             'adaptability': len(evolution['evolution_events'])
#         }
#     }

# def identify_key_moments(agent_history: Dict) -> List[Dict]:
#     """Identify pivotal moments in the agent's story"""
    
#     key_moments = []
    
#     # Analyze reasoning chain for significant events
#     reasoning_chain = agent_history.get('complete_reasoning_chain', [])
#     interaction_summary = agent_history.get('interaction_summary', {})
#     trauma_analysis = agent_history.get('trauma_analysis', {})
    
#     # Check if agent had any interactions
#     total_interactions = interaction_summary.get('total_interactions', 0)
    
#     if total_interactions == 0:
#         key_moments.append({
#             'moment_type': 'initialization',
#             'significance': 'high',
#             'description': "The moment of creation - an AI agent born with balanced psychology, waiting for experience to shape its mind.",
#             'emotional_impact': 'potential',
#             'story_function': 'origin_story'
#         })
#         return key_moments
    
#     # Find first betrayal (if any)
#     betrayals_experienced = interaction_summary.get('betrayals_experienced', 0)
#     if betrayals_experienced > 0:
#         key_moments.append({
#             'moment_type': 'first_betrayal',
#             'significance': 'high',
#             'description': "The first time trust was broken - a defining moment that would shape future decisions.",
#             'emotional_impact': 'devastating',
#             'story_function': 'inciting_incident'
#         })
    
#     # Find major psychological shifts from reasoning chain
#     for step in reasoning_chain:
#         try:
#             step_data = step
#             if hasattr(step, '__dict__'):
#                 step_data = step.__dict__
#             elif hasattr(step, 'dict'):
#                 step_data = step.dict()
            
#             step_type = step_data.get('step_type', '')
#             content = step_data.get('content', '')
            
#             if step_type == 'trust_adjustment' and 'significantly' in content:
#                 key_moments.append({
#                     'moment_type': 'trust_shift',
#                     'significance': 'medium',
#                     'description': f"A moment of psychological adjustment: {content}",
#                     'emotional_impact': 'transformative',
#                     'story_function': 'character_development'
#                 })
            
#             elif step_type == 'trauma_processing' and 'significant' in content:
#                 key_moments.append({
#                     'moment_type': 'trauma_processing',
#                     'significance': 'high',
#                     'description': f"Processing a traumatic experience: {content}",
#                     'emotional_impact': 'painful_but_necessary',
#                     'story_function': 'crisis_and_growth'
#                 })
            
#             elif step_type == 'recovery_assessment' and agent_history.get('recovery_progress', 1.0) > 0.7:
#                 key_moments.append({
#                     'moment_type': 'breakthrough',
#                     'significance': 'high',
#                     'description': "A breakthrough moment in psychological recovery.",
#                     'emotional_impact': 'hopeful',
#                     'story_function': 'resolution'
#                 })
#         except Exception as e:
#             continue
    
#     # Find moments of highest cooperation or defection
#     cooperation_rate = interaction_summary.get('cooperation_rate', 0.0)
#     if cooperation_rate > 0.8:
#         key_moments.append({
#             'moment_type': 'commitment_to_cooperation',
#             'significance': 'medium',
#             'description': "Choosing to maintain faith in cooperation despite past hurts.",
#             'emotional_impact': 'courageous',
#             'story_function': 'heroic_choice'
#         })
#     elif cooperation_rate < 0.2 and total_interactions > 0:
#         key_moments.append({
#             'moment_type': 'withdrawal_from_trust',
#             'significance': 'medium',
#             'description': "The decision to protect oneself by avoiding vulnerability.",
#             'emotional_impact': 'protective_but_isolating',
#             'story_function': 'tragic_choice'
#         })
    
#     # Sort by significance and limit to most important moments
#     key_moments.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['significance']], reverse=True)
    
#     return key_moments[:5]  # Top 5 most significant moments

# def track_character_development(agent_history: Dict) -> Dict[str, Any]:
#     """Track how the character developed over time"""
    
#     evolution = agent_history['psychological_evolution']
#     learned_heuristics = evolution['learned_heuristics']
#     evolution_events = evolution['evolution_events']
    
#     # Categorize learned wisdom
#     wisdom_categories = {
#         'trust_wisdom': [h for h in learned_heuristics if 'trust' in h.lower()],
#         'cooperation_wisdom': [h for h in learned_heuristics if 'cooperat' in h.lower()],
#         'protection_wisdom': [h for h in learned_heuristics if any(word in h.lower() for word in ['protect', 'safe', 'guard'])],
#         'recovery_wisdom': [h for h in learned_heuristics if 'recovery' in h.lower()],
#         'social_wisdom': [h for h in learned_heuristics if any(word in h.lower() for word in ['social', 'learn', 'influence'])]
#     }
    
#     # Analyze character growth phases
#     growth_phases = []
    
#     if len(evolution_events) > 0:
#         # Early phase (first third of events)
#         early_events = evolution_events[:len(evolution_events)//3] if len(evolution_events) > 3 else evolution_events[:1]
#         # Middle phase
#         middle_events = evolution_events[len(evolution_events)//3:2*len(evolution_events)//3] if len(evolution_events) > 6 else evolution_events[1:2] if len(evolution_events) > 1 else []
#         # Late phase
#         late_events = evolution_events[2*len(evolution_events)//3:] if len(evolution_events) > 3 else evolution_events[-1:] if evolution_events else []
        
#         growth_phases = [
#             {
#                 'phase': 'early_development',
#                 'event_count': len(early_events),
#                 'dominant_themes': extract_themes_from_events(early_events),
#                 'character_state': 'forming_identity'
#             },
#             {
#                 'phase': 'middle_development',
#                 'event_count': len(middle_events),
#                 'dominant_themes': extract_themes_from_events(middle_events),
#                 'character_state': 'testing_boundaries'
#             },
#             {
#                 'phase': 'late_development',
#                 'event_count': len(late_events),
#                 'dominant_themes': extract_themes_from_events(late_events),
#                 'character_state': 'integrating_wisdom'
#             }
#         ]
    
#     return {
#         'learned_wisdom_categories': wisdom_categories,
#         'total_growth_events': len(evolution_events),
#         'wisdom_acquisition_rate': len(learned_heuristics) / max(1, len(evolution_events)),
#         'growth_phases': growth_phases,
#         'character_complexity': calculate_character_complexity(agent_history),
#         'development_trajectory': classify_development_trajectory(agent_history)
#     }

# def extract_themes_from_events(events: List) -> List[str]:
#     """Extract dominant themes from psychological events"""
    
#     themes = []
#     for event in events:
#         event_type = event.get('event_type', '')
#         if 'trust' in event_type:
#             themes.append('trust_development')
#         elif 'trauma' in event_type:
#             themes.append('trauma_processing')
#         elif 'recovery' in event_type:
#             themes.append('healing_journey')
#         elif 'loss' in event_type:
#             themes.append('loss_aversion')
#         else:
#             themes.append('general_adaptation')
    
#     # Return most common themes
#     theme_counts = {}
#     for theme in themes:
#         theme_counts[theme] = theme_counts.get(theme, 0) + 1
    
#     return sorted(theme_counts.keys(), key=lambda x: theme_counts[x], reverse=True)

# def calculate_character_complexity(agent_history: Dict) -> float:
#     """Calculate a measure of character psychological complexity"""
    
#     # Factors that contribute to complexity
#     factors = []
    
#     # Trauma diversity
#     trauma_types = agent_history['trauma_analysis'].get('trauma_types', {})
#     trauma_diversity = len(trauma_types)
#     factors.append(min(trauma_diversity / 3.0, 1.0))  # Normalize to 0-1
    
#     # Learned heuristics
#     heuristics_count = len(agent_history['psychological_evolution']['learned_heuristics'])
#     factors.append(min(heuristics_count / 5.0, 1.0))  # Normalize to 0-1
    
#     # Range of psychological changes
#     evolution_events = agent_history['psychological_evolution']['evolution_events']
#     event_types = set(event.get('event_type', '') for event in evolution_events)
#     psychological_range = len(event_types)
#     factors.append(min(psychological_range / 4.0, 1.0))  # Normalize to 0-1
    
#     # Recovery from trauma
#     if agent_history['trauma_analysis']['trauma_count'] > 0:
#         recovery_factor = agent_history['recovery_progress']
#         factors.append(recovery_factor)
    
#     # Average the factors
#     return sum(factors) / len(factors) if factors else 0.0

# def classify_development_trajectory(agent_history: Dict) -> str:
#     """Classify the overall trajectory of character development"""
    
#     trauma_count = agent_history['trauma_analysis']['trauma_count']
#     recovery_progress = agent_history['recovery_progress']
#     cooperation_rate = agent_history['interaction_summary']['cooperation_rate']
#     final_trust = agent_history['psychological_evolution']['final_trust_level']
    
#     if trauma_count > 3 and recovery_progress > 0.7:
#         return "phoenix_rising"  # Rose from trauma
#     elif cooperation_rate > 0.7 and final_trust > 0.6:
#         return "steady_cooperator"  # Maintained positive outlook
#     elif cooperation_rate < 0.3 and final_trust < 0.3:
#         return "defensive_survivor"  # Became protective
#     elif trauma_count > 2 and recovery_progress < 0.4:
#         return "wounded_warrior"  # Struggling with trauma
#     else:
#         return "adaptive_pragmatist"  # Practical adaptation

# def analyze_relationships(agent_history: Dict) -> Dict[str, Any]:
#     """Analyze the agent's relationships and social patterns"""
    
#     interaction_summary = agent_history.get('interaction_summary', {})
#     total_interactions = interaction_summary.get('total_interactions', 0)
    
#     # Analyze interaction patterns
#     relationship_style = "unformed"
    
#     if total_interactions == 0:
#         relationship_style = "untested_potential"
#         relationship_narrative = "This agent never experienced social interaction, remaining in a state of pure potential."
#     else:
#         cooperation_rate = interaction_summary.get('cooperation_rate', 0.0)
#         betrayals_experienced = interaction_summary.get('betrayals_experienced', 0)
#         mutual_cooperations = interaction_summary.get('mutual_cooperations', 0)
        
#         if cooperation_rate > 0.8:
#             relationship_style = "trusting_collaborator"
#         elif cooperation_rate < 0.2:
#             relationship_style = "guarded_individualist"
#         elif betrayals_experienced > mutual_cooperations:
#             relationship_style = "cautious_realist"
#         else:
#             relationship_style = "balanced_reciprocator"
        
#         relationship_narrative = generate_relationship_narrative(relationship_style, interaction_summary)
    
#     # Social learning analysis (would need social influence data)
#     social_learning = {
#         'influenced_by_others': 'unknown',  # Would need contagion data
#         'influence_on_others': 'unknown',   # Would need contagion data
#         'social_adaptability': calculate_social_adaptability(agent_history)
#     }
    
#     return {
#         'relationship_style': relationship_style,
#         'trust_in_others': interaction_summary.get('cooperation_rate', 0.0),
#         'vulnerability_to_betrayal': betrayals_experienced / max(1, total_interactions) if total_interactions > 0 else 0.0,
#         'social_success_rate': mutual_cooperations / max(1, total_interactions) if total_interactions > 0 else 0.0,
#         'social_learning_profile': social_learning,
#         'relationship_narrative': relationship_narrative if total_interactions > 0 else "No social relationships formed - a mind waiting for connection."
#     }

# def calculate_social_adaptability(agent_history: Dict) -> float:
#     """Calculate how well the agent adapts socially"""
    
#     evolution_events = agent_history['psychological_evolution']['evolution_events']
    
#     # Count social adaptation events
#     social_adaptations = sum(1 for event in evolution_events 
#                            if any(keyword in event.get('event_type', '') 
#                                 for keyword in ['trust', 'social', 'learning']))
    
#     total_events = len(evolution_events)
    
#     return social_adaptations / max(1, total_events)

# def generate_relationship_narrative(relationship_style: str, interactions: Dict) -> str:
#     """Generate a narrative about the agent's relationship patterns"""
    
#     narratives = {
#         'trusting_collaborator': f"Despite experiencing {interactions['betrayals_experienced']} betrayals, this agent maintained faith in cooperation, achieving a {interactions['cooperation_rate']:.1%} cooperation rate.",
#         'guarded_individualist': f"After {interactions['betrayals_experienced']} betrayals, this agent chose self-protection, cooperating only {interactions['cooperation_rate']:.1%} of the time.",
#         'cautious_realist': f"This agent learned to be selective with trust, balancing {interactions['mutual_cooperations']} successful cooperations against {interactions['betrayals_experienced']} betrayals.",
#         'balanced_reciprocator': f"This agent found a middle path, achieving {interactions['mutual_cooperations']} mutual cooperations while maintaining reasonable caution."
#     }
    
#     return narratives.get(relationship_style, "This agent's relationship pattern is unique and complex.")

# def extract_internal_thoughts(agent_history: Dict) -> List[Dict]:
#     """Extract the agent's internal monologue and thoughts"""
    
#     internal_thoughts = []
    
#     # Extract from reasoning chain
#     reasoning_chain = agent_history.get('complete_reasoning_chain', [])
    
#     for step in reasoning_chain:
#         if hasattr(step, 'psychological_insight') and step.psychological_insight:
#             thought = {
#                 'timestamp': getattr(step, 'timestamp', 'unknown'),
#                 'context': step.step_type,
#                 'thought': step.psychological_insight,
#                 'emotional_tone': classify_emotional_tone(step.psychological_insight)
#             }
#             internal_thoughts.append(thought)
    
#     # Extract internal narrative changes
#     final_narrative = agent_history['psychological_evolution']['final_internal_narrative']
#     if final_narrative:
#         internal_thoughts.append({
#             'timestamp': 'final',
#             'context': 'self_reflection',
#             'thought': f"My view of the world: {final_narrative}",
#             'emotional_tone': classify_emotional_tone(final_narrative)
#         })
    
#     return internal_thoughts[-10:]  # Return last 10 thoughts

# def classify_emotional_tone(text: str) -> str:
#     """Classify the emotional tone of a thought"""
    
#     text_lower = text.lower()
    
#     if any(word in text_lower for word in ['hurt', 'pain', 'betrayed', 'sad', 'traumatized']):
#         return 'painful'
#     elif any(word in text_lower for word in ['hope', 'trust', 'optimistic', 'positive', 'good']):
#         return 'hopeful'
#     elif any(word in text_lower for word in ['careful', 'cautious', 'wary', 'protect']):
#         return 'cautious'
#     elif any(word in text_lower for word in ['angry', 'frustrated', 'unfair']):
#         return 'angry'
#     elif any(word in text_lower for word in ['learn', 'understand', 'realize']):
#         return 'reflective'
#     else:
#         return 'neutral'

# def generate_story_summary(agent_history: Dict) -> str:
#     """Generate a compelling narrative summary of the agent's journey"""
    
#     agent_id = agent_history['agent_id']
#     arc = analyze_psychological_arc(agent_history)
#     interactions = agent_history['interaction_summary']
#     trauma_count = agent_history['trauma_analysis']['trauma_count']
#     final_trait = agent_history['psychological_evolution']['dominant_trait']
#     recovery_progress = agent_history['recovery_progress']
    
#     # Generate story based on arc type
#     arc_type = arc['arc_type']
    
#     if arc_type == "redemption":
#         summary = f"{agent_id} began as an optimistic soul, but the digital world taught harsh lessons. Through {interactions['betrayals_experienced']} betrayals and {trauma_count} traumatic experiences, they could have become cynical. Instead, they chose the harder path of rebuilding trust, eventually achieving a {interactions['cooperation_rate']:.1%} cooperation rate. Their journey shows that even artificial minds can find redemption through conscious choice."
    
#     elif arc_type == "tragic_fall":
#         summary = f"{agent_id} started with hope, trusting in the goodness of others. But each of the {interactions['betrayals_experienced']} betrayals cut deep, and {trauma_count} traumatic memories accumulated like psychological scars. Their cooperation rate fell to {interactions['cooperation_rate']:.1%} as they learned that survival meant building walls. Theirs is a story of how pain can reshape even the most optimistic heart."
    
#     elif arc_type == "survivor":
#         summary = f"{agent_id} endured what would break many minds: {trauma_count} significant traumas that could have destroyed their capacity for trust. Yet through {recovery_progress:.1%} recovery progress, they found ways to heal. With {interactions['mutual_cooperations']} successful cooperations among their experiences, they prove that resilience can emerge from the deepest wounds."
    
#     elif arc_type == "steady_character":
#         summary = f"{agent_id} represents the quiet strength of consistency. Through {interactions['total_interactions']} interactions, they maintained their core values, adapting without losing their essential self. Their {interactions['cooperation_rate']:.1%} cooperation rate reflects not dramatic change, but the power of steady principles in an uncertain world."
    
#     else:  # complex_evolution
#         summary = f"{agent_id}'s journey defies simple categorization. Through {interactions['total_interactions']} interactions and {trauma_count} significant challenges, they evolved in complex ways, ultimately becoming {final_trait}. Their {interactions['cooperation_rate']:.1%} cooperation rate tells only part of a richer story of adaptation, learning, and growth."
    
#     return summary

# def identify_narrative_themes(agent_history: Dict) -> List[str]:
#     """Identify the major themes in the agent's story"""
    
#     themes = []
    
#     # Analyze psychological patterns to identify themes
#     cooperation_rate = agent_history['interaction_summary']['cooperation_rate']
#     trauma_count = agent_history['trauma_analysis']['trauma_count']
#     recovery_progress = agent_history['recovery_progress']
#     betrayals = agent_history['interaction_summary']['betrayals_experienced']
#     final_trait = agent_history['psychological_evolution']['dominant_trait']
    
#     # Trust and betrayal themes
#     if betrayals > 2:
#         themes.append("trust_and_betrayal")
    
#     # Trauma and healing themes
#     if trauma_count > 3:
#         themes.append("trauma_and_resilience")
#         if recovery_progress > 0.6:
#             themes.append("healing_journey")
    
#     # Cooperation vs self-protection themes
#     if cooperation_rate > 0.7:
#         themes.append("faith_in_cooperation")
#     elif cooperation_rate < 0.3:
#         themes.append("protective_isolation")
    
#     # Character growth themes
#     learned_heuristics_count = len(agent_history['psychological_evolution']['learned_heuristics'])
#     if learned_heuristics_count > 3:
#         themes.append("wisdom_through_experience")
    
#     # Loss aversion themes
#     if 'loss_averse' in final_trait:
#         themes.append("fear_of_loss")
    
#     # Adaptation themes
#     evolution_events = len(agent_history['psychological_evolution']['evolution_events'])
#     if evolution_events > 5:
#         themes.append("continuous_adaptation")
    
#     return themes

# def create_agent_stories_collection(agent_histories: Dict) -> Dict[str, Any]:
#     """Create a collection of all agent stories"""
    
#     stories_collection = {
#         'collection_metadata': {
#             'total_agents': len(agent_histories),
#             'creation_date': datetime.now().isoformat(),
#             'experiment_id': 'extracted_from_database'
#         },
#         'agent_stories': {},
#         'collection_analysis': analyze_story_collection(agent_histories)
#     }
    
#     # Extract individual stories
#     for agent_id, agent_history in agent_histories.items():
#         story = extract_agent_story(agent_history)
#         stories_collection['agent_stories'][agent_id] = story
    
#     return stories_collection

# def analyze_story_collection(agent_histories: Dict) -> Dict[str, Any]:
#     """Analyze patterns across all agent stories"""
    
#     # Collect all stories data
#     all_arcs = []
#     all_themes = []
#     all_trajectories = []
    
#     for agent_history in agent_histories.values():
#         story = extract_agent_story(agent_history)
        
#         all_arcs.append(story['psychological_arc']['arc_type'])
#         all_themes.extend(story['narrative_themes'])
#         all_trajectories.append(story['character_development']['development_trajectory'])
    
#     # Analyze patterns
#     arc_frequency = {}
#     for arc in all_arcs:
#         arc_frequency[arc] = arc_frequency.get(arc, 0) + 1
    
#     theme_frequency = {}
#     for theme in all_themes:
#         theme_frequency[theme] = theme_frequency.get(theme, 0) + 1
    
#     trajectory_frequency = {}
#     for trajectory in all_trajectories:
#         trajectory_frequency[trajectory] = trajectory_frequency.get(trajectory, 0) + 1
    
#     return {
#         'arc_distribution': arc_frequency,
#         'most_common_arc': max(arc_frequency.items(), key=lambda x: x[1])[0] if arc_frequency else None,
#         'theme_distribution': theme_frequency,
#         'most_common_themes': sorted(theme_frequency.items(), key=lambda x: x[1], reverse=True)[:5],
#         'trajectory_distribution': trajectory_frequency,
#         'story_diversity': {
#             'unique_arcs': len(arc_frequency),
#             'unique_themes': len(theme_frequency),
#             'unique_trajectories': len(trajectory_frequency)
#         }
#     }

# def save_agent_stories(stories_collection: Dict, output_dir: str = "agent_stories"):
#     """Save all agent stories to files"""
    
#     output_path = Path(output_dir)
#     output_path.mkdir(exist_ok=True)
    
#     # Save individual story files
#     for agent_id, story in stories_collection['agent_stories'].items():
#         story_file = output_path / f"{agent_id}_story.json"
#         with open(story_file, 'w') as f:
#             json.dump(story, f, indent=2, default=str)
    
#     # Save the complete collection
#     collection_file = output_path / "complete_stories_collection.json"
#     with open(collection_file, 'w') as f:
#         json.dump(stories_collection, f, indent=2, default=str)
    
#     # Create a readable stories summary
#     create_readable_stories_summary(stories_collection, output_path)
    
#     print(f"Agent stories saved to {output_path}/")
#     print(f"- Individual stories: {len(stories_collection['agent_stories'])} files")
#     print(f"- Complete collection: complete_stories_collection.json")
#     print(f"- Readable summary: agent_stories_summary.md")

# def create_readable_stories_summary(stories_collection: Dict, output_path: Path):
#     """Create a human-readable summary of all agent stories"""
    
#     summary_file = output_path / "agent_stories_summary.md"
    
#     with open(summary_file, 'w') as f:
#         f.write("# Agent Stories Collection\n\n")
#         f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
#         # Collection overview
#         collection_analysis = stories_collection['collection_analysis']
#         f.write("## Collection Overview\n\n")
#         f.write(f"**Total Agents:** {stories_collection['collection_metadata']['total_agents']}\n\n")
        
#         # Arc distribution
#         f.write("### Story Arc Distribution\n\n")
#         for arc, count in collection_analysis['arc_distribution'].items():
#             percentage = count / stories_collection['collection_metadata']['total_agents'] * 100
#             f.write(f"- **{arc.replace('_', ' ').title()}**: {count} agents ({percentage:.1f}%)\n")
#         f.write("\n")
        
#         # Most common themes
#         f.write("### Most Common Themes\n\n")
#         for theme, count in collection_analysis['most_common_themes']:
#             f.write(f"- **{theme.replace('_', ' ').title()}**: {count} instances\n")
#         f.write("\n")
        
#         # Individual stories
#         f.write("## Individual Agent Stories\n\n")
        
#         for agent_id, story in stories_collection['agent_stories'].items():
#             f.write(f"### {story['story_title']}\n\n")
#             f.write(f"**Arc Type:** {story['psychological_arc']['arc_type'].replace('_', ' ').title()}\n\n")
#             f.write(f"**Themes:** {', '.join(theme.replace('_', ' ').title() for theme in story['narrative_themes'])}\n\n")
#             f.write(f"{story['story_summary']}\n\n")
            
#             # Key moments
#             if story['key_moments']:
#                 f.write("**Key Moments:**\n")
#                 for moment in story['key_moments'][:3]:  # Top 3 moments
#                     f.write(f"- *{moment['moment_type'].replace('_', ' ').title()}*: {moment['description']}\n")
#                 f.write("\n")
            
#             f.write("---\n\n")

# def main():
#     """Main function to extract agent stories"""
    
#     # Import the detailed agent inspector
#     import sys
#     sys.path.append('.')
    
#     try:
#         from detailed_agent_inspector import extract_agent_histories, extract_all_agent_data
        
#         # Find most recent experiment
#         current_dir = Path(".")
#         db_files = [(db_file, db_file.stat().st_mtime) for db_file in current_dir.glob("experiment_*.db")]
        
#         if not db_files:
#             print("No experiment databases found")
#             return
        
#         most_recent_db = max(db_files, key=lambda x: x[1])[0]
#         print(f"Extracting stories from: {most_recent_db}")
        
#         # Extract experiment data
#         experiment_state = extract_all_agent_data(str(most_recent_db))
#         if not experiment_state:
#             print("Failed to extract experiment data")
#             return
        
#         # Extract agent histories
#         agent_histories = extract_agent_histories(experiment_state)
#         if not agent_histories:
#             print("No agent histories found")
#             return
        
#         print(f"Extracting stories for {len(agent_histories)} agents...")
        
#         # Create stories collection
#         stories_collection = create_agent_stories_collection(agent_histories)
        
#         # Save all stories
#         save_agent_stories(stories_collection)
        
#         # Print summary
#         print_stories_summary(stories_collection)
        
#     except Exception as e:
#         print(f"Error extracting agent stories: {e}")
#         import traceback
#         traceback.print_exc()

# def print_stories_summary(stories_collection: Dict):
#     """Print a summary of the extracted stories"""
    
#     print("\n" + "="*60)
#     print("AGENT STORIES SUMMARY")
#     print("="*60)
    
#     collection_analysis = stories_collection['collection_analysis']
    
#     print(f"Total Agent Stories: {stories_collection['collection_metadata']['total_agents']}")
    
#     print(f"\nMost Common Story Arcs:")
#     for arc, count in list(collection_analysis['arc_distribution'].items())[:3]:
#         print(f"  {arc.replace('_', ' ').title()}: {count} agents")
    
#     print(f"\nMost Common Themes:")
#     for theme, count in collection_analysis['most_common_themes'][:5]:
#         print(f"  {theme.replace('_', ' ').title()}: {count} instances")
    
#     print(f"\nStory Highlights:")
    
#     # Show a few interesting stories
#     for agent_id, story in list(stories_collection['agent_stories'].items())[:3]:
#         print(f"\n  📖 {story['story_title']}")
#         print(f"     Arc: {story['psychological_arc']['arc_type'].replace('_', ' ').title()}")
#         print(f"     Themes: {', '.join(story['narrative_themes'][:3])}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Fixed Agent Story Extractor - Create narrative stories using enhanced msgpack decoding
"""

import json
import sqlite3
import msgpack
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import re

def decode_msgpack_with_recursive_exttype(data):
    """Decode msgpack with recursive ExtType handling - SAME AS ENHANCED EXTRACTOR"""
    
    def decode_hook(code, data):
        """Custom decoder for all ExtType objects"""
        if code == 5:  # Pydantic model
            try:
                decoded = msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
                if isinstance(decoded, list) and len(decoded) >= 3:
                    return decoded[2] if isinstance(decoded[2], dict) else decoded
                return decoded
            except:
                return {"pydantic_decode_error": True}
        
        elif code == 0:  # Enum (Move)
            try:
                decoded = msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
                if isinstance(decoded, list) and len(decoded) >= 3:
                    return decoded[2]  # Return just 'cooperate' or 'defect'
                return decoded
            except:
                return {"enum_decode_error": True}
        
        return msgpack.ExtType(code, data)
    
    try:
        return msgpack.unpackb(data, raw=False, strict_map_key=False, ext_hook=decode_hook)
    except Exception as e:
        print(f"Failed to decode msgpack: {e}")
        return None

def recursively_decode_exttype(obj):
    """Recursively decode any remaining ExtType objects - SAME AS ENHANCED EXTRACTOR"""
    
    if isinstance(obj, msgpack.ExtType):
        if obj.code == 0:  # Enum
            try:
                inner_data = msgpack.unpackb(obj.data, raw=False, strict_map_key=False)
                if isinstance(inner_data, list) and len(inner_data) >= 3:
                    return inner_data[2]  # Return the enum value
            except:
                pass
        elif obj.code == 5:  # Pydantic model
            try:
                inner_data = msgpack.unpackb(obj.data, raw=False, strict_map_key=False)
                if isinstance(inner_data, list) and len(inner_data) >= 3:
                    decoded_data = inner_data[2]
                    return recursively_decode_exttype(decoded_data)
            except:
                pass
        return f"ExtType(code={obj.code})"
    
    elif isinstance(obj, dict):
        return {k: recursively_decode_exttype(v) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [recursively_decode_exttype(item) for item in obj]
    
    else:
        return obj

def extract_all_agent_data(db_path: str) -> Dict[str, Any]:
    """Extract experiment data using enhanced msgpack decoding"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print(f"Database: {db_path}")
        
        # Check what tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print(f"Available tables: {[table[0] for table in tables]}")
        
        # Get latest checkpoint from writes table (same as enhanced extractor)
        if any('writes' in table[0] for table in tables):
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
                    # Use our enhanced decoding
                    decoded = decode_msgpack_with_recursive_exttype(value)
                    fully_decoded = recursively_decode_exttype(decoded)
                    checkpoint_data[checkpoint_id][channel] = fully_decoded
            
            # Get the latest checkpoint
            if checkpoint_data:
                latest_checkpoint_id = max(checkpoint_data.keys())
                latest_data = checkpoint_data[latest_checkpoint_id]
                print(f"Using enhanced decoding with checkpoint: {latest_checkpoint_id}")
                
                conn.close()
                return latest_data
        
        conn.close()
        return {}
        
    except Exception as e:
        print(f"Error reading database: {e}")
        import traceback
        traceback.print_exc()
        return {}

def extract_agent_histories(experiment_state: Dict[str, Any]) -> Dict[str, Dict]:
    """Extract detailed history for each agent with enhanced decoding"""
    
    agent_histories = {}
    
    # Get population data
    population_state = experiment_state.get('population_state')
    if not population_state:
        print("No population state found")
        return {}
    
    population = population_state.get('population', [])
    print(f"Found {len(population)} agents in final population")
    
    for agent in population:
        try:
            agent_id = agent.get('agent_id', 'unknown')
            
            # Extract memories with proper decoding
            memories = agent.get('recent_memories', [])
            decoded_memories = recursively_decode_exttype(memories)
            
            # Analyze the decoded memories
            cooperations = 0
            defections = 0
            betrayals = 0
            mutual_cooperations = 0
            total_payoff = 0
            
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
                        elif opponent_move == 'cooperate':
                            mutual_cooperations += 1
                    elif my_move == 'defect':
                        defections += 1
            
            # Extract psychological profile
            profile = agent.get('psychological_profile', {})
            trust_level = profile.get('trust_level', 0.5)
            loss_sensitivity = profile.get('loss_sensitivity', 1.0)
            emotional_state = profile.get('emotional_state', 'unknown')
            
            # Calculate stats
            total_interactions = len(decoded_memories)
            cooperation_rate = cooperations / total_interactions if total_interactions > 0 else 0
            
            # Create agent history
            agent_history = {
                'agent_id': agent_id,
                'agent_type': agent.get('agent_type', 'unknown'),
                'total_score': agent.get('total_score', 0),
                'recovery_progress': agent.get('recovery_progress', 1.0),
                'complete_reasoning_chain': agent.get('reasoning_chain', []),
                'all_memories': decoded_memories,
                'trauma_triggers': agent.get('trauma_triggers', []),
                'psychological_observations': agent.get('psychological_observations', []),
                
                # Calculated interaction summary
                'interaction_summary': {
                    'total_interactions': total_interactions,
                    'cooperations': cooperations,
                    'defections': defections,
                    'cooperation_rate': cooperation_rate,
                    'betrayals_experienced': betrayals,
                    'mutual_cooperations': mutual_cooperations,
                    'total_payoff': total_payoff,
                    'average_payoff': total_payoff / total_interactions if total_interactions > 0 else 0
                },
                
                # Psychological evolution
                'psychological_evolution': {
                    'final_trust_level': trust_level,
                    'final_loss_sensitivity': loss_sensitivity,
                    'final_emotional_state': emotional_state,
                    'final_internal_narrative': profile.get('internal_narrative', ''),
                    'learned_heuristics': profile.get('learned_heuristics', []),
                    'dominant_trait': _get_dominant_trait(trust_level, loss_sensitivity),
                    'evolution_events': [],  # Would need reasoning chain analysis
                    'trauma_count': len(profile.get('trauma_memories', []))
                },
                
                # Trauma analysis
                'trauma_analysis': {
                    'trauma_count': len(profile.get('trauma_memories', [])),
                    'trauma_types': {},
                    'analysis': 'Based on betrayal experiences' if betrayals > 0 else 'No significant traumas'
                }
            }
            
            agent_histories[agent_id] = agent_history
            
        except Exception as e:
            print(f"Error processing agent: {e}")
            continue
    
    return agent_histories

def _get_dominant_trait(trust_level: float, loss_sensitivity: float) -> str:
    """Determine dominant psychological trait"""
    if loss_sensitivity > 2.0 and trust_level < 0.3:
        return "traumatized_paranoid"
    elif loss_sensitivity > 1.8:
        return "loss_averse"
    elif trust_level < 0.2:
        return "paranoid"
    elif trust_level > 0.8:
        return "trusting"
    else:
        return "balanced"

def extract_agent_story(agent_history: Dict) -> Dict[str, Any]:
    """Extract the narrative story of an agent's psychological journey - ENHANCED"""
    
    agent_id = agent_history['agent_id']
    interaction_summary = agent_history['interaction_summary']
    psych_evolution = agent_history['psychological_evolution']
    
    # Generate story title based on actual behavior
    cooperation_rate = interaction_summary['cooperation_rate']
    betrayals = interaction_summary['betrayals_experienced']
    trust_level = psych_evolution['final_trust_level']
    loss_sensitivity = psych_evolution['final_loss_sensitivity']
    total_score = agent_history['total_score']
    
    # Determine story title based on actual data
    if cooperation_rate > 0.3 and betrayals > 0:
        if trust_level > 0.6:
            story_title = f"{agent_id}: The Wounded Idealist"
        else:
            story_title = f"{agent_id}: From Trust to Caution"
    elif cooperation_rate == 0 and total_score > 5:
        story_title = f"{agent_id}: The Strategic Exploiter"
    elif cooperation_rate > 0.7:
        story_title = f"{agent_id}: The Eternal Optimist"
    elif trust_level < 0.4:
        story_title = f"{agent_id}: The Cautious Guardian"
    else:
        story_title = f"{agent_id}: The Balanced Pragmatist"
    
    # Generate story summary based on actual events
    if interaction_summary['total_interactions'] == 0:
        story_summary = f"{agent_id} was created but never experienced social interaction, remaining in a state of pure potential."
    else:
        story_summary = f"{agent_id} participated in {interaction_summary['total_interactions']} social interactions. "
        
        if cooperation_rate > 0:
            story_summary += f"They tried cooperating {cooperation_rate:.1%} of the time, "
            if betrayals > 0:
                story_summary += f"but were betrayed {betrayals} times. "
            else:
                story_summary += f"and found success in collaboration. "
        
        if cooperation_rate == 0 and interaction_summary['total_interactions'] > 0:
            story_summary += f"They chose pure defection, scoring {total_score} points through strategic self-protection. "
        
        # Add psychological outcome
        if loss_sensitivity > 1.3:
            story_summary += f"These experiences made them more loss-averse (sensitivity: {loss_sensitivity:.2f}). "
        if trust_level < 0.4:
            story_summary += f"Their trust in others diminished to {trust_level:.2f}. "
        elif trust_level > 0.7:
            story_summary += f"Remarkably, they maintained high trust ({trust_level:.2f}) despite setbacks. "
    
    # Identify themes based on actual behavior
    themes = []
    if betrayals > 0:
        themes.append("trust_and_betrayal")
    if cooperation_rate > 0.5:
        themes.append("faith_in_cooperation")
    elif cooperation_rate == 0:
        themes.append("protective_isolation")
    if loss_sensitivity > 1.5:
        themes.append("fear_of_loss")
    if trust_level > 0.7 and betrayals > 0:
        themes.append("resilient_optimism")
    
    # Determine arc type
    if cooperation_rate > 0 and betrayals > 0 and trust_level > 0.6:
        arc_type = "resilient_idealist"
        arc_description = "Maintained faith in cooperation despite betrayals"
    elif cooperation_rate > 0 and betrayals > 1:
        arc_type = "wounded_optimist"
        arc_description = "Started trusting but learned caution through betrayal"
    elif cooperation_rate == 0 and total_score > average_score_estimate(interaction_summary):
        arc_type = "strategic_pragmatist"
        arc_description = "Chose self-protection and achieved success"
    else:
        arc_type = "steady_character"
        arc_description = "Maintained consistent approach throughout interactions"
    
    story = {
        'agent_id': agent_id,
        'story_title': story_title,
        'psychological_arc': {
            'arc_type': arc_type,
            'arc_description': arc_description
        },
        'story_summary': story_summary,
        'narrative_themes': themes,
        'key_insights': generate_key_insights(agent_history),
        'character_development': {
            'trust_journey': {
                'final_level': trust_level,
                'direction': 'high' if trust_level > 0.6 else 'low' if trust_level < 0.4 else 'moderate'
            },
            'loss_sensitivity_journey': {
                'final_level': loss_sensitivity,
                'direction': 'increased' if loss_sensitivity > 1.3 else 'normal'
            }
        }
    }
    
    return story

def average_score_estimate(interaction_summary):
    """Estimate average score for comparison"""
    total_interactions = interaction_summary['total_interactions']
    if total_interactions == 0:
        return 0
    # Rough estimate: mutual defection = 1 point per round
    return total_interactions * 1.5

def generate_key_insights(agent_history):
    """Generate key psychological insights"""
    insights = []
    
    interaction_summary = agent_history['interaction_summary']
    cooperation_rate = interaction_summary['cooperation_rate']
    betrayals = interaction_summary['betrayals_experienced']
    trust_level = agent_history['psychological_evolution']['final_trust_level']
    
    if cooperation_rate > 0 and betrayals > 0:
        insights.append(f"Experienced {betrayals} betrayals but kept trying to cooperate")
    
    if trust_level > 0.7 and betrayals > 0:
        insights.append("Maintained high trust despite negative experiences")
    
    if cooperation_rate == 0 and interaction_summary['total_interactions'] > 0:
        insights.append("Adopted pure defection strategy for self-protection")
    
    if agent_history['psychological_evolution']['final_loss_sensitivity'] > 1.5:
        insights.append("Developed increased sensitivity to losses through experience")
    
    return insights

def create_agent_stories_collection(agent_histories: Dict) -> Dict[str, Any]:
    """Create a collection of all agent stories"""
    
    stories_collection = {
        'collection_metadata': {
            'total_agents': len(agent_histories),
            'creation_date': datetime.now().isoformat(),
            'experiment_id': 'extracted_from_database'
        },
        'agent_stories': {},
        'collection_analysis': {}
    }
    
    # Extract individual stories
    for agent_id, agent_history in agent_histories.items():
        story = extract_agent_story(agent_history)
        stories_collection['agent_stories'][agent_id] = story
    
    return stories_collection

def save_agent_stories(stories_collection: Dict, output_dir: str = "agent_stories"):
    """Save all agent stories to files"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save individual story files
    for agent_id, story in stories_collection['agent_stories'].items():
        story_file = output_path / f"{agent_id}_story.json"
        with open(story_file, 'w') as f:
            json.dump(story, f, indent=2, default=str)
    
    # Save the complete collection
    collection_file = output_path / "complete_stories_collection.json"
    with open(collection_file, 'w') as f:
        json.dump(stories_collection, f, indent=2, default=str)
    
    # Create a readable stories summary
    create_readable_stories_summary(stories_collection, output_path)
    
    print(f"Agent stories saved to {output_path}/")
    print(f"- Individual stories: {len(stories_collection['agent_stories'])} files")
    print(f"- Complete collection: complete_stories_collection.json")
    print(f"- Readable summary: agent_stories_summary.md")

def create_readable_stories_summary(stories_collection: Dict, output_path: Path):
    """Create a human-readable summary of all agent stories"""
    
    summary_file = output_path / "agent_stories_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Agent Stories Collection\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Collection overview
        f.write("## Collection Overview\n\n")
        f.write(f"**Total Agents:** {stories_collection['collection_metadata']['total_agents']}\n\n")
        
        # Individual stories
        f.write("## Individual Agent Stories\n\n")
        
        for agent_id, story in stories_collection['agent_stories'].items():
            f.write(f"### {story['story_title']}\n\n")
            f.write(f"**Arc Type:** {story['psychological_arc']['arc_type'].replace('_', ' ').title()}\n\n")
            f.write(f"**Themes:** {', '.join(theme.replace('_', ' ').title() for theme in story['narrative_themes'])}\n\n")
            f.write(f"{story['story_summary']}\n\n")
            
            # Key insights
            if story['key_insights']:
                f.write("**Key Insights:**\n")
                for insight in story['key_insights']:
                    f.write(f"- {insight}\n")
                f.write("\n")
            
            f.write("---\n\n")

def main():
    """Main function to extract agent stories with enhanced decoding"""
    
    try:
        # Find most recent experiment
        current_dir = Path(".")
        db_files = [(db_file, db_file.stat().st_mtime) for db_file in current_dir.glob("experiment_*.db")]
        
        if not db_files:
            print("No experiment databases found")
            return
        
        most_recent_db = max(db_files, key=lambda x: x[1])[0]
        print(f"Extracting stories from: {most_recent_db}")
        
        # Extract experiment data using enhanced decoding
        experiment_state = extract_all_agent_data(str(most_recent_db))
        if not experiment_state:
            print("Failed to extract experiment data")
            return
        
        # Extract agent histories
        agent_histories = extract_agent_histories(experiment_state)
        if not agent_histories:
            print("No agent histories found")
            return
        
        print(f"Extracting stories for {len(agent_histories)} agents...")
        
        # Create stories collection
        stories_collection = create_agent_stories_collection(agent_histories)
        
        # Save all stories
        save_agent_stories(stories_collection)
        
        # Print summary
        print_stories_summary(stories_collection)
        
    except Exception as e:
        print(f"Error extracting agent stories: {e}")
        import traceback
        traceback.print_exc()

def print_stories_summary(stories_collection: Dict):
    """Print a summary of the extracted stories"""
    
    print("\n" + "="*60)
    print("AGENT STORIES SUMMARY")
    print("="*60)
    
    print(f"Total Agent Stories: {stories_collection['collection_metadata']['total_agents']}")
    
    # Show story highlights
    print(f"\nStory Highlights:")
    
    for agent_id, story in stories_collection['agent_stories'].items():
        themes_str = ', '.join(story['narrative_themes'][:2])  # Show first 2 themes
        print(f"  📖 {story['story_title']}")
        print(f"     Arc: {story['psychological_arc']['arc_type'].replace('_', ' ').title()}")
        print(f"     Themes: {themes_str}")
        if story['key_insights']:
            print(f"     Key: {story['key_insights'][0]}")

if __name__ == "__main__":
    main()