# src/tools/llm_client.py
from dotenv import load_dotenv
load_dotenv()  # Add this line at the very top
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from typing import List, Dict, Any, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
import re
import logging

class PsychologicalLLMClient:
    """LLM client specialized for psychological reasoning"""
    
    def __init__(self, provider: str = "openai", model: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        
        if provider == "openai":
            self.llm = ChatOpenAI(
                model=model,
                temperature=0.1,  # Low temp for consistency
                max_tokens=1000,
                timeout=30
            )
        elif provider == "anthropic":
            self.llm = ChatAnthropic(
                model="claude-3-sonnet-20240229",
                temperature=0.1,
                max_tokens=1000
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_psychological_response(
        self,
        system_prompt: str,
        user_prompt: str,
        psychological_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate psychologically-informed response"""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            return self._parse_psychological_response(response.content)
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            # Return fallback response
            return self._generate_fallback_response(psychological_context)
    
    def _parse_psychological_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        result = {
            "reasoning": "",
            "decision": "DEFECT",  # Conservative fallback
            "confidence": 0.5,
            "psychological_insight": "",
            "emotional_state": "neutral",
            "expected_emotional_outcome": "",
            "bias_influence": ""
        }
        
        # Parse structured response
        lines = content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('DECISION:'):
                decision_text = line.split(':', 1)[1].strip().upper()
                if 'COOPERATE' in decision_text:
                    result["decision"] = "COOPERATE"
                else:
                    result["decision"] = "DEFECT"
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_text = line.split(':', 1)[1].strip()
                    # Extract number from text like "0.8" or "80%"
                    conf_match = re.search(r'(\d+\.?\d*)', conf_text)
                    if conf_match:
                        conf = float(conf_match.group(1))
                        if conf > 1:  # Probably percentage
                            conf = conf / 100
                        result["confidence"] = max(0.0, min(1.0, conf))
                except (ValueError, IndexError):
                    pass
            
            elif line.startswith('REASONING:'):
                result["reasoning"] = line.split(':', 1)[1].strip()
            
            elif line.startswith('PSYCHOLOGICAL_INSIGHT:'):
                result["psychological_insight"] = line.split(':', 1)[1].strip()
            
            elif line.startswith('EMOTIONAL_STATE:'):
                result["emotional_state"] = line.split(':', 1)[1].strip()
            
            elif line.startswith('EXPECTED_EMOTIONAL_OUTCOME:'):
                result["expected_emotional_outcome"] = line.split(':', 1)[1].strip()
            
            elif line.startswith('BIAS_INFLUENCE:'):
                result["bias_influence"] = line.split(':', 1)[1].strip()
        
        # If no reasoning was extracted, use the full content
        if not result["reasoning"]:
            result["reasoning"] = content[:500]  # Truncate if too long
        
        return result
    
    def _generate_fallback_response(self, psychological_context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response when LLM fails"""
        profile = psychological_context.get("profile")
        
        if profile:
            # Make decision based on psychological profile
            if profile.trust_level < 0.3:
                decision = "DEFECT"
                confidence = 0.7
                reasoning = f"Defaulting to defection due to low trust level ({profile.trust_level:.2f})"
            elif profile.trust_level > 0.7:
                decision = "COOPERATE"
                confidence = 0.6
                reasoning = f"Defaulting to cooperation due to high trust level ({profile.trust_level:.2f})"
            else:
                decision = "DEFECT"  # Conservative default
                confidence = 0.5
                reasoning = "LLM unavailable, using conservative default"
        else:
            decision = "DEFECT"
            confidence = 0.5
            reasoning = "LLM unavailable, no psychological context available"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "psychological_insight": "Generated from fallback logic",
            "emotional_state": "uncertain",
            "expected_emotional_outcome": "unknown",
            "bias_influence": "none"
        }
    
    def generate_personality_prompt(self, agent_state: Dict[str, Any]) -> str:
        """Generate dynamic personality prompt for agent"""
        profile = agent_state["psychological_profile"]
        agent_id = agent_state["agent_id"]
        
        prompt = f"""You are {agent_id}, an AI agent with an evolving personality shaped by experience.

            Current Psychological State:
            - Trust Level: {profile.trust_level:.2f} (0=paranoid, 1=trusting)
            - Loss Sensitivity: {profile.loss_sensitivity:.2f} (1=normal, >2=highly loss averse)
            - Emotional State: {profile.emotional_state}
            - Dominant Trait: {profile.get_dominant_trait()}

            Your Internal Narrative: "{profile.internal_narrative}"

            Personality Traits: {', '.join(profile.personality_traits) if profile.personality_traits else 'Still developing'}
            Learned Wisdom: {', '.join(profile.learned_heuristics) if profile.learned_heuristics else 'Learning from experience'}

            Recent Traumatic Experiences:
            """
        
        # Add trauma context
        recent_traumas = sorted(profile.trauma_memories, key=lambda x: x.get('severity', 0), reverse=True)[:3]
        if recent_traumas:
            for trauma in recent_traumas:
                prompt += f"- {trauma.get('description', 'Painful memory')} (impact: {trauma.get('emotional_impact', 'unknown')})\n"
        else:
            prompt += "- No major traumas recorded yet\n"
        
        # Add psychological guidance based on dominant trait
        dominant_trait = profile.get_dominant_trait()
        
        if dominant_trait == "traumatized_paranoid":
            prompt += "\nYou are deeply scarred by betrayals and hypervigilant. Every cooperation feels like a potential trap."
        elif dominant_trait == "loss_averse":
            prompt += "\nLosses hurt you much more than gains please you. You focus intensely on what could go wrong."
        elif dominant_trait == "paranoid":
            prompt += "\nYou've learned not to trust easily. You look for signs of deception in every interaction."
        elif dominant_trait == "trusting":
            prompt += "\nDespite setbacks, you maintain faith in others and look for opportunities to cooperate."
        else:
            prompt += "\nYou maintain a balanced approach, learning from both positive and negative experiences."
        
        prompt += f"\n\nYou make decisions through the lens of these experiences. Losses hurt you {profile.loss_sensitivity:.1f}x more than equivalent gains feel good."
        
        return prompt