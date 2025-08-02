from langgraph.graph import StateGraph
from typing import AsyncIterator, Dict, Any
from datetime import datetime
import asyncio
import logging

logger = logging.getLogger(__name__)

async def stream_experiment_progress(
    graph: StateGraph,
    initial_state: Dict[str, Any]
) -> AsyncIterator[Dict[str, Any]]:
    """Stream experiment progress in real-time"""
    
    try:
        async for event in graph.astream(initial_state):
            # Process different types of events
            if isinstance(event, dict):
                yield {
                    "event_type": "state_update",
                    "data": _sanitize_event_data(event),
                    "timestamp": datetime.now().isoformat(),
                    "status": "running"
                }
            else:
                yield {
                    "event_type": "node_execution",
                    "data": str(event),
                    "timestamp": datetime.now().isoformat(),
                    "status": "running"
                }
    
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield {
            "event_type": "error",
            "data": {"error": str(e)},
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

async def stream_population_metrics(
    population_state: Dict[str, Any],
    update_interval: float = 1.0
) -> AsyncIterator[Dict[str, Any]]:
    """Stream real-time population metrics"""
    
    while True:
        try:
            # Calculate current population metrics
            population = population_state.get("population", [])
            
            if population:
                trust_levels = [agent["psychological_profile"].trust_level for agent in population]
                loss_sensitivities = [agent["psychological_profile"].loss_sensitivity for agent in population]
                
                # Calculate trait distribution
                trait_counts = {}
                for agent in population:
                    trait = agent["psychological_profile"].get_dominant_trait()
                    trait_counts[trait] = trait_counts.get(trait, 0) + 1
                
                metrics = {
                    "generation": population_state.get("generation", 0),
                    "population_size": len(population),
                    "avg_trust_level": sum(trust_levels) / len(trust_levels),
                    "avg_loss_sensitivity": sum(loss_sensitivities) / len(loss_sensitivities),
                    "trait_distribution": trait_counts,
                    "contagion_events": len(population_state.get("contagion_events", [])),
                    "timestamp": datetime.now().isoformat()
                }
                
                yield {
                    "event_type": "population_metrics",
                    "data": metrics,
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                }
            
            await asyncio.sleep(update_interval)
            
        except Exception as e:
            logger.error(f"Population streaming error: {e}")
            yield {
                "event_type": "error",
                "data": {"error": str(e)},
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
            break

async def stream_agent_decisions(
    agent_states: list,
    decision_buffer_size: int = 100
) -> AsyncIterator[Dict[str, Any]]:
    """Stream agent decision-making in real-time"""
    
    decision_buffer = []
    
    try:
        for agent_state in agent_states:
            agent_id = agent_state.get("agent_id")
            reasoning_chain = agent_state.get("reasoning_chain", [])
            
            if reasoning_chain:
                latest_reasoning = reasoning_chain[-1]
                
                decision_event = {
                    "agent_id": agent_id,
                    "step_type": latest_reasoning.step_type,
                    "confidence": latest_reasoning.confidence,
                    "emotional_state": agent_state["psychological_profile"].emotional_state,
                    "decision": agent_state.get("current_decision"),
                    "timestamp": latest_reasoning.timestamp.isoformat()
                }
                
                decision_buffer.append(decision_event)
                
                # Keep buffer size manageable
                if len(decision_buffer) > decision_buffer_size:
                    decision_buffer = decision_buffer[-decision_buffer_size:]
                
                yield {
                    "event_type": "agent_decision",
                    "data": decision_event,
                    "timestamp": datetime.now().isoformat(),
                    "status": "active"
                }
    
    except Exception as e:
        logger.error(f"Agent decision streaming error: {e}")
        yield {
            "event_type": "error",
            "data": {"error": str(e)},
            "timestamp": datetime.now().isoformat(),
            "status": "error"
        }

def _sanitize_event_data(event_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize event data for streaming (remove large objects, etc.)"""
    
    sanitized = {}
    
    for key, value in event_data.items():
        if key in ["population", "reasoning_chain", "psychological_observations"]:
            # Replace large arrays with summaries
            if isinstance(value, list):
                sanitized[f"{key}_count"] = len(value)
            else:
                sanitized[key] = "large_object_omitted"
        elif key == "psychological_profile":
            # Keep psychological profile but summarize
            if hasattr(value, 'get_dominant_trait'):
                sanitized[key] = {
                    "dominant_trait": value.get_dominant_trait(),
                    "trust_level": value.trust_level,
                    "loss_sensitivity": value.loss_sensitivity,
                    "emotional_state": value.emotional_state
                }
            else:
                sanitized[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            sanitized[key] = value
        else:
            sanitized[key] = str(value)[:200]  # Truncate long strings
    
    return sanitized

class ExperimentStreamer:
    """Manages multiple streaming channels for an experiment"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.active_streams = {}
        self.logger = logging.getLogger(f"{__name__}.{experiment_id}")
    
    async def start_experiment_stream(self, graph: StateGraph, initial_state: Dict[str, Any]):
        """Start streaming the main experiment progress"""
        stream_id = f"{self.experiment_id}_main"
        self.active_streams[stream_id] = stream_experiment_progress(graph, initial_state)
        return stream_id
    
    async def start_population_stream(self, population_state: Dict[str, Any]):
        """Start streaming population metrics"""
        stream_id = f"{self.experiment_id}_population"
        self.active_streams[stream_id] = stream_population_metrics(population_state)
        return stream_id
    
    async def start_decision_stream(self, agent_states: list):
        """Start streaming agent decisions"""
        stream_id = f"{self.experiment_id}_decisions"
        self.active_streams[stream_id] = stream_agent_decisions(agent_states)
        return stream_id
    
    async def get_stream_events(self, stream_id: str, max_events: int = 10):
        """Get events from a specific stream"""
        if stream_id not in self.active_streams:
            return []
        
        events = []
        stream = self.active_streams[stream_id]
        
        try:
            for _ in range(max_events):
                event = await anext(stream)
                events.append(event)
        except StopAsyncIteration:
            self.logger.info(f"Stream {stream_id} completed")
        except Exception as e:
            self.logger.error(f"Error reading from stream {stream_id}: {e}")
        
        return events
    
    def stop_stream(self, stream_id: str):
        """Stop a specific stream"""
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            self.logger.info(f"Stopped stream {stream_id}")
    
    def stop_all_streams(self):
        """Stop all active streams"""
        for stream_id in list(self.active_streams.keys()):
            self.stop_stream(stream_id)
        self.logger.info(f"Stopped all streams for experiment {self.experiment_id}")

# Real-time dashboard data aggregator
class DashboardDataAggregator:
    """Aggregates streaming data for real-time dashboard display"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.event_buffer = []
        self.population_history = []
        self.decision_history = []
        self.error_log = []
    
    def add_event(self, event: Dict[str, Any]):
        """Add an event to the appropriate buffer"""
        event_type = event.get("event_type")
        
        if event_type == "population_metrics":
            self.population_history.append(event["data"])
            self._trim_buffer(self.population_history)
        
        elif event_type == "agent_decision":
            self.decision_history.append(event["data"])
            self._trim_buffer(self.decision_history)
        
        elif event_type == "error":
            self.error_log.append(event)
            self._trim_buffer(self.error_log)
        
        # Always add to general event buffer
        self.event_buffer.append(event)
        self._trim_buffer(self.event_buffer)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get aggregated data for dashboard display"""
        
        latest_population = self.population_history[-1] if self.population_history else {}
        recent_decisions = self.decision_history[-20:] if self.decision_history else []
        recent_errors = self.error_log[-5:] if self.error_log else []
        
        return {
            "current_population_metrics": latest_population,
            "recent_decisions": recent_decisions,
            "recent_errors": recent_errors,
            "population_trend": self._calculate_population_trend(),
            "decision_summary": self._summarize_recent_decisions(),
            "system_status": "healthy" if not recent_errors else "warning",
            "last_update": datetime.now().isoformat()
        }
    
    def _trim_buffer(self, buffer: list):
        """Trim buffer to maximum size"""
        if len(buffer) > self.buffer_size:
            buffer[:] = buffer[-self.buffer_size:]
    
    def _calculate_population_trend(self) -> Dict[str, Any]:
        """Calculate trends in population metrics"""
        if len(self.population_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent = self.population_history[-10:]
        
        trust_trend = "stable"
        loss_sensitivity_trend = "stable"
        
        if len(recent) >= 5:
            early_trust = sum(p.get("avg_trust_level", 0.5) for p in recent[:len(recent)//2]) / (len(recent)//2)
            late_trust = sum(p.get("avg_trust_level", 0.5) for p in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            if late_trust > early_trust + 0.05:
                trust_trend = "increasing"
            elif late_trust < early_trust - 0.05:
                trust_trend = "decreasing"
            
            early_loss = sum(p.get("avg_loss_sensitivity", 1.0) for p in recent[:len(recent)//2]) / (len(recent)//2)
            late_loss = sum(p.get("avg_loss_sensitivity", 1.0) for p in recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
            
            if late_loss > early_loss + 0.1:
                loss_sensitivity_trend = "increasing"
            elif late_loss < early_loss - 0.1:
                loss_sensitivity_trend = "decreasing"
        
        return {
            "trust_level_trend": trust_trend,
            "loss_sensitivity_trend": loss_sensitivity_trend,
            "data_points": len(self.population_history)
        }
    
    def _summarize_recent_decisions(self) -> Dict[str, Any]:
        """Summarize recent agent decisions"""
        if not self.decision_history:
            return {"summary": "no_decisions"}
        
        recent = self.decision_history[-50:]  # Last 50 decisions
        
        cooperation_count = sum(1 for d in recent if d.get("decision") == "COOPERATE")
        defection_count = sum(1 for d in recent if d.get("decision") == "DEFECT")
        
        avg_confidence = sum(d.get("confidence", 0.5) for d in recent) / len(recent)
        
        emotional_states = {}
        for decision in recent:
            state = decision.get("emotional_state", "unknown")
            emotional_states[state] = emotional_states.get(state, 0) + 1
        
        return {
            "cooperation_rate": cooperation_count / len(recent) if recent else 0,
            "avg_confidence": avg_confidence,
            "dominant_emotional_state": max(emotional_states.items(), key=lambda x: x[1])[0] if emotional_states else "unknown",
            "total_decisions": len(recent)
        }