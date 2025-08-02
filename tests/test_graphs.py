import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from src.graphs.agents.decision_graph import create_agent_decision_graph
from src.graphs.agents.evolution_graph import create_psychological_evolution_graph
from src.graphs.population.contagion_graph import create_contagion_graph
from src.state.agent_state import AgentState, PsychologicalProfile
from src.state.population_state import PopulationState

@pytest.fixture
def sample_agent_state():
    """Create a sample agent state for testing"""
    return AgentState(
        agent_id="test_agent",
        psychological_profile=PsychologicalProfile(
            trust_level=0.5,
            loss_sensitivity=1.5,
            emotional_state="neutral"
        ),
        current_round=1,
        game_context={},
        reasoning_chain=[],
        psychological_observations=[],
        current_decision=None,
        decision_confidence=0.5,
        expected_outcomes={},
        recent_memories=[],
        trauma_triggers=[],
        recovery_progress=1.0,
        agent_type="test",
        total_score=0.0,
        reference_point=0.0
    )

@pytest.fixture
def sample_population_state():
    """Create a sample population state for testing"""
    return PopulationState(
        population=[],
        generation=0,
        interaction_results=[],
        contagion_events=[],
        population_metrics={},
        current_experiment="test_experiment",
        experiment_parameters={},
        should_continue=True,
        psychological_distribution={},
        dominant_traits=[],
        avg_trust_level=0.5,
        avg_loss_sensitivity=1.0,
        avg_cooperation_rate=0.5,
        successful_agents=[],
        struggling_agents=[],
        trait_transmission_matrix={}
    )

def test_create_agent_decision_graph():
    """Test that agent decision graph is created properly"""
    graph = create_agent_decision_graph()
    
    # Check that graph was created
    assert graph is not None
    
    # Check that it has the expected structure
    # Note: Specific graph structure testing would depend on LangGraph internals

def test_create_psychological_evolution_graph():
    """Test that psychological evolution graph is created properly"""
    graph = create_psychological_evolution_graph()
    
    assert graph is not None

def test_create_contagion_graph():
    """Test that contagion graph is created properly"""
    graph = create_contagion_graph()
    
    assert graph is not None

@pytest.mark.asyncio
async def test_agent_decision_graph_execution(sample_agent_state, monkeypatch):
    """Test executing the agent decision graph"""
    
    # Mock the LLM client to avoid actual API calls
    mock_client = MagicMock()
    mock_client.generate_psychological_response = AsyncMock(return_value={
        "reasoning": "Test reasoning",
        "decision": "COOPERATE",
        "confidence": 0.7,
        "psychological_insight": "Test insight",
        "emotional_state": "hopeful"
    })
    mock_client.generate_personality_prompt = MagicMock(return_value="Test prompt")
    
    # Patch the LLM client
    monkeypatch.setattr("src.nodes.psychological_nodes.PsychologicalLLMClient", lambda: mock_client)
    
    graph = create_agent_decision_graph()
    
    # This should execute the full decision pipeline
    result = await graph.ainvoke(sample_agent_state)
    
    # Check that a decision was made
    assert result["current_decision"] in ["COOPERATE", "DEFECT"]
    assert 0 <= result["decision_confidence"] <= 1
    
    # Check that reasoning chain was populated
    assert len(result["reasoning_chain"]) > 0

@pytest.mark.asyncio
async def test_psychological_evolution_graph_execution(sample_agent_state):
    """Test executing the psychological evolution graph"""
    
    # Add a recent memory to process
    from src.state.agent_state import Memory, Move
    sample_agent_state["recent_memories"] = [
        Memory(
            round_number=1,
            my_move=Move.COOPERATE,
            opponent_move=Move.DEFECT,
            my_payoff=0.0,
            opponent_payoff=5.0,
            emotional_impact=-2.0,
            timestamp=datetime.now()
        )
    ]
    
    graph = create_psychological_evolution_graph()
    
    result = await graph.ainvoke(sample_agent_state)
    
    # Check that psychological state was updated
    assert result["psychological_profile"] is not None
    
    # Check that reasoning chain was populated
    assert len(result["reasoning_chain"]) > 0

@pytest.mark.asyncio
async def test_contagion_graph_execution(sample_population_state):
    """Test executing the contagion graph"""
    
    # Add some agents to the population
    from src.state.agent_state import AgentState, PsychologicalProfile
    
    agent1 = AgentState(
        agent_id="agent1",
        psychological_profile=PsychologicalProfile(trust_level=0.8, loss_sensitivity=1.0),
        current_round=0,
        game_context={},
        reasoning_chain=[],
        psychological_observations=[],
        current_decision=None,
        decision_confidence=0.5,
        expected_outcomes={},
        recent_memories=[],
        trauma_triggers=[],
        recovery_progress=1.0,
        agent_type="test",
        total_score=100.0,  # High score = influential
        reference_point=0.0
    )
    
    agent2 = AgentState(
        agent_id="agent2",
        psychological_profile=PsychologicalProfile(trust_level=0.2, loss_sensitivity=2.0),
        current_round=0,
        game_context={},
        reasoning_chain=[],
        psychological_observations=[],
        current_decision=None,
        decision_confidence=0.5,
        expected_outcomes={},
        recent_memories=[],
        trauma_triggers=[],
        recovery_progress=0.5,
        agent_type="test",
        total_score=50.0,  # Lower score = less influential
        reference_point=0.0
    )
    
    sample_population_state["population"] = [agent1, agent2]
    
    graph = create_contagion_graph()
    
    result = await graph.ainvoke(sample_population_state)
    
    # Check that the graph executed successfully
    assert result is not None
    assert "population" in result
    
    # Check that some interaction results were recorded
    assert len(result["interaction_results"]) > 0

if __name__ == "__main__":
    pytest.main([__file__])