# quick_data_check.py
from detailed_agent_inspector import extract_all_agent_data
from pathlib import Path

# Find your most recent experiment database
db_files = list(Path(".").glob("experiment_*.db"))
latest_db = max(db_files, key=lambda f: f.stat().st_mtime)

print(f"Checking: {latest_db}")
experiment_data = extract_all_agent_data(str(latest_db))

if experiment_data and 'population_state' in experiment_data:
    population = experiment_data['population_state'].get('population', [])
    print(f"Found {len(population)} agents in database")
    
    for agent in population[:2]:  # Check first 2 agents
        print(f"\nAgent: {agent.get('agent_id', 'unknown')}")
        print(f"  Score: {agent.get('total_score', 'unknown')}")
        print(f"  Memories: {len(agent.get('recent_memories', []))}")
        print(f"  Reasoning steps: {len(agent.get('reasoning_chain', []))}")
        
        # Check if memories have actual data
        memories = agent.get('recent_memories', [])
        if memories:
            first_memory = memories[0]
            print(f"  First memory type: {type(first_memory)}")
            if hasattr(first_memory, 'my_move'):
                print(f"  First memory move: {first_memory.my_move}")
            else:
                print(f"  First memory data: {first_memory}")
else:
    print("No population data found!")