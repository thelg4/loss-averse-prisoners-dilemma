# Loss-Averse Prisoner's Dilemma: When AI Agents Learn to Feel Pain

A cutting-edge LangGraph-based research project exploring how cognitive biases emerge and spread in populations of AI agents playing iterated prisoner's dilemma games.

## 🎯 Project Overview

This project demonstrates how psychological traits like loss aversion can:
- Emerge naturally in AI agents through experience
- Spread through populations via social learning
- Evolve over multiple generations
- Impact strategic decision-making in profound ways

## 🏗️ Architecture

Built on **LangGraph** for sophisticated workflow orchestration:

```
src/
├── graphs/          # LangGraph workflow definitions
│   ├── agents/      # Agent-level decision & evolution graphs
│   ├── population/  # Population-level contagion graphs
│   └── experiments/ # Experiment orchestration graphs
├── nodes/           # Individual workflow nodes
├── state/           # LangGraph state definitions
├── tools/           # LLM clients & external tools
└── utils/           # Streaming, checkpointing, parallel execution
```

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd loss_averse_pd
pip install -r requirements.txt
```

### Environment Setup

```bash
cp .env.example .env
# Edit .env with your API keys:
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here (optional)
```

### Run Basic Experiment

```bash
# Run the main loss aversion study
python main.py --experiment loss_aversion_study

# Run with real-time streaming
python main.py --experiment loss_aversion_study --stream

# Run multiple parallel replications
python main.py --experiment loss_aversion_study --parallel 3
```

### Resume from Checkpoint

```bash
# List available experiments
python main.py --list-experiments

# Resume a specific experiment
python main.py --resume experiment_20241201_143022
```

## 🧠 Key Features

### LLM-Powered Psychological Reasoning
- Agents develop rich psychological profiles through experience
- Dynamic personality prompts based on trauma history
- Biased decision-making through loss aversion lens

### Population-Level Emergence
- Psychological traits spread via social learning
- Successful strategies influence struggling agents
- Evolutionary pressure shapes population psychology

### Real-Time Monitoring
- Stream experiment progress live
- Monitor population psychological evolution
- Track individual agent decision-making

### Robust Experiment Management
- Automatic checkpointing for long experiments
- Resume interrupted experiments seamlessly
- Parallel execution of multiple replications

## 📊 Experiment Types

### 1. Baseline Study
Compares rational vs. loss-averse agents in controlled tournaments:
```yaml
baseline_experiment:
  agents:
    - type: "rational"
      count: 10
    - type: "loss_averse" 
      count: 10
      loss_coefficient: 2.25
```

### 2. Emergent Bias Study
Observes how biases emerge in initially neutral populations:
```yaml
emergent_experiment:
  population_size: 20
  generations: 100
  interactions_per_generation: 50
```

### 3. Contagion Study
Tracks how psychological traits spread between agents:
```yaml
contagion_experiment:
  population_size: 30
  seed_agents:
    - type: "paranoid"
      trust_level: 0.1
      loss_sensitivity: 3.0
```

## 🔬 Research Applications

### Psychology & Behavioral Economics
- Study emergence of cognitive biases
- Understand social learning mechanisms
- Model trauma and recovery cycles

### AI Alignment & Safety
- Explore how AI systems develop biases
- Understand multi-agent emergent behaviors
- Design more robust AI populations

### Game Theory & Strategy
- Analyze non-rational strategic behavior
- Study population-level strategy evolution
- Model psychological influences on cooperation

## 📈 Sample Results

Recent experiments show:
- **Loss aversion emerges** in 75% of initially neutral agents after 50 generations
- **Psychological contagion** spreads traits to neighboring agents with 30% success rate
- **Population cooperation** increases when trauma-aware agents develop recovery mechanisms
- **Evolutionary pressure** favors moderately loss-averse strategies in mixed populations

## 🛠️ Advanced Usage

### Custom Experiments

Create custom experiment configurations:

```yaml
# config/my_experiment.yaml
experiments:
  custom_study:
    name: "My Custom Study"
    emergent_experiment:
      population_size: 50
      generations: 200
      # ... custom parameters
```

Run with:
```bash
python main.py --experiment custom_study --config config/my_experiment.yaml
```

### Programmatic Usage

```python
from src.graphs.experiments.master_experiment_graph import create_master_experiment_graph
from src.state.experiment_state import Ex