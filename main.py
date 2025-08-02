"""
Main entry point for the Loss-Averse Prisoner's Dilemma LangGraph experiment.

Example usage:
    python main.py --experiment loss_aversion_study
    python main.py --experiment custom --config my_config.yaml
    python main.py --resume experiment_20241201_143022
"""

import asyncio
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path

from src.graphs.experiments.master_experiment_graph import create_master_experiment_graph
from src.state.experiment_state import ExperimentState
from src.utils.checkpointing import ExperimentCheckpointManager, AutoCheckpointScheduler
from src.utils.streaming import ExperimentStreamer
from src.utils.parallel import ExperimentScheduler
from src.tools.llm_client import PsychologicalLLMClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_experiment(
    experiment_config: dict,
    experiment_id: str = None,
    resume: bool = False,
    stream: bool = True
):
    """Run a complete experiment"""
    
    if not experiment_id:
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment: {experiment_id}")
    
    # Initialize checkpoint manager
    checkpoint_manager = ExperimentCheckpointManager()
    auto_checkpoint = AutoCheckpointScheduler(checkpoint_manager)
    
    # Try to resume from checkpoint
    initial_state = None
    if resume:
        initial_state = checkpoint_manager.load_experiment_checkpoint(experiment_id)
        if initial_state:
            logger.info(f"Resuming experiment from {initial_state['current_phase']}")
        else:
            logger.warning(f"No checkpoint found for {experiment_id}, starting fresh")
    
    # Create initial state if not resuming
    if not initial_state:
        initial_state = ExperimentState(
            experiment_id=experiment_id,
            current_phase="initializing",
            population_state={},
            results=[],
            experiment_config=experiment_config,
            baseline_complete=False,
            emergent_complete=False,
            contagion_complete=False,
            analysis_complete=False,
            baseline_results=[],
            emergent_results=[],
            contagion_results=[],
            statistical_results=[],
            start_time=datetime.now(),
            current_time=datetime.now(),
            estimated_completion=None,
            progress_percentage=0.0,
            total_generations=experiment_config.get("emergent_experiment", {}).get("generations", 50),
            interactions_per_generation=experiment_config.get("emergent_experiment", {}).get("interactions_per_generation", 50),
            rounds_per_interaction=experiment_config.get("emergent_experiment", {}).get("rounds_per_interaction", 100),
            population_size=experiment_config.get("emergent_experiment", {}).get("population_size", 20)
        )
    
    # Create experiment graph with checkpointing
    experiment_graph = create_master_experiment_graph()
    checkpointed_graph = checkpoint_manager.create_checkpointed_graph(experiment_graph)
    
    # Setup streaming if requested
    streamer = None
    if stream:
        streamer = ExperimentStreamer(experiment_id)
        await streamer.start_experiment_stream(checkpointed_graph, initial_state)
        logger.info("Started experiment streaming")
    
    try:
        # Run the experiment
        logger.info("Executing experiment graph...")
        
        if stream:
            # Stream execution
            async for event in checkpointed_graph.astream(initial_state):
                # Auto-checkpoint periodically
                if isinstance(event, dict):
                    auto_checkpoint.auto_checkpoint(experiment_id, event)
                
                # Process streaming events
                logger.info(f"Experiment progress: {event.get('current_phase', 'unknown')}")
        else:
            # Regular execution
            final_result = await checkpointed_graph.ainvoke(initial_state)
            
            # Final checkpoint
            checkpoint_manager.save_experiment_checkpoint(experiment_id, final_result)
            
            logger.info(f"Experiment completed: {experiment_id}")
            return final_result
    
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
    
    finally:
        if streamer:
            streamer.stop_all_streams()

def load_experiment_config(config_path: str) -> dict:
    """Load experiment configuration from YAML file"""
    
    config_file = Path(config_path)
    if not config_file.exists():
        # Try default config location
        config_file = Path("config/experiments.yaml")
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

async def run_multiple_experiments(configs: list, max_concurrent: int = 2):
    """Run multiple experiments with scheduling"""
    
    scheduler = ExperimentScheduler(max_concurrent_experiments=max_concurrent)
    
    # Schedule all experiments
    experiment_ids = []
    for i, config in enumerate(configs):
        experiment_id = await scheduler.schedule_experiment(
            config, priority=len(configs) - i  # Higher index = higher priority
        )
        experiment_ids.append(experiment_id)
    
    # Wait for all to complete
    completed_experiments = []
    for experiment_id in experiment_ids:
        result = await scheduler.wait_for_completion(experiment_id, timeout=3600)  # 1 hour timeout
        if result:
            completed_experiments.append(result)
    
    return completed_experiments

async def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description="Run Loss-Averse Prisoner's Dilemma Experiments")
    parser.add_argument("--experiment", default="loss_aversion_study", 
                       help="Experiment name from config file")
    parser.add_argument("--config", default="config/experiments.yaml",
                       help="Path to configuration file")
    parser.add_argument("--resume", 
                       help="Resume experiment from checkpoint (provide experiment ID)")
    parser.add_argument("--no-stream", action="store_true",
                       help="Disable real-time streaming")
    parser.add_argument("--parallel", type=int, default=1,
                       help="Number of parallel experiment replications")
    parser.add_argument("--list-experiments", action="store_true",
                       help="List available experiments and exit")
    
    args = parser.parse_args()
    
    # List experiments if requested
    if args.list_experiments:
        checkpoint_manager = ExperimentCheckpointManager()
        experiments = checkpoint_manager.list_experiments()
        
        print("\nAvailable Experiments:")
        print("-" * 50)
        for exp in experiments:
            print(f"ID: {exp['experiment_id']}")
            print(f"  Status: {exp['status']}")
            print(f"  Phase: {exp['current_phase']}")
            print(f"  Progress: {exp['progress_percentage']:.1f}%")
            print(f"  Last Update: {exp['last_checkpoint']}")
            print()
        return
    
    # Load configuration
    try:
        config = load_experiment_config(args.config)
        
        if args.experiment not in config["experiments"]:
            logger.error(f"Experiment '{args.experiment}' not found in config")
            return
        
        experiment_config = config["experiments"][args.experiment]
        
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return
    
    # Handle resume
    if args.resume:
        try:
            await run_experiment(
                experiment_config,
                experiment_id=args.resume,
                resume=True,
                stream=not args.no_stream
            )
        except Exception as e:
            logger.error(f"Failed to resume experiment: {e}")
        return
    
    # Run experiment(s)
    try:
        if args.parallel > 1:
            # Run multiple parallel replications
            logger.info(f"Running {args.parallel} parallel experiment replications")
            
            configs = [experiment_config] * args.parallel
            results = await run_multiple_experiments(configs, max_concurrent=2)
            
            logger.info(f"Completed {len(results)} parallel experiments")
            
        else:
            # Run single experiment
            result = await run_experiment(
                experiment_config,
                stream=not args.no_stream
            )
            
            if result:
                logger.info("Experiment completed successfully!")
                
                # Print summary
                if "results" in result and result["results"]:
                    final_report = result["results"][-1]
                    print("\n" + "="*60)
                    print("EXPERIMENT SUMMARY")
                    print("="*60)
                    print(f"Experiment ID: {result['experiment_id']}")
                    print(f"Duration: {result.get('current_time', datetime.now()) - result['start_time']}")
                    print(f"Final Phase: {result['current_phase']}")
                    print(f"Progress: {result['progress_percentage']:.1f}%")
                    
                    if "key_insights" in final_report:
                        print("\nKey Insights:")
                        for insight in final_report["key_insights"]:
                            print(f"  • {insight}")
                    
                    print("\n" + "="*60)
    
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}")
        raise

if __name__ == "__main__":
    # Set up environment
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Verify required environment variables
    required_vars = ["OPENAI_API_KEY"]  # Add ANTHROPIC_API_KEY if using Anthropic
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please create a .env file based on .env.example")
        exit(1)
    
    # Run main
    asyncio.run(main())