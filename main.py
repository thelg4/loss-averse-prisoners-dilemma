import asyncio
import argparse
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Use the new async checkpointing function
from src.utils.checkpointing import run_experiment_with_async_checkpointing, ExperimentCheckpointManager
from src.utils.streaming import ExperimentStreamer
from src.utils.parallel import ExperimentScheduler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """Main entry point with async checkpointing"""
    
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
    parser.add_argument("--test-checkpointing", action="store_true",
                       help="Run checkpointing tests and exit")
    
    args = parser.parse_args()
    
    # Run checkpointing tests if requested
    if args.test_checkpointing:
        logger.info("Running async checkpointing tests...")
        try:
            # Test basic async checkpointing
            from src.utils.checkpointing import create_async_sqlite_checkpointer
            from langgraph.graph import StateGraph, START, END
            from typing_extensions import TypedDict
            
            class TestState(TypedDict):
                value: int
            
            def increment(state: TestState) -> TestState:
                return {"value": state["value"] + 1}
            
            async with create_async_sqlite_checkpointer("test_async.db") as checkpointer:
                builder = StateGraph(TestState)
                builder.add_node("increment", increment)
                builder.add_edge(START, "increment")
                builder.add_edge("increment", END)
                
                graph = builder.compile(checkpointer=checkpointer)
                
                config = {"configurable": {"thread_id": "test_async"}}
                result = await graph.ainvoke({"value": 0}, config)
                
                logger.info(f"✅ Async checkpointing test successful: {result}")
            
            # Clean up
            Path("test_async.db").unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"❌ Async checkpointing test failed: {e}")
            import traceback
            traceback.print_exc()
        return
    
    # List experiments if requested
    if args.list_experiments:
        checkpoint_manager = ExperimentCheckpointManager()
        experiments = checkpoint_manager.list_experiments()
        
        print("\nAvailable Experiments:")
        print("-" * 50)
        for exp in experiments:
            print(f"ID: {exp['experiment_id']}")
            print(f"  Status: {exp.get('status', 'unknown')}")
            print(f"  Phase: {exp.get('current_phase', 'unknown')}")
            print(f"  Progress: {exp.get('progress_percentage', 0):.1f}%")
            print(f"  Last Update: {exp.get('last_checkpoint', 'unknown')}")
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
    
    # Handle resume (simplified for now)
    if args.resume:
        logger.warning("Resume functionality not yet implemented with async checkpointing")
        logger.info("Starting new experiment instead...")
    
    # Run experiment(s)
    try:
        if args.parallel > 1:
            # Run multiple parallel replications
            logger.info(f"Running {args.parallel} parallel experiment replications")
            
            # Run them sequentially for now (proper parallel async would be more complex)
            results = []
            for i in range(args.parallel):
                logger.info(f"Starting experiment replication {i+1}/{args.parallel}")
                result = await run_experiment_with_async_checkpointing(
                    experiment_config=experiment_config,
                    experiment_id=f"parallel_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    stream=not args.no_stream
                )
                if result:
                    results.append(result)
            
            logger.info(f"Completed {len(results)} parallel experiments")
            
        else:
            # Run single experiment
            result = await run_experiment_with_async_checkpointing(
                experiment_config=experiment_config,
                stream=not args.no_stream
            )
            
            if result:
                logger.info("Experiment completed successfully!")
                
                # Print summary
                print("\n" + "="*60)
                print("EXPERIMENT SUMMARY")
                print("="*60)
                print(f"Experiment ID: {result.get('experiment_id', 'unknown')}")
                print(f"Final Phase: {result.get('current_phase', 'unknown')}")
                print(f"Progress: {result.get('progress_percentage', 0):.1f}%")
                
                if "results" in result and result["results"]:
                    final_report = result["results"][-1]
                    if "key_insights" in final_report:
                        print("\nKey Insights:")
                        for insight in final_report["key_insights"]:
                            print(f"  • {insight}")
                
                print("\n" + "="*60)
    
    except KeyboardInterrupt:
        logger.info("Experiment interrupted by user")
    except Exception as e:
        logger.error(f"Experiment execution failed: {e}")
        import traceback
        traceback.print_exc()

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