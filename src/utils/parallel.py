import asyncio
from typing import List, Dict, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

async def run_parallel_experiments(
    experiment_configs: List[Dict[str, Any]],
    max_concurrent: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Run multiple experiments in parallel"""
    
    if max_concurrent is None:
        max_concurrent = min(len(experiment_configs), multiprocessing.cpu_count())
    
    logger.info(f"Running {len(experiment_configs)} experiments with max {max_concurrent} concurrent")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_single_experiment_with_semaphore(config):
        async with semaphore:
            return await run_single_experiment(config)
    
    # Create tasks for all experiments
    tasks = [
        run_single_experiment_with_semaphore(config)
        for config in experiment_configs
    ]
    
    # Run all experiments concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Separate successful results from exceptions
    successful_results = []
    failed_experiments = []
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Experiment {i} failed: {result}")
            failed_experiments.append({
                "experiment_index": i,
                "config": experiment_configs[i],
                "error": str(result)
            })
        else:
            successful_results.append(result)
    
    logger.info(f"Completed {len(successful_results)} experiments successfully, {len(failed_experiments)} failed")
    
    return {
        "successful_results": successful_results,
        "failed_experiments": failed_experiments,
        "total_experiments": len(experiment_configs),
        "success_rate": len(successful_results) / len(experiment_configs)
    }

async def run_single_experiment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single experiment (placeholder - would use actual experiment graph)"""
    
    from ..graphs.experiments.master_experiment_graph import create_master_experiment_graph
    from ..state.experiment_state import ExperimentState
    
    experiment_id = config.get("experiment_id", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Create initial experiment state
    initial_state = ExperimentState(
        experiment_id=experiment_id,
        current_phase="initializing",
        population_state={},
        results=[],
        experiment_config=config,
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
        total_generations=config.get("total_generations", 50),
        interactions_per_generation=config.get("interactions_per_generation", 50),
        rounds_per_interaction=config.get("rounds_per_interaction", 100),
        population_size=config.get("population_size", 20)
    )
    
    # Create and run experiment graph
    experiment_graph = create_master_experiment_graph()
    result = await experiment_graph.ainvoke(initial_state)
    
    return result

class ParallelPopulationManager:
    """Manages parallel processing of population interactions"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.executor = None
        self.logger = logging.getLogger(__name__)
    
    async def run_parallel_interactions(
        self,
        agent_pairs: List[tuple],
        interaction_function: Callable,
        rounds_per_interaction: int = 100
    ) -> List[Dict[str, Any]]:
        """Run interactions between agent pairs in parallel"""
        
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"Running {len(agent_pairs)} interactions in parallel")
        
        # Create tasks for all interactions
        loop = asyncio.get_event_loop()
        tasks = []
        
        for i, (agent1, agent2) in enumerate(agent_pairs):
            task = loop.run_in_executor(
                self.executor,
                interaction_function,
                agent1, agent2, rounds_per_interaction, f"interaction_{i}"
            )
            tasks.append(task)
        
        # Run all interactions concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_interactions = []
        failed_interactions = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Interaction {i} failed: {result}")
                failed_interactions.append({
                    "interaction_index": i,
                    "agent_pair": agent_pairs[i],
                    "error": str(result)
                })
            else:
                successful_interactions.append(result)
        
        return {
            "successful_interactions": successful_interactions,
            "failed_interactions": failed_interactions,
            "total_interactions": len(agent_pairs)
        }
    
    async def parallel_agent_evolution(
        self,
        agents: List[Dict[str, Any]],
        evolution_function: Callable
    ) -> List[Dict[str, Any]]:
        """Apply evolution function to agents in parallel"""
        
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info(f"Evolving {len(agents)} agents in parallel")
        
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.executor, evolution_function, agent)
            for agent in agents
        ]
        
        evolved_agents = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        successful_evolutions = [
            agent for agent in evolved_agents 
            if not isinstance(agent, Exception)
        ]
        
        logger.info(f"Successfully evolved {len(successful_evolutions)}/{len(agents)} agents")
        
        return successful_evolutions
    
    def shutdown(self):
        """Shutdown the executor"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None

class BatchProcessor:
    """Process large datasets in batches with parallel execution"""
    
    def __init__(self, batch_size: int = 100, max_workers: Optional[int] = None):
        self.batch_size = batch_size
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.logger = logging.getLogger(__name__)
    
    async def process_population_in_batches(
        self,
        population: List[Dict[str, Any]],
        processing_function: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process population in batches for memory efficiency"""
        
        # Split population into batches
        batches = [
            population[i:i + self.batch_size]
            for i in range(0, len(population), self.batch_size)
        ]
        
        logger.info(f"Processing {len(population)} agents in {len(batches)} batches")
        
        all_results = []
        
        for i, batch in enumerate(batches):
            logger.info(f"Processing batch {i + 1}/{len(batches)}")
            
            # Process batch in parallel
            batch_results = await self._process_batch_parallel(
                batch, processing_function, **kwargs
            )
            
            all_results.extend(batch_results)
        
        return all_results
    
    async def _process_batch_parallel(
        self,
        batch: List[Dict[str, Any]],
        processing_function: Callable,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process a single batch in parallel"""
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            loop = asyncio.get_event_loop()
            
            tasks = [
                loop.run_in_executor(executor, processing_function, item, **kwargs)
                for item in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            successful_results = [
                result for result in results 
                if not isinstance(result, Exception)
            ]
            
            return successful_results

class ExperimentScheduler:
    """Schedule and manage multiple experiments with resource constraints"""
    
    def __init__(self, max_concurrent_experiments: int = 2):
        self.max_concurrent = max_concurrent_experiments
        self.running_experiments = {}
        self.experiment_queue = []
        self.completed_experiments = []
        self.logger = logging.getLogger(__name__)
    
    async def schedule_experiment(
        self,
        experiment_config: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """Schedule an experiment for execution"""
        
        experiment_id = experiment_config.get(
            "experiment_id", 
            f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        experiment_item = {
            "experiment_id": experiment_id,
            "config": experiment_config,
            "priority": priority,
            "scheduled_time": datetime.now()
        }
        
        # Add to queue (higher priority first)
        self.experiment_queue.append(experiment_item)
        self.experiment_queue.sort(key=lambda x: x["priority"], reverse=True)
        
        logger.info(f"Scheduled experiment {experiment_id} with priority {priority}")
        
        # Try to start experiment if there's capacity
        await self._try_start_next_experiment()
        
        return experiment_id
    
    async def _try_start_next_experiment(self):
        """Try to start the next experiment in queue if there's capacity"""
        
        if (len(self.running_experiments) < self.max_concurrent and 
            self.experiment_queue):
            
            experiment_item = self.experiment_queue.pop(0)
            experiment_id = experiment_item["experiment_id"]
            
            # Start experiment
            task = asyncio.create_task(
                self._run_experiment_with_cleanup(experiment_item)
            )
            
            self.running_experiments[experiment_id] = {
                "task": task,
                "start_time": datetime.now(),
                "config": experiment_item["config"]
            }
            
            logger.info(f"Started experiment {experiment_id}")
    
    async def _run_experiment_with_cleanup(self, experiment_item: Dict[str, Any]):
        """Run experiment and handle cleanup"""
        
        experiment_id = experiment_item["experiment_id"]
        
        try:
            result = await run_single_experiment(experiment_item["config"])
            
            # Mark as completed
            self.completed_experiments.append({
                "experiment_id": experiment_id,
                "result": result,
                "completion_time": datetime.now()
            })
            
            logger.info(f"Completed experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            
            self.completed_experiments.append({
                "experiment_id": experiment_id,
                "error": str(e),
                "completion_time": datetime.now()
            })
            
        finally:
            # Cleanup
            if experiment_id in self.running_experiments:
                del self.running_experiments[experiment_id]
            
            # Try to start next experiment
            await self._try_start_next_experiment()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status"""
        
        return {
            "running_experiments": len(self.running_experiments),
            "queued_experiments": len(self.experiment_queue),
            "completed_experiments": len(self.completed_experiments),
            "max_concurrent": self.max_concurrent,
            "running_experiment_ids": list(self.running_experiments.keys()),
            "queue_experiment_ids": [e["experiment_id"] for e in self.experiment_queue]
        }
    
    async def wait_for_completion(self, experiment_id: str, timeout: Optional[float] = None):
        """Wait for a specific experiment to complete"""
        
        if experiment_id in self.running_experiments:
            task = self.running_experiments[experiment_id]["task"]
            
            try:
                await asyncio.wait_for(task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Experiment {experiment_id} timed out after {timeout}s")
        
        # Check if completed
        for completed in self.completed_experiments:
            if completed["experiment_id"] == experiment_id:
                return completed
        
        return None
    
    async def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a running or queued experiment"""
        
        # Cancel if running
        if experiment_id in self.running_experiments:
            task = self.running_experiments[experiment_id]["task"]
            task.cancel()
            del self.running_experiments[experiment_id]
            logger.info(f"Cancelled running experiment {experiment_id}")
            return True
        
        # Remove from queue
        original_length = len(self.experiment_queue)
        self.experiment_queue = [
            e for e in self.experiment_queue 
            if e["experiment_id"] != experiment_id
        ]
        
        if len(self.experiment_queue) < original_length:
            logger.info(f"Cancelled queued experiment {experiment_id}")
            return True
        
        return False