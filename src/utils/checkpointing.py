
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from langgraph.graph import StateGraph
from contextlib import asynccontextmanager
import os
import json
import logging

# Try to import the current LangGraph checkpointer libraries
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    SQLITE_AVAILABLE = True
except ImportError:
    try:
        # Try the old import path for compatibility
        from langgraph.checkpoint.sqlite import SqliteSaver
        SQLITE_AVAILABLE = True
        AsyncSqliteSaver = None
    except ImportError:
        # Fall back to in-memory checkpointer
        try:
            from langgraph.checkpoint.memory import MemorySaver
            SQLITE_AVAILABLE = False
            SqliteSaver = None
            AsyncSqliteSaver = None
        except ImportError:
            # Try even older import
            from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
            SQLITE_AVAILABLE = False
            SqliteSaver = None
            AsyncSqliteSaver = None



logger = logging.getLogger(__name__)

def create_memory_checkpointer():
    """Create a memory checkpointer as fallback"""
    try:
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()
    except ImportError:
        from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
        return MemorySaver()

@asynccontextmanager
async def create_async_sqlite_checkpointer(db_path: str = "experiments.db"):
    """Create an async SQLite checkpointer using context manager with better error handling"""
    
    if not SQLITE_AVAILABLE or not AsyncSqliteSaver:
        logger.warning("AsyncSQLite not available, using memory checkpointer")
        yield create_memory_checkpointer()
        return
    
    try:
        # FIXED: Ensure directory exists and use absolute path
        db_file = Path(db_path).resolve()
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # FIXED: Use file:// prefix for SQLite connection string
        connection_string = f"file:{db_file}?mode=rwc"
        
        logger.info(f"Attempting to connect to SQLite database: {db_file}")
        
        # Use the context manager approach for AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(connection_string) as checkpointer:
            logger.info(f"Successfully connected to AsyncSQLite database: {db_file}")
            yield checkpointer
            
    except Exception as e:
        logger.warning(f"Failed to create AsyncSQLite checkpointer: {e}")
        logger.warning(f"Database path attempted: {db_path}")
        logger.warning("Falling back to in-memory checkpointing")
        yield create_memory_checkpointer()

def create_sync_checkpointer(db_path: str = "experiments.db"):
    """Create a synchronous checkpointer with better error handling"""
    
    if SQLITE_AVAILABLE and SqliteSaver:
        try:
            # FIXED: Ensure directory exists and use absolute path
            db_file = Path(db_path).resolve()
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # FIXED: Create connection with proper error handling
            import sqlite3
            conn = sqlite3.connect(str(db_file), check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"Using sync SQLite checkpointing with database: {db_file}")
            return checkpointer
        except Exception as e:
            logger.warning(f"Failed to create sync SQLite checkpointer: {e}")
    
    # Fall back to memory checkpointer
    logger.warning("Using in-memory checkpointing")
    return create_memory_checkpointer()

async def run_experiment_with_async_checkpointing(
    experiment_config: dict,
    experiment_id: str = None,
    resume: bool = False,
    stream: bool = True
):
    """Run experiment with proper async checkpointing and FIXED progress tracking"""
    
    if not experiment_id:
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment: {experiment_id}")
    
    # FIXED: Use simpler database path in current directory
    db_path = f"experiment_{experiment_id}.db"
    
    # Use async context manager for checkpointing
    async with create_async_sqlite_checkpointer(db_path) as checkpointer:
        
        # Create initial state (properly typed for ExperimentState)
        initial_state = {
            "experiment_id": experiment_id,
            "current_phase": "initializing",
            "progress_percentage": 0.0,
            "start_time": datetime.now(),
            "current_time": datetime.now(),
            "experiment_config": experiment_config,
            
            # Initialize all required fields with proper types
            "population_state": {
                "population": [],
                "generation": 0,
                "interaction_results": [],
                "contagion_events": [],
                "population_metrics": {},
                "current_experiment": experiment_id,
                "experiment_parameters": experiment_config,
                "should_continue": True,
                "psychological_distribution": {},
                "dominant_traits": [],
                "avg_trust_level": 0.5,
                "avg_loss_sensitivity": 1.0,
                "avg_cooperation_rate": 0.0,
                "successful_agents": [],
                "struggling_agents": [],
                "trait_transmission_matrix": {}
            },
            
            # Initialize all list fields as empty lists (these use operator.add)
            "results": [],
            "baseline_results": [],
            "emergent_results": [],
            "contagion_results": [],
            "statistical_results": [],
            
            # Initialize boolean completion flags
            "baseline_complete": False,
            "emergent_complete": False,
            "contagion_complete": False,
            "analysis_complete": False,
            
            # Initialize configuration from experiment_config
            "estimated_completion": None,
            "total_generations": experiment_config.get("emergent_experiment", {}).get("generations", 50),
            "interactions_per_generation": experiment_config.get("emergent_experiment", {}).get("interactions_per_generation", 50),
            "rounds_per_interaction": experiment_config.get("emergent_experiment", {}).get("rounds_per_interaction", 100),
            "population_size": experiment_config.get("emergent_experiment", {}).get("population_size", 20)
        }
        
        # Create and compile graph with checkpointer
        from src.graphs.experiments.master_experiment_graph import create_master_experiment_graph
        experiment_graph_builder = create_master_experiment_graph()
        graph = experiment_graph_builder.compile(checkpointer=checkpointer)
        
        # Execute with proper threading
        thread_config = {"configurable": {"thread_id": experiment_id}}
        
        try:
            if stream:
                # FIXED: Better streaming with proper state extraction
                final_result = None
                step_count = 0
                
                async for event in graph.astream(initial_state, thread_config):
                    step_count += 1
                    
                    # FIXED: Extract progress from the actual event structure
                    current_phase = "unknown"
                    progress = 0.0
                    
                    if isinstance(event, dict):
                        # LangGraph streaming returns: {node_name: {updates}}
                        if len(event) == 1:
                            node_name = list(event.keys())[0]
                            node_updates = event[node_name]
                            
                            if isinstance(node_updates, dict):
                                current_phase = node_updates.get('current_phase', current_phase)
                                progress = node_updates.get('progress_percentage', progress)
                        else:
                            # Direct state updates
                            current_phase = event.get('current_phase', current_phase)
                            progress = event.get('progress_percentage', progress)
                    
                    logger.info(f"Step {step_count}: {node_name if 'node_name' in locals() else 'node'} -> {current_phase} ({progress:.1f}%)")
                    
                    # Store the most recent event
                    final_result = event
                
                # IMPORTANT: Get the complete final state from the graph
                try:
                    complete_state = await graph.aget_state(thread_config)
                    if complete_state and hasattr(complete_state, 'values'):
                        logger.info("Retrieved complete final state from graph")
                        return complete_state.values
                    else:
                        logger.warning("Could not get complete state, using last event")
                        return final_result
                except Exception as e:
                    logger.warning(f"Could not retrieve complete state: {e}")
                    return final_result
                    
            else:
                # Regular execution
                result = await graph.ainvoke(initial_state, thread_config)
                return result
                
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            # Log more details about the error
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

# Legacy checkpoint manager class (for backward compatibility)
class ExperimentCheckpointManager:
    """Manages experiment checkpoints and recovery - ASYNC VERSION with better error handling"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.{db_path}")
        self.sqlite_available = SQLITE_AVAILABLE
        
        if not self.sqlite_available:
            self.logger.warning("SQLite checkpointing not available. Using file-based fallback.")
            self._ensure_fallback_dir()
    
    def _ensure_fallback_dir(self):
        """Ensure fallback directory exists"""
        fallback_dir = Path("checkpoints")
        fallback_dir.mkdir(exist_ok=True)
    
    async def create_checkpointed_graph_async(self, graph_builder):
        """Create a checkpointed version of a graph with async context manager"""
        
        @asynccontextmanager
        async def graph_with_checkpointer():
            async with create_async_sqlite_checkpointer(self.db_path) as checkpointer:
                graph = graph_builder.compile(checkpointer=checkpointer)
                yield graph
        
        return graph_with_checkpointer()
    
    def create_checkpointed_graph_sync(self, graph_builder):
        """Create a checkpointed version of a graph - SYNC VERSION"""
        checkpointer = create_sync_checkpointer(self.db_path)
        return graph_builder.compile(checkpointer=checkpointer)
    
    def save_experiment_checkpoint(
        self,
        experiment_id: str,
        experiment_state: Dict[str, Any],
        force: bool = False
    ) -> bool:
        """Save experiment checkpoint to file-based system"""
        return self._save_fallback_checkpoint(experiment_id, experiment_state)
    
    def _save_fallback_checkpoint(self, experiment_id: str, experiment_state: Dict[str, Any]) -> bool:
        """Save checkpoint using file-based fallback"""
        try:
            self._ensure_fallback_dir()
            fallback_file = Path("checkpoints") / f"{experiment_id}_checkpoint.json"
            
            # Convert to serializable format
            serializable_state = self._make_serializable(experiment_state)
            
            with open(fallback_file, 'w') as f:
                json.dump(serializable_state, f, indent=2)
            
            self.logger.info(f"Saved fallback checkpoint for {experiment_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save fallback checkpoint: {e}")
            return False
    
    def load_experiment_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest experiment checkpoint"""
        return self._load_fallback_checkpoint(experiment_id)
    
    def _load_fallback_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint using file-based fallback"""
        try:
            fallback_file = Path("checkpoints") / f"{experiment_id}_checkpoint.json"
            if fallback_file.exists():
                with open(fallback_file, 'r') as f:
                    data = json.load(f)
                    # Convert datetime strings back to datetime objects
                    if "start_time" in data and isinstance(data["start_time"], str):
                        data["start_time"] = datetime.fromisoformat(data["start_time"])
                    self.logger.info(f"Loaded fallback checkpoint for {experiment_id}")
                    return data
            
            self.logger.info(f"No fallback checkpoint found for {experiment_id}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load fallback checkpoint: {e}")
            return None
    
    def list_experiments(self) -> list:
        """List all experiments with their status"""
        return self._list_fallback_experiments()
    
    def _list_fallback_experiments(self) -> list:
        """List experiments from fallback files"""
        experiments = []
        
        try:
            checkpoints_dir = Path("checkpoints")
            if not checkpoints_dir.exists():
                return experiments
            
            for checkpoint_file in checkpoints_dir.glob("*_checkpoint.json"):
                exp_id = checkpoint_file.stem.replace("_checkpoint", "")
                stat = checkpoint_file.stat()
                
                try:
                    with open(checkpoint_file, 'r') as f:
                        data = json.load(f)
                        
                    experiments.append({
                        "experiment_id": exp_id,
                        "start_time": data.get("start_time", datetime.fromtimestamp(stat.st_mtime).isoformat()),
                        "last_checkpoint": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "current_phase": data.get("current_phase", "unknown"),
                        "progress_percentage": data.get("progress_percentage", 0.0),
                        "status": "file_fallback"
                    })
                except Exception as e:
                    self.logger.warning(f"Failed to read checkpoint file {checkpoint_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to list fallback experiments: {e}")
        
        return experiments
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete all data for an experiment"""
        try:
            fallback_file = Path("checkpoints") / f"{experiment_id}_checkpoint.json"
            if fallback_file.exists():
                fallback_file.unlink()
                self.logger.info(f"Deleted fallback data for {experiment_id}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to delete fallback data: {e}")
        return False
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            # Handle Pydantic models and other objects with __dict__
            if hasattr(obj, 'dict'):  # Pydantic model
                return self._make_serializable(obj.dict())
            else:
                return self._make_serializable(obj.__dict__)
        elif hasattr(obj, 'value'):  # Enum
            return obj.value
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            # Try to convert to string as last resort
            return str(obj)