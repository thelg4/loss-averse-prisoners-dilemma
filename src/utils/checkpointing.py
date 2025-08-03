# import logging
# from typing import Dict, Any, Optional
# from datetime import datetime
# from pathlib import Path
# import json
# import os

# # Try to import the current LangGraph checkpointer libraries
# try:
#     from langgraph.checkpoint.sqlite import SqliteSaver
#     from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
#     SQLITE_AVAILABLE = True
# except ImportError:
#     try:
#         # Try the old import path for compatibility
#         from langgraph.checkpoint.sqlite import SqliteSaver
#         SQLITE_AVAILABLE = True
#         AsyncSqliteSaver = None
#     except ImportError:
#         # Fall back to in-memory checkpointer
#         try:
#             from langgraph.checkpoint.memory import MemorySaver
#             SQLITE_AVAILABLE = False
#             SqliteSaver = None
#             AsyncSqliteSaver = None
#         except ImportError:
#             # Try even older import
#             from langgraph.checkpoint.memory import InMemorySaver as MemorySaver
#             SQLITE_AVAILABLE = False
#             SqliteSaver = None
#             AsyncSqliteSaver = None

# from langgraph.graph import StateGraph

# logger = logging.getLogger(__name__)

# def create_checkpointer(db_path: str = "experiments.db"):
#     """Create a checkpointer instance"""
    
#     if SQLITE_AVAILABLE and SqliteSaver:
#         # Use SQLite checkpointing
#         Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
#         try:
#             # Try the new connection string method first
#             checkpointer = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
#             logger.info(f"Using SQLite checkpointing with database: {db_path}")
#             return checkpointer
#         except Exception as e:
#             logger.warning(f"Failed to create SQLite checkpointer: {e}")
#             # Fall back to memory checkpointer
#             checkpointer = MemorySaver()
#             logger.warning("Falling back to in-memory checkpointing")
#             return checkpointer
#     else:
#         # Fall back to in-memory checkpointing
#         checkpointer = MemorySaver()
#         logger.warning("SQLite checkpointing not available, using in-memory checkpointing. "
#                       "Install langgraph-checkpoint-sqlite for persistent checkpointing: "
#                       "pip install langgraph-checkpoint-sqlite")
#         return checkpointer

# def create_checkpointed_graph(graph_builder: StateGraph, db_path: str = "experiments.db"):
#     """Compile a graph builder with checkpointing - USE ONLY WITH UNCOMPILED BUILDERS"""
    
#     checkpointer = create_checkpointer(db_path)
    
#     # This should only be called on uncompiled StateGraph builders
#     return graph_builder.compile(checkpointer=checkpointer)

# class ExperimentCheckpointManager:
#     """Manages experiment checkpoints and recovery"""
    
#     def __init__(self, db_path: str = "experiments.db"):
#         self.db_path = db_path
#         self.logger = logging.getLogger(f"{__name__}.{db_path}")
#         self.sqlite_available = SQLITE_AVAILABLE
        
#         if self.sqlite_available:
#             self._init_checkpoint_db()
#         else:
#             self.logger.warning("SQLite checkpointing not available. Using file-based fallback.")
#             # Use simple file-based fallback
#             self.fallback_data = {}
#             self._ensure_fallback_dir()
    
#     def _ensure_fallback_dir(self):
#         """Ensure fallback directory exists"""
#         fallback_dir = Path("checkpoints")
#         fallback_dir.mkdir(exist_ok=True)
    
#     def create_checkpointed_graph(self, graph_builder):
#         """Create a checkpointed version of a graph - ONLY FOR UNCOMPILED BUILDERS"""
#         checkpointer = create_checkpointer(self.db_path)
#         return graph_builder.compile(checkpointer=checkpointer)
    
#     def _init_checkpoint_db(self):
#         """Initialize checkpoint database with custom tables"""
#         if not self.sqlite_available or not SqliteSaver:
#             return
            
#         try:
#             import sqlite3
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
            
#             # Create experiment metadata table
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS experiment_metadata (
#                     experiment_id TEXT PRIMARY KEY,
#                     start_time TEXT,
#                     last_checkpoint TEXT,
#                     current_phase TEXT,
#                     progress_percentage REAL,
#                     config_json TEXT,
#                     status TEXT
#                 )
#             """)
            
#             # Create population snapshots table
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS population_snapshots (
#                     experiment_id TEXT,
#                     generation INTEGER,
#                     snapshot_data TEXT,
#                     timestamp TEXT,
#                     PRIMARY KEY (experiment_id, generation)
#                 )
#             """)
            
#             # Create agent states table
#             cursor.execute("""
#                 CREATE TABLE IF NOT EXISTS agent_states (
#                     experiment_id TEXT,
#                     agent_id TEXT,
#                     generation INTEGER,
#                     state_data TEXT,
#                     timestamp TEXT,
#                     PRIMARY KEY (experiment_id, agent_id, generation)
#                 )
#             """)
            
#             conn.commit()
#             self.logger.info(f"Initialized checkpoint database: {self.db_path}")
            
#         except Exception as e:
#             self.logger.error(f"Failed to initialize checkpoint database: {e}")
#             # Don't raise, fall back to file-based system
#             self.sqlite_available = False
#             self._ensure_fallback_dir()
#         finally:
#             if 'conn' in locals():
#                 conn.close()
    
#     def save_experiment_checkpoint(
#         self,
#         experiment_id: str,
#         experiment_state: Dict[str, Any],
#         force: bool = False
#     ) -> bool:
#         """Save experiment checkpoint"""
        
#         if not self.sqlite_available:
#             # Use simple file-based fallback
#             return self._save_fallback_checkpoint(experiment_id, experiment_state)
        
#         # Original SQLite implementation
#         try:
#             import sqlite3
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
            
#             # Extract metadata
#             current_phase = experiment_state.get("current_phase", "unknown")
#             progress = experiment_state.get("progress_percentage", 0.0)
#             config = experiment_state.get("experiment_config", {})
#             status = "running" if progress < 100 else "completed"
            
#             # Convert datetime objects to strings for JSON serialization
#             serializable_config = self._make_serializable(config)
            
#             # Save experiment metadata
#             cursor.execute("""
#                 INSERT OR REPLACE INTO experiment_metadata
#                 (experiment_id, start_time, last_checkpoint, current_phase, 
#                  progress_percentage, config_json, status)
#                 VALUES (?, ?, ?, ?, ?, ?, ?)
#             """, (
#                 experiment_id,
#                 self._serialize_datetime(experiment_state.get("start_time", datetime.now())),
#                 datetime.now().isoformat(),
#                 current_phase,
#                 progress,
#                 json.dumps(serializable_config),
#                 status
#             ))
            
#             # Save population snapshot if available
#             population_state = experiment_state.get("population_state")
#             if population_state:
#                 generation = population_state.get("generation", 0)
#                 # Use JSON instead of pickle for better compatibility
#                 snapshot_data = json.dumps(self._make_serializable(population_state))
                
#                 cursor.execute("""
#                     INSERT OR REPLACE INTO population_snapshots
#                     (experiment_id, generation, snapshot_data, timestamp)
#                     VALUES (?, ?, ?, ?)
#                 """, (
#                     experiment_id,
#                     generation,
#                     snapshot_data,
#                     datetime.now().isoformat()
#                 ))
                
#                 # Save individual agent states
#                 population = population_state.get("population", [])
#                 for agent_state in population:
#                     agent_id = agent_state.get("agent_id")
#                     if agent_id:
#                         agent_data = json.dumps(self._make_serializable(agent_state))
                        
#                         cursor.execute("""
#                             INSERT OR REPLACE INTO agent_states
#                             (experiment_id, agent_id, generation, state_data, timestamp)
#                             VALUES (?, ?, ?, ?, ?)
#                         """, (
#                             experiment_id,
#                             agent_id,
#                             generation,
#                             agent_data,
#                             datetime.now().isoformat()
#                         ))
            
#             conn.commit()
#             self.logger.info(f"Saved checkpoint for experiment {experiment_id} at {current_phase}")
#             return True
            
#         except Exception as e:
#             self.logger.error(f"Failed to save checkpoint for {experiment_id}: {e}")
#             # Try fallback
#             return self._save_fallback_checkpoint(experiment_id, experiment_state)
#         finally:
#             if 'conn' in locals():
#                 conn.close()
    
#     def _save_fallback_checkpoint(self, experiment_id: str, experiment_state: Dict[str, Any]) -> bool:
#         """Save checkpoint using file-based fallback"""
#         try:
#             fallback_file = Path("checkpoints") / f"{experiment_id}_checkpoint.json"
            
#             # Convert to serializable format
#             serializable_state = self._make_serializable(experiment_state)
            
#             with open(fallback_file, 'w') as f:
#                 json.dump(serializable_state, f, indent=2)
            
#             self.logger.info(f"Saved fallback checkpoint for {experiment_id}")
#             return True
#         except Exception as e:
#             self.logger.error(f"Failed to save fallback checkpoint: {e}")
#             return False
    
#     def load_experiment_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
#         """Load the latest experiment checkpoint"""
        
#         if not self.sqlite_available:
#             return self._load_fallback_checkpoint(experiment_id)
        
#         # Original SQLite implementation
#         try:
#             import sqlite3
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
            
#             # Load experiment metadata
#             cursor.execute("""
#                 SELECT start_time, last_checkpoint, current_phase, 
#                        progress_percentage, config_json, status
#                 FROM experiment_metadata
#                 WHERE experiment_id = ?
#             """, (experiment_id,))
            
#             metadata_row = cursor.fetchone()
#             if not metadata_row:
#                 self.logger.info(f"No checkpoint found for experiment {experiment_id}")
#                 return self._load_fallback_checkpoint(experiment_id)
            
#             start_time, last_checkpoint, current_phase, progress, config_json, status = metadata_row
            
#             # Load latest population snapshot
#             cursor.execute("""
#                 SELECT generation, snapshot_data
#                 FROM population_snapshots
#                 WHERE experiment_id = ?
#                 ORDER BY generation DESC
#                 LIMIT 1
#             """, (experiment_id,))
            
#             snapshot_row = cursor.fetchone()
#             population_state = None
            
#             if snapshot_row:
#                 generation, snapshot_data = snapshot_row
#                 population_state = json.loads(snapshot_data)
                
#                 # Load agent states for this generation
#                 cursor.execute("""
#                     SELECT agent_id, state_data
#                     FROM agent_states
#                     WHERE experiment_id = ? AND generation = ?
#                 """, (experiment_id, generation))
                
#                 agent_rows = cursor.fetchall()
#                 if agent_rows:
#                     population = []
#                     for agent_id, state_data in agent_rows:
#                         agent_state = json.loads(state_data)
#                         population.append(agent_state)
                    
#                     population_state["population"] = population
            
#             # Reconstruct experiment state
#             experiment_state = {
#                 "experiment_id": experiment_id,
#                 "start_time": datetime.fromisoformat(start_time),
#                 "current_time": datetime.now(),
#                 "current_phase": current_phase,
#                 "progress_percentage": progress,
#                 "experiment_config": json.loads(config_json),
#                 "population_state": population_state,
#                 "baseline_complete": progress >= 25,
#                 "emergent_complete": progress >= 75,
#                 "contagion_complete": progress >= 90,
#                 "analysis_complete": progress >= 95,
#                 "results": [],
#                 "baseline_results": [],
#                 "emergent_results": [],
#                 "contagion_results": [],
#                 "statistical_results": []
#             }
            
#             self.logger.info(f"Loaded checkpoint for experiment {experiment_id} from {current_phase}")
#             return experiment_state
            
#         except Exception as e:
#             self.logger.error(f"Failed to load checkpoint for {experiment_id}: {e}")
#             return self._load_fallback_checkpoint(experiment_id)
#         finally:
#             if 'conn' in locals():
#                 conn.close()
    
#     def _load_fallback_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
#         """Load checkpoint using file-based fallback"""
#         try:
#             fallback_file = Path("checkpoints") / f"{experiment_id}_checkpoint.json"
#             if fallback_file.exists():
#                 with open(fallback_file, 'r') as f:
#                     data = json.load(f)
#                     # Convert datetime strings back to datetime objects
#                     if "start_time" in data and isinstance(data["start_time"], str):
#                         data["start_time"] = datetime.fromisoformat(data["start_time"])
#                     self.logger.info(f"Loaded fallback checkpoint for {experiment_id}")
#                     return data
            
#             self.logger.info(f"No fallback checkpoint found for {experiment_id}")
#             return None
#         except Exception as e:
#             self.logger.error(f"Failed to load fallback checkpoint: {e}")
#             return None
    
#     def list_experiments(self) -> list:
#         """List all experiments with their status"""
        
#         if not self.sqlite_available:
#             return self._list_fallback_experiments()
        
#         # Original SQLite implementation
#         try:
#             import sqlite3
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
            
#             cursor.execute("""
#                 SELECT experiment_id, start_time, last_checkpoint, 
#                        current_phase, progress_percentage, status
#                 FROM experiment_metadata
#                 ORDER BY last_checkpoint DESC
#             """)
            
#             experiments = []
#             for row in cursor.fetchall():
#                 experiment_id, start_time, last_checkpoint, phase, progress, status = row
#                 experiments.append({
#                     "experiment_id": experiment_id,
#                     "start_time": start_time,
#                     "last_checkpoint": last_checkpoint,
#                     "current_phase": phase,
#                     "progress_percentage": progress,
#                     "status": status
#                 })
            
#             # Also check fallback experiments
#             fallback_experiments = self._list_fallback_experiments()
#             experiments.extend(fallback_experiments)
            
#             return experiments
            
#         except Exception as e:
#             self.logger.error(f"Failed to list experiments: {e}")
#             return self._list_fallback_experiments()
#         finally:
#             if 'conn' in locals():
#                 conn.close()
    
#     def _list_fallback_experiments(self) -> list:
#         """List experiments from fallback files"""
#         experiments = []
        
#         try:
#             checkpoints_dir = Path("checkpoints")
#             if not checkpoints_dir.exists():
#                 return experiments
            
#             for checkpoint_file in checkpoints_dir.glob("*_checkpoint.json"):
#                 exp_id = checkpoint_file.stem.replace("_checkpoint", "")
#                 stat = checkpoint_file.stat()
                
#                 try:
#                     with open(checkpoint_file, 'r') as f:
#                         data = json.load(f)
                        
#                     experiments.append({
#                         "experiment_id": exp_id,
#                         "start_time": data.get("start_time", datetime.fromtimestamp(stat.st_mtime).isoformat()),
#                         "last_checkpoint": datetime.fromtimestamp(stat.st_mtime).isoformat(),
#                         "current_phase": data.get("current_phase", "unknown"),
#                         "progress_percentage": data.get("progress_percentage", 0.0),
#                         "status": "file_fallback"
#                     })
#                 except Exception as e:
#                     self.logger.warning(f"Failed to read checkpoint file {checkpoint_file}: {e}")
            
#         except Exception as e:
#             self.logger.error(f"Failed to list fallback experiments: {e}")
        
#         return experiments
    
#     def delete_experiment(self, experiment_id: str) -> bool:
#         """Delete all data for an experiment"""
        
#         success = True
        
#         if self.sqlite_available:
#             # SQLite cleanup
#             try:
#                 import sqlite3
#                 conn = sqlite3.connect(self.db_path)
#                 cursor = conn.cursor()
                
#                 cursor.execute("DELETE FROM experiment_metadata WHERE experiment_id = ?", (experiment_id,))
#                 cursor.execute("DELETE FROM population_snapshots WHERE experiment_id = ?", (experiment_id,))
#                 cursor.execute("DELETE FROM agent_states WHERE experiment_id = ?", (experiment_id,))
                
#                 conn.commit()
#                 self.logger.info(f"Deleted experiment {experiment_id} from SQLite")
                
#             except Exception as e:
#                 self.logger.error(f"Failed to delete experiment from SQLite {experiment_id}: {e}")
#                 success = False
#             finally:
#                 if 'conn' in locals():
#                     conn.close()
        
#         # Fallback cleanup
#         try:
#             fallback_file = Path("checkpoints") / f"{experiment_id}_checkpoint.json"
#             if fallback_file.exists():
#                 fallback_file.unlink()
#                 self.logger.info(f"Deleted fallback data for {experiment_id}")
#         except Exception as e:
#             self.logger.error(f"Failed to delete fallback data: {e}")
#             success = False
        
#         return success
    
#     def _serialize_datetime(self, dt) -> str:
#         """Serialize datetime object to string"""
#         if isinstance(dt, datetime):
#             return dt.isoformat()
#         return str(dt)
    
#     def _make_serializable(self, obj):
#         """Convert objects to JSON-serializable format"""
#         if isinstance(obj, datetime):
#             return obj.isoformat()
#         elif isinstance(obj, dict):
#             return {k: self._make_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [self._make_serializable(item) for item in obj]
#         elif hasattr(obj, '__dict__'):
#             # Handle Pydantic models and other objects with __dict__
#             if hasattr(obj, 'dict'):  # Pydantic model
#                 return self._make_serializable(obj.dict())
#             else:
#                 return self._make_serializable(obj.__dict__)
#         elif hasattr(obj, 'value'):  # Enum
#             return obj.value
#         elif isinstance(obj, (str, int, float, bool, type(None))):
#             return obj
#         else:
#             # Try to convert to string as last resort
#             return str(obj)

# class AutoCheckpointScheduler:
#     """Automatically saves checkpoints at regular intervals"""
    
#     def __init__(self, checkpoint_manager: ExperimentCheckpointManager, interval_minutes: int = 5):
#         self.checkpoint_manager = checkpoint_manager
#         self.interval_minutes = interval_minutes
#         self.last_checkpoint = {}
#         self.logger = logging.getLogger(__name__)
    
#     def should_checkpoint(self, experiment_id: str, experiment_state: Dict[str, Any]) -> bool:
#         """Determine if a checkpoint should be saved"""
#         current_time = datetime.now()
#         last_time = self.last_checkpoint.get(experiment_id, datetime.min)
        
#         # Checkpoint every N minutes
#         time_based = (current_time - last_time).total_seconds() > (self.interval_minutes * 60)
        
#         # Checkpoint at phase transitions
#         current_phase = experiment_state.get("current_phase", "")
#         last_phase = self.last_checkpoint.get(f"{experiment_id}_phase", "")
#         phase_changed = current_phase != last_phase
        
#         # Checkpoint at major progress milestones
#         progress = experiment_state.get("progress_percentage", 0)
#         last_progress = self.last_checkpoint.get(f"{experiment_id}_progress", 0)
#         major_progress = progress >= last_progress + 10  # Every 10%
        
#         return time_based or phase_changed or major_progress
    
#     def auto_checkpoint(self, experiment_id: str, experiment_state: Dict[str, Any]) -> bool:
#         """Automatically checkpoint if needed"""
        
#         if self.should_checkpoint(experiment_id, experiment_state):
#             success = self.checkpoint_manager.save_experiment_checkpoint(experiment_id, experiment_state)
            
#             if success:
#                 current_time = datetime.now()
#                 self.last_checkpoint[experiment_id] = current_time
#                 self.last_checkpoint[f"{experiment_id}_phase"] = experiment_state.get("current_phase", "")
#                 self.last_checkpoint[f"{experiment_id}_progress"] = experiment_state.get("progress_percentage", 0)
                
#                 self.logger.info(f"Auto-checkpointed experiment {experiment_id}")
            
#             return success
        
#         return False
# src/utils/checkpointing.py - ASYNC FIX VERSION
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from contextlib import asynccontextmanager

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

from langgraph.graph import StateGraph

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
    """Create an async SQLite checkpointer using context manager"""
    
    if not SQLITE_AVAILABLE or not AsyncSqliteSaver:
        logger.warning("AsyncSQLite not available, using memory checkpointer")
        yield create_memory_checkpointer()
        return
    
    try:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Use the context manager approach for AsyncSqliteSaver
        async with AsyncSqliteSaver.from_conn_string(f"sqlite:///{db_path}") as checkpointer:
            logger.info(f"Using AsyncSQLite checkpointing with database: {db_path}")
            yield checkpointer
            
    except Exception as e:
        logger.warning(f"Failed to create AsyncSQLite checkpointer: {e}")
        logger.warning("Falling back to in-memory checkpointing")
        yield create_memory_checkpointer()

def create_sync_checkpointer(db_path: str = "experiments.db"):
    """Create a synchronous checkpointer"""
    
    if SQLITE_AVAILABLE and SqliteSaver:
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            import sqlite3
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
            logger.info(f"Using sync SQLite checkpointing with database: {db_path}")
            return checkpointer
        except Exception as e:
            logger.warning(f"Failed to create sync SQLite checkpointer: {e}")
    
    # Fall back to memory checkpointer
    logger.warning("Using in-memory checkpointing")
    return create_memory_checkpointer()

# Modified run_experiment function for main.py
async def run_experiment_with_async_checkpointing(
    experiment_config: dict,
    experiment_id: str = None,
    resume: bool = False,
    stream: bool = True
):
    """Run experiment with proper async checkpointing"""
    
    if not experiment_id:
        experiment_id = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting experiment: {experiment_id}")
    
    # Use async context manager for checkpointing
    async with create_async_sqlite_checkpointer(f"experiments_{experiment_id}.db") as checkpointer:
        
        # Create initial state (you'll need to adapt this to your ExperimentState structure)
        initial_state = {
            "experiment_id": experiment_id,
            "current_phase": "initializing",
            "progress_percentage": 0.0,
            "start_time": datetime.now(),
            "experiment_config": experiment_config,
            # Add other required fields from your ExperimentState
            "population_state": {},
            "results": [],
            "baseline_complete": False,
            "emergent_complete": False,
            "contagion_complete": False,
            "analysis_complete": False,
            "baseline_results": [],
            "emergent_results": [],
            "contagion_results": [],
            "statistical_results": [],
            "current_time": datetime.now(),
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
                # Stream execution
                final_result = None
                async for event in graph.astream(initial_state, thread_config):
                    logger.info(f"Progress: {event.get('current_phase', 'unknown')}")
                    final_result = event
                return final_result
            else:
                # Regular execution
                result = await graph.ainvoke(initial_state, thread_config)
                return result
                
        except Exception as e:
            logger.error(f"Experiment execution failed: {e}")
            raise

# Legacy checkpoint manager class (for backward compatibility)
class ExperimentCheckpointManager:
    """Manages experiment checkpoints and recovery - ASYNC VERSION"""
    
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