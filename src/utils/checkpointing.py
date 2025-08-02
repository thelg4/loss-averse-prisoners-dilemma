from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph
from typing import Dict, Any, Optional
import sqlite3
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

def create_checkpointed_graph(graph: StateGraph, db_path: str = "experiments.db") -> StateGraph:
    """Add checkpointing to graph for long-running experiments"""
    
    # Ensure the database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create SQLite saver for checkpointing
    memory = SqliteSaver.from_conn_string(f"sqlite:///{db_path}")
    
    # Return graph with checkpointing enabled
    return graph.with_checkpointer(memory)

class ExperimentCheckpointManager:
    """Manages experiment checkpoints and recovery"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(f"{__name__}.{db_path}")
        self._init_checkpoint_db()
    
    def _init_checkpoint_db(self):
        """Initialize checkpoint database with custom tables"""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Create experiment metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_metadata (
                    experiment_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    last_checkpoint TEXT,
                    current_phase TEXT,
                    progress_percentage REAL,
                    config_json TEXT,
                    status TEXT
                )
            """)
            
            # Create population snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS population_snapshots (
                    experiment_id TEXT,
                    generation INTEGER,
                    snapshot_data BLOB,
                    timestamp TEXT,
                    PRIMARY KEY (experiment_id, generation)
                )
            """)
            
            # Create agent states table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS agent_states (
                    experiment_id TEXT,
                    agent_id TEXT,
                    generation INTEGER,
                    state_data BLOB,
                    timestamp TEXT,
                    PRIMARY KEY (experiment_id, agent_id, generation)
                )
            """)
            
            conn.commit()
            self.logger.info(f"Initialized checkpoint database: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize checkpoint database: {e}")
            raise
        finally:
            conn.close()
    
    def save_experiment_checkpoint(
        self,
        experiment_id: str,
        experiment_state: Dict[str, Any],
        force: bool = False
    ) -> bool:
        """Save experiment checkpoint"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract metadata
            current_phase = experiment_state.get("current_phase", "unknown")
            progress = experiment_state.get("progress_percentage", 0.0)
            config = experiment_state.get("experiment_config", {})
            status = "running" if progress < 100 else "completed"
            
            # Save experiment metadata
            cursor.execute("""
                INSERT OR REPLACE INTO experiment_metadata
                (experiment_id, start_time, last_checkpoint, current_phase, 
                 progress_percentage, config_json, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                experiment_state.get("start_time", datetime.now()).isoformat(),
                datetime.now().isoformat(),
                current_phase,
                progress,
                json.dumps(config),
                status
            ))
            
            # Save population snapshot if available
            population_state = experiment_state.get("population_state")
            if population_state:
                generation = population_state.get("generation", 0)
                snapshot_data = pickle.dumps(population_state)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO population_snapshots
                    (experiment_id, generation, snapshot_data, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    experiment_id,
                    generation,
                    snapshot_data,
                    datetime.now().isoformat()
                ))
                
                # Save individual agent states
                population = population_state.get("population", [])
                for agent_state in population:
                    agent_id = agent_state.get("agent_id")
                    if agent_id:
                        agent_data = pickle.dumps(agent_state)
                        
                        cursor.execute("""
                            INSERT OR REPLACE INTO agent_states
                            (experiment_id, agent_id, generation, state_data, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            experiment_id,
                            agent_id,
                            generation,
                            agent_data,
                            datetime.now().isoformat()
                        ))
            
            conn.commit()
            self.logger.info(f"Saved checkpoint for experiment {experiment_id} at {current_phase}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for {experiment_id}: {e}")
            return False
        finally:
            conn.close()
    
    def load_experiment_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest experiment checkpoint"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Load experiment metadata
            cursor.execute("""
                SELECT start_time, last_checkpoint, current_phase, 
                       progress_percentage, config_json, status
                FROM experiment_metadata
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            metadata_row = cursor.fetchone()
            if not metadata_row:
                self.logger.info(f"No checkpoint found for experiment {experiment_id}")
                return None
            
            start_time, last_checkpoint, current_phase, progress, config_json, status = metadata_row
            
            # Load latest population snapshot
            cursor.execute("""
                SELECT generation, snapshot_data
                FROM population_snapshots
                WHERE experiment_id = ?
                ORDER BY generation DESC
                LIMIT 1
            """, (experiment_id,))
            
            snapshot_row = cursor.fetchone()
            population_state = None
            
            if snapshot_row:
                generation, snapshot_data = snapshot_row
                population_state = pickle.loads(snapshot_data)
                
                # Load agent states for this generation
                cursor.execute("""
                    SELECT agent_id, state_data
                    FROM agent_states
                    WHERE experiment_id = ? AND generation = ?
                """, (experiment_id, generation))
                
                agent_rows = cursor.fetchall()
                if agent_rows:
                    population = []
                    for agent_id, state_data in agent_rows:
                        agent_state = pickle.loads(state_data)
                        population.append(agent_state)
                    
                    population_state["population"] = population
            
            # Reconstruct experiment state
            experiment_state = {
                "experiment_id": experiment_id,
                "start_time": datetime.fromisoformat(start_time),
                "current_time": datetime.now(),
                "current_phase": current_phase,
                "progress_percentage": progress,
                "experiment_config": json.loads(config_json),
                "population_state": population_state,
                "baseline_complete": progress >= 25,
                "emergent_complete": progress >= 75,
                "contagion_complete": progress >= 90,
                "analysis_complete": progress >= 95,
                "results": [],
                "baseline_results": [],
                "emergent_results": [],
                "contagion_results": [],
                "statistical_results": []
            }
            
            self.logger.info(f"Loaded checkpoint for experiment {experiment_id} from {current_phase}")
            return experiment_state
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for {experiment_id}: {e}")
            return None
        finally:
            conn.close()
    
    def list_experiments(self) -> list:
        """List all experiments with their status"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT experiment_id, start_time, last_checkpoint, 
                       current_phase, progress_percentage, status
                FROM experiment_metadata
                ORDER BY last_checkpoint DESC
            """)
            
            experiments = []
            for row in cursor.fetchall():
                experiment_id, start_time, last_checkpoint, phase, progress, status = row
                experiments.append({
                    "experiment_id": experiment_id,
                    "start_time": start_time,
                    "last_checkpoint": last_checkpoint,
                    "current_phase": phase,
                    "progress_percentage": progress,
                    "status": status
                })
            
            return experiments
            
        except Exception as e:
            self.logger.error(f"Failed to list experiments: {e}")
            return []
        finally:
            conn.close()
    
    def delete_experiment(self, experiment_id: str) -> bool:
        """Delete all data for an experiment"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete from all tables
            cursor.execute("DELETE FROM experiment_metadata WHERE experiment_id = ?", (experiment_id,))
            cursor.execute("DELETE FROM population_snapshots WHERE experiment_id = ?", (experiment_id,))
            cursor.execute("DELETE FROM agent_states WHERE experiment_id = ?", (experiment_id,))
            
            conn.commit()
            self.logger.info(f"Deleted experiment {experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete experiment {experiment_id}: {e}")
            return False
        finally:
            conn.close()
    
    def get_experiment_progress(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress of an experiment"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_phase, progress_percentage, status, last_checkpoint
                FROM experiment_metadata
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            row = cursor.fetchone()
            if row:
                phase, progress, status, last_checkpoint = row
                return {
                    "current_phase": phase,
                    "progress_percentage": progress,
                    "status": status,
                    "last_checkpoint": last_checkpoint
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get progress for {experiment_id}: {e}")
            return None
        finally:
            conn.close()
    
    def cleanup_old_checkpoints(self, keep_generations: int = 50):
        """Clean up old checkpoints to save space"""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Keep only the latest N generations per experiment
            cursor.execute("""
                DELETE FROM population_snapshots
                WHERE (experiment_id, generation) NOT IN (
                    SELECT experiment_id, generation
                    FROM population_snapshots ps1
                    WHERE (
                        SELECT COUNT(*)
                        FROM population_snapshots ps2
                        WHERE ps2.experiment_id = ps1.experiment_id
                        AND ps2.generation >= ps1.generation
                    ) <= ?
                )
            """, (keep_generations,))
            
            cursor.execute("""
                DELETE FROM agent_states
                WHERE (experiment_id, generation) NOT IN (
                    SELECT experiment_id, generation
                    FROM population_snapshots
                )
            """)
            
            deleted_snapshots = cursor.rowcount
            conn.commit()
            
            self.logger.info(f"Cleaned up {deleted_snapshots} old checkpoint entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")
            return False
        finally:
            conn.close()

# Automatic checkpoint scheduling
class AutoCheckpointScheduler:
    """Automatically saves checkpoints at regular intervals"""
    
    def __init__(self, checkpoint_manager: ExperimentCheckpointManager, interval_minutes: int = 5):
        self.checkpoint_manager = checkpoint_manager
        self.interval_minutes = interval_minutes
        self.last_checkpoint = {}
        self.logger = logging.getLogger(__name__)
    
    def should_checkpoint(self, experiment_id: str, experiment_state: Dict[str, Any]) -> bool:
        """Determine if a checkpoint should be saved"""
        
        current_time = datetime.now()
        last_time = self.last_checkpoint.get(experiment_id, datetime.min)
        
        # Checkpoint every N minutes
        time_based = (current_time - last_time).total_seconds() > (self.interval_minutes * 60)
        
        # Checkpoint at phase transitions
        current_phase = experiment_state.get("current_phase", "")
        last_phase = self.last_checkpoint.get(f"{experiment_id}_phase", "")
        phase_changed = current_phase != last_phase
        
        # Checkpoint at major progress milestones
        progress = experiment_state.get("progress_percentage", 0)
        last_progress = self.last_checkpoint.get(f"{experiment_id}_progress", 0)
        major_progress = progress >= last_progress + 10  # Every 10%
        
        return time_based or phase_changed or major_progress
    
    def auto_checkpoint(self, experiment_id: str, experiment_state: Dict[str, Any]) -> bool:
        """Automatically checkpoint if needed"""
        
        if self.should_checkpoint(experiment_id, experiment_state):
            success = self.checkpoint_manager.save_experiment_checkpoint(experiment_id, experiment_state)
            
            if success:
                current_time = datetime.now()
                self.last_checkpoint[experiment_id] = current_time
                self.last_checkpoint[f"{experiment_id}_phase"] = experiment_state.get("current_phase", "")
                self.last_checkpoint[f"{experiment_id}_progress"] = experiment_state.get("progress_percentage", 0)
                
                self.logger.info(f"Auto-checkpointed experiment {experiment_id}")
            
            return success
        
        return False