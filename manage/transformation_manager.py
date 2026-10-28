import contextlib
import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd

from core.lineage import TRACE_SCHEMA, genesis_hash, seal_step, verify_trace
from core.steps import STEPS
from core.workspace import UserScope
from utils.data_paths import get_traces_dir
from utils.settings import get_settings

logger = logging.getLogger(__name__)

class TransformationManager:
    def __init__(self, trace_dir: str | None = None, scope: UserScope | None = None) -> None:
        """
        Initialize the TransformationManager.
        
        Parameters:
        -----------
        trace_dir : str, optional
             Directory where transformation traces will be stored.
             Defaults to the traces folder of ``scope``, else DATA_ROOT/traces via data_paths utility.
        scope : UserScope, optional
             The user's workspace, passed by the session's services (manage.services); None outside a
             session.
        """
        self.scope: UserScope | None = scope
        if trace_dir:
            self.trace_dir: str = trace_dir
        elif self.scope is not None:
            self.trace_dir = str(self.scope.traces_dir())  # per-user workspace (audit E4)
        else:
            self.trace_dir = get_traces_dir()
        self.history: list[dict[str, Any]] = []
        self.source_dataset: str | None = None
        self.session_id: str | None = None
        # Trace schema 2: every step is chained to the previous one by a hash (core.lineage)
        self.schema_version: int = TRACE_SCHEMA
        self.created_at: str | None = None
        self.parent: dict[str, Any] | None = None
        self.head_hash: str | None = None
        self.read_only: bool = False

        # Ensure trace directory exists (get_traces_dir already does this)

    def initialize_session(self, dataset_name: str, username: str | None = None, *, parent: dict[str, Any] | None = None) -> None:
        """Initializes a new session trace (``parent`` links it to the trace of a saved version it starts from)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c for c in dataset_name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()

        # Legacy name-prefix isolation only when no workspace scope is active
        if self.scope is None and username and username != 'admin':
            self.session_id = f"{username}_session_{timestamp}_{safe_name}"
        else:
            self.session_id = f"session_{timestamp}_{safe_name}"

        self.source_dataset = dataset_name
        self.history = []
        self.schema_version = TRACE_SCHEMA
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.parent = parent
        self.head_hash = genesis_hash(self.session_id)
        self.read_only = False
        logger.info("TransformationManager: Session initialized %s", self.session_id)
        self.save_trace()

    def get_trace_path(self) -> str | None:
        """Generates the trace file path based on the session ID."""
        if not self.session_id:
            return None
        return os.path.join(self.trace_dir, f"{self.session_id}.json")

    def add_step(self, function_name, params, description, output_dataset_path=None, stats_cat_path=None,
                 stats_num_path=None, corr_matrix_path=None, *, data=None, population=None, replayable=None,
                 recorded_at=None):
        """
        Records a transformation step with optional references to intermediate data.

        Parameters:
        -----------
        function_name : str
            Name of the function/transformation applied.
        params : dict
             Dictionary of parameters used in the transformation.
        description : str
             Human-readable description of the step.
        output_dataset_path : str, optional
             Path to the output dataset (snapshot).
        stats_cat_path : str, optional
             Path to categorical statistics file.
        stats_num_path : str, optional
             Path to numerical statistics file.
        corr_matrix_path : str, optional
             Path to correlation matrix file.
        data : dict, optional
             What the step did to the table (rows/columns in and out, fingerprints): core.lineage.Change.
        population : dict, optional
             The analysis population the step used (criteria, counts, rows kept).
        replayable : bool, optional
             False when the replay engine cannot redo the step.
        recorded_at : str, optional
             When the change was made, if it was applied later (an Enrich draft); the step then also records
             ``applied_at``.
        """
        if self.read_only:
            raise RuntimeError("This trace was loaded for review and cannot be extended.")
        if function_name not in STEPS:
            # every kind of step is declared in core/steps.py, so that the replay and the report know it
            if get_settings().development.strict_state:  # the test suite
                STEPS.get(function_name)  # raises UnknownIdError, which lists the registered steps
            logger.warning("Trace step %r is not registered in core/steps.py", function_name)
        # Convert params to serializable format if necessary
        serializable_params = self._make_serializable(params)
        now = datetime.now().isoformat()
        step = {
            "index": len(self.history) + 1,
            "timestamp": recorded_at or now,
            "function": function_name,
            "params": serializable_params,
            "description": description,
            "output_dataset_path": output_dataset_path,
            "stats_cat_path": stats_cat_path,
            "stats_num_path": stats_num_path,
            "corr_matrix_path": corr_matrix_path
        }
        if recorded_at:
            step["applied_at"] = now
        if data is not None:
            step["data"] = data
        if population is not None:
            step["population"] = population
        if replayable is not None:
            step["replayable"] = bool(replayable)
        if self.head_hash is None:
            self.head_hash = genesis_hash(self.session_id)
        sealed = seal_step(step, self.head_hash)
        self.history.append(sealed)
        self.head_hash = sealed["hash"]
        self.save_trace() # Auto-save
        logger.info("TransformationManager: Added step %s", function_name)

    def _make_serializable(self, obj):
        """Recursively converts objects to JSON-serializable formats."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return "DataFrame/Series (Not Serializable)"
        elif hasattr(obj, 'item'): # numpy types
            return obj.item()
        else:
            return obj

    def get_trace(self):
        """Returns the current transformation trace."""
        trace = {
            "session_id": self.session_id,
            "source_dataset": self.source_dataset,
            "steps": self.history
        }
        if self.schema_version >= TRACE_SCHEMA:
            trace = {
                "schema_version": self.schema_version,
                **trace,
                "created_at": self.created_at,
                "parent": self.parent,
                "head_hash": self.head_hash,
            }
        return trace

    def verify(self):
        """Integrity of the hash chain (core.lineage.TraceVerification)."""
        return verify_trace(self.get_trace())

    def save_trace(self, filepath=None):
        """Saves the trace to a JSON file (UTF-8, atomic replace).

        A relative ``filepath`` is reduced to its file name and placed in ``trace_dir``, so a trace never
        lands outside the user's workspace (for example in the process working directory).
        """
        if filepath is None:
            filepath = self.get_trace_path()

        if not filepath:
            return False, "No session initialized, cannot determine trace path."
        if not os.path.isabs(filepath):
            filepath = os.path.join(self.trace_dir, os.path.basename(filepath))

        trace = self.get_trace()
        tmp_path = f"{filepath}.tmp"
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(trace, f, indent=4)
            os.replace(tmp_path, filepath)
            return True, f"Trace saved to {filepath}"
        except Exception as e:
            with contextlib.suppress(OSError):
                os.remove(tmp_path)
            return False, str(e)

    def load_trace(self, filepath):
        """Loads a trace from a JSON file."""
        if not filepath or not os.path.exists(filepath):
            return False, "Trace file not found"

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                trace = json.load(f)
            
            self.session_id = trace.get("session_id")
            self.source_dataset = trace.get("source_dataset")
            self.history = trace.get("steps", [])
            self.schema_version = int(trace.get("schema_version") or 1)
            self.created_at = trace.get("created_at")
            self.parent = trace.get("parent")
            self.head_hash = trace.get("head_hash")
            self.read_only = True  # a loaded trace documents the past: new steps would break its chain
            return True, "Trace loaded successfully"
        except Exception as e:
            return False, str(e)

    def get_available_datasets_from_traces(self, username=None):
        """
        Scans all trace files in the trace directory and returns a list of available datasets.
        Each entry contains: session_id, timestamp, step_description, file_path.
        Filters by username if provided (unless admin).
        """
        datasets = []
        if not os.path.exists(self.trace_dir):
            return datasets

        trace_files = [f for f in os.listdir(self.trace_dir) if f.endswith('.json')]
        
        # Legacy name-prefix filtering only when no workspace scope is active
        if self.scope is None and username and username != 'admin':
            trace_files = [f for f in trace_files if f.startswith(f"{username}_")]
        
        for trace_file in trace_files:
            try:
                with open(os.path.join(self.trace_dir, trace_file), 'r', encoding='utf-8') as f:
                    trace = json.load(f)
                    
                session_id = trace.get("session_id", "Unknown Session")
                
                for i, step in enumerate(trace.get("steps", [])):
                    if step.get("output_dataset_path"):
                        datasets.append({
                            "session_id": session_id,
                            "step_number": i + 1,
                            "timestamp": step.get("timestamp"),
                            "description": step.get("description"),
                            "function": step.get("function"),
                            "file_path": step.get("output_dataset_path")
                        })
            except Exception as e:
                logging.error(f"Error reading trace file {trace_file}: {e}")
                
        # Sort by timestamp descending
        datasets.sort(key=lambda x: x["timestamp"], reverse=True)
        return datasets
