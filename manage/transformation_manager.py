import json
import logging
from datetime import datetime
import pandas as pd
import streamlit as st
import os
from utils.data_paths import get_traces_dir

class TransformationManager:
    def __init__(self, trace_dir=None):
        """
        Initialize the TransformationManager.
        
        Parameters:
        -----------
        trace_dir : str, optional
             Directory where transformation traces will be stored.
             Defaults to DATA_ROOT/traces via data_paths utility.
        """
        self.trace_dir = trace_dir if trace_dir else get_traces_dir()
        self.history = []
        self.source_dataset = None
        self.session_id = None
        
        # Ensure trace directory exists (get_traces_dir already does this)

    def initialize_session(self, dataset_name, username=None):
        """Initializes a new session trace."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join([c for c in dataset_name if c.isalnum() or c in (' ', '.', '_', '-')]).strip()
        
        # User Isolation: Prefix session ID
        if username and username != 'admin':
            self.session_id = f"{username}_session_{timestamp}_{safe_name}"
        else:
            self.session_id = f"session_{timestamp}_{safe_name}"

        self.source_dataset = dataset_name
        self.history = []
        logging.info(f"TransformationManager: Session initialized {self.session_id}")
        self.save_trace()

    def get_trace_path(self):
        """Generates the trace file path based on the session ID."""
        if not self.session_id:
            return None
        return os.path.join(self.trace_dir, f"{self.session_id}.json")

    def add_step(self, function_name, params, description, output_dataset_path=None, stats_cat_path=None, stats_num_path=None, corr_matrix_path=None):
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
        """
        # Convert params to serializable format if necessary
        serializable_params = self._make_serializable(params)
        
        step = {
            "timestamp": datetime.now().isoformat(),
            "function": function_name,
            "params": serializable_params,
            "description": description,
            "output_dataset_path": output_dataset_path,
            "stats_cat_path": stats_cat_path,
            "stats_num_path": stats_num_path,
            "corr_matrix_path": corr_matrix_path
        }
        self.history.append(step)
        self.save_trace() # Auto-save
        logging.info(f"TransformationManager: Added step {function_name}")

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
        return {
            "session_id": self.session_id,
            "source_dataset": self.source_dataset,
            "steps": self.history
        }

    def save_trace(self, filepath=None):
        """Saves the trace to a JSON file."""
        if filepath is None:
            filepath = self.get_trace_path()
            
        if not filepath:
            return False, "No session initialized, cannot determine trace path."

        trace = self.get_trace()
        try:
            with open(filepath, 'w') as f:
                json.dump(trace, f, indent=4)
            return True, f"Trace saved to {filepath}"
        except Exception as e:
            return False, str(e)

    def load_trace(self, filepath):
        """Loads a trace from a JSON file."""
        if not filepath or not os.path.exists(filepath):
            return False, "Trace file not found"

        try:
            with open(filepath, 'r') as f:
                trace = json.load(f)
            
            self.session_id = trace.get("session_id")
            self.source_dataset = trace.get("source_dataset")
            self.history = trace.get("steps", [])
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
        
        # User Isolation Filtering
        if username and username != 'admin':
            trace_files = [f for f in trace_files if f.startswith(f"{username}_")]
        
        for trace_file in trace_files:
            try:
                with open(os.path.join(self.trace_dir, trace_file), 'r') as f:
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
