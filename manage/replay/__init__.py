"""
The replay of a trace, in the app and outside it (M8, the study package): ``manage.replay.engine`` runs the recorded
steps again on a dataset, for the Reproduce page, the session's own checks and the package's verification.

Nothing is imported here, so that the study package's command line (``python -m manage.replay``) starts without
loading the engine, its statistics and its file readers. Like every headless entry point, the package reaches neither
Streamlit nor the AI packages (``tests/test_layering.py``, R5).
"""
