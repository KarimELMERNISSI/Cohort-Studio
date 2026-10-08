import json
from pathlib import Path
from docx import Document
from manage.trace_documenter import TraceDocumenter


def test_trace_documenter(tmp_path: Path):
    """Test generating docx reports from trace files."""
    trace_data = {
        "session_id": "test_session",
        "source_dataset": "test_source.csv",
        "steps": [
            {
                "timestamp": "2026-01-14T12:00:00",
                "function": "enrichment",
                "description": "Test Enrichment",
                "params": {"file": "external.csv", "cols": ["A", "B"]},
            }
        ],
    }
    trace_path = tmp_path / "test_trace.json"
    trace_path.write_text(json.dumps(trace_data), encoding="utf-8")

    doc_generator = TraceDocumenter(str(trace_path))
    buffer = doc_generator.generate_report()

    assert buffer is not None
    assert buffer.getbuffer().nbytes > 0

    # Verify generated DOCX structure
    doc = Document(buffer)
    text = "\n".join([p.text for p in doc.paragraphs])
    assert "Data Transformation Report" in text
    assert "Session Information" in text
