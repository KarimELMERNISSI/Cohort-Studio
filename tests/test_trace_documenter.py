
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from manage.trace_documenter import TraceDocumenter
from docx import Document
import json

def test_trace_documenter():
    trace_path = "calculated_data_v5_trace.json"
    
    if not os.path.exists(trace_path):
        # Create a dummy trace if not found
        trace_data = {
            "session_id": "test_session",
            "source_dataset": "test_source.csv",
            "steps": [
                {
                    "timestamp": "2026-01-14T12:00:00",
                    "function": "enrichment",
                    "description": "Test Enrichment",
                    "params": {"file": "external.csv", "cols": ["A", "B"]}
                }
            ]
        }
        with open("test_trace_temp.json", "w") as f:
            json.dump(trace_data, f)
        trace_path = "test_trace_temp.json"
    
    print(f"Testing with trace: {trace_path}")
    
    try:
        doc_generator = TraceDocumenter(trace_path)
        buffer = doc_generator.generate_report()
        
        print(f"Generated buffer size: {buffer.getbuffer().nbytes} bytes")
        
        # Verify it's a valid docx
        doc = Document(buffer)
        print("Successfully opened generated DOCX.")
        
        text = "\n".join([p.text for p in doc.paragraphs])
        # print("Document Content Preview:", text[:200])
        
        assert "Data Transformation Report" in text
        assert "Session Information" in text
        
        print("Verification Passed!")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_trace_documenter()
