from pathlib import Path

from manage.file_handling import detect_encoding


def test_detect_encoding_utf8(tmp_path: Path):
    """Test encoding detection for UTF-8 encoded files."""
    test_file = tmp_path / "test_utf8.txt"
    test_file.write_text("This is a test file with UTF-8 encoding. éàç", encoding="utf-8")
    
    encoding = detect_encoding(str(test_file))
    assert encoding is not None
    normalized = encoding.lower().replace("_", "-")
    assert normalized in ["utf-8", "ascii"]


def test_detect_encoding_latin1(tmp_path: Path):
    """Test encoding detection for ISO-8859-1 encoded files."""
    test_file = tmp_path / "test_iso.txt"
    test_file.write_text("Ceci est un fichier de test avec encodage ISO-8859-1. éàç", encoding="iso-8859-1")
    
    encoding = detect_encoding(str(test_file))
    assert encoding is not None
    normalized = encoding.lower().replace("_", "-")
    assert normalized in ["iso-8859-1", "windows-1252", "latin-1", "cp1250", "cp1252"]
