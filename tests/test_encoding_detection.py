"""Encoding detection of the import engine (ingest.loaders), the one every file goes through."""

from ingest.loaders import detect_encoding


def test_detect_encoding_utf8() -> None:
    encoding = detect_encoding("This is a test file with UTF-8 encoding. éàç".encode())
    assert encoding.lower().replace("_", "-") in ["utf-8", "ascii"]


def test_detect_encoding_latin1() -> None:
    encoding = detect_encoding("Ceci est un fichier de test avec encodage ISO-8859-1. éàç".encode("iso-8859-1"))
    assert encoding.lower().replace("_", "-") in ["iso-8859-1", "windows-1252", "latin-1", "cp1250", "cp1252"]
