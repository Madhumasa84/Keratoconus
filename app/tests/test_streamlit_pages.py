"""
Streamlit Page compilation and smoke test.
Validates that all Streamlit pages and modules parse and load cleanly.
"""
import py_compile
from pathlib import Path
import pytest

PAGES_DIR = Path(__file__).parent.parent / "pages"
MAIN_APP = Path(__file__).parent.parent / "streamlit_app.py"


def test_main_app_syntax():
    """Verify main streamlit_app.py compiles without syntax errors."""
    assert MAIN_APP.exists()
    py_compile.compile(str(MAIN_APP), doraise=True)


def test_all_pages_syntax():
    """Verify all 8 Streamlit page scripts compile without syntax errors."""
    pages = list(PAGES_DIR.glob("*.py"))
    assert len(pages) >= 8
    for page in pages:
        py_compile.compile(str(page), doraise=True)
