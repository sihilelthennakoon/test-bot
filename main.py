#!/usr/bin/env python
"""
Main entry point for the RAG Bot application.

This script provides the primary interface for running the RAG chatbot service,
ingesting documents, and managing the vector store.

Usage:
    python main.py serve                  # Start the API server
    python main.py ingest --source FILE   # Ingest a text file
    python main.py test                   # Run tests
    python main.py eval                   # Run evaluation on test prompts
"""

import sys
from pathlib import Path

# Add src to path so we can import ragbot
PROJECT_ROOT = Path(__file__).parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ragbot.cli import main as cli_main


if __name__ == "__main__":
    cli_main()
