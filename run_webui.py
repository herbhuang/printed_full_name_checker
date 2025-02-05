"""
Script to run the Assignment Scanner Web UI.
"""

# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#     "streamlit==1.31.0",
#     "streamlit-drawable-canvas==0.9.3",
#     "PyPDF2==3.0.1",
#     "pytesseract==0.3.10",
#     "Pillow==9.5.0",
#     "numpy==1.24.3",
#     "pandas==2.0.3",
#     "thefuzz[speedup]==0.19.0",
#     "pdf2image==1.16.3",
#     "opencv-python-headless==4.8.1.78",
#     "scikit-image==0.21.0"
# ]
# ///

import streamlit.web.cli as stcli
import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    webui_path = Path("src/assignment_scanner/webui.py")
    sys.argv = ["streamlit", "run", str(webui_path)]
    sys.exit(stcli.main()) 