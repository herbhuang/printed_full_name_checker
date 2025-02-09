# Printed Full Name Checker

A specialized tool for scanning documents and checking printed full names using advanced OCR and vision-language models. Perfect for verifying names on official documents, forms, and assignments. Initially developed for my class assignments.

## Features

- 📄 Document Processing
  - Multi-page document support
  - Stacked preview for easy name detection
  - Automatic page caching for better performance

- 🎯 Name Region Selection
  - Interactive drawing interface for name fields
  - Multiple name region selection per page
  - Name region preview and editing capabilities
  - Flexible region management

- 🖼️ Image Processing
  - Advanced preprocessing options
  - Region stitching modes:
    - Individual processing
    - Vertical stacking
    - Horizontal stacking
    - Grid layout

- 🤖 OCR Capabilities
  - Multiple OCR Models:
    - Local Models:
      - [Qwen-2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) ⬅️ acceptable performance
      - [Florence-2](https://huggingface.co/microsoft/Florence-2-large-ft)
    - 🚧 API Models:
      - [GOT-OCR-2.0-hf](https://huggingface.co/stepfun-ai/GOT-OCR-2.0-hf)
      - [Llama-3.2-11B-Vision-Instruct](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
      - [Qwen2.5-VL-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)
  - Customizable prompts
  - Region-based text detection
  - Batch processing support

## Requirements

### Hardware Requirements
- **CUDA-capable GPU with at least 16GB VRAM** (for local models)
- At least 16GB RAM
- SSD storage recommended for better performance

### Software Requirements
- Python 3.11
- CUDA Toolkit 11.8 or higher
- PDF processing tools (poppler-utils)

### Package Management

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd path-to-project
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

3. Install dependencies with uv:
```bash
uv pip install -e .
```


## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/printed_full_name_checker.git
cd printed_full_name_checker
```

2. Install system dependencies:
   - Ubuntu/Debian:
     ```bash
     sudo apt-get update
     sudo apt-get install -y poppler-utils
     ```
   - Windows:
     - Download and install [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/)
     - Add the poppler `bin` directory to your system PATH


## Usage

1. Start the web interface:
```bash
uv run python streamlit run src/assignment_scanner/webui.py
```

2. Follow the step-by-step process:
   1. Upload PDF file
   2. Draw regions of interest
   3. Configure image processing
   4. Run OCR with selected model

## Project Structure

```
src/
├── assignment_scanner/
│   ├── webui.py           # Main web interface
│   ├── scanner.py         # Core scanning logic
│   ├── state_manager.py   # State management
│   ├── region_processor.py # Region processing
│   ├── image_processor.py # Image preprocessing
│   └── ui_manager.py      # UI components
├── ocr/
│   ├── processor.py       # OCR orchestration
│   ├── base.py           # Base classes
│   ├── huggingface_api.py # API model implementation
│   ├── qwen.py           # Qwen model implementation
│   └── florence.py       # Florence model implementation
```

## Contributing

Working on this section.

## License

This project is licensed under the MIT License.
