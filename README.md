# Assignment Scanner

A Python tool for automating the process of scanning and processing student assignments. This tool uses OCR to extract student names from scanned assignments and can be extended to support more features in the future.

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system
  - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - On macOS: `brew install tesseract`
  - On Windows: Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)

## Installation

1. Clone this repository
2. Install dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```

## Usage

The tool can be used from the command line:

```bash
python -m assignment_scanner.cli path/to/your/assignments.pdf --output results.csv
```

Optional arguments:
- `--output` or `-o`: Specify the output CSV file (default: results.csv)
- `--region` or `-r`: Specify the region to look for names in format "x1,y1,x2,y2" (optional)

Example with region:
```bash
python -m assignment_scanner.cli assignments.pdf -o results.csv -r "100,200,500,300"
```

## Features

Current:
- Extract student names from scanned assignments
- Save results to CSV file
- Configurable region for name detection

Planned:
- Response analysis
- Signature verification
- Multiple page support
- Custom OCR configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
