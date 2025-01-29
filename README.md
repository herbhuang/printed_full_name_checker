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

### Basic Arguments
- `--output` or `-o`: Specify the output CSV file (default: results.csv)
- `--region` or `-r`: Specify the region to look for names in format "x1,y1,x2,y2" (optional)

### Roster Matching
You can provide a class roster CSV file to match extracted names against known student names:

```bash
python -m assignment_scanner.cli assignments.pdf -o results.csv --roster class_roster.csv
```

Roster-related arguments:
- `--roster` or `-R`: Path to CSV file containing class roster
- `--name-column` or `-n`: Name of the column in roster CSV containing student names (default: "name")
- `--threshold` or `-t`: Minimum similarity score (0-100) for fuzzy name matching (default: 80)

### Example Roster CSV Format
```csv
name,student_id,email
John Smith,12345,john@example.com
Jane Doe,67890,jane@example.com
```

The tool will use fuzzy string matching to match extracted names with the roster, accounting for OCR errors or slight variations in name format. The output CSV will include:
- Matched student name
- Raw OCR text
- Confidence score for the match
- Whether the name was successfully matched to the roster
- Additional columns from the roster (if matched)

## Features

Current:
- Extract student names from scanned assignments
- Fuzzy name matching against class roster
- Configurable matching threshold
- Save results to CSV file
- Configurable region for name detection
- Detailed match confidence scores

Planned:
- Response analysis
- Signature verification
- Multiple page support
- Custom OCR configurations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
