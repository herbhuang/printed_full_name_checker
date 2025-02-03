# Assignment Scanner

A Python tool for automating the process of scanning and processing student assignments. This tool uses OCR to extract student names from scanned assignments and can be extended to support more features in the future.

## Prerequisites

- Python 3.8 or higher
- Tesseract OCR engine installed on your system
  - On Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - On macOS: `brew install tesseract`
  - On Windows: Download and install from [GitHub Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki)
- `uv` package manager installed (`pip install uv`)

## Installation

There are two ways to install and run this tool:

### Method 1: Development Installation
1. Clone this repository
2. Install dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```
3. Run using the module:
   ```bash
   python -m assignment_scanner.cli [arguments]
   ```

### Method 2: Direct Script Execution
You can run the script directly using `uv run` without installing:
1. Clone this repository
2. Run using `uv`:
   ```bash
   uv run scan_assignments.py [arguments]
   ```

This method automatically handles dependencies and virtual environment creation.

## Usage

The tool can be used in two ways:

### Using installed package:
```bash
python -m assignment_scanner.cli path/to/your/assignments.pdf --output results.csv
```

### Using uv run (recommended):
```bash
uv run scan_assignments.py path/to/your/assignments.pdf --output results.csv
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
- `--threshold` or `-t`: Minimum similarity score (0-100) for fuzzy name matching (default: 80)

#### Single-Column Name Format
For rosters with a single column containing full names:
- `--name-column` or `-n`: Name of the column containing full names (default: "name")

Example roster format:
```csv
name,student_id,email
John Smith,12345,john@example.com
Jane Doe,67890,jane@example.com
```

Example command:
```bash
python -m assignment_scanner.cli assignments.pdf --roster roster.csv --name-column "full_name"
```

#### Split-Name Format
For rosters with separate columns for first and last names:
- `--first-name-column` or `-f`: Name of the column containing first names
- `--last-name-column` or `-l`: Name of the column containing last names

Example roster format:
```csv
first_name,last_name,student_id,email
John,Smith,12345,john@example.com
Jane,Doe,67890,jane@example.com
```

Example command:
```bash
python -m assignment_scanner.cli assignments.pdf --roster roster.csv --first-name-column "first_name" --last-name-column "last_name"
```

Note: You must provide both first and last name columns when using split names. You cannot mix single-column and split-name formats.

### Output Format
The tool will use fuzzy string matching to match extracted names with the roster, accounting for OCR errors or slight variations in name format. The output CSV will include:
- Matched student name
- Raw OCR text
- Confidence score for the match
- Whether the name was successfully matched to the roster
- First and last names (if using split-name format)
- Additional columns from the roster (if matched)

## Features

Current:
- Extract student names from scanned assignments
- Support for both single-column and split-name formats in roster
- Fuzzy name matching against class roster
- Separate matching for first/last names when using split format
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
