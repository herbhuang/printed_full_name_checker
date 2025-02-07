#!/usr/bin/env python3

"""
Main entry point for the Assignment Scanner application.
Provides both CLI and web interface options.
"""

import sys
import argparse
from src.assignment_scanner.webui import main as webui_main
from src.assignment_scanner.cli import main as cli_main

def parse_args():
    parser = argparse.ArgumentParser(description='Assignment Scanner - Process and analyze student assignments')
    parser.add_argument('--mode', choices=['web', 'cli'], default='web',
                       help='Run mode: web interface (default) or command line interface')
    parser.add_argument('--input', type=str, help='Input PDF file or directory (required for CLI mode)')
    parser.add_argument('--output', type=str, help='Output directory (required for CLI mode)')
    parser.add_argument('--config', type=str, help='Configuration file path (optional)')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        if args.mode == 'web':
            print("Starting web interface...")
            webui_main()
        else:  # CLI mode
            if not args.input or not args.output:
                print("Error: --input and --output are required for CLI mode")
                sys.exit(1)
            print(f"Processing in CLI mode: {args.input} -> {args.output}")
            cli_main(args.input, args.output, args.config)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 