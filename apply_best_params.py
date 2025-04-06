#!/usr/bin/env python3
"""
Apply Best Parameters Script

This script applies the best parameters found during hyperparameter tuning 
to the original trading strategy file.
"""

import re
import argparse
import pandas as pd
import shutil
import sys
from pathlib import Path
from datetime import datetime


def backup_original_file(file_path: str) -> str:
    """Create a backup of the original file before modifying it."""
    original_path = Path(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = original_path.with_name(f"{original_path.stem}_backup_{timestamp}{original_path.suffix}")
    
    try:
        shutil.copy2(original_path, backup_path)
        print(f"Created backup at {backup_path}")
        return str(backup_path)
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        sys.exit(1)


def apply_parameters(file_path: str, timespan: int, make_width: float, take_width: float) -> None:
    """Apply the best parameters to the trading strategy file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Replace the parameters
        content = re.sub(
            r'self\.timespan = \d+',
            f'self.timespan = {timespan}',
            content
        )
        content = re.sub(
            r'self\.make_width = \d+\.\d+',
            f'self.make_width = {make_width}',
            content
        )
        content = re.sub(
            r'self\.take_width = \d+\.\d+',
            f'self.take_width = {take_width}',
            content
        )
        
        with open(file_path, 'w') as f:
            f.write(content)
            
        print(f"Successfully applied parameters to {file_path}:")
        print(f"  timespan = {timespan}")
        print(f"  make_width = {make_width}")
        print(f"  take_width = {take_width}")
            
    except Exception as e:
        print(f"Error applying parameters: {str(e)}")
        sys.exit(1)


def get_best_parameters_from_csv(csv_path: str) -> tuple[int, float, float]:
    """Extract the best parameter combination from the CSV results file."""
    try:
        df = pd.read_csv(csv_path)
        best_row = df.loc[df['total_profit'].idxmax()]
        
        timespan = int(best_row['timespan'])
        make_width = float(best_row['make_width'])
        take_width = float(best_row['take_width'])
        
        print(f"Found best parameters in {csv_path}:")
        print(f"  timespan = {timespan}")
        print(f"  make_width = {make_width}")
        print(f"  take_width = {take_width}")
        print(f"  total_profit = {best_row['total_profit']}")
        
        return timespan, make_width, take_width
        
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
        sys.exit(1)


def main():
    """Main function to apply best parameters from tuning results."""
    parser = argparse.ArgumentParser(description='Apply best parameters from tuning results')
    parser.add_argument('--csv', 
                       help='Path to the CSV file containing tuning results')
    parser.add_argument('--timespan', type=int,
                       help='Timespan parameter value')
    parser.add_argument('--make-width', type=float,
                       help='Make width parameter value')
    parser.add_argument('--take-width', type=float,
                       help='Take width parameter value')
    parser.add_argument('--file', default='round0/tutorial.py',
                       help='Path to the trading strategy file to modify')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating a backup of the original file')
    
    args = parser.parse_args()
    
    # Validate input
    if args.csv and (args.timespan or args.make_width or args.take_width):
        print("Error: Provide either a CSV file OR specific parameter values, not both.")
        sys.exit(1)
    
    if not args.csv and not (args.timespan and args.make_width and args.take_width):
        print("Error: Either provide a CSV file with tuning results or specify all three parameters.")
        sys.exit(1)
    
    # Determine parameters to apply
    if args.csv:
        timespan, make_width, take_width = get_best_parameters_from_csv(args.csv)
    else:
        timespan = args.timespan
        make_width = args.make_width
        take_width = args.take_width
    
    # Create backup if requested
    if not args.no_backup:
        backup_original_file(args.file)
    
    # Apply parameters
    apply_parameters(args.file, timespan, make_width, take_width)
    print("Parameters applied successfully!")


if __name__ == "__main__":
    main() 