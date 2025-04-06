#!/usr/bin/env python3
"""
Test Hyperparameter Tuning Script

This is a smaller version of the hyperparameter tuning script with fewer parameter
combinations for testing purposes.
"""

import os
import re
import csv
import subprocess
import tempfile
import shutil
from datetime import datetime
from itertools import product
import time
import sys
from pathlib import Path

# Define a smaller set of parameter values for testing
TIMESPAN_VALUES = [10, 15]
MAKE_WIDTH_VALUES = [3.0, 3.5]
TAKE_WIDTH_VALUES = [0.5, 1.0]

def modify_trading_params(original_file, temp_dir, timespan, make_width, take_width):
    """Creates a modified version of the trading file in the temp directory."""
    # Get the original file name
    original_path = Path(original_file)
    file_name = original_path.name
    
    # Copy datamodel.py to temp directory
    datamodel_path = original_path.parent / "datamodel.py"
    if datamodel_path.exists():
        shutil.copy2(datamodel_path, temp_dir / "datamodel.py")
    
    # Create the modified file
    temp_file_path = temp_dir / file_name
    
    with open(original_file, 'r') as f:
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
        
        with open(temp_file_path, 'w') as tmp_f:
            tmp_f.write(content)
            
    return temp_file_path

# Function to run the backtest and extract profit information
def run_backtest(temp_file_path):
    """Runs backtest with given parameters and returns profit metrics."""
    # Run backtest and capture output
    print(f"Running backtest with {temp_file_path}...")
    result = subprocess.run(
        [
            "prosperity3bt", 
            str(temp_file_path), 
            "0", 
            "--match-trades", 
            "worse", 
            "--no-progress", 
            "--no-out"
        ],
        capture_output=True,
        text=True
    )
    
    output = result.stdout
    
    # Print output for debugging
    print("Backtest output:")
    print(output)
    
    # Extract profit information using regex
    kelp_profit = 0
    resin_profit = 0
    total_profit = 0
    
    # Updated regex patterns to match the exact format
    kelp_match = re.search(r'KELP: ([\d,]+)', output)
    resin_match = re.search(r'RAINFOREST_RESIN: ([\d,]+)', output)
    total_match = re.search(r'Total profit: ([\d,]+)', output)
    
    if kelp_match:
        kelp_profit = int(kelp_match.group(1).replace(',', ''))
    if resin_match:
        resin_profit = int(resin_match.group(1).replace(',', ''))
    if total_match:
        total_profit = int(total_match.group(1).replace(',', ''))
    
    return {
        'kelp_profit': kelp_profit,
        'resin_profit': resin_profit,
        'total_profit': total_profit
    }

def main():
    """Main function to run hyperparameter tuning."""
    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    # CSV file for results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = results_dir / f"test_hyperparameter_results_{timestamp}.csv"
    
    # Original trader file
    original_file = "round0/tutorial.py"
    
    # Set up CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'timespan', 
            'make_width', 
            'take_width', 
            'kelp_profit', 
            'resin_profit', 
            'total_profit'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Generate parameter combinations
        param_combinations = list(product(
            TIMESPAN_VALUES, 
            MAKE_WIDTH_VALUES, 
            TAKE_WIDTH_VALUES
        ))
        
        total_combinations = len(param_combinations)
        print(f"Starting test hyperparameter tuning with {total_combinations} combinations")
        
        # Create a temporary directory for all files
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            # Run backtests for each parameter combination
            for i, (timespan, make_width, take_width) in enumerate(param_combinations):
                print(f"Progress: {i+1}/{total_combinations} combinations")
                
                # Create temporary file with modified parameters
                temp_file = modify_trading_params(
                    original_file, 
                    temp_dir,
                    timespan, 
                    make_width, 
                    take_width
                )
                
                try:
                    print(f"Testing parameters: timespan={timespan}, make_width={make_width}, take_width={take_width}")
                    
                    # Run backtest
                    profits = run_backtest(temp_file)
                    
                    print(f"Results: kelp={profits['kelp_profit']}, resin={profits['resin_profit']}, total={profits['total_profit']}")
                    
                    # Write results to CSV
                    writer.writerow({
                        'timespan': timespan,
                        'make_width': make_width,
                        'take_width': take_width,
                        'kelp_profit': profits['kelp_profit'],
                        'resin_profit': profits['resin_profit'],
                        'total_profit': profits['total_profit']
                    })
                    
                    # Flush to save progress incrementally
                    csvfile.flush()
                    
                except Exception as e:
                    print(f"Error with parameters (timespan={timespan}, "
                        f"make_width={make_width}, take_width={take_width}): {str(e)}")
            
    print(f"\nTest hyperparameter tuning completed. Results saved to {csv_path}")
    
    # Find and print the best parameter combination
    best_params = None
    max_profit = -float('inf')
    
    with open(csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_profit = int(row['total_profit'])
            if total_profit > max_profit:
                max_profit = total_profit
                best_params = {
                    'timespan': int(row['timespan']),
                    'make_width': float(row['make_width']),
                    'take_width': float(row['take_width']),
                    'total_profit': total_profit
                }
    
    print("\nBest parameters:")
    print(f"  Timespan: {best_params['timespan']}")
    print(f"  Make Width: {best_params['make_width']}")
    print(f"  Take Width: {best_params['take_width']}")
    print(f"  Total Profit: {best_params['total_profit']}")

if __name__ == "__main__":
    main() 