#!/usr/bin/env python3
"""
Results Analysis Script for Hyperparameter Tuning

This script analyzes CSV results from hyperparameter tuning and generates 
visualizations to help identify optimal parameter combinations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import sys


def load_results(csv_path: str) -> pd.DataFrame:
    """Load and validate the hyperparameter tuning results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        required_columns = [
            'timespan', 'make_width', 'take_width', 
            'kelp_profit', 'resin_profit', 'total_profit'
        ]
        
        # Check that all required columns exist
        for col in required_columns:
            if col not in df.columns:
                print(f"Error: CSV file is missing required column '{col}'")
                sys.exit(1)
                
        return df
    except Exception as e:
        print(f"Error loading CSV file: {str(e)}")
        sys.exit(1)


def create_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmaps for each parameter pair to visualize profit impact."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create pivot tables for each parameter pair
    param_pairs = [
        ('timespan', 'make_width'),
        ('timespan', 'take_width'),
        ('make_width', 'take_width')
    ]
    
    for x_param, y_param in param_pairs:
        # Create pivot table with average profit
        pivot = df.pivot_table(
            index=y_param, 
            columns=x_param, 
            values='total_profit', 
            aggfunc='mean'
        )
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='viridis')
        plt.title(f'Average Profit by {x_param} and {y_param}')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_dir / f'heatmap_{x_param}_{y_param}.png')
        plt.close()


def plot_parameter_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot distribution of profits for each parameter value."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    params = ['timespan', 'make_width', 'take_width']
    
    for param in params:
        plt.figure(figsize=(12, 6))
        
        # Group by parameter and calculate statistics
        grouped = df.groupby(param)['total_profit'].agg(['mean', 'std'])
        
        # Sort by parameter value
        grouped = grouped.sort_index()
        
        # Create bar chart with error bars
        ax = grouped['mean'].plot(kind='bar', yerr=grouped['std'], capsize=5)
        
        plt.title(f'Average Profit by {param}')
        plt.xlabel(param)
        plt.ylabel('Total Profit')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels on top of each bar
        for i, v in enumerate(grouped['mean']):
            ax.text(i, v + 50, f'{v:.0f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'distribution_{param}.png')
        plt.close()


def plot_profit_components(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot kelp vs resin profit contribution for best parameter combinations."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get top 10 parameter combinations
    top10 = df.sort_values('total_profit', ascending=False).head(10)
    
    # Create parameter combination labels
    top10['params'] = top10.apply(
        lambda row: f"T={row['timespan']}, M={row['make_width']}, T={row['take_width']}", 
        axis=1
    )
    
    # Create stacked bar chart
    plt.figure(figsize=(14, 8))
    top10.plot(
        x='params',
        y=['kelp_profit', 'resin_profit'],
        kind='bar',
        stacked=True,
        ax=plt.gca()
    )
    
    plt.title('Profit Components for Top 10 Parameter Combinations')
    plt.xlabel('Parameter Combination')
    plt.ylabel('Profit')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Product')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add total profit values on top of each bar
    for i, v in enumerate(top10['total_profit']):
        plt.text(i, v + 50, f'{v:.0f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'profit_components.png')
    plt.close()


def generate_summary_stats(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary statistics and save to text file."""
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find best parameter combination
    best_params = df.loc[df['total_profit'].idxmax()]
    
    # Calculate parameter importance (correlation with profit)
    correlations = df.corr()['total_profit'].drop('total_profit')
    correlations = correlations.drop(['kelp_profit', 'resin_profit'], errors='ignore')
    
    # Calculate average profit for each parameter value
    param_summaries = {}
    for param in ['timespan', 'make_width', 'take_width']:
        param_summaries[param] = df.groupby(param)['total_profit'].mean().sort_values(ascending=False)
    
    # Write summary to file
    with open(output_dir / 'summary_stats.txt', 'w') as f:
        f.write("Hyperparameter Tuning Summary\n")
        f.write("============================\n\n")
        
        f.write("Best Parameter Combination:\n")
        f.write(f"  Timespan: {best_params['timespan']}\n")
        f.write(f"  Make Width: {best_params['make_width']}\n")
        f.write(f"  Take Width: {best_params['take_width']}\n")
        f.write(f"  Total Profit: {best_params['total_profit']}\n")
        f.write(f"  Kelp Profit: {best_params['kelp_profit']}\n")
        f.write(f"  Resin Profit: {best_params['resin_profit']}\n\n")
        
        f.write("Parameter Importance (Correlation with Total Profit):\n")
        for param, corr in correlations.items():
            f.write(f"  {param}: {corr:.4f}\n")
        f.write("\n")
        
        f.write("Average Profit by Parameter Value:\n")
        for param, summary in param_summaries.items():
            f.write(f"  {param}:\n")
            for value, profit in summary.items():
                f.write(f"    {value}: {profit:.2f}\n")
            f.write("\n")
        
        # Top 10 parameter combinations
        f.write("Top 10 Parameter Combinations:\n")
        top10 = df.sort_values('total_profit', ascending=False).head(10)
        for i, row in top10.iterrows():
            f.write(f"  {i+1}. Timespan={row['timespan']}, Make Width={row['make_width']}, ")
            f.write(f"Take Width={row['take_width']}: {row['total_profit']}\n")


def main():
    """Main function to analyze hyperparameter tuning results."""
    parser = argparse.ArgumentParser(description='Analyze hyperparameter tuning results')
    parser.add_argument('csv_file', help='Path to the CSV file containing tuning results')
    parser.add_argument(
        '--output-dir', 
        default='analysis_results', 
        help='Directory to save analysis results'
    )
    
    args = parser.parse_args()
    
    # Load results
    df = load_results(args.csv_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"Analyzing results from {args.csv_file}...")
    
    # Generate visualizations and statistics
    create_heatmaps(df, output_dir / 'heatmaps')
    plot_parameter_distributions(df, output_dir / 'distributions')
    plot_profit_components(df, output_dir / 'profit_components')
    generate_summary_stats(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    print(f"Best parameter combination:")
    best_params = df.loc[df['total_profit'].idxmax()]
    print(f"  Timespan: {best_params['timespan']}")
    print(f"  Make Width: {best_params['make_width']}")
    print(f"  Take Width: {best_params['take_width']}")
    print(f"  Total Profit: {best_params['total_profit']}")


if __name__ == "__main__":
    main() 