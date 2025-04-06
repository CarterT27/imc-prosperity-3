# Hyperparameter Tuning Framework

This framework allows you to tune the hyperparameters for the trading strategy in `round0/tutorial.py` by running backtests with different parameter combinations and analyzing the results.

## Overview

The framework consists of three main scripts:

1. `hyperparameter_tuning.py` - Runs backtests with different parameter combinations and saves results to a CSV file
2. `analyze_results.py` - Analyzes the CSV results and generates visualizations and statistics
3. `apply_best_params.py` - Applies the best parameters found back to the original trading strategy file

## Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install prosperity3bt pandas numpy matplotlib seaborn
```

## Hyperparameter Tuning

The tuning script will modify the following parameters in the trading strategy:
- `timespan` - How many historical price points to use
- `make_width` - Spread width for market making
- `take_width` - How aggressive to be when taking orders

To run the hyperparameter tuning:

```bash
python hyperparameter_tuning.py
```

This will:
1. Create a temporary modified version of `round0/tutorial.py` for each parameter combination
2. Run a backtest for each combination using the `prosperity3bt` tool
3. Extract profit information from the backtest output
4. Save the results to a CSV file in the `results` directory

The script will test the following parameter ranges:
- `timespan`: [5, 10, 15, 20]
- `make_width`: [2.0, 3.0, 3.5, 4.0, 5.0]
- `take_width`: [0.5, 1.0, 1.5, 2.0]

This gives a total of 80 parameter combinations to test.

## Analyzing Results

Once the tuning is complete, you can analyze the results using the analysis script:

```bash
python analyze_results.py results/hyperparameter_results_<timestamp>.csv
```

This will generate:
1. Heatmaps showing the impact of each parameter pair on profit
2. Bar charts showing profit distribution for each parameter value
3. A stacked bar chart showing profit components for the top 10 parameter combinations
4. A summary text file with detailed statistics

All analysis results will be saved to the `analysis_results` directory by default. You can change this with the `--output-dir` option.

## Applying Best Parameters

After analyzing the results, you can apply the best parameters found to your original trading strategy file:

```bash
# Apply best parameters from a CSV file
python apply_best_params.py --csv results/hyperparameter_results_<timestamp>.csv

# Or apply specific parameter values
python apply_best_params.py --timespan 15 --make-width 3.5 --take-width 1.0
```

The script will:
1. Create a backup of the original file (unless `--no-backup` is specified)
2. Apply the best parameters to the trading strategy file
3. Print a confirmation message with the applied parameter values

By default, it modifies `round0/tutorial.py`, but you can specify a different file with the `--file` option.

## Understanding the Results

The analysis will help you identify:
- Which parameters have the largest impact on profit
- The optimal value for each parameter
- How different parameters interact with each other
- The best overall parameter combination

The heatmaps are particularly useful for visualizing how parameters interact with each other, while the bar charts provide a clearer picture of each parameter's individual impact.

## Example Usage

Complete workflow example:

```bash
# Run hyperparameter tuning
python hyperparameter_tuning.py

# Analyze the results (replace with your CSV filename)
python analyze_results.py results/hyperparameter_results_2025-04-06_14-00-00.csv

# View the summary statistics
cat analysis_results/summary_stats.txt

# Apply the best parameters
python apply_best_params.py --csv results/hyperparameter_results_2025-04-06_14-00-00.csv

# Run a final backtest with the optimized parameters
prosperity3bt round0/tutorial.py 0 --match-trades worse
```

## Modifying the Parameter Ranges

If you want to test different parameter ranges, modify the following variables in `hyperparameter_tuning.py`:

```python
TIMESPAN_VALUES = [5, 10, 15, 20]
MAKE_WIDTH_VALUES = [2.0, 3.0, 3.5, 4.0, 5.0]
TAKE_WIDTH_VALUES = [0.5, 1.0, 1.5, 2.0]
```

For instance, if you want to test narrower ranges around promising values, you could change them to:

```python
TIMESPAN_VALUES = [9, 10, 11, 12]
MAKE_WIDTH_VALUES = [3.3, 3.4, 3.5, 3.6, 3.7]
TAKE_WIDTH_VALUES = [0.8, 0.9, 1.0, 1.1, 1.2]
```

Then run the tuning process again to zoom in on the optimal values.

## Tips for Efficient Tuning

- Start with a wide range of parameter values to get a general idea
- Run subsequent tuning with narrower ranges around promising values
- Use the analysis visualizations to guide your tuning process
- Consider focusing on the parameters that show the strongest correlation with profit
- Pay attention to parameter interactions in the heatmaps

## Extending the Framework

This framework can be extended in several ways:
- Add more hyperparameters to tune
- Implement more sophisticated tuning algorithms (e.g., Bayesian optimization)
- Add cross-validation by running on different days/rounds
- Incorporate risk metrics in addition to profit (e.g., Sharpe ratio, maximum drawdown)
- Add visualization of performance over time for the best parameter combinations 