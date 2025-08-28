# Multivariate Anomaly Detection 

## Project Overview

This project implements a multivariate anomaly detection system using an LSTM-based autoencoder and Isolation Forest. The system analyzes time-series data for individual customers, detecting anomalies where at least two out of three specified metrics that exhibit anomalous behavior that persists for at least 2 minutes. Anomalies are visualized as red scatter points on line graphs, highlighting specific metrics that are anomalous during persistent periods.


## Features

- **Multivariate Analysis**: Detects anomalies when at least two of the three metrics are anomalous, based on reconstruction errors.
- **Time-Series Windowing**: Uses sliding windows (sequence length = 2) to capture temporal dependencies.
- **LSTM Autoencoder**: Models sequential patterns with a 3-layer LSTM (64 hidden units, learning rate = 0.0005, 100 epochs).
- **Isolation Forest**: Detects anomalies in reconstruction errors with a contamination rate of 0.02 for high sensitivity.
- **Persistence Filtering**: Identifies anomalies lasting at least 2 minutes using timestamp differences.
- **Visualization**: Plots each metric's time series with blue lines and red scatter points for anomalous data points in persistent periods.

## Requirements

- Python 3.8+
- Libraries:
  - `pandas`
  - `numpy`
  - `sklearn` (scikit-learn)
  - `torch` (PyTorch)
  - `matplotlib`
- Install dependencies:

  ```bash
  pip install pandas numpy scikit-learn torch matplotlib
  ```

## Usage

1. **Prepare Input Data**:

   - Place the CSV file (`MetricsAnomalyMonitoring.csv`) in the appropriate directory (e.g., `/Users/tharunmr/Documents/workspace/`).
   - Update the file path in the script if necessary.
   - The CSV must include `CustomerName`, `Timestamp`, `metric_4`, `metric_5`, and `metric_6`. Missing values in `Timestamp` are dropped, and invalid timestamps are handled as `NaT`.

2. **Run the Script**:

   - Execute the main script (e.g., `multivariate_anomaly_detection.py`):

     ```bash
     python multivariate_anomaly_detection.py
     ```
   - The script processes each customer’s data, trains an LSTM autoencoder, detects anomalies, and generates plots.

3. **Output**:

   - Console output: Summary of persistent anomalies for each customer (e.g., number of anomalous data points where 2+ metrics are anomalous).
   - Visualizations: Line graphs for each metric with red scatter points indicating anomalies in persistent periods (≥2 minutes).

## Code Structure

- **Main Script**: `multivariate_anomaly_detection.py`
  - Loads and preprocesses the CSV data, dropping rows with invalid `Timestamp`.
  - Groups data by `CustomerName` for customer-specific analysis.
  - Processes three metrics (`metric_4`, `metric_5`, `metric_6`):
    - Scales data using `MinMaxScaler`.
    - Creates sequences (window size = 2) for time-series analysis.
    - Trains an LSTM autoencoder (`hidden_dim=64`, `num_layers=3`, `epochs=100`, `lr=0.0005`).
    - Uses `IsolationForest` (contamination = 0.02) to detect anomalies in sequence-level reconstruction errors.
    - Computes feature-specific anomalies using a threshold of `mean + 2 * std`.
    - Marks data points as anomalies if at least 2 metrics are anomalous and persist for ≥2 minutes.
    - Plots time series with red scatter points for anomalous metrics in persistent periods.
- **Key Functions**:
  - `create_sequences`: Generates sliding window sequences for time-series input.
  - `find_persistent_anomalies`: Filters anomalies based on a minimum duration (2 minutes).
  - `LSTMAutoencoder`: Defines the LSTM-based autoencoder model for multivariate data.

## Configuration

- **Tunable Parameters**:
  - `seq_length=2`: Number of timesteps in each sequence. Increase for more temporal context (requires more data).
  - `min_duration_minutes=2`: Minimum duration for persistent anomalies. Adjust to capture shorter/longer spikes.
  - `contamination=0.02`: Isolation Forest’s expected anomaly proportion. Decrease (e.g., 0.01) for higher sensitivity.
  - `hidden_dim=64`, `num_layers=3`: LSTM architecture parameters. Adjust for model complexity.
  - `epochs=100`, `lr=0.0005`: Training parameters. Increase epochs or decrease learning rate for better convergence.
  - `feature_thresholds multiplier=2`: Standard deviation multiplier for feature-specific anomaly detection. Lower (e.g., 1.5) for higher sensitivity.
  - `anomaly_threshold=2`: Number of anomalous metrics required (out of 3). Lower to 1 for higher sensitivity.
- Modify these in the script to balance sensitivity and specificity.

## Notes

- **Data Requirements**:
  - Ensure at least 2 data points per customer for sequence creation.
  - `Timestamp` must be parseable by `pd.to_datetime`. Invalid timestamps are handled by dropping rows.
  - Handle missing values in numerical columns (`metric_4`, `metric_5`, `metric_6`) before running if needed (e.g., add `group = group.dropna(subset=num_cols)`).
- **Performance**:
  - LSTM training may be slow for large datasets. Consider reducing `epochs` or using a GPU.
  - Subsample data for testing if processing time is an issue.
- **Sensitivity Tuning**:
  - To increase sensitivity, reduce `contamination` (e.g., to 0.01), lower the feature threshold multiplier (e.g., to 1.5), or reduce `anomaly_threshold` to 1.
  - To reduce false positives, increase `min_duration_minutes` or `contamination`.
- **Visualization**:
  - Red scatter points highlight specific metrics that are anomalous during persistent periods (≥2 minutes).
  - Consider switching to shaded regions (using `axvspan`) for better visualization of anomaly duration.

## Example Output

For each customer, the script generates:

- A console summary, e.g.:

  ```
  Processing customer: CustomerA
  Persistent anomalies (2+ features, time delta >= 2 min) detected for CustomerA: 25 out of 1000 data points
  ```
- A multi-panel plot with one subplot per metric, showing the time series (blue line) and red scatter points for anomalous data points.

## Future Improvements

- Switch to shaded regions (`axvspan`) for anomaly visualization to highlight duration.
- Add ROC-AUC evaluation if labeled anomaly data is available.
- Implement early stopping for LSTM training to optimize epochs.
- Support irregular timestamps by interpolating data.
- Add feature weighting for domain-specific metrics (e.g., prioritize `metric_6` if it represents critical failures).
