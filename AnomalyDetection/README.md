# Anomaly Detection Visualization Project

This project provides a Python-based tool for visualizing and detecting anomalies in time series data, specifically designed for multiple customers and metrics. The tool generates plots and exports anomaly data to a CSV file, with features to ignore single spikes and ensure thresholds are not influenced by anomalies.

## Overview

The script processes CSV data containing timestamps, customer names, and multiple metric columns, identifying consecutive anomalies using rolling percentile thresholds (75th/45th) and Z-score checks. It produces per-customer plots with customizable parameters and saves the results.


## Usage

### Input Data Format
The script expects a CSV file (`your_data.csv` by default) with the following columns:
- `CustomerName`: Name of the customer (string).
- `Timestamp`: Date and time in ISO format with UTC (e.g., `2024-05-09T12:58:00.000Z`).
- `metric_1` to `metric_6`: Numerical values for six metrics.

Example:
```
CustomerName,Timestamp,metric_1,metric_2,metric_3,metric_4,metric_5,metric_6
CustomerA,2024-05-09T12:58:00.000Z,100.5,200.3,150.7,175.2,225.1,300.4
CustomerB,2024-05-09T13:00:00.000Z,120.0,220.0,160.0,180.0,230.0,310.0
```

### Running the Script
1. Update the file path in the script if necessary (e.g., replace `'your_data.csv'` with your file name).
2. Run the script:
   ```bash
   python fixed_thresholds_anomaly_detection.py
   ```

### Customization
Modify the following parameters in the script to suit your needs:
- `window_size = 50`: Size of the rolling window for threshold calculation.
- `upper_percentile = 75`: Upper percentile threshold (e.g., 90th percentile).
- `lower_percentile = 45`: Lower percentile threshold (e.g., 10th percentile).
- `z_score_threshold = 3`: Z-score threshold for extreme values.
- `min_consecutive = 20`: Minimum number of consecutive anomalies to consider.
- `nbins` in `MaxNLocator(nbins=10)`: Number of ticks on the x-axis (adjust for readability).

### Output
- **Plots**: For each customer, a PNG file (`anomaly_plots_<CustomerName>.png`) is generated with six subplots, one per metric. Each plot shows:
  - Time series: Blue for normal points/single spikes, red for consecutive anomalies.
  - Green dashed line: Upper threshold (90th percentile).
  - Orange dashed line: Lower threshold (10th percentile).
  - Red points: Consecutive anomalies.
- **CSV**: An `anomalies_all_customers.csv` file containing anomaly flags for each metric per customer.

## Features
- **Anomaly Detection**: Uses fixed rolling 90th/10th percentile thresholds and Z-score checks, unaffected by anomalies.
- **Spike Filtering**: Ignores single spikes, marking only consecutive anomalies (default: 2 or more).
- **Visualization**: Hour-level x-axis with controlled ticks (max 10 per subplot).
- **No Legends or Shading**: Clean plots without legends or shaded regions.

## Troubleshooting
- **Warning: MAXTICKS Exceeded**: The script uses `MaxNLocator` to limit ticks to 10, resolving the `Locator attempting to generate 567073 ticks` warning. If issues persist, increase `nbins` in `MaxNLocator`.
- **Threshold Issues**: If thresholds still appear influenced by anomalies, verify the data range and adjust `window_size`.
- **Missing Data**: Ensure all required columns are present in the CSV; the script skips invalid or empty data.

## Contributing
Feel free to fork this repository, submit issues, or create pull requests for enhancements.
