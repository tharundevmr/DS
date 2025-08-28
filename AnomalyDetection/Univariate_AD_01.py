import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.dates import DateFormatter
from matplotlib.ticker import MaxNLocator

# Load data from CSV file (replace 'your_file.csv' with the actual file path)
df = pd.read_csv('/Users/tharunmr/Documents/workspace/MADTool/MetricsAnomalyMonitoring.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)  # Ensure Timestamp is tz-aware (UTC)

# Parameters
window_size = 150  # Reduced window size for sensitivity to local spikes
upper_percentile = 75  # Tighter upper threshold
lower_percentile = 45  # Tighter lower threshold
z_score_threshold = 3  # Z-score check for extreme spikes
max_iterations = 1  # Number of iterations to stabilize thresholds
min_consecutive = 20  # Minimum consecutive anomalies to consider
metric_columns = [ 'metric_1', 'metric_2', 'metric_3', 'metric_4', 'metric_5', 'metric_6']


# Function to compute moving thresholds and detect anomalies, including Z-score check
def detect_anomalies(series, window_size, upper_percentile, lower_percentile, z_score_threshold):
    # Compute fixed thresholds on original data
    upper_threshold = series.rolling(window=window_size, center=True, min_periods=1).quantile(upper_percentile / 100)
    lower_threshold = series.rolling(window=window_size, center=True, min_periods=1).quantile(lower_percentile / 100)
    rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
    rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
    z_scores = np.abs((series - rolling_mean) / rolling_std.replace(0, np.nan))

    # Interpolate thresholds for consistency
    upper_threshold = upper_threshold.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    lower_threshold = lower_threshold.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
    z_scores = z_scores.fillna(0)

    # Detect anomalies based on fixed thresholds
    anomalies = (series > upper_threshold) | (series < lower_threshold) | (z_scores > z_score_threshold)

    return upper_threshold, lower_threshold, anomalies


# Function to filter single spikes and return consecutive anomalies
def filter_consecutive_anomalies(anomalies, min_consecutive):
    if anomalies.sum() == 0:
        return anomalies
    groups = (anomalies != anomalies.shift()).cumsum()
    group_sizes = anomalies.groupby(groups).transform('size')
    return anomalies & (group_sizes >= min_consecutive)


# Function to create segmented line with correct coloring based on thresholds
def plot_colored_line(ax, timestamps, y, upper_threshold, lower_threshold, anomalies):
    x_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().values
    points = np.array([x_seconds, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    consecutive_anomalies = filter_consecutive_anomalies(anomalies, min_consecutive)
    colors = []
    for i in range(len(x_seconds) - 1):
        start_y, end_y = y.iloc[i], y.iloc[i + 1]
        start_above = start_y > upper_threshold.iloc[i] or start_y < lower_threshold.iloc[i]
        end_above = end_y > upper_threshold.iloc[i + 1] or end_y < lower_threshold.iloc[i + 1]
        if (start_above or end_above) and (consecutive_anomalies.iloc[i] or consecutive_anomalies.iloc[i + 1]):
            colors.append('red')
        else:
            colors.append('blue')

    lc = LineCollection(segments, colors=colors, linewidth=2, label=None)
    ax.add_collection(lc)
    return lc, x_seconds


# Initialize list to collect anomaly data
all_anomaly_data = []

# Ensure unique customers are processed
unique_customers = df['CustomerName'].unique()
print(f"Processing {len(unique_customers)} customers: {unique_customers}")

# Group by CustomerName and analyze each customer
for customer in unique_customers:
    customer_df = df[df['CustomerName'] == customer].sort_values('Timestamp')

    # Skip empty or invalid customer data
    if customer_df.empty or len(customer_df) < window_size:
        print(f"Skipping customer {customer}: insufficient data (length={len(customer_df)})")
        continue

    # Set up plot for this customer
    fig, axes = plt.subplots(nrows=6, figsize=(12, 18), sharex=True)

    # Process each metric for this customer
    anomaly_data = pd.DataFrame({'Timestamp': customer_df['Timestamp'],
                                 'CustomerName': customer_df['CustomerName']})
    for idx, column in enumerate(metric_columns):
        if column not in customer_df.columns or customer_df[column].isna().all():
            print(f"Skipping metric {column} for customer {customer}: missing or invalid data")
            continue

        series = customer_df[column]

        # Detect anomalies using fixed thresholds
        upper_threshold, lower_threshold, anomalies = detect_anomalies(
            series, window_size, upper_percentile, lower_percentile, z_score_threshold)

        # Plot time series with correct coloring
        lc, x_seconds = plot_colored_line(axes[idx], customer_df['Timestamp'], series, upper_threshold, lower_threshold,
                                          anomalies)

        # Plot thresholds using relative seconds
        axes[idx].plot(x_seconds, upper_threshold, color='green', linestyle='--')
        axes[idx].plot(x_seconds, lower_threshold, color='orange', linestyle='--')

        # Plot only consecutive anomalies as red points using relative seconds
        consecutive_anomalies = filter_consecutive_anomalies(anomalies, min_consecutive)
        anomalies_x = x_seconds[consecutive_anomalies.reset_index(drop=True)]
        anomalies_y = series[consecutive_anomalies]
        axes[idx].scatter(anomalies_x, anomalies_y, color='red')

        axes[idx].set_title(f'{column} for {customer}')
        axes[idx].set_ylabel('Metric Value')
        axes[idx].grid(True)

        # Set x-axis with controlled tick count using MaxNLocator
        axes[idx].xaxis.set_major_locator(MaxNLocator(nbins=10, prune='both'))  # Limit to 10 ticks
        axes[idx].xaxis.set_major_formatter(DateFormatter('%H:%M'))
        axes[idx].tick_params(axis='x', rotation=45)  # Rotate for readability

    # Format x-axis for the bottom subplot
    axes[-1].set_xlabel('Timestamp')
    plt.gcf().autofmt_xdate()  # Rotate timestamps for readability
    plt.suptitle(
        f'Anomaly Detection for {customer} (Stable {upper_percentile}th/{lower_percentile}th Percentile Thresholds)',
        y=1.02)
    plt.tight_layout()
    plt.show()