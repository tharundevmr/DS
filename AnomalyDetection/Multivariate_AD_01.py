import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Load data from CSV file (replace 'your_file.csv' with the actual file path)
data = pd.read_csv('/Users/tharunmr/Documents/workspace/MetricsAnomalyMonitoring.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')  # Convert to datetime, handle invalid as NaT

# Drop rows with NaT in Timestamp
data = data.dropna(subset=['Timestamp'])

num_cols = ['metric_4','metric_5','metric_6']


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences), np.arange(seq_length - 1, len(data))


# Function to detect persistent anomalies based on time deltas
def find_persistent_anomalies(anomalies, timestamps, min_duration_minutes):
    persistent_anomalies = np.zeros_like(anomalies, dtype=bool)
    i = 0
    while i < len(anomalies):
        if anomalies[i]:
            j = i
            while j < len(anomalies) and anomalies[j]:
                j += 1
            time_diff = (timestamps.iloc[j - 1] - timestamps.iloc[i]).total_seconds() / 60.0 if j > i else 0
            if time_diff >= min_duration_minutes:
                persistent_anomalies[i:j] = True
            i = j
        else:
            i += 1
    return persistent_anomalies


# Group by CustomerName and perform analysis separately for each
for customer, group in data.groupby('CustomerName'):
    print(f"Processing customer: {customer}")

    # Sort by Timestamp and reset index for consistency
    group = group.sort_values('Timestamp').reset_index(drop=True)

    # Extract features and timestamps
    features = group[num_cols].values
    timestamps = group['Timestamp']

    # Check if enough data for sequences
    seq_length = 2
    if len(features) < seq_length:
        print(f"Skipping customer {customer}: Not enough data points ({len(features)} < {seq_length}).")
        continue

    # Scale the features to [0, 1]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Create sequences and track corresponding indices
    sequences, seq_indices = create_sequences(features_scaled, seq_length)

    # Convert to PyTorch tensor
    tensor = torch.FloatTensor(sequences)

    # Create DataLoader for training
    loader = DataLoader(TensorDataset(tensor, tensor), batch_size=32, shuffle=True)


    # Define LSTM Autoencoder model
    class LSTMAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim=64, num_layers=3):
            super(LSTMAutoencoder, self).__init__()
            self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            hidden = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
            decoded, _ = self.decoder(hidden)
            decoded = self.fc(decoded)
            return decoded


    # Initialize model, optimizer, and loss
    input_dim = len(num_cols)
    model = LSTMAutoencoder(input_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.MSELoss()

    # Train the autoencoder
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for inputs, _ in loader:
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Compute reconstruction errors for each sequence
    model.eval()
    with torch.no_grad():
        reconstructed = model(tensor)
        squared_errors = (tensor - reconstructed) ** 2
        feature_errors = torch.mean(squared_errors, dim=2).numpy()

    # Apply Isolation Forest for anomaly detection
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    anomaly_scores = iso_forest.fit_predict(feature_errors)
    sequence_anomalies = anomaly_scores == -1

    # Map sequence anomalies to original data points
    anomalies = np.zeros(len(features), dtype=bool)
    for i, idx in enumerate(seq_indices):
        anomalies[idx] = sequence_anomalies[i]

    # Filter for persistent anomalies (lasting at least 30 minutes)
    min_duration_minutes = 2
    persistent_anomalies = find_persistent_anomalies(anomalies, timestamps, min_duration_minutes)

    # Compute feature-specific anomalies for visualization
    with torch.no_grad():
        single_tensor = torch.FloatTensor(features_scaled).unsqueeze(1)
        single_reconstructed = model(single_tensor).squeeze(1)
        feature_errors_single = ((single_tensor.squeeze(1) - single_reconstructed) ** 2).numpy()
        feature_thresholds = np.mean(feature_errors_single, axis=0) + 2 * np.std(feature_errors_single, axis=0)
        feature_anomalies = feature_errors_single > feature_thresholds

    # Mark data points as anomalies if at least 4 features are anomalous
    four_feature_anomalies = np.sum(feature_anomalies, axis=1) >= 2
    final_anomalies = four_feature_anomalies & persistent_anomalies

    # Print anomaly detection summary
    print(
        f"Persistent anomalies (4+ features, time delta >= {min_duration_minutes} min) detected for {customer}: {np.sum(final_anomalies)} out of {len(anomalies)} data points")

    # Plot line graphs for each feature with only anomalous metrics marked
    fig, axs = plt.subplots(len(num_cols), 1, figsize=(12, 3 * len(num_cols)), sharex=True)
    times = group['Timestamp']

    for i, col in enumerate(num_cols):
        # Plot the feature values
        axs[i].plot(times, group[col], label=col, color='blue')

        # Select data points where the specific feature is anomalous and the data point is a persistent anomaly
        feature_anomaly_idx = feature_anomalies[:, i] & final_anomalies
        anomaly_times = times[feature_anomaly_idx]
        anomaly_values = group[col][feature_anomaly_idx]

        # Plot red markers only for the specific feature's anomalies
        axs[i].scatter(anomaly_times, anomaly_values, color='red', label='Anomaly')
        axs[i].set_ylabel(col)
        axs[i].grid(True)

    axs[-1].set_xlabel('Timestamp')
    plt.suptitle(
        f'Persistent Multivariate Anomaly Detection (LSTM, Isolation Forest, 4+ Features) for Customer: {customer}')
    plt.tight_layout()
    plt.show()