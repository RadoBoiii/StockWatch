# Improved Stock Market Anomaly Detection
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, losses, callbacks
from tensorflow.keras.models import Model
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from scipy import stats
import json
import os

# Set plotting defaults
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # Try updated style name first
except:
    try:
        plt.style.use('seaborn-whitegrid')  # Try legacy name
    except:
        plt.style.use('grid')  # Fallback to basic grid style

plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Function to fetch and preprocess stock data
def fetch_stock_data(ticker="^GSPC", period="10y", rolling_window=4):
    """
    Fetch stock data and perform initial preprocessing
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period to fetch
        rolling_window (int): Rolling window size for smoothing
    
    Returns:
        pd.Series: Preprocessed stock data
    """
    print(f"Fetching data for {ticker} over {period} period...")
    
    # Download data
    stock_data = yf.Ticker(ticker)
    data = stock_data.history(period=period)
    
    # Extract close prices
    close_prices = data['Close']
    
    # Apply rolling window smoothing
    smoothed_data = close_prices.rolling(rolling_window).mean()
    
    # Drop NaN values
    smoothed_data = smoothed_data.dropna()
    
    # Format dates for better readability
    smoothed_data.index = pd.to_datetime(smoothed_data.index)
    
    return smoothed_data

# Enhanced visualization function for time series data
def plot_time_series(data, title="Stock Price Time Series", show_statistics=True):
    """
    Create an enhanced visualization of time series data
    
    Args:
        data (pd.Series): Time series data
        title (str): Plot title
        show_statistics (bool): Whether to show statistics
    """
    fig = plt.figure(figsize=(14, 8))
    
    # Plot the data
    plt.plot(data.index, data.values, linewidth=2, color='#1f77b4')
    
    # Add trend line
    z = np.polyfit(range(len(data)), data.values, 1)
    p = np.poly1d(z)
    plt.plot(data.index, p(range(len(data))), "r--", linewidth=1.5, alpha=0.7)
    
    # Add volatility bands (1 standard deviation)
    rolling_std = data.rolling(30).std()
    rolling_mean = data.rolling(30).mean()
    plt.fill_between(data.index, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2, color='#2ca02c')
    
    # Add labels and title
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    
    # Add statistics annotation if requested
    if show_statistics:
        stats_text = f"Mean: {data.mean():.2f}\n"
        stats_text += f"Std Dev: {data.std():.2f}\n"
        stats_text += f"Min: {data.min():.2f}\n"
        stats_text += f"Max: {data.max():.2f}\n"
        stats_text += f"Current: {data.iloc[-1]:.2f}"
        
        plt.annotate(stats_text, xy=(0.02, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc='#f8f9fa', ec='#cccccc', alpha=0.8),
                    fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# Improved data normalization and windowing function
def create_windows(data, window_size=30, step=5, scale=True):
    """
    Create windowed dataset with improved normalization
    
    Args:
        data (pd.Series): Input time series data
        window_size (int): Size of each window
        step (int): Step size between windows
        scale (bool): Whether to scale the data
    
    Returns:
        tuple: X windows, scalers (if scaling was applied), and dates
    """
    X = []
    scalers = []
    dates = []
    
    for i in range(0, len(data) - window_size, step):
        # Extract window
        x_window = data.iloc[i:i+window_size]
        
        if scale:
            # Normalize the window
            scaler = MinMaxScaler(feature_range=(0, 1))
            x_window_values = np.array(x_window).reshape(-1, 1)
            x_window_scaled = scaler.fit_transform(x_window_values).flatten()
            
            X.append(x_window_scaled)
            scalers.append(scaler)
        else:
            X.append(x_window.values)
        
        # Store the date for the end of this window
        dates.append(data.index[i+window_size-1])
    
    return np.array(X), scalers, dates

# Enhanced Autoencoder model
class EnhancedAnomalyDetector(Model):
    def __init__(self, input_dim=30, latent_dim=8, dropout_rate=0.2, regularization=0.001):
        """
        Enhanced Autoencoder model for anomaly detection
        
        Args:
            input_dim (int): Input dimension
            latent_dim (int): Latent dimension
            dropout_rate (float): Dropout rate
            regularization (float): L2 regularization factor
        """
        super(EnhancedAnomalyDetector, self).__init__()
        
        # L2 regularizer
        regularizer = tf.keras.regularizers.l2(regularization)
        
        # Encoder network
        self.encoder = tf.keras.Sequential([
            layers.Dense(24, activation="relu", kernel_regularizer=regularizer, name="encoder_1"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(16, activation="relu", kernel_regularizer=regularizer, name="encoder_2"),
            layers.BatchNormalization(),
            layers.Dense(latent_dim, activation="relu", name="latent_encoding")
        ])
        
        # Decoder network
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation="relu", kernel_regularizer=regularizer, name="decoder_1"),
            layers.BatchNormalization(),
            layers.Dropout(dropout_rate),
            layers.Dense(24, activation="relu", kernel_regularizer=regularizer, name="decoder_2"),
            layers.BatchNormalization(),
            layers.Dense(input_dim, activation="sigmoid", name="output")
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

# Function to train the model with better settings and callbacks
def train_model(X_train, X_val, batch_size=32, epochs=100, learning_rate=0.001):
    """
    Train the anomaly detection model with improved settings
    
    Args:
        X_train (np.array): Training data
        X_val (np.array): Validation data
        batch_size (int): Batch size
        epochs (int): Maximum epochs
        learning_rate (float): Learning rate
    
    Returns:
        tuple: Trained model and training history
    """
    input_dim = X_train.shape[1]
    
    # Create model
    model = EnhancedAnomalyDetector(input_dim=input_dim)
    
    # Adam optimizer with learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile model
    model.compile(optimizer=optimizer, loss='mae')
    
    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        callbacks.TensorBoard(log_dir=f"./logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history

# Function to plot training history
def plot_training_history(history):
    """
    Plot training history with improved visualization
    
    Args:
        history: Training history object
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training & validation loss
    ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    
    # Add labels and title
    ax.set_title('Model Training History', fontsize=16, fontweight='bold')
    ax.set_ylabel('Loss (MAE)', fontsize=12)
    ax.set_xlabel('Epoch', fontsize=12)
    
    # Add grid and legend
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(frameon=True, fontsize=12)
    
    # Annotate lowest validation loss
    min_val_loss = min(history.history['val_loss'])
    min_val_epoch = history.history['val_loss'].index(min_val_loss)
    
    ax.scatter(min_val_epoch, min_val_loss, s=100, color='red', zorder=5)
    ax.annotate(f'Best: {min_val_loss:.4f}', 
                xy=(min_val_epoch, min_val_loss),
                xytext=(min_val_epoch+5, min_val_loss+0.001),
                fontsize=10,
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
    
    plt.tight_layout()
    plt.show()

# Improved function to detect anomalies with multiple threshold options
def detect_anomalies(model, data, dates, method='dynamic', contamination=0.01, z_score=3):
    """
    Detect anomalies using multiple threshold methods
    
    Args:
        model: Trained autoencoder model
        data (np.array): Input data
        dates (list): Dates corresponding to each window
        method (str): 'dynamic', 'static', 'zscore', or 'isolation_forest'
        contamination (float): Expected proportion of anomalies
        z_score (float): Z-score threshold for anomaly detection
    
    Returns:
        tuple: reconstruction errors, anomaly predictions, threshold
    """
    # Get reconstructions
    reconstructions = model.predict(data)
    
    # Calculate reconstruction errors
    reconstruction_errors = np.mean(np.abs(reconstructions - data), axis=1)
    
    # Determine threshold based on method
    if method == 'static':
        # Static threshold based on training data
        threshold = np.mean(reconstruction_errors) + z_score * np.std(reconstruction_errors)
        anomalies = reconstruction_errors > threshold
    
    elif method == 'zscore':
        # Z-score based threshold
        z_scores = stats.zscore(reconstruction_errors)
        anomalies = z_scores > z_score
        threshold = np.mean(reconstruction_errors) + z_score * np.std(reconstruction_errors)
    
    elif method == 'dbscan':
        # DBSCAN clustering for anomaly detection
        errors_2d = np.column_stack((range(len(reconstruction_errors)), reconstruction_errors))
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(errors_2d)
        anomalies = clusters == -1  # DBSCAN labels outliers with -1
        threshold = np.min(reconstruction_errors[anomalies]) if np.any(anomalies) else np.max(reconstruction_errors)
    
    else:  # dynamic (default)
        # Dynamic threshold based on top n% of errors
        threshold = np.percentile(reconstruction_errors, 100 * (1 - contamination))
        anomalies = reconstruction_errors > threshold
    
    # Create results DataFrame
    results = pd.DataFrame({
        'date': dates,
        'reconstruction_error': reconstruction_errors,
        'is_anomaly': anomalies,
        'threshold': threshold
    })
    
    return results

# Function to visualize reconstruction errors and anomalies
def plot_anomalies(results, title="Anomaly Detection Results"):
    """
    Visualize reconstruction errors and detected anomalies
    
    Args:
        results (pd.DataFrame): Results from anomaly detection
        title (str): Plot title
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot 1: Reconstruction errors with threshold
    ax1.plot(results['date'], results['reconstruction_error'], label='Reconstruction Error')
    ax1.axhline(y=results['threshold'].iloc[0], color='r', linestyle='--', label=f'Threshold: {results["threshold"].iloc[0]:.4f}')
    
    # Highlight anomalies
    anomalies = results[results['is_anomaly']]
    ax1.scatter(anomalies['date'], anomalies['reconstruction_error'], 
                color='red', label=f'Anomalies ({len(anomalies)} detected)', s=50, zorder=5)
    
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Reconstruction Error', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Binary anomaly indicators
    ax2.stem(results['date'], results['is_anomaly'], linefmt='r-', markerfmt='ro', basefmt='r-')
    ax2.set_ylabel('Anomaly', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Normal', 'Anomaly'])
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Function to visualize original vs. reconstructed data for specific anomalies
def plot_reconstructions(model, data, results, num_plots=3):
    """
    Visualize original vs. reconstructed data for specific anomalies
    
    Args:
        model: Trained autoencoder model
        data (np.array): Input data
        results (pd.DataFrame): Results from anomaly detection
        num_plots (int): Number of anomalies to plot
    """
    # Get anomalous data
    anomalies = results[results['is_anomaly']]
    
    # If there are anomalies, plot them
    if len(anomalies) > 0:
        # Select random anomalies to plot (or all if there are fewer than num_plots)
        indices_to_plot = anomalies.index[:min(num_plots, len(anomalies))]
        
        for i, idx in enumerate(indices_to_plot):
            # Get original and reconstructed data
            original = data[idx]
            reconstructed = model.predict(np.array([original]))[0]
            
            # Create figure
            plt.figure(figsize=(12, 6))
            
            # Plot original and reconstructed data
            plt.plot(original, 'b', label='Original', linewidth=2)
            plt.plot(reconstructed, 'r', label='Reconstructed', linewidth=2)
            
            # Fill the area between the lines
            plt.fill_between(range(len(original)), original, reconstructed, 
                            color='lightcoral', alpha=0.5, label='Error')
            
            # Add labels and title
            plt.title(f'Anomaly {i+1} (Date: {anomalies.iloc[i]["date"].strftime("%Y-%m-%d")}, Error: {anomalies.iloc[i]["reconstruction_error"]:.4f})', 
                    fontsize=14, fontweight='bold')
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('Normalized Value', fontsize=12)
            plt.legend(loc='best')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.show()
    else:
        print("No anomalies detected to plot.")

# Function to visualize latent space representations
def plot_latent_space(model, data, results):
    """
    Visualize latent space representations
    
    Args:
        model: Trained autoencoder model
        data (np.array): Input data
        results (pd.DataFrame): Results from anomaly detection
    """
    # Get encoded representations
    encoded_data = model.encode(data).numpy()
    
    # If the latent dimension is > 2, use PCA to reduce to 2D
    if encoded_data.shape[1] > 2:
        pca = PCA(n_components=2)
        encoded_data_2d = pca.fit_transform(encoded_data)
    else:
        encoded_data_2d = encoded_data
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Plot normal points
    normal_mask = ~results['is_anomaly'].values
    plt.scatter(encoded_data_2d[normal_mask, 0], encoded_data_2d[normal_mask, 1], 
                c='blue', label='Normal', alpha=0.5)
    
    # Plot anomalous points
    anomaly_mask = results['is_anomaly'].values
    if np.any(anomaly_mask):
        plt.scatter(encoded_data_2d[anomaly_mask, 0], encoded_data_2d[anomaly_mask, 1], 
                    c='red', label='Anomaly', alpha=0.5)
    
    plt.title('Latent Space Visualization', fontsize=16, fontweight='bold')
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

# Function to generate interactive visualization with Plotly
def create_interactive_dashboard(original_data, results):
    """
    Create an interactive dashboard with Plotly
    
    Args:
        original_data (pd.Series): Original time series data
        results (pd.DataFrame): Results from anomaly detection
    """
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Price with Anomalies', 'Reconstruction Error')
    )
    
    # Match dates between original data and results
    # Convert results dates to string format for matching
    results['date_str'] = results['date'].astype(str).str[:10]
    
    # Convert original data index to string format for matching
    original_data_df = original_data.reset_index()
    original_data_df['date_str'] = original_data_df['Date'].astype(str).str[:10]
    
    # Merge datasets
    merged_data = pd.merge(
        original_data_df,
        results[['date_str', 'is_anomaly', 'reconstruction_error']],
        on='date_str',
        how='left'
    )
    
    # Fill NaN values in is_anomaly with False
    merged_data['is_anomaly'] = merged_data['is_anomaly'].fillna(False)
    
    # Add trace for stock price
    fig.add_trace(
        go.Scatter(
            x=merged_data['Date'],
            y=merged_data['Close'],
            mode='lines',
            name='Stock Price',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add trace for anomalies
    anomalies = merged_data[merged_data['is_anomaly']]
    if len(anomalies) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomalies['Date'],
                y=anomalies['Close'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='circle')
            ),
            row=1, col=1
        )
    
    # Add trace for reconstruction error
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=results['reconstruction_error'],
            mode='lines',
            name='Reconstruction Error',
            line=dict(color='orange', width=1)
        ),
        row=2, col=1
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=results['date'],
            y=[results['threshold'].iloc[0]] * len(results),
            mode='lines',
            name='Threshold',
            line=dict(color='red', width=1, dash='dash')
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Interactive Stock Anomaly Dashboard',
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axes labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Error", row=2, col=1)
    
    # Enable zooming
    fig.update_layout(
        xaxis=dict(rangeslider=dict(visible=True)),
        xaxis2=dict(rangeslider=dict(visible=False))
    )
    
    return fig

# Function to export results to JSON for the dashboard
def export_results_to_json(sp500_data, results, output_path="anomaly_results.json"):
    """
    Export anomaly detection results to JSON for the React dashboard
    
    Args:
        sp500_data (pd.Series): Original stock price data
        results (pd.DataFrame): Results from anomaly detection
        output_path (str): Path to save the JSON file
    """
    # Convert dates to strings for JSON serialization
    stock_prices = [
        {
            "date": date.strftime("%Y-%m-%d"),
            "price": float(price),
            "trend": float(price) * 1.05  # Simple trend line approximation
        }
        for date, price in zip(sp500_data.index, sp500_data.values)
    ]
    
    # Prepare anomalies data for export
    anomalies_df = results[results['is_anomaly']]
    anomalies = [
        {
            "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
            "error": float(row["reconstruction_error"]),
            "threshold": float(row["threshold"])
        }
        for _, row in anomalies_df.iterrows()
    ]
    
    # Prepare reconstruction error data
    errors = [
        {
            "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
            "error": float(row["reconstruction_error"]),
            "threshold": float(row["threshold"]),
            "is_anomaly": bool(row["is_anomaly"])
        }
        for _, row in results.iterrows()
    ]
    
    # Create a data structure for export
    data_for_export = {
        "stockPrices": stock_prices,
        "anomalies": anomalies,
        "reconstructionErrors": errors,
        "statistics": {
            "mean": float(sp500_data.mean()),
            "std": float(sp500_data.std()),
            "min": float(sp500_data.min()),
            "max": float(sp500_data.max()),
            "current": float(sp500_data.iloc[-1]),
            "totalAnomalies": int(sum(results['is_anomaly'])),
            "anomalyRate": float((sum(results['is_anomaly']) / len(results)) * 100),
            "threshold": float(results["threshold"].iloc[0])
        }
    }
    
    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(data_for_export, f, indent=2)
    
    print(f"Results exported to {output_path}")
    return data_for_export

# Main function to run the entire workflow
def main(export_json=True):
    """
    Main function to run the entire anomaly detection workflow
    
    Args:
        export_json (bool): Whether to export results to JSON
    
    Returns:
        tuple: model, results_list, sp500_data
    """
    try:
        # 1. Fetch and preprocess data
        sp500_data = fetch_stock_data(ticker="^GSPC", period="5y")
        
        # 2. Visualize the time series
        plot_time_series(sp500_data, title="S&P 500 Close Price")
        
        # 3. Create windowed dataset
        X, scalers, dates = create_windows(sp500_data, window_size=30, step=5)
        
        # 4. Split data
        train_size = int(0.7 * X.shape[0])
        val_size = int(0.15 * X.shape[0])
        test_size = X.shape[0] - train_size - val_size
        
        X_train = X[:train_size]
        X_val = X[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        
        test_dates = dates[train_size+val_size:]
        
        print(f"Training set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # 5. Train model
        model, history = train_model(
            X_train, X_val,
            batch_size=32,
            epochs=50,  # Reduced for faster execution
            learning_rate=0.001
        )
        
        # 6. Plot training history
        plot_training_history(history)
        
        # 7. Detect anomalies (using multiple methods for comparison)
        methods = ['dynamic', 'static', 'zscore', 'dbscan']
        results_list = []
        
        for method in methods:
            results = detect_anomalies(
                model, X_test, test_dates,
                method=method,
                contamination=0.05,  # Expect 5% anomalies
                z_score=2.5  # Lower threshold for more sensitivity
            )
            
            results_list.append((method, results))
            
            # 8. Plot anomalies for each method
            plot_anomalies(results, title=f"Anomaly Detection Results ({method.capitalize()} Method)")
        
        # 9. Plot reconstructions for the best method (using dynamic as default)
        best_method, best_results = results_list[0]  # Dynamic method
        plot_reconstructions(model, X_test, best_results, num_plots=3)
        
        # 10. Visualize latent space
        plot_latent_space(model, X_test, best_results)
        
        # 11. Create interactive dashboard
        dashboard = create_interactive_dashboard(sp500_data, best_results)
        dashboard.show()  # This should open a browser window
        
        # 12. Print summary of findings
        anomaly_counts = [(method, sum(results['is_anomaly'])) for method, results in results_list]
        
        print("\nSummary of Anomaly Detection Results:")
        print("-" * 40)
        for method, count in anomaly_counts:
            print(f"{method.capitalize()} method detected {count} anomalies ({count/len(X_test)*100:.2f}%)")
        
        # 13. Export results to JSON for the dashboard
        if export_json:
            export_results_to_json(sp500_data, best_results)
        
        # Return results for potential further analysis
        return model, results_list, sp500_data

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# If this script is run directly
if __name__ == "__main__":
    main(export_json=True)