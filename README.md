# Stock Market Anomaly Detection Dashboard

An interactive React dashboard for visualizing stock market anomalies detected by a TensorFlow autoencoder model.

## Features

- Time series visualization with anomaly detection
- Reconstruction error analysis
- Latent space visualization
- Original vs. reconstructed data comparison for anomalies

## Getting Started

### Prerequisites

- Node.js (v14 or newer)
- npm or yarn
- Python 3.6+ (for the anomaly detection model)

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/stock-anomaly-dashboard.git
cd stock-anomaly-dashboard
```

2. Install the React app dependencies

```bash
npm install
# or 
yarn install
```

3. Install Python dependencies (optional, for running the model)

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn plotly yfinance
```

### Running the Dashboard

1. Start the React development server

```bash
npm start
# or
yarn start
```

This will open the dashboard in your browser at http://localhost:3000.

2. Run the Python anomaly detection model (optional)

```bash
python stockwatch_updated.py
```

This will:
- Download stock data
- Train the autoencoder model
- Detect anomalies
- Export results to `public/data/anomaly_results.json`

## Integration

The dashboard will automatically load data from `public/data/anomaly_results.json` if available. If not, it will use sample data for demonstration purposes.

To connect your own anomaly detection results:

1. Export your results as a JSON file in the following format:
```json
{
  "stockPrices": [
    {"date": "2023-01-01", "price": 3000.0, "trend": 3050.0},
    ...
  ],
  "anomalies": [
    {"date": "2023-01-15", "error": 0.075},
    ...
  ],
  "reconstructionErrors": [
    {"date": "2023-01-01", "error": 0.02, "is_anomaly": false, "threshold": 0.05},
    ...
  ],
  "statistics": {
    "mean": 3050.0,
    "std": 150.5,
    "min": 2800.0,
    "max": 3300.0,
    "current": 3100.0,
    "totalAnomalies": 8,
    "anomalyRate": 5.2,
    "threshold": 0.05
  }
}
```

2. Place the file at `public/data/anomaly_results.json`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- TensorFlow for the autoencoder model
- Recharts for the visualization components