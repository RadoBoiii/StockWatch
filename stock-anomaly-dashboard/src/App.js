import React, { useState, useEffect } from 'react';
import './App.css';
import axios from 'axios';
import TimeSeriesChart from './components/TimeSeriesChart';
import ReconstructionErrorChart from './components/ReconstructionErrorChart';
import LatentSpaceChart from './components/LatentSpaceChart';
import ComparisonChart from './components/ComparisonChart';
import { generateSampleData } from './utils/sampleData';

function App() {
  const [activeTab, setActiveTab] = useState("time-series");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Try to load real data from your Python script's output
        const response = await axios.get('/data/anomaly_results.json');
        setData(response.data);
      } catch (err) {
        console.log("Could not load data from JSON file, using sample data instead");
        // Generate sample data if the JSON file is not available
        setData(generateSampleData());
        setError("Using sample data (couldn't load anomaly_results.json)");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return <div className="loading">Loading data...</div>;
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Stock Market Anomaly Detection Dashboard</h1>
        {error && <div className="error-message">{error}</div>}
      </header>
      <main>
        <div className="tabs">
          <div className="tab-list">
            <button 
              className={activeTab === "time-series" ? "active" : ""} 
              onClick={() => setActiveTab("time-series")}
            >
              Time Series
            </button>
            <button 
              className={activeTab === "reconstruction-errors" ? "active" : ""} 
              onClick={() => setActiveTab("reconstruction-errors")}
            >
              Reconstruction Errors
            </button>
            <button 
              className={activeTab === "latent-space" ? "active" : ""} 
              onClick={() => setActiveTab("latent-space")}
            >
              Latent Space
            </button>
            <button 
              className={activeTab === "comparisons" ? "active" : ""} 
              onClick={() => setActiveTab("comparisons")}
            >
              Anomaly Comparisons
            </button>
          </div>
          
          <div className="tab-content">
            {activeTab === "time-series" && (
              <div className="card">
                <div className="card-header">
                  <h2>Stock Price Time Series with Anomalies</h2>
                </div>
                <div className="card-content">
                  <TimeSeriesChart data={data} />
                  <div className="stats-container">
                    <div className="stat-card">
                      <p className="stat-label">Total Anomalies</p>
                      <p className="stat-value">{data.statistics?.totalAnomalies || 0}</p>
                    </div>
                    <div className="stat-card">
                      <p className="stat-label">Anomaly Rate</p>
                      <p className="stat-value">{data.statistics?.anomalyRate?.toFixed(2) || 0}%</p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === "reconstruction-errors" && (
              <div className="card">
                <div className="card-header">
                  <h2>Reconstruction Errors Analysis</h2>
                </div>
                <div className="card-content">
                  <ReconstructionErrorChart data={data} />
                  <div className="stats-container">
                    <div className="stat-card">
                      <p className="stat-label">Mean Error</p>
                      <p className="stat-value">
                        {data.statistics ? (data.statistics.meanError || 0.032).toFixed(3) : 0.032}
                      </p>
                    </div>
                    <div className="stat-card">
                      <p className="stat-label">Threshold</p>
                      <p className="stat-value">
                        {data.statistics ? (data.statistics.threshold || 0.05).toFixed(3) : 0.05}
                      </p>
                    </div>
                    <div className="stat-card">
                      <p className="stat-label">Max Error</p>
                      <p className="stat-value">
                        {data.statistics ? (data.statistics.maxError || 0.089).toFixed(3) : 0.089}
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === "latent-space" && (
              <div className="card">
                <div className="card-header">
                  <h2>Latent Space Visualization</h2>
                </div>
                <div className="card-content">
                  <LatentSpaceChart data={data} />
                  <div className="analysis-box">
                    <h3>Latent Space Analysis</h3>
                    <p>This visualization shows the 2D projection of the autoencoder's latent space. Normal data points cluster together, while anomalies (in red) tend to be isolated or form separate clusters.</p>
                    <p className="note">The distance between points represents their similarity in the encoded representation.</p>
                  </div>
                </div>
              </div>
            )}
            
            {activeTab === "comparisons" && (
              <div className="card">
                <div className="card-header">
                  <h2>Original vs. Reconstructed Comparison</h2>
                </div>
                <div className="card-content">
                  <ComparisonChart data={data} />
                  <div className="analysis-box">
                    <h3>Reconstruction Analysis</h3>
                    <p>This visualization compares the original window of data (purple) with the autoencoder's reconstruction (green). The difference between these lines represents the reconstruction error.</p>
                    <p className="note">Large gaps between the original and reconstructed signals indicate unusual patterns that the model couldn't properly learn, suggesting anomalous behavior.</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      <footer>
        <p>Stock Market Anomaly Detection Dashboard · Powered by TensorFlow and React</p>
      </footer>
    </div>
  );
}

export default App;