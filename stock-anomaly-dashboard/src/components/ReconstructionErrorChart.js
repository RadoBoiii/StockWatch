import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Scatter, ReferenceLine } from 'recharts';

const ReconstructionErrorChart = ({ data }) => {
  // If no data is provided, return placeholder
  if (!data || !data.reconstructionErrors) {
    return <div className="chart-container">No reconstruction error data available</div>;
  }

  // Prepare data for chart
  const chartData = data.reconstructionErrors.map(item => ({
    date: item.date,
    error: item.error,
    anomalyError: item.is_anomaly ? item.error : null,
    threshold: item.threshold
  }));

  // Calculate statistics
  const threshold = data.statistics?.threshold || chartData[0]?.threshold || 0.05;
  const errors = chartData.map(item => item.error);
  const meanError = errors.reduce((sum, val) => sum + val, 0) / errors.length;
  const maxError = Math.max(...errors);

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }} 
            tickFormatter={(value) => {
              const date = new Date(value);
              return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
            }}
          />
          <YAxis />
          <Tooltip 
            formatter={(value, name) => [
              value.toFixed(4), 
              name === 'error' ? 'Reconstruction Error' : 
              name === 'anomalyError' ? 'Anomaly' : 
              'Threshold'
            ]}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="error" 
            stroke="#4caf50" 
            dot={false} 
            name="Reconstruction Error" 
          />
          <ReferenceLine 
            y={threshold} 
            stroke="#ff5722" 
            strokeDasharray="3 3" 
            label={{ 
              value: `Threshold: ${threshold.toFixed(4)}`, 
              position: 'insideBottomRight',
              fill: '#ff5722',
              fontSize: 12
            }} 
          />
          <Scatter 
            name="Anomalies" 
            dataKey="anomalyError" 
            fill="#e91e63" 
            shape="circle" 
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="stats-container">
        <div className="stat-card">
            <p className="stat-label">Mean Error</p>
            <p className="stat-value">{meanError.toFixed(3)}</p>
        </div>
        <div className="stat-card">
            <p className="stat-label">Threshold</p>
            <p className="stat-value">{threshold.toFixed(3)}</p>
        </div>
        <div className="stat-card">
            <p className="stat-label">Max Error</p>
            <p className="stat-value">{maxError.toFixed(3)}</p>
        </div>
    </div>

      <div className="analysis-box">
        <h3>Reconstruction Error Analysis</h3>
        <p>This chart shows the reconstruction errors from the autoencoder model. When an input cannot be reconstructed accurately, it results in a high error value.</p>
        <p>Points above the threshold (red line) are classified as anomalies, indicating unusual market behavior that the model failed to reconstruct properly.</p>
        <p className="note">Mean Error: {meanError.toFixed(4)} | Max Error: {maxError.toFixed(4)} | Threshold: {threshold.toFixed(4)}</p>
      </div>
    </div>
  );
};

export default ReconstructionErrorChart;