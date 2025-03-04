import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

const ComparisonChart = ({ data }) => {
  const [selectedAnomaly, setSelectedAnomaly] = useState(0);
  
  // Generate sample comparison data if not provided
  const generateComparisonData = () => {
    const windowSize = 30;
    const samples = [];
    
    // Generate a few anomaly samples
    for (let s = 0; s < 3; s++) {
      const sample = [];
      
      // Base pattern (sine wave)
      for (let i = 0; i < windowSize; i++) {
        const original = Math.sin(i / 5) * 0.5 + 0.5;
        
        // Add reconstruction error - more error for anomalies in certain regions
        const error = i > 20 ? Math.random() * 0.25 : Math.random() * 0.05;
        const reconstructed = original + (Math.random() > 0.5 ? error : -error);
        
        sample.push({
          timeStep: i,
          original: original,
          reconstructed: reconstructed,
          error: Math.abs(original - reconstructed)
        });
      }
      
      samples.push({
        data: sample,
        date: `2023-${(s + 1).toString().padStart(2, '0')}-15`,
        maxError: Math.max(...sample.map(item => Math.abs(item.original - item.reconstructed)))
      });
    }
    
    return samples;
  };

  // Use provided comparison data or generate sample data
  const comparisonSamples = data?.comparisonSamples || generateComparisonData();
  
  const handleAnomalyChange = (event) => {
    setSelectedAnomaly(Number(event.target.value));
  };

  const selectedSample = comparisonSamples[selectedAnomaly];
  
  return (
    <div className="chart-container">
      <div className="comparison-controls">
        <label htmlFor="anomaly-select">Select Anomaly: </label>
        <select 
          id="anomaly-select"
          value={selectedAnomaly}
          onChange={handleAnomalyChange}
          className="anomaly-select"
        >
          {comparisonSamples.map((sample, index) => (
            <option key={index} value={index}>
              Anomaly {index + 1} {sample.date ? `(${sample.date})` : ''}
            </option>
          ))}
        </select>
      </div>
      
      <ResponsiveContainer width="100%" height="80%">
        <AreaChart 
          data={selectedSample.data} 
          margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timeStep" label={{ value: 'Time Step', position: 'bottom', offset: 0 }} />
          <YAxis label={{ value: 'Value', angle: -90, position: 'insideLeft' }} />
          <Tooltip 
            formatter={(value, name) => [
              value.toFixed(4), 
              name === 'original' ? 'Original' : 
              name === 'reconstructed' ? 'Reconstructed' : 
              'Error'
            ]}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="original" 
            stroke="#8884d8" 
            strokeWidth={2}
            name="Original" 
            dot={false} 
          />
          <Line 
            type="monotone" 
            dataKey="reconstructed" 
            stroke="#4caf50" 
            strokeWidth={2}
            name="Reconstructed" 
            dot={false} 
          />
          <Area 
            type="monotone" 
            dataKey="error" 
            fill="#f44336" 
            stroke="none"
            opacity={0.2}
            name="Error" 
          />
        </AreaChart>
      </ResponsiveContainer>
      
      <div className="anomaly-details">
        <p>
          <strong>Anomaly Date:</strong> {selectedSample.date || 'N/A'}
          <span className="separator">|</span>
          <strong>Max Error:</strong> {selectedSample.maxError?.toFixed(4) || 'N/A'}
        </p>
      </div>
    </div>
  );
};

export default ComparisonChart;