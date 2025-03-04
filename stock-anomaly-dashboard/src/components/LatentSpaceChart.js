import React from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const LatentSpaceChart = ({ data }) => {
  // Generate sample latent space data if not provided in the JSON
  // In a real implementation, this would come from your Python model's encoder
  const generateLatentSpaceData = () => {
    const latentData = [];
    
    // Generate normal cluster
    for (let i = 0; i < 50; i++) {
      latentData.push({
        x: Math.random() * 2 - 1,
        y: Math.random() * 2 - 1,
        z: 100,
        isAnomaly: false
      });
    }
    
    // Generate anomalies
    for (let i = 0; i < 8; i++) {
      latentData.push({
        x: Math.random() * 6 - 3,
        y: Math.random() * 6 - 3,
        z: 100,
        isAnomaly: true
      });
    }
    
    return latentData;
  };

  // Use provided data or generate sample data
  const latentSpaceData = data?.latentSpace || generateLatentSpaceData();
  
  // Split data into normal and anomalous points
  const normalData = latentSpaceData.filter(point => !point.isAnomaly);
  const anomalyData = latentSpaceData.filter(point => point.isAnomaly);

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid />
          <XAxis 
            type="number" 
            dataKey="x" 
            name="Component 1" 
            domain={['auto', 'auto']} 
            label={{ value: 'Component 1', position: 'bottom', offset: 0 }}
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name="Component 2" 
            domain={['auto', 'auto']} 
            label={{ value: 'Component 2', angle: -90, position: 'left' }}
          />
          <ZAxis type="number" dataKey="z" range={[60, 200]} />
          <Tooltip 
            cursor={{ strokeDasharray: '3 3' }}
            formatter={(value, name) => [value.toFixed(3), name === 'x' ? 'Component 1' : 'Component 2']}
            labelFormatter={() => 'Latent Point'}
          />
          <Legend />
          <Scatter 
            name="Normal Data" 
            data={normalData} 
            fill="#3f51b5" 
            shape="circle" 
          />
          <Scatter 
            name="Anomalies" 
            data={anomalyData} 
            fill="#e91e63" 
            shape="diamond" 
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export default LatentSpaceChart;