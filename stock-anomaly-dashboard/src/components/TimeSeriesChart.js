import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Scatter } from 'recharts';

const TimeSeriesChart = ({ data }) => {
  // If no data is provided, return placeholder
  if (!data || !data.stockPrices) {
    return <div className="chart-container">No data available</div>;
  }

  // Prepare data for chart
  const chartData = data.stockPrices.map(item => {
    const isAnomaly = data.anomalies && data.anomalies.some(a => a.date === item.date);
    return {
      date: item.date,
      price: item.price,
      trend: item.trend,
      anomalyPrice: isAnomaly ? item.price : null
    };
  });

  return (
    <div className="chart-container">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="date" 
            tick={{ fontSize: 12 }} 
            tickFormatter={(value) => {
              // Format date for better readability
              const date = new Date(value);
              return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
            }}
          />
          <YAxis />
          <Tooltip 
            formatter={(value, name) => [
              name === 'anomalyPrice' ? 'Anomaly' : value.toFixed(2), 
              name === 'anomalyPrice' ? 'Anomaly' : name
            ]}
          />
          <Legend />
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke="#3f51b5" 
            dot={false} 
            name="Stock Price" 
            strokeWidth={2}
          />
          <Line 
            type="monotone" 
            dataKey="trend" 
            stroke="#ff9800" 
            strokeDasharray="5 5" 
            dot={false} 
            name="Trend" 
          />
          <Scatter 
            name="Anomalies" 
            dataKey="anomalyPrice" 
            fill="#e91e63" 
            shape="circle"
            legendType="circle"
          />
        </LineChart>
      </ResponsiveContainer>
      <div className="analysis-box">
        <h3>Time Series Analysis</h3>
        <p>This visualization shows the stock price over time with detected anomalies highlighted in red. The dashed orange line shows the overall trend.</p>
        <p>Anomalies represent points where the model detected unusual market behavior that deviates significantly from the expected pattern.</p>
      </div>
    </div>
  );
};

export default TimeSeriesChart;