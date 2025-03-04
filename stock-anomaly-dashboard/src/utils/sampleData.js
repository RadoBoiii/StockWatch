export const generateSampleData = () => {
    // Generate sample stock price data
    const generateStockPrices = () => {
      const data = [];
      let price = 3000;
      const trendSlope = 1.2;
      
      // Start date
      const startDate = new Date(2023, 0, 1);
      
      for (let i = 0; i < 100; i++) {
        // Calculate date
        const currentDate = new Date(startDate);
        currentDate.setDate(startDate.getDate() + i);
        
        // Add a slight upward trend
        const trend = 3000 + (i * trendSlope);
        
        // Random walk with occasional spikes
        price = price + (Math.random() - 0.48) * 30;
        
        // Add occasional anomalies
        const isAnomaly = Math.random() > 0.95;
        if (isAnomaly) {
          price = price + (Math.random() > 0.5 ? 150 : -150);
        }
        
        data.push({
          date: currentDate.toISOString().split('T')[0],
          price: price,
          trend: trend
        });
      }
      
      return data;
    };
  
    // Generate sample anomalies
    const generateAnomalies = (stockPrices) => {
      return stockPrices
        .filter((_, index) => Math.random() > 0.95)
        .map(item => ({
          date: item.date,
          error: Math.random() * 0.05 + 0.05
        }));
    };
  
    // Generate sample reconstruction errors
    const generateReconstructionErrors = (stockPrices, anomalies) => {
      return stockPrices.map(item => {
        const isAnomaly = anomalies.some(a => a.date === item.date);
        const error = Math.random() * 0.04 + 0.01;
        const threshold = 0.05;
        
        return {
          date: item.date,
          error: isAnomaly ? threshold + (Math.random() * 0.04) : error,
          is_anomaly: isAnomaly,
          threshold: threshold
        };
      });
    };
  
    // Generate sample comparison data for anomalies
    const generateComparisonSamples = (anomalies) => {
      return anomalies.slice(0, 3).map(anomaly => {
        const windowSize = 30;
        const sample = [];
        
        // Generate original and reconstructed patterns
        for (let i = 0; i < windowSize; i++) {
          const original = Math.sin(i / 5) * 0.5 + 0.5;
          const error = i > 20 ? Math.random() * 0.25 : Math.random() * 0.05;
          const reconstructed = original + (Math.random() > 0.5 ? error : -error);
          
          sample.push({
            timeStep: i,
            original: original,
            reconstructed: reconstructed,
            error: Math.abs(original - reconstructed)
          });
        }
        
        return {
          data: sample,
          date: anomaly.date,
          maxError: Math.max(...sample.map(item => Math.abs(item.original - item.reconstructed)))
        };
      });
    };
  
    // Generate latent space data
    const generateLatentSpace = (anomalies) => {
      const latentData = [];
      
      // Generate normal cluster
      for (let i = 0; i < 60; i++) {
        latentData.push({
          x: Math.random() * 2 - 1,
          y: Math.random() * 2 - 1,
          z: 100,
          isAnomaly: false
        });
      }
      
      // Generate anomalies
      for (let i = 0; i < anomalies.length; i++) {
        latentData.push({
          x: Math.random() * 6 - 3,
          y: Math.random() * 6 - 3,
          z: 100,
          isAnomaly: true
        });
      }
      
      return latentData;
    };
  
    // Generate full dataset
    const stockPrices = generateStockPrices();
    const anomalies = generateAnomalies(stockPrices);
    const reconstructionErrors = generateReconstructionErrors(stockPrices, anomalies);
    const comparisonSamples = generateComparisonSamples(anomalies);
    const latentSpace = generateLatentSpace(anomalies);
    
    // Calculate statistics
    const prices = stockPrices.map(item => item.price);
    const errors = reconstructionErrors.map(item => item.error);
    
    return {
      stockPrices,
      anomalies,
      reconstructionErrors,
      comparisonSamples,
      latentSpace,
      statistics: {
        mean: prices.reduce((sum, val) => sum + val, 0) / prices.length,
        std: Math.sqrt(prices.reduce((sum, val) => sum + Math.pow(val - (prices.reduce((sum, val) => sum + val, 0) / prices.length), 2), 0) / prices.length),
        min: Math.min(...prices),
        max: Math.max(...prices),
        current: prices[prices.length - 1],
        totalAnomalies: anomalies.length,
        anomalyRate: (anomalies.length / stockPrices.length) * 100,
        threshold: 0.05,
        meanError: errors.reduce((sum, val) => sum + val, 0) / errors.length,
        maxError: Math.max(...errors)
      }
    };
  };