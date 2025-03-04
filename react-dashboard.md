# Running the Stock Market Anomaly Detection Dashboard

Follow these step-by-step instructions to set up and run the React dashboard on your local machine.

## Step 1: Set up the project structure

1. Create a new directory for the React app:

```bash
mkdir -p stock-anomaly-dashboard
cd stock-anomaly-dashboard
```

2. Create the necessary folder structure:

```bash
mkdir -p src/components src/utils public/data
```

## Step 2: Create the component files

Copy all the provided code into the appropriate files:

1. `src/App.js` - Main application component
2. `src/App.css` - Main application styles
3. `src/index.js` - React entry point
4. `src/index.css` - Global styles
5. `src/components/TimeSeriesChart.js` - Time series visualization component
6. `src/components/ReconstructionErrorChart.js` - Reconstruction error visualization
7. `src/components/LatentSpaceChart.js` - Latent space visualization
8. `src/components/ComparisonChart.js` - Original vs. reconstructed comparison
9. `src/utils/sampleData.js` - Sample data generator
10. `package.json` - Project dependencies
11. `README.md` - Project documentation

## Step 3: Copy the Python output to the React app (optional)

If you've run the Python analysis script (`stockwatch_updated.py`), copy the JSON output to the public directory:

```bash
mkdir -p public/data
cp anomaly_results.json public/data/
```

## Step 4: Install dependencies

Install all required dependencies using npm:

```bash
npm install
```

This will install React, Recharts, Axios, and other dependencies defined in the package.json file.

## Step 5: Start the development server

Start the React development server:

```bash
npm start
```

This will automatically open your browser to http://localhost:3000 where you can view the dashboard.

## Troubleshooting

If you encounter any issues:

1. **Node version issues**: Make sure you have Node.js v14 or newer installed
   ```bash
   node -v
   ```

2. **Dependency issues**: Try cleaning npm cache and reinstalling
   ```bash
   npm cache clean --force
   npm install
   ```

3. **Cannot find module errors**: Make sure all files are in the correct locations according to the imports

4. **Data loading issues**: Check that the anomaly_results.json file is in the public/data directory, or the app will use sample data

5. **Port already in use**: If port 3000 is already in use, npm will prompt you to use a different port

## Additional Configuration

- To use a different port, you can set it before starting the server:
  ```bash
  PORT=8080 npm start
  ```

- To build for production deployment:
  ```bash
  npm run build
  ```
  This creates an optimized build in the `build` folder that you can deploy to a web server.