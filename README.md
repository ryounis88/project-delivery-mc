# Project Delivery Method Selection Tool

This application uses Monte Carlo simulation to help select the optimal project delivery method based on key parameters:
- Duration (months)
- Cost ($ millions)
- Quality (PQI)

## Project Delivery Methods

The tool evaluates three common project delivery methods:
1. Design-Bid-Build (DBB)
2. Design-Build (DB)
3. Construction Management at Risk (CMaR)

## Features

- Triangular distribution modeling for each parameter
- User-adjustable weights for criteria importance
- Comprehensive statistical analysis including:
  - Probability Density Function (PDF) visualization
  - Cumulative Distribution Function (CDF) visualization
  - Statistical significance testing (p-values)
  - Practical significance assessment (time and cost savings)
- Interactive UI for parameter adjustment
- Clear result interpretation for non-technical users

## How to Use

1. Set the number of simulations using the slider
2. Adjust the weights for each criterion (Duration, Cost, Quality)
3. Input parameter estimates for each delivery method:
   - Minimum, most likely, and maximum values for Duration
   - Minimum, most likely, and maximum values for Cost
   - Minimum, most likely, and maximum values for Quality
4. Click "Run Monte Carlo Simulation"
5. Review the results, visualizations, and recommendation

## Running the Application

```bash
streamlit run app.py
```

## Requirements

See requirements.txt for a full list of dependencies.