# Exoplanet Equilibrium Temperature Prediction using Physics-Informed Machine Learning

## Overview

This project predicts the equilibrium temperature of exoplanets using stellar and orbital parameters from the NASA Exoplanet Archive. The objective is to evaluate how physics-inspired feature engineering (log transformation) impacts the performance of a linear regression model.

Two models are implemented and compared:
- Model A: Linear Regression using raw features
- Model B: Linear Regression using log-transformed features **(predicting normal temperature, not log temperature)**

---

## Dataset

### Source
NASA Exoplanet Archive  
https://exoplanetarchive.ipac.caltech.edu/

### Features Used
- pl_orbsmax: Orbital semi-major axis  
- st_teff: Stellar effective temperature  
- st_rad: Stellar radius  

### Target
- pl_eqt: Planet equilibrium temperature (Kelvin)

### Preprocessing
- Selected relevant numerical features
- Removed unnecessary columns
- Verified all values are positive for log transformation
- No missing values in selected subset

---

## Physical Motivation

The equilibrium temperature is approximated by:

`T_eq ∝ T_* √(R_* / a)`

Taking logarithm:

`log(T_eq) = log(T_*) + 0.5 log(R_*) − 0.5 log(a) + C`

This motivates the use of log-transformed features.

---

## Model A: Linear Regression (Raw Features)

### Features
- pl_orbsmax
- st_teff
- st_rad

### Results
- MAE: 274.65  
- MSE: 127380.72  
- RMSE: 356.90  
- R²: 0.21  

### Observation
Model fails to capture non-linear relationships in the data.

---

## Feature Engineering

Log transformation applied to:
- pl_orbsmax
- st_teff
- st_rad

This converts multiplicative relationships into additive form.

---

## Model B: Linear Regression (Log Features)

### Important note on model structure
Model B uses **log-transformed inputs** but predicts **normal temperature (not log temperature)**.  
The equation is:

`Teq = w1·log(x1) + w2·log(x2) + w3·log(x3) + b`

Because a linear combination can produce any real number, **this model can sometimes predict negative temperatures** — which is physically impossible in Kelvin. This is a known limitation of the current approach.

### Features
- log(pl_orbsmax)
- log(st_teff)
- log(st_rad)

### Results
- MAE: 96.80  
- MSE: 34855.62  
- RMSE: 186.69  
- R²: 0.78  

### Observation
Significant improvement in predictive performance due to partial linearization of the underlying physical relationship.

### Known Limitation & Suggested Fix
To avoid non-physical negative predictions, train on the **log-transformed target** instead:

`log(Teq) = w1·log(x1) + w2·log(x2) + w3·log(x3) + b`

Then exponentiate predictions: `Teq_pred = exp(predicted_log)`. This guarantees all predictions are positive and fully aligns with the multiplicative physics.

---

## Results Comparison

| Metric | Model A | Model B |
|--------|--------|--------|
| MAE | 274.65 | 96.80 |
| RMSE | 356.90 | 186.69 |
| R² | 0.21 | 0.78 |

---

## Key Insights

- Feature engineering has a greater impact than model complexity in this case
- Log transformation aligns data with physical relationships
- Linear regression performs well in log-linear space, but mismatched target transformation can produce non-physical outputs
- Residual analysis shows remaining non-linear structure

---

## Residual Analysis

- Model B significantly reduces residual variance
- Slight curvature remains, indicating missing physical variables or higher-order effects

---

## Conclusion

This project demonstrates that physics-informed feature engineering can significantly improve machine learning performance. A simple linear regression model improved from R² = 0.21 to R² = 0.78 by transforming the feature space using domain knowledge. However, careful alignment between input and target transformations is necessary to ensure physically meaningful predictions.

---

## Future Work

- Apply full log–log model (log-transformed target) to eliminate negative predictions
- Use non-linear models (Random Forest, Gradient Boosting)
- Add interaction terms between features
- Include additional astrophysical parameters
- Develop full predictive pipeline

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
