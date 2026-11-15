# CodeAlpha_CarPricePrediction

**CodeAlpha Data Science Internship – Task 3: Car Price Prediction with Machine Learning**

Predicts the selling price of a used car (in lakh INR) from its original price, age, kilometres driven, fuel type, seller type, transmission and number of owners.

## Files
| File | Purpose |
|---|---|
| `Car_Price_Prediction.ipynb` | Full workflow: audit, cleaning, EDA, model comparison, evaluation, interpretation |
| `car_price_model.joblib` | Trained final model (fitted on all cars) |
| `data/car_data.csv` | Dataset |
| `images/` | Charts exported from the notebook |

## Approach
1. **Data audit:** found that ~1/3 of the rows are **two-wheelers** (Royal Enfield, Bajaj, Honda Activa, ...) priced ~10x lower. Since the task is car price, I model **cars only** (198 cars after removing 2 duplicates).
2. **Feature engineering:** car age, log of new price, log of kilometres; brand mapped from model name (tested, not needed).
3. **Target:** `log(selling price)`, because resale value changes by percentages.
4. **Models compared** with leak-free pipelines and 5-fold x 10-repeat cross-validation: baseline linear, log-log linear, Ridge, Random Forest, Gradient Boosting (tuned with GridSearchCV).
5. **Robustness:** hold-out test, unseen-car-model check (GroupKFold), sensitivity to the Land Cruiser outlier.

## Results (cars only)
| Model | CV R² | CV RMSE (lakh) | CV median % error |
|---|---|---|---|
| Baseline linear (raw) | 0.816 | 2.01 | 17.0 |
| **Linear, log-log (final)** | **0.941** | **1.13** | 11.5 |
| Ridge | 0.941 | 1.13 | 11.6 |
| Random Forest | 0.843 | 1.88 | 9.8 |
| Gradient Boosting | 0.866 | 1.72 | 10.2 |

Hold-out test R² ≈ 0.91; unseen-model R² ≈ 0.91.

**Findings:** new price dominates; each extra year of age ≈ -10% price; doubling km ≈ -5%; diesel ≈ +20% vs comparable petrol; brand adds no reliable signal.

## Charts
![Model comparison](images/07_model_comparison.png)
![Diagnostics](images/08_diagnostics.png)
![Depreciation](images/04_depreciation.png)

## Predict a new car
```python
import joblib, numpy as np, pandas as pd
model = joblib.load("car_price_model.joblib")
car = pd.DataFrame([{"LogPP": np.log(9.5), "Age": 2020 - 2015, "LogKm": np.log1p(45000),
                     "Owner": 0, "Fuel_Type": "Petrol", "Selling_type": "Dealer", "Transmission": "Manual"}])
print(model.predict(car))   # lakh INR
```

## Limitations
Small dataset (198 cars), prices from the 2003-2018 era, very few luxury or CNG/multi-owner cars. Retrain on fresh data before real use.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Car_Price_Prediction.ipynb
```
