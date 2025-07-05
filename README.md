# ML-Flight-Price-Predictor

This project predicts airline Ticket Prices using various regression models - Linear regression, Decision Tree regressor, and Random Forest regressor. 

It includes Data Analysis, Feature Engineering, Model Training, and Evaluation using metrics like r2 Score, MSE, RMSE, MAE and RMAE.

## How to Use
```bash
pip install -r requirements.txt
python main.ipynb
```

## Project Structure
```
data/
  raw/
    Flight_Booking.csv

src/
  data_loader.py
  evaluate.py
  features.py
  model.py
  utils.py

EDA.ipynb
main.ipynb
config/config.yaml
```