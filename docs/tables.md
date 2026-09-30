## Selected model (from model_card.json)

- Churn model: **Logistic Regression** (threshold 0.6709)
- Value model: **LightGBM Tweedie**
- Validation ROC-AUC: 0.7967 | Test ROC-AUC: 0.7559 | Test recall: 0.7579
- CLV validation MAE: £283.76 | top-decile value capture: 0.6391


## Churn model comparison (validation-ranked)

| Model               |   Val ROC-AUC |   Val Recall |   Val Precision |   Val F1 |   Val Accuracy |   CV ROC-AUC (fwd-chain) | use_scaled   |
|:--------------------|--------------:|-------------:|----------------:|---------:|---------------:|-------------------------:|:-------------|
| Logistic Regression |        0.7967 |       0.7552 |          0.8099 |   0.7816 |         0.7325 |                   0.7815 | True         |
| Random Forest       |        0.7897 |       0.8143 |          0.794  |   0.804  |         0.7484 |                   0.8347 | False        |
| Decision Tree       |        0.7873 |       0.8041 |          0.8003 |   0.8022 |         0.7487 |                   0.774  | False        |
| XGBoost             |        0.7617 |       0.8077 |          0.7698 |   0.7883 |         0.7251 |                   0.8869 | False        |
| LightGBM            |        0.7602 |       0.7807 |          0.7805 |   0.7806 |         0.7218 |                   0.8613 | False        |
| KNN                 |        0.7562 |       0.8158 |          0.7697 |   0.7921 |         0.7286 |                   0.7976 | True         |


## Value model comparison

| Model                       |   R² Score |   MAE (£) |   RMSE (£) |   Training Time (s) |
|:----------------------------|-----------:|----------:|-----------:|--------------------:|
| Ridge Regression            |     0.8447 |    370.22 |    1180.75 |               0.031 |
| Random Forest Regressor     |     0.6678 |    412.71 |    1726.63 |               0.326 |
| LightGBM Regressor          |     0.5793 |    449.5  |    1943.2  |               0.053 |
| XGBoost Regressor           |     0.517  |    421.91 |    2082.15 |               0.675 |
| Gradient Boosting Regressor |     0.3884 |    452.01 |    2342.81 |               2.481 |