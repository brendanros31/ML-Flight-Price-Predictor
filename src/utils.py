import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor



# Variance Inflation Factor
def vif(X):
    vif_data = pd.DataFrame()

    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    print(vif_data)
    return vif_data



# Heatmap for selected Features
def heatmap(df, show = True):
    if show == True:
        grouped_data = df.drop(columns=['flight'])
        plt.figure(figsize=(10,5))
        sns.heatmap(grouped_data.corr(), annot=True, cmap='coolwarm')

        plt.show()
