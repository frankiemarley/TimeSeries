import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

# Load the dataset
csv_file_path = 'Aquifer_Auser.csv'
df = pd.read_csv(csv_file_path)

# Convert 'Date' column to datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

# Display basic information about the dataset
print(df.describe())
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Exclude non-numeric columns for correlation heatmap
numeric_df = df.select_dtypes(include=[float, int])

# Plot the distribution of Volume_POL
plt.figure(figsize=(12, 6))
sns.histplot(df['Volume_POL'].dropna(), kde=True)
plt.title('Distribution of Volume_POL')
plt.show()

# Correlation heatmap
plt.figure(figsize=(20, 16))
sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df.drop(columns=['Date'])), columns=df.columns[1:])

# Feature engineering
df_imputed['Month'] = df['Date'].dt.month
df_imputed['Season'] = df['Date'].dt.month.map({1:1, 2:1, 3:2, 4:2, 5:2, 6:3, 7:3, 8:3, 9:4, 10:4, 11:4, 12:1})
df_imputed['Year'] = df['Date'].dt.year
df_imputed['DayOfYear'] = df['Date'].dt.dayofyear
df_imputed['WeekOfYear'] = df['Date'].dt.isocalendar().week

# Define target variable and features
# Assuming 'Volume_POL' is the target variable for prediction
X = df_imputed.drop(columns=['Volume_POL'])
y = df_imputed['Volume_POL']

# Feature selection using Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
selector = RFE(rf, n_features_to_select=15, step=1)
selector = selector.fit(X, y)

selected_features = X.columns[selector.support_]
X = X[selected_features]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(alpha=1.0),
    'Ridge': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'\n{name}:')
    print(f'Mean Squared Error: {mse}')
    print(f'R-squared: {r2}')
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f'Cross-validation R-squared: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')

# Feature importance for Random Forest
rf_model = models['Random Forest']
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Time Series Analysis
df.set_index('Date', inplace=True)

# Time Series Decomposition
decomposition = seasonal_decompose(df['Volume_POL'].dropna(), model='additive', period=365)

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
decomposition.trend.plot(ax=ax1)
ax1.set_title('Trend')
decomposition.seasonal.plot(ax=ax2)
ax2.set_title('Seasonality')
decomposition.resid.plot(ax=ax3)
ax3.set_title('Residuals')
df['Volume_POL'].plot(ax=ax4)
ax4.set_title('Original')
plt.tight_layout()
plt.show()

# Autocorrelation and Partial Autocorrelation Analysis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
plot_acf(df['Volume_POL'].dropna(), ax=ax1)
plot_pacf(df['Volume_POL'].dropna(), ax=ax2)
plt.show()

# ARIMA Modeling
model = ARIMA(df['Volume_POL'].dropna(), order=(1, 1, 1))
results = model.fit()

print(results.summary())

forecast = results.get_forecast(steps=365)
forecast_ci = forecast.conf_int()

ax = df['Volume_POL'].plot(figsize=(12, 6), label='Original')
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='k', alpha=.25)
ax.set_xlabel('Date')
ax.set_ylabel('Volume_POL')
plt.title('ARIMA Forecast')
plt.legend()
plt.show()

# Time Series Cross-Validation
tscv = TimeSeriesSplit(n_splits=5)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
mse_scores = []
r2_scores = []

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Fit the model
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mse_scores.append(mse)
    r2_scores.append(r2)

# Print average performance metrics
print("Time Series Cross-Validation Results:")
print(f"Average MSE: {np.mean(mse_scores):.2f} (+/- {np.std(mse_scores) * 2:.2f})")
print(f"Average R-squared: {np.mean(r2_scores):.4f} (+/- {np.std(r2_scores) * 2:.4f})")

# Plot the predictions vs actual for the last fold
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted')
plt.title('Time Series Cross-Validation: Last Fold Predictions')
plt.xlabel('Date')
plt.ylabel('Volume_POL')
plt.legend()
plt.show()

# Feature importance for the final model
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title("Feature Importances (Time Series CV)")
plt.bar(range(X.shape[1]), importances[indices])
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()
