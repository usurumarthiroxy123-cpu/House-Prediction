# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Load dataset
# Example columns: ['Area', 'Bedrooms', 'Bathrooms', 'Price']
df = pd.read_csv("house_prices.csv")

# 2. Features and target
X = df[['Area', 'Bedrooms', 'Bathrooms']]  # independent variables
y = df['Price']  # target variable

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Predict a new house price
sample_house = [[2000, 3, 2]]  # Area=2000 sq.ft, 3 bedrooms, 2 bathrooms
predicted_price = model.predict(sample_house)[0]
print(f"Predicted Price: ₹{predicted_price:,.2f}")
