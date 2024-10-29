import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the data
data = {
    "Year": [
        2020,
        2019,
        2018,
        2017,
        2016,
        2015,
        2010,
        2005,
        2000,
        1995,
        1990,
        1985,
        1980,
        1975,
        1970,
        1965,
        1960,
        1955,
    ],
    "Population": [
        1380004385,
        1366417754,
        1352642280,
        1338676785,
        1324517249,
        1310152403,
        1234281170,
        1147609927,
        1056575549,
        963922588,
        873277798,
        784360008,
        698952844,
        623102897,
        555189792,
        499123324,
        450547679,
        409880595,
    ],
}

df = pd.DataFrame(data)

# Preparing the data
X = df[["Year"]]
y = df["Population"]

# Training the model
model = LinearRegression()
model.fit(X, y)

# Predicting the population for future years
future_years = pd.DataFrame({"Year": [2025, 2030, 2035, 2040]})
predictions = model.predict(future_years)

# Prediction output
for year, population in zip(future_years["Year"], predictions):
    print(f"Predicted population for {year}: {int(population)}")

# Plotting the data and the predictions
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Linear Fit")
plt.scatter(future_years, predictions, color="green", label="Predictions")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population Prediction of India")
plt.legend()
plt.show()
