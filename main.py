import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt


#Loading datas on csv file

df = pd.read_csv("dados.csv");
print("Datas loaded")
print(df)

#Training Linear regression model

X = df[['horas_estudo']] #inputing
y = df['nota'] #output

#creating model
model = LinearRegression()

#training model
model.fit(X, y)

#making prevision for student that studie 6.5 hours

hours = [[6.5]]
prevision_note = model.predict(hours)

print(f"Pridict value for 6.5h studie: {prevision_note[0]:.2f}")

#model view

plt.scatter(X, y, color="blue", label="Real datas")
plt.plot(X, model.predict(X), color="red", label="Linear Regression")
plt.xlabel("Study Hours")

plt.ylabel("Value")

plt.title("Predict value with linear Regressesion")

plt.legend()

plt.grid(True)

plt.show()