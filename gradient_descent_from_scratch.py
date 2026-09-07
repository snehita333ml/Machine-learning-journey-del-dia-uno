import numpy as np
import matplotlib.pyplot as plt 

#creating fake dataset on my own
np.random.seed(42)
X = 2 * np.random.rand(100,1) #creates 100 random house sizes
y = 4 + 3 * X + np.random.randn(100,1) # creates prices based on a formula

print("X (first 5 house sizes):", X[:5])
print("y(first 5 prices):", y[:5])
print("Total data points: ", len(X))

#visualising the data

plt.figure(figsize=(8,6))
plt.scatter(X,y, color='blue', alpha=0.5)
plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("House Size vs Price")
plt.show()

#initializing weights randomly
w = 0.0 #slope
b = 0.0 #intercept
learning_rate = 0.01
epochs = 1000

# Gradient Descent Loop
for epoch in range(epochs):
    #make predictions
    y_pred = w * X + b

    #calculate gradients

    dw = (-2/len(X)) * np.sum(X * (y - y_pred))
    db = (-2/len(X)) * np.sum(y - y_pred)

    #update weights

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        loss = np.mean((y-y_pred) ** 2)
        print(f"Epoch{epoch} | Loss: {loss :.4f} | w: {w:.4f} | b: {b:.4f}")

print(f"\nFinal: w = {w: .4f}, b = {b:.4f}")
print(f"True Values were : w = 3, b = 4") 


#plot the result by visualisation

plt.figure(figsize = (8,6))
plt.scatter(X,y, color='blue', alpha= 0.5, label = 'REAL DATA')
plt.plot(X, w * X + b, color ='red', label=f'Learned Line: y = {w:.2f}x + {b:.2f}')
plt.xlabel("House Size")
plt.ylabel("Price")
plt.title("Gradient Descent - Learned Line")
plt.legend()
plt.show()
