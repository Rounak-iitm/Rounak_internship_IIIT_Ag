import numpy as np
import matplotlib.pyplot as plt
class KLMS:
    def __init__(self, eta=0.1, sigma=1.0):
        self.eta = eta
        self.sigma = sigma
        self.weights = []
        self.centers = []

    def kernel(self, x, c):
        return np.exp(-np.linalg.norm(x - c)**2 / (2 * self.sigma**2))

    def predict(self, x):
        return sum(w * self.kernel(x, c) for w, c in zip(self.weights, self.centers))

    def update(self, x, d):
        y = self.predict(x)
        err = d - y
        self.weights.append(self.eta * err)
        self.centers.append(x)
# Example usage
klms = KLMS(eta=0.1, sigma=1.0)
data_points = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
desired_outputs = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
for x, d in zip(data_points, desired_outputs):
    klms.update(np.array([x]), d)
predicted_outputs = [klms.predict(np.array([x])) for x in data_points]
# Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(data_points, desired_outputs, 'bo-', label="Desired Output")
plt.plot(data_points, predicted_outputs, 'r*-', label="Predicted Output")
plt.xlabel("Input Data")
plt.ylabel("Output")
plt.title("KLMS Prediction vs Desired Output")
plt.legend()
plt.grid()
plt.show()