import numpy as np
import matplotlib.pyplot as plt

# Define parameters for the stability region
lambda_real = np.linspace(-10, 10, 400)
lambda_imag = np.linspace(-10, 10, 400)

Lambda_real, Lambda_imag = np.meshgrid(lambda_real, lambda_imag)
Lambda = Lambda_real + 1j*Lambda_imag

# Euler method stability condition |1 + lambda*dt| < 1
dt = .1  # time step size
stability_condition = np.abs(1 + Lambda * dt) < 1

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(Lambda_real, Lambda_imag, stability_condition, levels=[0,1], colors=['white', 'blue'])
plt.title('Stability Region for Euler Method')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.grid(True)
plt.show()
