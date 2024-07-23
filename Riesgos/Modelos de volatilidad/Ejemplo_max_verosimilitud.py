import numpy as np
from scipy.optimize import minimize

# Definir la función de verosimilitud
def likelihood(parameters):
    # Calcular la verosimilitud (aquí sería tu función de verosimilitud real)
    mu, sigma = parameters
    ll = -np.sum(np.log(np.exp(-(data - mu)**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))))
    return ll

# Datos de ejemplo
data = np.random.normal(loc=5, scale=2, size=100)

# Estimación de parámetros
initial_guess = [0, 1]  # Valores iniciales para mu y sigma
result = minimize(likelihood, initial_guess, method='Nelder-Mead')

# Imprimir resultados
print("Parámetros estimados:", result.x)
print("Log-verosimilitud:", -result.fun)