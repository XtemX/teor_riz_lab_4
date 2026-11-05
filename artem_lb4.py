import numpy as np
from scipy.optimize import minimize

m = np.array([0.20, 0.40, 0.60])
sigma = np.array([0.10, 0.18, 0.30])
rho = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])

cov = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        cov[i, j] = sigma[i] * sigma[j] * rho[i, j]


def objective(x):
    return -np.dot(x, m)


def constraint_risk(x):
    return np.dot(x, np.dot(cov, x))


def constraint_sum(x):
    return np.sum(x) - 1.0


x0 = [0.33, 0.33, 0.34]
bounds = [(0, 1), (0, 1), (0, 1)]
constraints = [
    {"type": "eq", "fun": constraint_sum},
    {"type": "eq", "fun": constraint_risk},
]

res = minimize(
    objective, x0, method="SLSQP", bounds=bounds, constraints=constraints
)

print("--- ОПТИМАЛЬНА СТРУКТУРА ПОРТФЕЛЯ ---")
print(f"Частка x1 (A1): {res.x[0]:.4f} ({res.x[0]*100:.2f}%)")
print(f"Частка x2 (A2): {res.x[1]:.4f} ({res.x[1]*100:.2f}%)")
print(f"Частка x3 (A3): {res.x[2]:.4f} ({res.x[2]*100:.2f}%)")
print("\n--- ХАРАКТЕРИСТИКИ ПОРТФЕЛЯ ---")
print(f"Сподівана норма прибутку m_pi: {-res.fun*100:.2f}%")
print(f"Оцінка ризику sigma_pi: {np.sqrt(constraint_risk(res.x))*100:.4f}%")