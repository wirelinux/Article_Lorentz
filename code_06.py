# Spyder (Python3) – Code 06
# Poincaré Mapping in the xy-plane
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
# Parâmetros do sistema
sigma = 10.0
r = 200
b = 8/3
dt = 0.005
t_total = 50
transiente = 30
# Condições iniciais
CIs = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]

@jit(nopython=True)
def lorenz(X):
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (r - z) - y
    dzdt = x * y - b * z
    return np.array([dxdt, dydt, dzdt])
@jit(nopython=True)
def RK4(X):
    k1 = lorenz(X)
    k2 = lorenz(X + 0.5*dt*k1)
    k3 = lorenz(X + 0.5*dt*k2)
    k4 = lorenz(X + dt*k3)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
@jit(nopython=True)
def simular(CI):
    X = np.copy(CI)
    z_plano = r - 1
    max_points = int((t_total - transiente)/dt)
    pontos_cruzamento = np.zeros((max_points, 3))
    pontos_contagem = 0

    # Fase transiente
    for _ in range(int(transiente/dt)):
        X = RK4(X)
    # Simulação principal
    trajetoria = np.zeros((int((t_total - transiente)/dt), 3))
    z_prev = X[2]
    for i in range(trajetoria.shape[0]):
        X = RK4(X)
        trajetoria[i] = X
        # Detectar cruzamentos e interpolação
        z_atual = X[2]
        if (z_prev < z_plano and z_atual >= z_plano) or (z_prev > z_plano and z_atual <= z_plano):
            alpha = (z_plano - z_prev)/(z_atual - z_prev)
            x_interp = trajetoria[i-1,0] + alpha*(X[0] - trajetoria[i-1,0])
            y_interp = trajetoria[i-1,1] + alpha*(X[1] - trajetoria[i-1,1])
            pontos_cruzamento[pontos_contagem] = np.array([x_interp, y_interp, z_plano])
            pontos_contagem += 1
        z_prev = z_atual
    return pontos_cruzamento[:pontos_contagem]
# Simular e plotar
resultados = [simular(CI) for CI in CIs]
cores = ['darkblue', 'crimson']

plt.figure(figsize=(7, 4))
for i, pontos in enumerate(resultados):
    if len(pontos) > 0:
        plt.scatter(pontos[:,0], pontos[:,1], s=3, alpha=0.9, color=cores[i], label=f'C.I. {i+1}')
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.legend()
plt.xlim(-60, 50)
plt.ylim(-120, 120)
plt.grid(True)
plt.tight_layout()
plt.legend(loc='upper left')
plt.savefig('fig_6.png', dpi=300)
plt.show()