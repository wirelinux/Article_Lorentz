# Spyder (Python3) – Code 02
# Espectro dos Expoentes de Lyapunov do Sistema de Lorenz
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from numba import njit
# Parâmetros
param_r = np.linspace(0, 250, 5000)
sigma = 10.0
b = 8/3
CI = np.array([0.0, 1.0, 0.0])  # Garantir tipo float64
delta = 0.01
t_final = 2000
nits = int(t_final/delta)  # Pré-calcular o número de iterações
# Função de Lorenz compilada com Numba
@njit(nogil=True, fastmath=True)
def lorenz(t, X, sigma, r, b):
    x, y, z = X
    dxdt = sigma*(y - x)
    dydt = x*(r - z) - y
    dzdt = x*y - b*z
    return np.array([dxdt, dydt, dzdt])
# Método RK4 otimizado com Numba
@njit(nogil=True, fastmath=True)
def RK4(f, t, X, dt, *args):
    k1 = f(t, X, *args)
    k2 = f(t + dt*0.5, X + dt*0.5*k1, *args)
    k3 = f(t + dt*0.5, X + dt*0.5*k2, *args)
    k4 = f(t + dt, X + dt * k3, *args)
    return X + dt*(k1 + 2*k2 + 2*k3 + k4)/6
# Derivada da matriz Y (para o cálculo do expoente de Lyapunov)
@njit(nogil=True, fastmath=True)
def Y_derivada(t, Y_flat, Jac):
    Y = Y_flat.reshape(3, 3)
    dYdt = Jac @ Y
    return dYdt.flatten()
# Jacobiana atualizada (evitar alocações)
@njit(nogil=True, fastmath=True)
def update_jacobian(Jac, x, y, z, r):
    Jac[0, 0], Jac[0, 1], Jac[0, 2] = -sigma, sigma, 0.0
    Jac[1, 0], Jac[1, 1], Jac[1, 2] = r - z, -1.0, -x
    Jac[2, 0], Jac[2, 1], Jac[2, 2] = y, x, -b
# Cálculo da trajetória e expoentes de Lyapunov
@njit(nogil=True, fastmath=True)
def trajetoria_lyap(r):
    acum = np.zeros(3)
    y = CI.copy()
    Y = np.eye(3) # Matriz identidade
    Jac = np.zeros((3, 3))

    # Transiente inicial - descartar primeiras iterações
    for _ in range(1000):
        y = RK4(lorenz, 0, y, delta, sigma, r, b)
    # Cálculo dos expoentes
    for _ in range(nits):
        y = RK4(lorenz, 0, y, delta, sigma, r, b)
        x, y_val, z = y
        update_jacobian(Jac, x, y_val, z, r)
        Y = RK4(Y_derivada, 0, Y.flatten(), delta, Jac).reshape(3, 3)
        Q, R = np.linalg.qr(Y)
        acum += np.log(np.abs(np.diag(R)))
        Y = Q
    return acum/(nits*delta)
# Processamento paralelo e plotagem
if __name__ == '__main__':
    start_time = time.time()
    # Para ambientes que não suportam multiprocessing, use esta versão serial:
    lyap_espectro = []
    for r in param_r:
        lyap_espectro.append(trajetoria_lyap(r))
    lyap_espectro = np.array(lyap_espectro)
    plt.figure(figsize=(12, 6))
    plt.plot(param_r, lyap_espectro[:, 0], 'b', label='$λ_1$', linewidth=0.8)
    plt.plot(param_r, lyap_espectro[:, 1], 'r', label='$λ_2$', linewidth=0.8)
    plt.plot(param_r, lyap_espectro[:, 2], 'g', label='$λ_3$', linewidth=0.8)
    plt.xlabel('$r$', fontsize=14)
    plt.ylabel('$λ_i$', fontsize=14)
    plt.legend(loc='center right')
    plt.ylim(-16, 2.5)
    plt.grid(True, alpha=0.3)
    plt.savefig('fig_2_exp.png', dpi=300)
    plt.tight_layout()
    plt.show()
    print(f"Tempo de execução: {time.time() - start_time:.2f} segundos")