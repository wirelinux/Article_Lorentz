# Spyder (Python3) – Code 01
# Bifurcation Diagram in Lorenz
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
# Sistema de Lorenz e Integração Numérica RK4
@jit(nopython=True)
def lorenz(X, r):
    x, y, z = X
    dxdt = 10.0*(y - x)
    dydt = r*x - y - x*z
    dzdt = x*y - (8.0/3.0)*z
    return np.array([dxdt, dydt, dzdt])
@jit(nopython=True)
def rk4_step(f, X, dt, r):
    k1 = f(X, r)
    k2 = f(X + 0.5*dt*k1, r)
    k3 = f(X + 0.5*dt*k2, r)
    k4 = f(X + dt*k3, r)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
# Simulação com Detecção de Poincaré
@jit(nopython=True)
def simulate(r, dt=0.01, t_final=100):
    t_transiente = 80
    CIs = [np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0])]
    poincare = []
    z_plano = r - 1
    for estado_inicial in CIs:
        estado = estado_inicial.copy()
        t = 0.0
        z_prev = estado[2]
        for _ in range(int(t_final/dt)):
            estado_anterior = estado.copy()
            estado = rk4_step(lorenz, estado, dt, r)
            t += dt
            if t > t_transiente:
                z_atual = estado[2]
                if (z_prev < z_plano <= z_atual) or (z_prev > z_plano >= z_atual):
                    alpha = (z_plano - z_prev)/(z_atual - z_prev)
                    x_interp = estado_anterior[0] + alpha*(estado[0] - estado_anterior[0])
                    poincare.append(x_interp)
                z_prev = z_atual
    return r, np.array(poincare)
# Função Principal Paralelizada
def main():
    start_time = time.time()
    # Definir r_val com resolução adaptativa
    r_val = np.concatenate([np.arange(0, 14, 0.001),np.arange(14, 23, 0.1),
        np.arange(23, 250, 0.025)])
    # Paralelização
    with Pool() as pool:
        results = pool.map(simulate, r_val)
    r_plot, x_plot = [], []
    for r, poincare in results:
        r_plot.extend([r] * len(poincare))
        x_plot.extend(poincare)
    # Plot
    plt.figure(figsize=(12, 6))
    plt.scatter(r_plot, x_plot, s=0.2, alpha=0.01, color='black')
    #plt.xlabel('$r$', fontsize=14)
    plt.ylabel('$x$', fontsize=14)
    plt.ylim(-60,60)
    plt.grid(True)
    plt.savefig('fig_2_diag.png', dpi=300)
    plt.show()
    print(f"Tempo total: {time.time() - start_time:.2f} segundos")
if __name__ == "__main__":
    main()