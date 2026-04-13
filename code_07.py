# Spyder (Python3) – Code 07
#  Poincaré Section and Lorenz (Poincaré) Map
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
# Parâmetros do sistema de Lorenz
sigma, r, b = 10.0, 28, 8/3
dt, t_total, transiente = 0.005, 100, 0
CI = np.array([0.0, 1.0, 0.0])
@jit(nopython=True)
def lorenz(X):
    x, y, z = X
    return np.array([sigma*(y - x), x*(r - z) - y, x*y - b*z])
@jit(nopython=True)
def RK4(X):
    k1 = lorenz(X)
    k2 = lorenz(X + 0.5*dt*k1)
    k3 = lorenz(X + 0.5*dt*k2)
    k4 = lorenz(X + dt*k3)
    return X + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
def simular_maximos_z():
    X = np.copy(CI)
    n = int(t_total/dt)
    trajetoria = np.empty((n, 3))
    Z_max, t, z_prev, dz_prev = [], 0, X[2], lorenz(X)[2]
    crossing_points = []
    for i in range(n):
        X_prev = np.copy(X)
        X = RK4(X)
        trajetoria[i] = X
        t += dt
        if t < transiente:
            continue
        dz_atual = lorenz(X)[2]
        if dz_prev > 0 and dz_atual <= 0 and np.abs(
            dz_atual - dz_prev) > 1e-10:
            alpha = -dz_prev/(dz_atual - dz_prev)
            z_interp = z_prev + alpha*(X[2] - z_prev)
            x_interp = X_prev[0] + alpha*(X[0] - X_prev[0])
            y_interp = X_prev[1] + alpha*(X[1] - X_prev[1])
            Z_max.append(z_interp)
            crossing_points.append([x_interp, y_interp, z_interp])
        z_prev, dz_prev = X[2], dz_atual
    return trajetoria[transiente:], np.array(
        Z_max), np.array(crossing_points)
trajetoria, Z_max, crossing_points = simular_maximos_z()
fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot(trajetoria[:, 0], trajetoria[:, 1], trajetoria[:, 2],
         lw=0.6, alpha=0.6, color='darkblue')
# Separação das asas por sinal de X e construção das superfícies
if len(crossing_points) > 0:
    # Filtrar pontos por asa
    right_wing = crossing_points[crossing_points[:,0] > 0]
    left_wing = crossing_points[crossing_points[:,0] < 0]
    # Função para criar superfícies adaptativas
    def create_poincare_surface(points, ax, color):
        if len(points) < 2:
            return
        x_min, x_max = points[:,0].min(), points[:,0].max()
        y_min, y_max = points[:,1].min(), points[:,1].max()
        # Grid adaptativo baseado nos pontos coletados
        x_grid = np.linspace(x_min - 2, x_max + 2, 15)
        y_grid = np.linspace(y_min - 2, y_max + 2, 15)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = (X * Y)/b  # Superfície z = xy/b
        ax.plot_surface(X, Y, Z, color=color, alpha=0.4,
                        edgecolor='none', zorder=2)
    # Criar superfícies para cada asa
    create_poincare_surface(right_wing, ax1, 'gray')
    create_poincare_surface(left_wing, ax1, 'gray')
    # Plotar pontos de cruzamento
    ax1.scatter(crossing_points[:,0], crossing_points[:,1],
                crossing_points[:,2], marker='o', s=2,
                color='k', alpha=1, zorder=3)
ax1.set_xlabel('$x$', fontsize=14)
ax1.set_ylabel('$y$', fontsize=14)
ax1.set_zlabel('$z$', fontsize=14)
ax1.view_init(elev=5, azim=300)
# Mapa de retorno
ax2 = fig.add_subplot(122)
if len(Z_max) > 1:
    ax2.scatter(Z_max[:-1], Z_max[1:], s=15, color='crimson', alpha=0.8)
    ax2.plot([25, 50], [25, 50], 'k--', lw=1, label='$Z_{n+1} = Z_n$')
    ax2.set_xlabel('$Z_n$', fontsize=14)
    ax2.set_ylabel('$Z_{n+1}$', fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.9)
plt.tight_layout()
plt.savefig('fig_7.png', dpi=300)
plt.show()