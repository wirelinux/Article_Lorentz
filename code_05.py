# Spyder (Python3) – Code 05
# Poincaré section in the z = 199 plane
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
# Parâmetros do sistema
sigma = 10.0
r = 200
b = 8/3
dt = 0.001
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
    # Pré-alocar arrays
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
        if (z_prev < z_plano and z_atual >= z_plano) or (z_prev > z_plano
                                                         and z_atual <= z_plano):
            alpha = (z_plano - z_prev)/(z_atual - z_prev)
            x_interp = trajetoria[i-1,0] + alpha*(X[0] - trajetoria[i-1,0])
            y_interp = trajetoria[i-1,1] + alpha*(X[1] - trajetoria[i-1,1])
            pontos_cruzamento[pontos_contagem] = np.array([x_interp,
                                                           y_interp, z_plano])
            pontos_contagem += 1
        z_prev = z_atual
    return trajetoria, pontos_cruzamento[:pontos_contagem]
resultados = [simular(CI) for CI in CIs]
fig = plt.figure(figsize=(12, 8))
# Subplot 1: Trajetórias no atrator de Lorenz
ax1 = fig.add_subplot(121, projection='3d')
cores = ['darkblue', 'crimson']
# Subplot 2: Pontos de cruzamento
ax2 = fig.add_subplot(122, projection='3d')
# Configuração do plano para ambos os subplots
z_plano = r - 1
x_val = np.linspace(-60, 60, 60)
y_val = np.linspace(-120, 120, 120)
Xs, Ys = np.meshgrid(x_val, y_val)
Zp = z_plano*np.ones_like(Xs)
for i, (traj, pontos) in enumerate(resultados):
    # Plotar trajetórias
    ax1.plot(traj[:,0], traj[:,1], traj[:,2],
            lw=1, alpha=0.6, color=cores[i], label=f'C.I. {i+1}')
    ax1.view_init(elev=20, azim=110)
    # Plotar pontos de cruzamento originais
    if len(pontos) > 0:
        ax2.scatter(pontos[:,0], pontos[:,1], pontos[:,2],
                   s=3, alpha=0.9, color=cores[i], label=f'C.I. {i+1}')
    ax2.view_init(elev=20, azim=110)
# Configurações visuais comuns
for ax in [ax1, ax2]:
    ax.plot_surface(Xs, Ys, Zp, alpha=0.3, color='gray')
    ax.set_xlabel('$x$', fontsize=14)
    ax.set_ylabel('$y$', fontsize=14)
    ax.set_zlabel('$z$', fontsize=14)
    ax.legend(loc=(0.15,0.30),frameon=0)
    ax.set_xlim(-60, 60)
    ax.set_ylim(-120, 120)
    ax.set_zlim(100, 300)
plt.tight_layout()
plt.savefig('fig_5.png', dpi=300)
plt.show()