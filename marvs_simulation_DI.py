import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# SEÇÃO 1: DEFINIÇÃO DO SISTEMA, PARAMETRIZAÇÃO E CONFIGURAÇÃO DO AMBIENTE
# =============================================================================

# Parâmetros Geométricos do Manipulador
L1 = 0.0935
L2 = 0.09
H_INITIAL = 0.16

# Parâmetros da Trajetória e Temporização
V_MAX_XY, A_MAX_XY, TF_XY = 0.05, 0.05, 2.414
V_MAX_Z, A_MAX_Z, TF_Z = 0.1, 0.05, 7.5

# Ganhos PD (Matrizes Diagonais) - PARA CONTROLE POR DINÂMICA INVERSA
Kp = np.diag([25, 25, 25])
Kd = np.diag([10, 10, 10])

# Limites de Hardware
TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3 = 0.294, 0.294, 29.4

# Parâmetros Dinâmicos
params = {
    'M1': 0.1253, 'M2': 0.1464, 'M3': 0.2037,
    'I1': 1.57e-4, 'I2': 1.67e-4, 'I3': 6.78e-6,
    'L1': L1, 'L2': L2, 'LC1': 0.067, 'LC2': 0.045, 'G': 9.81
}

# Parâmetros de Simulação
T_TOTAL = TF_XY + TF_Z
DT = 0.01

# Pontos da Trajetória
P_INICIAL = np.array([0.1, 0.1, 0.15])
P_INTER = np.array([0.15, 0.05, 0.15])
P_FINAL = np.array([0.15, 0.05, -0.40])

# =============================================================================
# SEÇÃO 2: GERAÇÃO DA TRAJETÓRIA E CINEMÁTICA
# =============================================================================

def generate_trapezoidal_profile(dist, vmax, amax, t_vec):
    t_accel = vmax / amax
    d_accel = 0.5 * amax * t_accel**2
    if dist > 2 * d_accel:
        t_final = dist / vmax + vmax / amax
        t_const = t_final - 2 * t_accel
    else:
        t_accel = np.sqrt(dist / amax)
        t_const = 0
        t_final = 2 * t_accel
        vmax = amax * t_accel
    s, s_dot, s_ddot = np.zeros_like(t_vec), np.zeros_like(t_vec), np.zeros_like(t_vec)
    for i, t in enumerate(t_vec):
        if t <= t_accel:
            s[i], s_dot[i], s_ddot[i] = 0.5 * amax * t**2, amax * t, amax
        elif t <= t_accel + t_const:
            s[i], s_dot[i], s_ddot[i] = d_accel + vmax * (t - t_accel), vmax, 0
        elif t <= t_final:
            t_dec = t - t_accel - t_const
            s[i] = d_accel + vmax * t_const + vmax * t_dec - 0.5 * amax * t_dec**2
            s_dot[i] = vmax - amax * t_dec
            s_ddot[i] = -amax
        else:
            s[i], s_dot[i], s_ddot[i] = dist, 0, 0
    return s, s_dot, s_ddot

def generate_cartesian_trajectory():
    t = np.arange(0, T_TOTAL, DT)
    p_d, pd_d, pdd_d = np.zeros((len(t), 3)), np.zeros((len(t), 3)), np.zeros((len(t), 3))
    
    t_xy = np.arange(0, TF_XY, DT)
    dist_xy = np.linalg.norm(P_INTER[:2] - P_INICIAL[:2])
    s_xy, sd_xy, sdd_xy = generate_trapezoidal_profile(dist_xy, V_MAX_XY, A_MAX_XY, t_xy)
    dir_xy = (P_INTER - P_INICIAL) / np.linalg.norm(P_INTER - P_INICIAL)
    p_d[:len(t_xy)] = P_INICIAL + s_xy[:, np.newaxis] * dir_xy
    pd_d[:len(t_xy)] = sd_xy[:, np.newaxis] * dir_xy
    pdd_d[:len(t_xy)] = sdd_xy[:, np.newaxis] * dir_xy

    t_z = np.arange(0, TF_Z, DT)
    dist_z = abs(P_FINAL[2] - P_INTER[2])
    s_z, sd_z, sdd_z = generate_trapezoidal_profile(dist_z, V_MAX_Z, A_MAX_Z, t_z)
    start_idx = len(t_xy)
    end_idx = start_idx + len(t_z)
    if end_idx > len(t): end_idx = len(t)
    
    p_d[start_idx:end_idx] = P_INTER + np.array([0, 0, -1]) * s_z[:end_idx-start_idx, np.newaxis]
    pd_d[start_idx:end_idx] = np.array([0, 0, -1]) * sd_z[:end_idx-start_idx, np.newaxis]
    pdd_d[start_idx:end_idx] = np.array([0, 0, -1]) * sdd_z[:end_idx-start_idx, np.newaxis]
    
    if end_idx < len(t):
        p_d[end_idx:] = P_FINAL
    return t, p_d, pd_d, pdd_d

def inverse_kinematics(P):
    Px, Py, Pz = P
    cos_theta2 = np.clip((Px**2 + Py**2 - L1**2 - L2**2) / (2 * L1 * L2), -1, 1)
    theta2 = np.arccos(cos_theta2)
    k1 = L1 + L2 * np.cos(theta2)
    k2 = L2 * np.sin(theta2)
    theta1 = np.arctan2(Py, Px) - np.arctan2(k2, k1)
    d3 = H_INITIAL - Pz
    return np.array([theta1, theta2, d3])

def calculate_jacobian(q):
    theta1, theta2, _ = q
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)
    return np.array([
        [-L1*s1 - L2*s12, -L2*s12, 0],
        [L1*c1 + L2*c12, L2*c12, 0],
        [0, 0, -1]
    ])

def calculate_jacobian_derivative(q, qd):
    """
    Calcula a derivada da matriz Jacobiana no tempo (VERSÃO FINAL CORRIGIDA)
    """
    theta1, theta2, _ = q
    theta1_dot, theta2_dot, _ = qd
    
    # Parâmetros Geométricos (acessados do escopo global)
    # L1, L2
    
    # Calcular termos trigonométricos
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)
    
    # Derivadas dos termos trigonométricos
    s1_dot = c1 * theta1_dot
    c1_dot = -s1 * theta1_dot
    s12_dot = c12 * (theta1_dot + theta2_dot)
    c12_dot = -s12 * (theta1_dot + theta2_dot)
    
    # Construir derivada da matriz Jacobiana usando as variáveis corretas
    J_dot = np.array([
        [-L1 * s1_dot - L2 * s12_dot, -L2 * s12_dot, 0],
        [ L1 * c1_dot + L2 * c12_dot,  L2 * c12_dot, 0],
        [0, 0, 0]
    ])
    
    return J_dot

def convert_cartesian_to_joint_trajectory(t, p_d, pd_d, pdd_d):
    q_d, qd_d, qdd_d = np.zeros_like(p_d), np.zeros_like(pd_d), np.zeros_like(pdd_d)
    for i in range(len(t)):
        q_d[i] = inverse_kinematics(p_d[i])
        J = calculate_jacobian(q_d[i])
        try:
            J_inv = np.linalg.pinv(J)
            qd_d[i] = J_inv @ pd_d[i]
            J_dot = calculate_jacobian_derivative(q_d[i], qd_d[i])
            qdd_d[i] = J_inv @ (pdd_d[i] - J_dot @ qd_d[i])
        except np.linalg.LinAlgError:
            if i > 0: qd_d[i], qdd_d[i] = qd_d[i-1], qdd_d[i-1]
    return q_d, qd_d, qdd_d

# =============================================================================
# SEÇÃO 3: FUNÇÕES DE DINÂMICA
# =============================================================================

def calculate_inertia_matrix(q, p):
    _, th2, _ = q
    c2 = np.cos(th2)
    b11 = p['M1']*p['LC1']**2 + p['I1'] + p['M2']*(p['L1']**2 + p['LC2']**2) + p['M3']*(p['L1']**2 + p['L2']**2) + p['I2'] + p['I3'] + 2*(p['M2']*p['L1']*p['LC2'] + p['M3']*p['L1']*p['L2'])*c2
    b12 = p['M2']*p['LC2']**2 + p['M3']*p['L2']**2 + p['I2'] + p['I3'] + (p['M2']*p['L1']*p['LC2'] + p['M3']*p['L1']*p['L2'])*c2
    b22 = p['M2']*p['LC2']**2 + p['M3']*p['L2']**2 + p['I2'] + p['I3']
    return np.array([[b11, b12, 0], [b12, b22, 0], [0, 0, p['M3']]])

def calculate_coriolis_matrix(q, qd, p):
    _, th2, _ = q
    th1_d, th2_d, _ = qd
    s2 = np.sin(th2)
    h = -(p['M2']*p['L1']*p['LC2'] + p['M3']*p['L1']*p['L2']) * s2
    return np.array([[h*th2_d, h*(th1_d + th2_d), 0], [-h*th1_d, 0, 0], [0, 0, 0]])

def calculate_gravity_vector(q, p):
    return np.array([0, 0, p['M3'] * p['G']])

# =============================================================================
# SEÇÃO 4: SIMULAÇÃO DINÂMICA
# =============================================================================

def system_dynamics(t, y, trajectory_data, params, Kp_m, Kd_m):
    q_obt, qd_obt = y[:3], y[3:]
    t_traj, q_d_traj, qd_d_traj, qdd_d_traj = trajectory_data
    
    idx = min(int(t / DT), len(t_traj) - 1)
    q_d, qd_d, qdd_d = q_d_traj[idx], qd_d_traj[idx], qdd_d_traj[idx]

    # --- LEI DE CONTROLE: DINÂMICA INVERSA ---
    error_q = q_d - q_obt
    error_qd = qd_d - qd_obt
    y_cmd = qdd_d + Kd_m @ error_qd + Kp_m @ error_q
    
    B = calculate_inertia_matrix(q_obt, params)
    C = calculate_coriolis_matrix(q_obt, qd_obt, params)
    g = calculate_gravity_vector(q_obt, params)
    
    tau = B @ y_cmd + C @ qd_obt + g
    tau = np.clip(tau, [-TAU_MAX_J1, -TAU_MAX_J2, -TAU_MAX_J3], [TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3])
    
    # --- SIMULAÇÃO DA PLANTA REAL ---
    B_inv = np.linalg.pinv(B)
    qdd_obt = B_inv @ (tau - C @ qd_obt - g)
    
    return np.concatenate([qd_obt, qdd_obt])

def run_simulation_and_plot():
    print("Iniciando simulação do controlador por Dinâmica Inversa...")
    t, p_d, pd_d, pdd_d = generate_cartesian_trajectory()
    q_d, qd_d, qdd_d = convert_cartesian_to_joint_trajectory(t, p_d, pd_d, pdd_d)
    
    y0 = np.concatenate([q_d[0], np.zeros(3)])
    
    print("Executando simulação dinâmica...")
    solution = solve_ivp(
        fun=system_dynamics, t_span=[0, T_TOTAL], y0=y0, method='RK45',
        t_eval=t, args=((t, q_d, qd_d, qdd_d), params, Kp, Kd), rtol=1e-5, atol=1e-8
    )
    
    print("Simulação concluída. Gerando gráficos...")
    t_sim, y_sim = solution.t, solution.y
    q_obt, qd_obt = y_sim[:3, :].T, y_sim[3:, :].T
    
    # Recalcular histórico de torques para plotagem
    tau_history = []
    for i in range(len(t_sim)):
        error_q = q_d[i] - q_obt[i]
        error_qd = qd_d[i] - qd_obt[i]
        y_cmd = qdd_d[i] + Kd @ error_qd + Kp @ error_q
        B = calculate_inertia_matrix(q_obt[i], params)
        C = calculate_coriolis_matrix(q_obt[i], qd_obt[i], params)
        g = calculate_gravity_vector(q_obt[i], params)
        tau = B @ y_cmd + C @ qd_obt[i] + g
        tau = np.clip(tau, [-TAU_MAX_J1, -TAU_MAX_J2, -TAU_MAX_J3], [TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3])
        tau_history.append(tau)
    tau_history = np.array(tau_history)
    
    plot_trajectory_tracking(t_sim, q_d, q_obt)
    plot_error_evolution(t_sim, q_d, q_obt)
    plot_control_signals(t_sim, tau_history)
    analyze_performance(t_sim, q_d, q_obt, tau_history)

# =============================================================================
# SEÇÃO 5: FUNÇÕES DE PLOTAGEM
# =============================================================================
def plot_trajectory_tracking(t, q_d, q_obt):
    labels = [r'Junta 1 ($\theta_1$)', r'Junta 2 ($\theta_2$)', r'Junta 3 ($d_3$)']
    units = ['rad', 'rad', 'm']
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Seguimento da Trajetória - Posição Desejada vs. Obtida', fontsize=16, weight='bold')
    for i in range(3):
        axes[i].plot(t, q_d[:, i], 'b-', lw=2, label='Desejada')
        axes[i].plot(t, q_obt[:, i], 'r--', lw=2, label='Obtida')
        axes[i].set_title(labels[i])
        axes[i].set_ylabel(f'Posição ({units[i]})')
        axes[i].legend()
        axes[i].grid(alpha=0.6)
    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_error_evolution(t, q_d, q_obt):
    labels = [r'Junta 1 ($\theta_1$)', r'Junta 2 ($\theta_2$)', r'Junta 3 ($d_3$)']
    units = ['rad', 'rad', 'm']
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Evolução do Erro de Posição', fontsize=16, weight='bold')
    for i in range(3):
        error = q_d[:, i] - q_obt[:, i]
        axes[i].plot(t, error, 'g-', lw=2)
        axes[i].set_title(f'Erro - {labels[i]}')
        axes[i].set_ylabel(f'Erro ({units[i]})')
        axes[i].grid(alpha=0.6)
        axes[i].axhline(0, color='k', linestyle=':', alpha=0.8)
    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_control_signals(t, tau_history):
    labels = [r'Junta 1 ($\theta_1$)', r'Junta 2 ($\theta_2$)', r'Junta 3 ($d_3$)']
    units = ['Nm', 'Nm', 'N']
    limits = [TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3]
    fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Sinais de Controle (Torques/Força)', fontsize=16, weight='bold')
    for i in range(3):
        axes[i].plot(t, tau_history[:, i], 'b-', lw=2, label='Torque/Força Aplicado')
        axes[i].axhline(limits[i], color='r', ls='--', label=f'Limite ({limits[i]:.2f})')
        axes[i].axhline(-limits[i], color='r', ls='--')
        axes[i].set_title(labels[i])
        axes[i].set_ylabel(f'Torque/Força ({units[i]})')
        axes[i].legend()
        axes[i].grid(alpha=0.6)
    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def analyze_performance(t, q_d, q_obt, tau_history):
    print("\n" + "="*60 + "\nANÁLISE DE DESEMPENHO\n" + "="*60)
    labels = ['Junta 1 (θ₁)', 'Junta 2 (θ₂)', 'Junta 3 (d₃)']
    for i in range(3):
        error = q_d[:, i] - q_obt[:, i]
        print(f"\n{labels[i]}:\n  - RMSE: {np.sqrt(np.mean(error**2)):.6f}\n  - Erro Máximo: {np.max(np.abs(error)):.6f}")

if __name__ == "__main__":
    run_simulation_and_plot()