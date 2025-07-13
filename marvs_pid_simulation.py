import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =============================================================================
# SEÇÃO 1: DEFINIÇÃO DO SISTEMA, PARAMETRIZAÇÃO E CONFIGURAÇÃO DO AMBIENTE
# =============================================================================

# Parâmetros Geométricos do Manipulador
L1 = 0.0935  # m - Comprimento do Elo 1
L2 = 0.09    # m - Comprimento do Elo 2
H_INITIAL = 0.16  # m - Altura inicial da junta prismática

# Parâmetros da Trajetória e Temporização
V_MAX_XY = 0.05   # m/s - Velocidade máxima segmento XY
A_MAX_XY = 0.05   # m/s² - Aceleração máxima segmento XY
TF_XY = 2.414     # s - Tempo total segmento XY

V_MAX_Z = 0.1     # m/s - Velocidade máxima segmento Z
A_MAX_Z = 0.05    # m/s² - Aceleração máxima segmento Z
TF_Z = 7.5        # s - Tempo total segmento Z

# Ganhos PID (Matrizes Diagonais) - CORRIGIDOS PARA ESTABILIDADE
Kp = np.diag([32.78, 9.33, 896.28])
Ki = np.diag([298.0, 84.8, 8148])
Kd = np.diag([1.043, 0.297, 28.52])
# Limites de Hardware
TAU_MAX_J1 = 0.294  # Nm - Torque máximo junta 1
TAU_MAX_J2 = 0.294  # Nm - Torque máximo junta 2
TAU_MAX_J3 = 29.4   # N - Força máxima junta 3

# Parâmetros Dinâmicos - CONFORME RELATÓRIO
M1 = 0.1253   # kg - Massa do elo 1
M2 = 0.1464   # kg - Massa do elo 2
M3 = 0.2037   # kg - Massa do elo 3
I1 = 1.57e-4  # kg·m² - Momento de inércia do elo 1
I2 = 1.67e-4  # kg·m² - Momento de inércia do elo 2
I3 = 6.78e-6  # kg·m² - Momento de inércia do elo 3
LC1 = 0.067   # m - Distância ao centro de massa do elo 1
LC2 = 0.045   # m - Distância ao centro de massa do elo 2
G = 9.81      # m/s² - Aceleração da gravidade

# Parâmetros de Simulação
T_TOTAL = TF_XY + TF_Z  # s - Tempo total da simulação
DT = 0.01               # s - Passo de tempo

# Pontos da Trajetória
P_INICIAL = np.array([0.1, 0.1, 0.15])
P_INTER = np.array([0.15, 0.05, 0.15])
P_FINAL = np.array([0.15, 0.05, -0.40])

print("Configuração do ambiente e parâmetros do sistema MARVS carregados com sucesso!")
print(f"Tempo total de simulação: {T_TOTAL:.2f} s")
print(f"Ganhos PID (ajustados para estabilidade): Kp diagonal = {np.diag(Kp)}")

# =============================================================================
# SEÇÃO 2: GERAÇÃO DA TRAJETÓRIA DESEJADA
# =============================================================================

def generate_trapezoidal_profile(total_distance, max_velocity, max_acceleration, time_vector):
    """
    Gera um perfil de velocidade trapezoidal
    """
    # Calcular tempo de aceleração
    t_accel = max_velocity / max_acceleration
    
    # Calcular distância durante aceleração
    d_accel = 0.5 * max_acceleration * t_accel**2
    
    # Verificar se é perfil trapezoidal ou triangular
    if total_distance > 2 * d_accel:
        # Perfil trapezoidal
        t_final = total_distance / max_velocity + max_velocity / max_acceleration
        t_const = t_final - 2 * t_accel
    else:
        # Perfil triangular
        t_accel = np.sqrt(total_distance / max_acceleration)
        t_const = 0
        t_final = 2 * t_accel
        max_velocity = max_acceleration * t_accel
    
    # Inicializar vetores
    s_t = np.zeros_like(time_vector)
    s_dot_t = np.zeros_like(time_vector)
    s_ddot_t = np.zeros_like(time_vector)
    
    for i, t in enumerate(time_vector):
        if t <= t_accel:
            # Fase de aceleração
            s_t[i] = 0.5 * max_acceleration * t**2
            s_dot_t[i] = max_acceleration * t
            s_ddot_t[i] = max_acceleration
        elif t <= t_accel + t_const:
            # Fase de velocidade constante
            s_t[i] = 0.5 * max_acceleration * t_accel**2 + max_velocity * (t - t_accel)
            s_dot_t[i] = max_velocity
            s_ddot_t[i] = 0
        elif t <= t_final:
            # Fase de desaceleração
            t_dec = t - t_accel - t_const
            s_t[i] = (0.5 * max_acceleration * t_accel**2 +
                     max_velocity * t_const +
                     max_velocity * t_dec -
                     0.5 * max_acceleration * t_dec**2)
            s_dot_t[i] = max_velocity - max_acceleration * t_dec
            s_ddot_t[i] = -max_acceleration
        else:
            # Após o movimento
            s_t[i] = total_distance
            s_dot_t[i] = 0
            s_ddot_t[i] = 0
    
    return s_t, s_dot_t, s_ddot_t

def generate_cartesian_trajectory():
    """
    Gera a trajetória cartesiana desejada para os dois segmentos
    """
    t_total = np.arange(0, T_TOTAL, DT)
    p_d = np.zeros((len(t_total), 3))
    pd_d = np.zeros((len(t_total), 3))
    pdd_d = np.zeros((len(t_total), 3))
    
    # SEGMENTO 1: Movimento XY
    t_xy = np.arange(0, TF_XY, DT)
    L_xy = np.linalg.norm(P_INTER[:2] - P_INICIAL[:2])
    s_xy, s_dot_xy, s_ddot_xy = generate_trapezoidal_profile(L_xy, V_MAX_XY, A_MAX_XY, t_xy)
    direction_xy = (P_INTER - P_INICIAL) / np.linalg.norm(P_INTER - P_INICIAL)
    
    for i in range(len(t_xy)):
        p_d[i] = P_INICIAL + s_xy[i] * direction_xy
        pd_d[i] = s_dot_xy[i] * direction_xy
        pdd_d[i] = s_ddot_xy[i] * direction_xy
    
    # SEGMENTO 2: Movimento Z
    t_z = np.arange(0, TF_Z, DT)
    L_z = abs(P_FINAL[2] - P_INTER[2])
    s_z, s_dot_z, s_ddot_z = generate_trapezoidal_profile(L_z, V_MAX_Z, A_MAX_Z, t_z)
    
    start_idx = len(t_xy)
    for i in range(len(t_z)):
        if start_idx + i < len(t_total):
            p_d[start_idx + i] = P_INTER + np.array([0, 0, -s_z[i]])
            pd_d[start_idx + i] = np.array([0, 0, -s_dot_z[i]])
            pdd_d[start_idx + i] = np.array([0, 0, -s_ddot_z[i]])

    # Preencher o final da trajetória se o tempo total for maior
    final_idx = len(t_xy) + len(t_z)
    if final_idx < len(t_total):
        p_d[final_idx:] = P_FINAL
        pd_d[final_idx:] = 0
        pdd_d[final_idx:] = 0

    return t_total, p_d, pd_d, pdd_d

# =============================================================================
# SEÇÃO 2.2: CINEMÁTICA INVERSA E DERIVADAS
# =============================================================================

def inverse_kinematics(P):
    """
    Calcula a cinemática inversa para uma posição cartesiana
    """
    Px, Py, Pz = P
    
    cos_theta2 = (Px**2 + Py**2 - L1**2 - L2**2) / (2 * L1 * L2)
    
    if abs(cos_theta2) > 1: # Inalcançável
        cos_theta2 = np.clip(cos_theta2, -1, 1)

    theta2 = np.arccos(cos_theta2) # Solução "cotovelo para cima"
    
    # Calcular theta1
    s2 = np.sin(theta2)
    k1 = L1 + L2 * cos_theta2
    k2 = L2 * s2
    theta1 = np.arctan2(Py, Px) - np.arctan2(k2, k1)

    d3 = H_INITIAL - Pz
    
    return np.array([theta1, theta2, d3])

def calculate_jacobian(q):
    """
    Calcula a matriz Jacobiana analítica
    """
    theta1, theta2, _ = q
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)
    
    J = np.array([
        [-L1*s1 - L2*s12, -L2*s12, 0],
        [L1*c1 + L2*c12, L2*c12, 0],
        [0, 0, -1]
    ])
    return J

# =============================================================================
# CÓDIGO CORRIGIDO
# Substitua esta função no seu script
# =============================================================================

def calculate_jacobian_derivative(q, qd):
    """
    Calcula a derivada da matriz Jacobiana no tempo (VERSÃO CORRIGIDA)
    
    Args:
        q: Configuração da junta [theta1, theta2, d3]
        qd: Velocidades da junta [theta1_dot, theta2_dot, d3_dot]
    
    Returns:
        J_dot: Derivada da matriz Jacobiana
    """
    theta1, theta2, _ = q
    theta1_dot, theta2_dot, _ = qd
    
    # Calcular termos trigonométricos
    s1, c1 = np.sin(theta1), np.cos(theta1)
    s12, c12 = np.sin(theta1 + theta2), np.cos(theta1 + theta2)
    
    # Derivadas dos termos trigonométricos
    s1_dot = c1 * theta1_dot
    c1_dot = -s1 * theta1_dot
    s12_dot = c12 * (theta1_dot + theta2_dot)
    c12_dot = -s12 * (theta1_dot + theta2_dot)
    
    # Construir derivada da matriz Jacobiana (corrigida)
    # J_dot = d/dt(J_p)
    J_dot_linear = np.array([
        [-L1*c1_dot - L2*c12_dot, -L2*c12_dot, 0], # Derivada de [-L1*s1 - L2*s12, -L2*s12, 0]
        [L1*c1_dot + L2*c12_dot, L2*c12_dot, 0], # Derivada de [L1*c1 + L2*c12, L2*c12, 0]
        [0, 0, 0]                               # Derivada de [0, 0, -1]
    ])
    
    return J_dot_linear

def convert_cartesian_to_joint_trajectory(t, p_d, pd_d, pdd_d):
    """
    Converte trajetória cartesiana para trajetória no espaço das juntas
    """
    n_points = len(t)
    q_d = np.zeros((n_points, 3))
    qd_d = np.zeros((n_points, 3))
    qdd_d = np.zeros((n_points, 3))
    
    for i in range(n_points):
        q_d[i] = inverse_kinematics(p_d[i])
        J_linear = calculate_jacobian(q_d[i])
        
        try:
            J_inv = np.linalg.pinv(J_linear)
            qd_d[i] = J_inv @ pd_d[i]
            
            J_dot_linear = calculate_jacobian_derivative(q_d[i], qd_d[i])
            qdd_d[i] = J_inv @ (pdd_d[i] - J_dot_linear @ qd_d[i])
        except np.linalg.LinAlgError:
            if i > 0:
                qd_d[i] = qd_d[i-1]
                qdd_d[i] = qdd_d[i-1]
    
    return q_d, qd_d, qdd_d

# =============================================================================
# SEÇÃO 3: SIMULAÇÃO DINÂMICA DO SISTEMA CONTROLADO
# =============================================================================

# Dicionário com parâmetros físicos (CORRIGIDOS)
params = {
    'M1': M1, 'M2': M2, 'M3': M3,
    'I1': I1, 'I2': I2, 'I3': I3,
    'L1': L1, 'L2': L2,
    'LC1': LC1, 'LC2': LC2,
    'G': G
}

def calculate_inertia_matrix(q, params):
    """
    Calcula a matriz de inércia B(q) do manipulador
    """
    _, theta2, _ = q
    m1, m2, m3 = params['M1'], params['M2'], params['M3']
    I1, I2, I3 = params['I1'], params['I2'], params['I3']
    L1, L2 = params['L1'], params['L2']
    lc1, lc2 = params['LC1'], params['LC2']
    c2 = np.cos(theta2)
    
    b11 = (m1*lc1**2 + I1 + m2*(L1**2 + lc2**2) + m3*(L1**2 + L2**2) + I2 + I3 +
           2 * (m2*L1*lc2 + m3*L1*L2) * c2)
    b12 = m2*lc2**2 + m3*L2**2 + I2 + I3 + (m2*L1*lc2 + m3*L1*L2) * c2
    b22 = m2*lc2**2 + m3*L2**2 + I2 + I3
    b33 = m3
    
    B = np.array([[b11, b12, 0], [b12, b22, 0], [0, 0, b33]])
    return B

def calculate_coriolis_matrix(q, qd, params):
    """
    Calcula a matriz de Coriolis C(q,qd) do manipulador
    
    Args:
        q: Configuração da junta [theta1, theta2, d3]
        qd: Velocidades da junta [theta1_dot, theta2_dot, d3_dot]
        params: Dicionário com parâmetros físicos
    
    Returns:
        C: Matriz de Coriolis 3x3
    """
    _, theta2, _ = q
    theta1_dot, theta2_dot, _ = qd
    
    # Extrair parâmetros
    m2, m3 = params['M2'], params['M3']
    L1, L2 = params['L1'], params['L2']
    lc2 = params['LC2']
    s2 = np.sin(theta2)

    # Termo principal de Coriolis
    h = -(m2*L1*lc2 + m3*L1*L2) * s2
    
    # Construir matriz de Coriolis (com o sinal corrigido em C[1,0])
    C = np.array([
        [h * theta2_dot, h * (theta1_dot + theta2_dot), 0],
        [-h * theta1_dot, 0, 0], # <--- CORREÇÃO APLICADA AQUI
        [0, 0, 0]
    ])
    
    return C

def calculate_gravity_vector(q, params):
    """
    Calcula o vetor de gravidade g(q) do manipulador
    """
    # Para o SCARA, a gravidade afeta apenas a junta prismática
    g_vec = np.array([0, 0, params['M3'] * params['G']])
    return g_vec

class PIDController:
    """
    Implementação do controlador PID
    """
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)
    
    def calculate_torque(self, q_d, q_obt, qd_d, qd_obt):
        """
        Calcula o torque/força de controle
        """
        error = q_d - q_obt
        self.integral_error += error * self.dt
        
        # O termo derivativo usa o erro de velocidade para evitar "derivative kick"
        derivative_term = self.Kd @ (qd_d - qd_obt)
        
        tau = self.Kp @ error + self.Ki @ self.integral_error + derivative_term
        return tau

def system_dynamics(t, y, pid_controller, trajectory_data, params):
    """
    Define a dinâmica do sistema para integração numérica
    """
    q_obt = y[:3]
    qd_obt = y[3:]
    t_traj, q_d_traj, qd_d_traj, _ = trajectory_data
    
    t_clip = np.clip(t, t_traj[0], t_traj[-1])
    q_d = np.array([np.interp(t_clip, t_traj, q_d_traj[:, i]) for i in range(3)])
    qd_d = np.array([np.interp(t_clip, t_traj, qd_d_traj[:, i]) for i in range(3)])
    
    tau = pid_controller.calculate_torque(q_d, q_obt, qd_d, qd_obt)
    
    tau_limits = np.array([TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3])
    tau = np.clip(tau, -tau_limits, tau_limits)
    
    B = calculate_inertia_matrix(q_obt, params)
    C = calculate_coriolis_matrix(q_obt, qd_obt, params)
    g = calculate_gravity_vector(q_obt, params)
    
    try:
        B_inv = np.linalg.inv(B)
        qdd_obt = B_inv @ (tau - C @ qd_obt - g)
    except np.linalg.LinAlgError:
        B_inv = np.linalg.pinv(B)
        qdd_obt = B_inv @ (tau - C @ qd_obt - g)
    
    return np.concatenate([qd_obt, qdd_obt])

# =============================================================================
# FUNÇÃO PRINCIPAL DE SIMULAÇÃO E PLOTAGEM
# =============================================================================
def run_simulation_and_plot():
    """
    Executa a simulação e gera os gráficos de validação
    """
    print("Iniciando simulação do controlador PID do manipulador MARVS...")
    t, p_d, pd_d, pdd_d = generate_cartesian_trajectory()
    q_d, qd_d, _ = convert_cartesian_to_joint_trajectory(t, p_d, pd_d, pdd_d)
    
    pid = PIDController(Kp, Ki, Kd, DT)
    y0 = np.concatenate([q_d[0], np.zeros(3)])
    
    print("Executando simulação dinâmica...")
    solution = solve_ivp(
        fun=system_dynamics, t_span=[0, T_TOTAL], y0=y0, method='RK45',
        t_eval=t, args=(pid, (t, q_d, qd_d, None), params), rtol=1e-5, atol=1e-8
    )
    
    print("Simulação concluída. Gerando gráficos...")
    t_sim, y_sim = solution.t, solution.y
    q_obt, qd_obt = y_sim[:3, :].T, y_sim[3:, :].T
    
    tau_history = []
    pid_plot = PIDController(Kp, Ki, Kd, DT) # Resetar o integrador para plotagem
    for i in range(len(t_sim)):
        tau = pid_plot.calculate_torque(q_d[i], q_obt[i], qd_d[i], qd_obt[i])
        tau_limits = np.array([TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3])
        tau = np.clip(tau, -tau_limits, tau_limits)
        tau_history.append(tau)
    tau_history = np.array(tau_history)
    
    plot_trajectory_tracking(t_sim, q_d, q_obt)
    plot_error_evolution(t_sim, q_d, q_obt)
    plot_control_signals(t_sim, tau_history)
    analyze_performance(t_sim, q_d, q_obt, tau_history)

# (As funções de plotagem e análise permanecem as mesmas do seu código original)
def plot_trajectory_tracking(t, q_d, q_obt):
    joint_names = ['Junta 1 (θ₁)', 'Junta 2 (θ₂)', 'Junta 3 (d₃)']
    joint_units = ['rad', 'rad', 'm']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Seguimento da Trajetória - Posição Desejada vs. Obtida', fontsize=16, fontweight='bold')
    for i in range(3):
        axes[i].plot(t, q_d[:, i], 'b-', linewidth=2, label='Desejada')
        axes[i].plot(t, q_obt[:, i], 'r--', linewidth=2.5, label='Obtida')
        axes[i].set_title(f'{joint_names[i]}')
        axes[i].set_ylabel(f'Posição ({joint_units[i]})')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_error_evolution(t, q_d, q_obt):
    joint_names = ['Junta 1 (θ₁)', 'Junta 2 (θ₂)', 'Junta 3 (d₃)']
    joint_units = ['rad', 'rad', 'm']
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Evolução do Erro de Posição', fontsize=16, fontweight='bold')
    for i in range(3):
        error = q_d[:, i] - q_obt[:, i]
        axes[i].plot(t, error, 'g-', linewidth=2)
        axes[i].set_title(f'Erro - {joint_names[i]}')
        axes[i].set_ylabel(f'Erro ({joint_units[i]})')
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].axhline(y=0, color='k', linestyle=':', alpha=0.7)
    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_control_signals(t, tau_history):
    joint_names = ['Junta 1 (θ₁)', 'Junta 2 (θ₂)', 'Junta 3 (d₃)']
    joint_units = ['Nm', 'Nm', 'N']
    tau_limits = [TAU_MAX_J1, TAU_MAX_J2, TAU_MAX_J3]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle('Sinais de Controle (Torques/Força)', fontsize=16, fontweight='bold')
    for i in range(3):
        axes[i].plot(t, tau_history[:, i], 'b-', linewidth=2, label='Torque/Força Aplicado')
        axes[i].axhline(y=tau_limits[i], color='r', linestyle='--', label=f'Limite ({tau_limits[i]:.2f})')
        axes[i].axhline(y=-tau_limits[i], color='r', linestyle='--')
        axes[i].set_title(f'{joint_names[i]}')
        axes[i].set_ylabel(f'Torque/Força ({joint_units[i]})')
        axes[i].legend()
        axes[i].grid(True, linestyle='--', alpha=0.6)
    axes[-1].set_xlabel('Tempo (s)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def analyze_performance(t, q_d, q_obt, tau_history):
    print("\n" + "="*60)
    print("ANÁLISE DE DESEMPENHO")
    print("="*60)
    joint_names = ['Junta 1 (θ₁)', 'Junta 2 (θ₂)', 'Junta 3 (d₃)']
    for i in range(3):
        error = q_d[:, i] - q_obt[:, i]
        rmse = np.sqrt(np.mean(error**2))
        max_error = np.max(np.abs(error))
        print(f"\n{joint_names[i]}:\n  - RMSE: {rmse:.6f}\n  - Erro Máximo: {max_error:.6f}")

if __name__ == "__main__":
    run_simulation_and_plot()