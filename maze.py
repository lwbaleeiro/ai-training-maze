import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Labirinto (0 = caminho, 1 = parede, 2 = objetivo)
maze = np.array([
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 0, 1, 0],
    [0, 0, 0, 0, 2]
])

# Configurações do ambiente e Q-Learning
start = (0, 0)  # Posição inicial do agente
goal = (4, 4)  # Objetivo
alpha = 0.1  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
epsilon = 1.0  # Taxa de exploração inicial
num_episodes = 1000
best_score = float('-inf')
best_q_table = None
actions = {
    0: (-1, 0),  # Cima
    1: (1, 0),   # Baixo
    2: (0, -1),  # Esquerda
    3: (0, 1)    # Direita
}

# Carregar ou inicializar Q-Table
if os.path.exists("best_q_table.npy"):
    best_q_table = np.load("best_q_table.npy")
    print("Q-Table carregada com sucesso.")
else:
    q_table = np.zeros((maze.shape[0], maze.shape[1], len(actions)))
    print("Inicializando nova Q-Table.")

def get_next_state(state, action):
    """Calcula o próximo estado baseado na ação, permanecendo no mesmo lugar se for inválida."""
    i, j = state  # Coordenadas atuais
    di, dj = actions[action]  # Deslocamento baseado na ação
    ni, nj = i + di, j + dj  # Próximas coordenadas

    # Verifica se o próximo estado está dentro dos limites do labirinto
    if ni < 0 or ni >= maze.shape[0] or nj < 0 or nj >= maze.shape[1]:
        return state  # Fora dos limites, permanece no mesmo lugar

    # Verifica se o próximo estado não é uma parede
    if maze[ni, nj] == 1:
        return state  # É uma parede, permanece no mesmo lugar

    return (ni, nj)  # Próximo estado válido

def get_reward(state):
    """Retorna a recompensa para um estado."""
    if maze[state] == 1:
        return -10  # Penalidade para parede
    elif maze[state] == 2:
        return 10  # Recompensa por objetivo
    return -1  # Penalidade por movimento

def simulate_episode(q_table_var):
    """Simula um episódio com o agente treinado."""
    state = start
    path = [state]

    while state != goal:
        action = np.argmax(q_table_var[state[0], state[1]])
        next_state = get_next_state(state, action)
        path.append(state)
    
        # Verifica se o proximo estado é uma parede
        if maze[next_state] == 1:
            print(f"Erro: Caminho inválido detectado em {next_state}. Corrigindo...")
            break

        state = next_state
        path.append(state)

    return path

def render_maze(path):
    """Renderiza o labirinto e a trajetória do agente."""
    fig, ax = plt.subplots()
    cmap = plt.colormaps.get_cmap('cool')

    ax.imshow(maze, cmap=cmap, origin='upper', extent=[-0.5, maze.shape[1]-0.5, maze.shape[0]-0.5, -0.5])
    ax.set_xticks(np.arange(-0.5, maze.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, maze.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Animação
    scat = ax.scatter([], [], c='red', s=200, label="Agente")
    ax.legend(loc="upper left")

    def update(frame):
        scat.set_offsets([path[frame][1], path[frame][0]])
        return scat,

    ani = animation.FuncAnimation(
        fig, update, frames=len(path), interval=150, blit=True, repeat=False
    )

    plt.title("Agente no labirinto")
    plt.show()

# Treinamento
if 'q_table' in locals():  # Treinar somente se a Q-Table não foi carregada
    for episode in range(num_episodes):
        state = start
        total_reward = 0
        steps = 0
        penalties = 0
        done = False

        while not done:
            # Escolher ação
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(actions) - 1)
            else:
                action = np.argmax(q_table[state[0], state[1]])

            # Executar ação
            next_state = get_next_state(state, action)
            reward = get_reward(next_state)
            total_reward += reward
            steps += 1

            if next_state == state:  # Colisão com parede
                penalties += 1

            # Atualizar Q-Table
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)
            
            state = next_state

            if maze[state] == 2:
                done = True

        # Decaimento do epsilon
        epsilon = max(0.1, epsilon * 0.995)

        # Avaliação do episódio
        min_steps = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        score = total_reward - penalties * 5 - max(0, steps - min_steps)
        if score > best_score:
            best_score = score
            best_q_table = q_table.copy()
            print(f"Novo melhor score: {score}, Penalidades: {penalties}")

    # Salvar Q-Table
    if best_q_table is not None:
        np.save("best_q_table.npy", best_q_table)
        print("Melhor Q-Table salva com sucesso.")

# Simulação e visualização
path = simulate_episode(best_q_table)
render_maze(path)
