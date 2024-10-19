import gymnasium as gym
from stable_baselines3 import PPO

# Carregar o modelo salvo
model = PPO.load("ppo_lunar_lander_recordista")

# Criar o ambiente com render_mode="human"
env = gym.make("LunarLander-v2", render_mode="human")

# Definir o número de episódios para renderizar (alterar conforme necessário)
N_RENDER = 100  # Renderizar a cada 100 episódios

# Avaliar o modelo por alguns episódios
num_episodes = 1000  # Número total de episódios
episode_count = 0

obs, _ = env.reset()  # Gymnasium retorna obs e info, mas SB3 precisa apenas de obs

for i in range(num_episodes):
    done = False
    while not done:
        # Prever a ação com base no estado de observação
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)

        # Renderizar apenas a cada N episódios
        if (episode_count % N_RENDER) == 0:
            env.render()

        # Verificar se o episódio terminou ou foi truncado
        if done or truncated:
            obs, _ = env.reset()  # Reiniciar o ambiente com obs e info, mas apenas obs será usado
            episode_count += 1

env.close()
