import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

class StopTrainingOnSuccessCallback(BaseCallback):
    """
    Callback customizado que interrompe o treinamento quando o agente
    atinge um recorde de pousos consecutivos bem-sucedidos.
    """

    def __init__(self, check_freq: int, threshold: float = 300, patience: int = 300, verbose: int = 1):
        super(StopTrainingOnSuccessCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.threshold = threshold
        self.patience = patience
        self.success_counter = 0
        self.best_reward = -np.inf  # Guarda a melhor recompensa até agora

    def _on_step(self) -> bool:
        # Verifica a cada "check_freq" passos
        if self.n_calls % self.check_freq == 0:
            if len(self.locals['infos']) > 0:
                episode_rewards = [info.get('episode', {}).get('r', 0) for info in self.locals['infos']]
                average_reward = np.mean(episode_rewards)

                # Atualiza a melhor recompensa
                if average_reward > self.best_reward:
                    self.best_reward = average_reward
                    if self.verbose > 0:
                        print(f"Nova melhor recompensa: {self.best_reward:.2f}")

                # Conta sucesso se a média for maior ou igual ao limite
                if average_reward >= self.threshold:
                    self.success_counter += 1
                    if self.verbose > 0:
                        print(f"Sucesso consecutivo {self.success_counter}/{self.patience}.")
                else:
                    self.success_counter = 0  # Reseta se não atingir o threshold

            # Interrompe se o agente atingir o número de sucessos consecutivos exigido
            if self.success_counter >= self.patience:
                print("Condição de sucesso recordista atingida. Interrompendo o treinamento...")
                return False

        return True  # Continue o treinamento

# Função para verificar e carregar ou criar o modelo
def get_or_create_model(env, model_path, **kwargs):
    if os.path.exists(model_path):
        print("Modelo encontrado. Carregando o modelo existente...")
        model = PPO.load(model_path, env=env)
    else:
        print("Modelo não encontrado. Criando novo modelo...")
        model = PPO('MlpPolicy', env, verbose=1, **kwargs)
    
    # Adicionar meta informações ao modelo
    model.meta_info = {
        "criador": "Raul Campos Nascimento",
        "algoritmo": "PPO",
        "ambiente": "LunarLander-v2",
        "data_criacao": "2024-10-18",
        "descricao": "Modelo otimizado para pousos recordistas",
        "versao": 1.0
    }
    return model

# Criar o ambiente
env = gym.make("LunarLander-v2")

# Caminho para salvar ou carregar o modelo
model_path = "ppo_lunar_lander_recordista.zip"

# Carregar ou criar o modelo
model = get_or_create_model(env, model_path,
                            n_steps=2048,              # Mais atualizações por episódio
                            batch_size=64,             # Lotes menores para aprendizado mais granular
                            gae_lambda=0.98,           # Ajuste de Generalized Advantage Estimation
                            gamma=0.999,               # Desconto maior para focar em recompensas de longo prazo
                            learning_rate=2.5e-4,      # Redução da taxa de aprendizado para estabilidade
                            ent_coef=0.01,             # Maior penalidade por entropia para explorar mais
                            clip_range=0.2)            # Aumenta a suavidade da atualização de políticas

# Definir o callback que vai monitorar pousos recordistas (com limite de 300)
callback = StopTrainingOnSuccessCallback(check_freq=1000, threshold=300, patience=300)

# Continuar o treinamento ou iniciar o treinamento do modelo
model.learn(total_timesteps=5000000, callback=callback)

# Salvar o modelo treinado
model.save(model_path)
print("Treinamento concluído e modelo recordista salvo!")

# Exibir as meta informações do modelo
print("Meta informações do modelo:", model.meta_info)
