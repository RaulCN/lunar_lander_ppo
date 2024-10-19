# Projeto de Aprendizado por Reforço: Lunar Lander

Este projeto implementa e treina um agente de Aprendizado por Reforço para resolver o problema do Lunar Lander usando o algoritmo PPO (Proximal Policy Optimization) da biblioteca Stable Baselines3.

## Arquivos

1. `treinamento_exigente_ainda_mais.py`: Script de treinamento
2. `agente.py`: Script de avaliação

## Requisitos

- Python 3.7+
- gymnasium
- stable-baselines3

Você pode instalar os pacotes necessários usando:
pip install gymnasium stable-baselines3
Copy
## Treinando o Agente

Para treinar o agente, execute:
python treinamento.py

Este script faz o seguinte:

- Cria ou carrega um modelo PPO
- Treina o modelo no ambiente LunarLander-v2
- Usa um callback personalizado para interromper o treinamento quando o agente atinge um número recorde de pousos consecutivos bem-sucedidos
- Salva o modelo treinado

### Características Principais:

- `StopTrainingOnSuccessCallback` personalizado para monitorar e interromper o treinamento com base no desempenho
- Hiperparâmetros otimizados para melhor aprendizado
- Metainformações adicionadas ao modelo para rastreamento

## Avaliando o Agente

Para avaliar o agente treinado, execute:
python agente.py

Este script:

- Carrega o modelo treinado
- Executa o agente no ambiente LunarLander-v2 com renderização visual
- Avalia o desempenho do agente em vários episódios

## Informações do Modelo

O modelo treinado inclui as seguintes metainformações:

- Criador: Raul Campos Nascimento
- Algoritmo: PPO
- Ambiente: LunarLander-v2
- Data de Criação: 18/10/2024
- Descrição: Modelo otimizado para pousos recordistas, o modelo só para de treinar quando obtêm a pontuação máxima.
- Versão: 1.0

## Personalização

Você pode ajustar vários parâmetros em ambos os scripts para experimentar diferentes configurações de treinamento e configurações de avaliação.
