"""Determinismo das runs: onde o acaso entra e como fixa-lo.

Sem isto, duas execucoes da mesma configuracao divergem sozinhas -- medimos
3,26 p.p. de diferenca no CIFAR-10 entre FedAvg puro e agrupado, que sao o
mesmo algoritmo. Uma diferenca dessa ordem afoga o efeito que qualquer eixo do
experimento produziria, e nao ha como separar "a mudanca funcionou" de "esta
run deu sorte".

Quatro fontes de aleatoriedade, e onde cada uma e tratada:

1. Pesos iniciais do modelo -- `seed_everything` no ServerApp, antes de
   `create_model()`. E o que faz o round 0 sair igual.
2. Ordem dos batches -- cada DataLoader com `shuffle=True` recebe um
   `torch.Generator` semeado, via `seeded_generator`.
3. Sorteio de clients (so com fraction < 1) -- o `sample_nodes` do Flower usa o
   `random` global do Python, entao `seed_everything` no ServerApp o alcanca.
4. Dropout e afins durante o treino -- `seed_everything` no ClientApp, com uma
   semente derivada de (base, particao, round), para cada client ser
   reproduzivel sem que todos sorteiem a mesma coisa.

O que este modulo NAO garante: a ordem em que o Ray devolve as respostas dos
clients. Como a agregacao e uma soma ponderada, ordem diferente muda o
arredondamento em float32 -- diferencas na casa de 1e-7, irrelevantes para
accuracy mas suficientes para dois CSVs nao saírem byte a byte iguais. Por isso
a verificacao de determinismo compara com tolerancia, nao por igualdade exata.
"""

import os
import random

import numpy as np
import torch

# Semente base do projeto. O mesmo 42 ja usado em group-seed e SUBSAMPLE_SEED,
# para nao haver dois numeros magicos diferentes rolando pelo codigo.
DEFAULT_SEED = 42


def seed_everything(seed: int) -> None:
    """Fix every RNG this process draws from."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN escolhe algoritmos por benchmark, e o vencedor pode variar entre
    # execucoes. Em CPU isto nao custa nada; deixa a run reproduzivel se um dia
    # rodar em GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeded_generator(seed: int) -> torch.Generator:
    """Build the generator a shuffling DataLoader should use."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def client_seed(base_seed: int, partition_id: int, server_round: int) -> int:
    """Derive one client's seed for one round.

    Precisa depender das tres coisas: da semente base (para a run inteira ser
    repetivel), da particao (para dois clients nao embaralharem identicamente) e
    do round (para o mesmo client nao repetir a mesma ordem de batches em todo
    round, o que enviesaria o treino).
    """
    return (base_seed * 1_000_003 + partition_id * 9_176 + server_round) % (2**31 - 1)
