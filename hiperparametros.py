"""Ficha tecnica de cada tarefa, em campos estruturados.

Antes isso era uma string solta por run ("lr=0,1  batch=32  1 epoca local"), o
que obriga o leitor a comparar quatro frases para descobrir o que muda entre as
tarefas. Em campos, a pagina consegue fazer isso sozinha: o que tem o mesmo
valor nas quatro sobe para o cabecalho da simulacao, e so o que difere fica no
cartao da tarefa.

Todos os valores vieram da leitura do codigo em 07/09/2026, nao da documentacao:
arquiteturas em pytorchexample/tasks/*.py, otimizadores nos metodos train(), e
as contagens de parametros medidas instanciando os modelos.

Fonte unica: o manifesto do painel e o sincronizador importam daqui, para a
ficha nao divergir entre os dois.
"""

# A ordem das chaves e a ordem em que aparecem na tela.
HYPER = {
    "cifar10": {
        "Modelo": (
            "CNN do quickstart Flower — conv(3→6, 5×5) · maxpool 2×2 · "
            "conv(6→16, 5×5) · maxpool 2×2 · fc 400→120→84→10"
        ),
        "Pesos iniciais": "aleatórios (treinada do zero)",
        "Otimizador": "SGD, momentum 0,9",
        "Learning rate": "0,1",
        "Batch size": "32",
        "Épocas locais": "1",
        "Dados por client": "4.000 imagens (partição de 5.000, split 80/20)",
        "Particionamento": "IID (IidPartitioner)",
        "Pré-processamento": "ToTensor · Normalize(0,5 | 0,5)",
        "Comunicado por round": "62.006 pesos — o modelo inteiro",
    },
    "cifar10-pretrained": {
        "Modelo": (
            "ResNet-18 da torchvision congelada (11,2M pesos, em eval, sem a "
            "cabeça ImageNet) → features 512-d normalizadas em L2 → "
            "head treinável nn.Linear(512, 10)"
        ),
        "Pesos iniciais": "ResNet18_Weights.IMAGENET1K_V1 (backbone) · head aleatório",
        "Otimizador": "SGD, momentum 0,9",
        "Learning rate": "1,0",
        "Batch size": "32",
        "Épocas locais": "1",
        "Dados por client": "500 imagens (MAX_IMAGES_PER_CLIENT)",
        "Particionamento": "IID (IidPartitioner)",
        "Pré-processamento": "Resize 224 · Normalize ImageNet (0,485/0,456/0,406)",
        "Comunicado por round": "5.130 pesos — só o head linear",
    },
    "stackexchange": {
        "Modelo": (
            "LSTM in-house — Embedding(5.000, 128, padding_idx=0) · "
            "LSTM(128→256, batch_first) · Linear(256→5.000)"
        ),
        "Pesos iniciais": "aleatórios (treinada do zero)",
        "Otimizador": "SGD, momentum 0,9",
        "Learning rate": "0,1",
        "Batch size": "32",
        "Épocas locais": "1",
        "Dados por client": "respostas de um autor (SEQ_LEN 32, vocab 5.000)",
        "Particionamento": "non-IID — um autor por client",
        "Pré-processamento": "tokenização própria · pad/trunca em 32 tokens",
        "Comunicado por round": "2.320.264 pesos — o modelo inteiro",
    },
    "stackexchange-pretrained": {
        "Modelo": (
            "DistilGPT-2 congelado (81.912.576 pesos) + adapters LoRA "
            "rank 8, α 16, aplicados em attn.c_attn (12 tensores)"
        ),
        "Pesos iniciais": "distilgpt2 do Hugging Face · LoRA com B zerado",
        "Otimizador": "AdamW",
        "Learning rate": "5e-4",
        "Batch size": "16",
        "Épocas locais": "1",
        "Dados por client": "250 blocos de 64 tokens (MAX_BLOCKS_PER_CLIENT)",
        "Particionamento": "non-IID — um autor por client",
        "Pré-processamento": "tokenizer do DistilGPT-2 · blocos de 64 tokens",
        "Comunicado por round": "147.456 pesos — só os adapters LoRA",
    },
}

# Metrica e conjunto de avaliacao ficam fora de HYPER: nao sao hiperparametros
# de treino, e o conjunto de avaliacao ja tem lugar proprio na pagina.
METRICA = {
    "cifar10": ("Accuracy de classificação (10 classes)", 0.1,
                "10.000 imagens (test split completo do CIFAR-10)"),
    "cifar10-pretrained": ("Accuracy de classificação (10 classes)", 0.1,
                           "1.000 imagens (MAX_CENTRALIZED_IMAGES = 1000)"),
    "stackexchange": ("Accuracy de próxima palavra", None,
                      "tokens dos autores reservados para avaliação"),
    "stackexchange-pretrained": ("Accuracy de próximo token", None,
                                 "800 blocos de 64 tokens (MAX_CENTRALIZED_BLOCKS = 800)"),
}

ROTULO = {
    "cifar10": ("CIFAR-10 — CNN do zero", "scratch"),
    "cifar10-pretrained": ("CIFAR-10 — ResNet-18 congelada", "pretrained"),
    "stackexchange": ("Stack Exchange — LSTM do zero", "scratch"),
    "stackexchange-pretrained": ("Stack Exchange — LoRA DistilGPT-2", "pretrained"),
}
