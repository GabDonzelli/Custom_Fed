"""Acrescenta a versao em ingles e a secao de achados ao manifesto.

O painel e bilingue por campos irmaos: cada texto traduzivel ganha um vizinho
com sufixo `_en`, e a pagina escolhe um ou outro conforme o idioma. Preferi isto
a manter dois manifestos porque os dados -- quais CSVs, quais runs, qual
configuracao -- sao os mesmos nos dois idiomas, e duplicar o arquivo inteiro
criaria duas verdades que divergiriam na primeira run nova.

Rodar de novo e seguro: o script sobrescreve os campos `_en` e a secao de
achados, sem tocar em runs, CSVs ou configuracao.
"""

import io
import json
import sys
from pathlib import Path

MANIFESTO = Path(__file__).parent / "simulacoes.json"

# ----------------------------------------------------------------- tarefas
TAREFAS_EN = {
    "cifar10": {
        "label": "CIFAR-10 — CNN from scratch",
        "metric": "Classification accuracy (10 classes)",
        "eval_set": "10,000 images (full CIFAR-10 test split)",
    },
    "cifar10-pretrained": {
        "label": "CIFAR-10 — frozen ResNet-18",
        "metric": "Classification accuracy (10 classes)",
        "eval_set": "1,000 images (MAX_CENTRALIZED_IMAGES = 1000)",
    },
    "stackexchange": {
        "label": "Stack Exchange — LSTM from scratch",
        "metric": "Next-word accuracy",
        "eval_set": "tokens from the held-out authors",
    },
    "stackexchange-pretrained": {
        "label": "Stack Exchange — LoRA DistilGPT-2",
        "metric": "Next-token accuracy",
        "eval_set": "800 blocks of 64 tokens (MAX_CENTRALIZED_BLOCKS = 800)",
    },
}

HYPER_EN = {
    "cifar10": {
        "Model": "Flower quickstart CNN — conv(3→6, 5×5) · maxpool 2×2 · "
                 "conv(6→16, 5×5) · maxpool 2×2 · fc 400→120→84→10",
        "Initial weights": "random (trained from scratch)",
        "Optimizer": "SGD, momentum 0.9",
        "Learning rate": "0.1",
        "Batch size": "32",
        "Local epochs": "1",
        "Data per client": "4,000 images (5,000-image partition, 80/20 split)",
        "Partitioning": "IID (IidPartitioner)",
        "Preprocessing": "ToTensor · Normalize(0.5 | 0.5)",
        "Communicated per round": "62,006 weights — the whole model",
    },
    "cifar10-pretrained": {
        "Model": "torchvision ResNet-18, frozen (11.2M weights, in eval, ImageNet "
                 "head removed) → L2-normalized 512-d features → trainable "
                 "nn.Linear(512, 10) head",
        "Initial weights": "ResNet18_Weights.IMAGENET1K_V1 (backbone) · random head",
        "Optimizer": "SGD, momentum 0.9",
        "Learning rate": "1.0",
        "Batch size": "32",
        "Local epochs": "1",
        "Data per client": "500 images (MAX_IMAGES_PER_CLIENT)",
        "Partitioning": "IID (IidPartitioner)",
        "Preprocessing": "Resize 224 · ImageNet Normalize (0.485/0.456/0.406)",
        "Communicated per round": "5,130 weights — the linear head only",
    },
    "stackexchange": {
        "Model": "in-house LSTM — Embedding(5,000, 128, padding_idx=0) · "
                 "LSTM(128→256, batch_first) · Linear(256→5,000)",
        "Initial weights": "random (trained from scratch)",
        "Optimizer": "SGD, momentum 0.9",
        "Learning rate": "0.1",
        "Batch size": "32",
        "Local epochs": "1",
        "Data per client": "one author's answers (SEQ_LEN 32, vocab 5,000)",
        "Partitioning": "non-IID — one author per client",
        "Preprocessing": "in-house tokenizer · pad/truncate to 32 tokens",
        "Communicated per round": "2,320,264 weights — the whole model",
    },
    "stackexchange-pretrained": {
        "Model": "frozen DistilGPT-2 (81,912,576 weights) + LoRA adapters "
                 "rank 8, α 16, on attn.c_attn (12 tensors)",
        "Initial weights": "distilgpt2 from Hugging Face · LoRA with B zeroed",
        "Optimizer": "AdamW",
        "Learning rate": "5e-4",
        "Batch size": "16",
        "Local epochs": "1",
        "Data per client": "250 blocks of 64 tokens (MAX_BLOCKS_PER_CLIENT)",
        "Partitioning": "non-IID — one author per client",
        "Preprocessing": "DistilGPT-2 tokenizer · 64-token blocks",
        "Communicated per round": "147,456 weights — the LoRA adapters only",
    },
}
# A ordem das chaves e a mesma nos dois idiomas, entao da para casar por posicao.
HYPER_CHAVES_EN = list(HYPER_EN["cifar10"].keys())

# ------------------------------------------------------------- simulacoes
SIMS_EN = {
    "full-10c-20r": (
        "Full participation",
        "All 10 clients train and evaluate in every round. This is the reference "
        "any partial-participation regime is read against.",
    ),
    "partial07-10c-20r": (
        "Partial participation 0.7",
        "7 of the 10 clients are sampled per round, redrawn independently every "
        "round, for both the train and the evaluate phase.",
    ),
    "baseline-full-10c-20r": (
        "Plain FedAvg — full participation",
        "Plain FedAvg, no grouping: a single example-weighted mean over all 10 "
        "clients. Same tasks, same hyperparameters, same code as the grouped "
        "runs — only the strategy changes.",
    ),
    "baseline-partial07-10c-20r": (
        "Plain FedAvg — partial 0.7",
        "Plain FedAvg with 7 of 10 clients sampled per round. Without groups "
        "there is no empty-group case: every sampled client lands in the same mean.",
    ),
    "seed-full-examples": (
        "Example-weighted groups — full",
        "Each group weighs in proportion to the examples it contributed. This "
        "two-level mean telescopes: the result is identical to plain FedAvg, and "
        "it serves here as the baseline against equal weighting.",
    ),
    "seed-full-equal": (
        "Equal-weight groups — full",
        "Every group weighs the same in the group→global step, regardless of how "
        "much data it holds. This is the setting where grouping stops being "
        "bookkeeping and starts changing the model.",
    ),
    "seed-partial07-examples": (
        "Example-weighted groups — partial 0.7",
        "Each group weighs in proportion to the examples it contributed. This "
        "two-level mean telescopes: the result is identical to plain FedAvg, and "
        "it serves here as the baseline against equal weighting.",
    ),
    "seed-partial07-equal": (
        "Equal-weight groups — partial 0.7",
        "Every group weighs the same in the group→global step, regardless of how "
        "much data it holds. This is the setting where grouping stops being "
        "bookkeeping and starts changing the model.",
    ),
}

GRUPOS_EN = {
    "participacao": "Participation",
    "estrategia": "Aggregation strategy",
    "otimizacao": "Optimization",
}

CHIPS_EN = {
    "min-train-nodes": "min-train-nodes",
    "min-evaluate-nodes": "min-evaluate-nodes",
    "grupos vazios": "empty groups",
    "agrupamento": "grouping",
    "branch": "branch",
    "determinismo": "determinism",
    "peso grupo→global": "group→global weight",
}
VALORES_EN = {
    "nenhum, por construção": "none, by construction",
    "0 de 80 rounds medidos": "0 of 80 rounds measured",
    "12 de 80 rounds medidos (15%)": "12 of 80 rounds measured (15%)",
    "nenhum": "none",
    "SEM semente": "NO seed",
    "semeado (seed 42)": "seeded (seed 42)",
    "igual": "equal",
    "proporcional aos exemplos": "proportional to examples",
}
ESTRATEGIA_EN = {
    "Grouped FedAvg": "Grouped FedAvg",
    "FedAvg puro (TrackedFedAvg)": "Plain FedAvg (TrackedFedAvg)",
}

# ---------------------------------------------------------------- achados
ACHADOS = [
    {
        "title": "Agrupar não muda o modelo — em nenhum regime",
        "title_en": "Grouping does not change the model — in any regime",
        "body":
            "Com o peso grupo→global proporcional aos exemplos, a média de dois "
            "níveis <b>telescopa</b>: o <code>N_g</code> que pondera o grupo cancela "
            "com o <code>N_g</code> que normaliza a média dentro dele, e sobra "
            "exatamente o FedAvg puro. É a diferença entre <code>(a+b)+(c+d)</code> e "
            "<code>a+b+c+d</code> — parênteses não mudam uma soma. Verificado de três "
            "formas: pela álgebra, numericamente contra o <code>TrackedFedAvg</code> "
            "(diferença máxima <code>1e-16</code>, o epsilon de máquina em float64), e "
            "rodando o treino inteiro com a mesma semente (idêntico até a última casa, "
            "com uma única divergência de <code>1e-6</code> na loss, que é ordem de "
            "soma em float32). Vale inclusive com participação parcial e grupos "
            "vazios — a crença anterior de que 0,7 quebraria a identidade estava "
            "errada.",
        "body_en":
            "With the group→global weight proportional to examples, the two-level "
            "mean <b>telescopes</b>: the <code>N_g</code> weighting the group cancels "
            "the <code>N_g</code> normalizing the mean inside it, leaving exactly "
            "plain FedAvg. It is the difference between <code>(a+b)+(c+d)</code> and "
            "<code>a+b+c+d</code> — parentheses do not change a sum. Verified three "
            "ways: algebraically, numerically against <code>TrackedFedAvg</code> (max "
            "difference <code>1e-16</code>, float64 machine epsilon), and by running "
            "the full training with the same seed (identical to the last digit, with a "
            "single <code>1e-6</code> divergence in the loss, which is float32 "
            "summation order). It holds under partial participation and empty groups "
            "too — the earlier belief that 0.7 would break the identity was wrong.",
    },
    {
        "title": "O peso do grupo é a alavanca — e ela funciona",
        "title_en": "The group weight is the lever — and it works",
        "body":
            "Trocar o peso grupo→global de proporcional para <b>igual entre grupos</b> "
            "faz o <code>N_g</code> parar de cancelar, e o agrupamento passa a mudar o "
            "modelo. Medido nas 16 runs semeadas: até <b>−3,27 p.p.</b> na CNN do zero "
            "em participação parcial e <b>−2,18 p.p.</b> na LSTM. Como tudo é semeado, "
            "cada ponto dessa diferença é do peso, sem ruído a descontar.",
        "body_en":
            "Switching the group→global weight from proportional to <b>equal across "
            "groups</b> stops the <code>N_g</code> from cancelling, and grouping starts "
            "changing the model. Measured across the 16 seeded runs: up to "
            "<b>−3.27 pp</b> on the from-scratch CNN under partial participation and "
            "<b>−2.18 pp</b> on the LSTM. Since everything is seeded, every point of "
            "that difference comes from the weight, with no noise to discount.",
    },
    {
        "title": "Sem semente, o ruído engolia o efeito",
        "title_en": "Without a seed, noise swamped the effect",
        "body":
            "Antes de 08/09/2026 o projeto não tinha <code>manual_seed</code> nenhum. "
            "Duas execuções do <i>mesmo algoritmo</i> — FedAvg puro e agrupado, que "
            "provamos idênticos — terminaram <b>3,26 p.p.</b> apart no CIFAR-10. Era "
            "quase o tamanho dos 3,76 p.p. que eu havia atribuído à participação "
            "parcial, e numa segunda amostra o sinal chegou a inverter. Depois de "
            "semear, duas execuções da mesma configuração batem em todas as casas, "
            "inclusive na loss. As simulações marcadas <i>SEM semente</i> continuam no "
            "painel porque são a régua de ruído, não para atribuir efeito.",
        "body_en":
            "Before 2026-09-08 the project had no <code>manual_seed</code> anywhere. "
            "Two runs of the <i>same algorithm</i> — plain and grouped FedAvg, which we "
            "proved identical — finished <b>3.26 pp</b> apart on CIFAR-10. That was "
            "nearly the size of the 3.76 pp I had attributed to partial participation, "
            "and on a second sample the sign even flipped. After seeding, two runs of "
            "the same configuration match to every digit, loss included. The "
            "simulations tagged <i>NO seed</i> stay in the dashboard as the noise "
            "ruler, not as evidence of effect.",
    },
    {
        "title": "Grupo vazio: calculado, medido, e só importa sob peso igual",
        "title_en": "Empty groups: computed, measured, and only relevant under equal weight",
        "body":
            "O sorteio de clients ignora os grupos. Com 7 de 10 sorteados e grupos de "
            "tamanho 3, 3, 2, 2, algum grupo fica vazio em <b>15,0%</b> dos rounds — "
            "valor exato pela combinatória, que bateu com os 12 de 80 rounds medidos "
            "nos logs. Um grupo de 2 fica vazio oito vezes mais que um de 3. Sob peso "
            "por exemplos isso não muda nada (peso zero dos dois lados); sob peso "
            "igual, os grupos sobreviventes sobem de <code>1/4</code> para "
            "<code>1/3</code> e a influência oscila round a round. Com fração 0,3 e 4 "
            "grupos, <b>100%</b> dos rounds têm grupo vazio — três clients não cobrem "
            "quatro grupos.",
        "body_en":
            "Client sampling ignores the groups. With 7 of 10 sampled and groups of "
            "size 3, 3, 2, 2, some group is empty in <b>15.0%</b> of rounds — the exact "
            "combinatorial value, which matched the 12 of 80 rounds measured in the "
            "logs. A group of 2 goes empty eight times more often than one of 3. Under "
            "example weighting this changes nothing (zero weight on both sides); under "
            "equal weighting the surviving groups rise from <code>1/4</code> to "
            "<code>1/3</code> and influence swings round to round. At fraction 0.3 with "
            "4 groups, <b>100%</b> of rounds have an empty group — three clients cannot "
            "cover four groups.",
    },
    {
        "title": "Pré-treinados quase não reagem ao peso; do zero reagem muito",
        "title_en": "Pretrained models barely react to the weight; from-scratch ones react a lot",
        "body":
            "LoRA DistilGPT-2 variou <b>−0,01</b> e <b>−0,18 p.p.</b> entre os dois "
            "pesos; o probe da ResNet-18, <b>+0,10</b> e <b>−0,70</b>. O que se treina "
            "neles é minúsculo — 147.456 e 5.130 pesos sobre uma representação já "
            "pronta — então mudar a média muda pouco. Na prática isso os torna quase "
            "imunes ao desequilíbrio de dados entre clients, o que é um argumento a "
            "favor deles em cenário federado real.",
        "body_en":
            "LoRA DistilGPT-2 moved <b>−0.01</b> and <b>−0.18 pp</b> between the two "
            "weightings; the ResNet-18 probe, <b>+0.10</b> and <b>−0.70</b>. What is "
            "trained in them is tiny — 147,456 and 5,130 weights on top of a "
            "ready-made representation — so changing the mean changes little. In "
            "practice that makes them nearly immune to data imbalance across clients, "
            "which is an argument in their favour for real federated settings.",
    },
    {
        "title": "Na LSTM, o peso igual custa accuracy — e isso é a métrica, não o método",
        "title_en": "On the LSTM, equal weighting costs accuracy — and that is the metric, not the method",
        "body":
            "A LSTM perde <b>1,58</b> e <b>2,18 p.p.</b> sob peso igual. É a tarefa em "
            "que um único autor detém <b>54%</b> dos exemplos. O peso por exemplos "
            "deixa esse autor dominar o modelo global — e isso <i>infla</i> a accuracy, "
            "porque a avaliação centralizada é dominada pelo mesmo perfil de texto. O "
            "peso igual tira essa vantagem: o modelo fica mais representativo dos "
            "autores pequenos e paga na métrica agregada. A leitura honesta é que a "
            "accuracy centralizada premia o desequilíbrio; uma accuracy média "
            "<i>por client</i> contaria outra história.",
        "body_en":
            "The LSTM loses <b>1.58</b> and <b>2.18 pp</b> under equal weighting. It is "
            "the task where a single author holds <b>54%</b> of the examples. Example "
            "weighting lets that author dominate the global model — and that "
            "<i>inflates</i> accuracy, because centralized evaluation is dominated by "
            "the same text profile. Equal weighting removes the advantage: the model "
            "becomes more representative of the small authors and pays for it in the "
            "aggregate metric. The honest reading is that centralized accuracy rewards "
            "imbalance; a <i>per-client</i> mean accuracy would tell a different story.",
    },
    {
        "title": "Os dois painéis de CIFAR-10 não medem no mesmo conjunto",
        "title_en": "The two CIFAR-10 panels are not measured on the same test set",
        "body":
            "A CNN do zero é avaliada nas <b>10.000</b> imagens do test split completo; "
            "o probe da ResNet-18, em <b>1.000</b> "
            "(<code>MAX_CENTRALIZED_IMAGES = 1000</code>, porque cada imagem tem de "
            "atravessar o backbone congelado na CPU). Descobri pelo denominador "
            "implícito: <code>0,840000</code> só sai de 1.000 exemplos. Consequência: "
            "erro padrão de <b>≈1,2 p.p.</b> no probe contra <b>≈0,5 p.p.</b> na CNN. "
            "A amostra de 1.000 é aleatória (<code>shuffle(seed=42)</code> antes do "
            "corte), então é estimativa não-enviesada — só menos precisa.",
        "body_en":
            "The from-scratch CNN is evaluated on the full <b>10,000</b>-image test "
            "split; the ResNet-18 probe on <b>1,000</b> "
            "(<code>MAX_CENTRALIZED_IMAGES = 1000</code>, because every image has to "
            "cross the frozen backbone on CPU). I found this from the implied "
            "denominator: <code>0.840000</code> can only come from 1,000 examples. "
            "Consequence: a standard error of <b>≈1.2 pp</b> on the probe against "
            "<b>≈0.5 pp</b> on the CNN. The 1,000-image sample is random "
            "(<code>shuffle(seed=42)</code> before the cut), so it is an unbiased "
            "estimate — just a less precise one.",
    },
    {
        "title": "Auditoria: as runs são reais e conferem com os logs",
        "title_en": "Audit: the runs are genuine and match their logs",
        "body":
            "Em 07/09/2026 confrontei cada CSV com o log de execução: <b>zero "
            "divergências</b> nas 21 accuracies e 21 losses de cada run, e cada log "
            "nomeia o CSV que escreveu. A coluna <code>num_clients_trained</code> lê "
            "exatamente 10 ou 7 conforme a fração, e a soma das participações por "
            "partição fecha em 200 e 140. As contagens de parâmetros foram medidas "
            "instanciando os modelos, não copiadas da documentação: <b>62.006</b>, "
            "<b>5.130</b>, <b>2.320.264</b> e <b>147.456</b>. Uma run de LoRA sem log "
            "preservado foi trocada por uma repetição auditável e equivalente.",
        "body_en":
            "On 2026-09-07 I checked every CSV against its execution log: <b>zero "
            "divergences</b> across the 21 accuracies and 21 losses of each run, and "
            "every log names the CSV it wrote. The <code>num_clients_trained</code> "
            "column reads exactly 10 or 7 per the fraction, and per-partition "
            "participation sums to 200 and 140. Parameter counts were measured by "
            "instantiating the models, not copied from docs: <b>62,006</b>, "
            "<b>5,130</b>, <b>2,320,264</b> and <b>147,456</b>. One LoRA run with no "
            "preserved log was swapped for an auditable, equivalent repeat.",
    },
]

ACHADOS_INTRO = (
    "O que as 30 runs deste painel mostraram, em ordem de importância para a "
    "pesquisa. Cada item foi medido, não inferido; os números vêm dos CSVs e dos "
    "logs de execução."
)
ACHADOS_INTRO_EN = (
    "What the 30 runs in this dashboard showed, ordered by importance to the "
    "research. Every item was measured, not inferred; the numbers come from the "
    "CSVs and the execution logs."
)

PLAN_INTRO_EN = (
    "The plan varies <b>one axis at a time</b> against a fixed reference. "
    "Everything else — model, hyperparameters, partitioning, code — stays "
    "identical, so an observed difference has a single cause. Since 2026-09-08 "
    "every new run is <b>seeded</b> (<code>seed = 42</code>): two runs of the same "
    "configuration give exactly the same result, so any difference between two "
    "simulations comes from the change, not from the draw."
)
AVISOS_EN = [
    "<b>The older runs have no seed.</b> The simulations tagged <i>NO seed</i> "
    "predate seeding, and two of them with the same configuration diverged on "
    "their own — we measured <code>3.26 pp</code> on CIFAR-10 between plain and "
    "grouped FedAvg, which are the same algorithm. They stay here as the noise "
    "ruler, but they cannot attribute an effect to anything."
]

PLANEJADAS_EN = {
    "FedAvg puro (sem grupos)": ("Plain FedAvg (no groups)", "Strategy",
        "Run on 2026-09-07 across the 4 tasks and both fractions. As the algebra "
        "predicted, it reproduced the grouped curves instead of contrasting with "
        "them — the run's real value turned out to be measuring run-to-run "
        "variance, not testing the strategy."),
    "Semear as runs": ("Seed the runs", "Methodology",
        "Done on 2026-09-08. <code>torch.manual_seed</code> on model creation and "
        "on the DataLoaders, plus the client deriving its own seed per partition "
        "and round. Two runs of the same configuration now match to every digit."),
    "Peso igual por grupo": ("Equal weight per group", "Strategy",
        "Done on 2026-09-08. Switching the group→global weight from proportional "
        "to equal is what makes grouping stop being bookkeeping. Confirmed: up to "
        "3.27 pp of difference across the 16 seeded runs."),
    "Participação parcial 0,3": ("Partial participation 0.3", "Participation",
        "3 of 10 clients per round, with <code>min-train-nodes = 3</code>. Extends "
        "the axis that already has 1.0 and 0.7. With 4 groups, 100% of rounds have "
        "an empty group — which only matters now that equal weighting exists."),
    "LR decay": ("LR decay", "Optimization",
        "Branch <code>lr-decay</code>. A decreasing step per round, to contain the "
        "noise that client resampling injects late in training. It is the only "
        "axis that genuinely changes training dynamics."),
}


def traduzir(manifesto: dict) -> None:
    manifesto["titulo_en"] = "Federated learning"
    manifesto["subtitulo_en"] = "Simulation dashboard"

    for grupo in manifesto.get("grupos", []):
        grupo["label_en"] = GRUPOS_EN.get(grupo["id"], grupo["label"])

    for sim in manifesto["simulacoes"]:
        par = SIMS_EN.get(sim["id"])
        if par:
            sim["label_en"], sim["summary_en"] = par
        config = sim.setdefault("config", {})
        if config.get("strategy"):
            config["strategy_en"] = ESTRATEGIA_EN.get(
                config["strategy"], config["strategy"]
            )
        for chip in config.get("extra", []):
            chip["k_en"] = CHIPS_EN.get(chip["k"], chip["k"])
            chip["v_en"] = VALORES_EN.get(chip["v"], chip["v"])

        for run in sim["runs"]:
            en = TAREFAS_EN.get(run["id"])
            if en:
                run["label_en"] = en["label"]
                run["metric_en"] = en["metric"]
                run["eval_set_en"] = en["eval_set"]
            if run.get("hyper"):
                run["hyper_en"] = dict(HYPER_EN[run["id"]])

    plano = manifesto.setdefault("planejamento", {})
    plano["intro_en"] = PLAN_INTRO_EN
    plano["avisos_en"] = AVISOS_EN

    for item in manifesto.get("planejadas", []):
        par = PLANEJADAS_EN.get(item["label"])
        if par:
            item["label_en"], item["eixo_en"], item["detail_en"] = par

    manifesto["achados"] = {
        "intro": ACHADOS_INTRO,
        "intro_en": ACHADOS_INTRO_EN,
        "itens": [json.loads(json.dumps(a)) for a in ACHADOS],
    }


def main() -> int:
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    traduzir(manifesto)
    MANIFESTO.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sims = len(manifesto["simulacoes"])
    runs = sum(len(s["runs"]) for s in manifesto["simulacoes"])
    print(f"traduzido: {sims} simulacoes, {runs} runs, "
          f"{len(manifesto['achados']['itens'])} achados")
    return 0


if __name__ == "__main__":
    sys.exit(main())
