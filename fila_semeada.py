"""Fila das runs semeadas, com registro ao vivo no painel.

Diferenca em relacao a fila.py: aqui cada run e lancada com Popen em vez de
subprocess.run, e o laco fica vigiando o diretorio ate o CSV novo aparecer. E
que o log do flwr demora a esvaziar o buffer -- procurar o nome do CSV dentro
dele fazia a run passar invisivel no painel ate terminar. O CSV, ao contrario, e
criado logo no inicio e o ResultsLogger da flush a cada round.

Assim que o CSV nasce, a run e registrada no manifesto e copiada para o
diretorio de dados do painel. Dai em diante o proprio servidor cuida do resto:
ele releo CSV a cada requisicao, entao a curva cresce na tela round a round.

Ordem da fila: as tarefas rapidas primeiro, e a LoRA por ultimo. As quatro
combinacoes de LoRA levariam ~8h sozinhas, entao a fila e escrita em ordem de
prioridade e roda ate onde der -- o que nao terminar fica registrado como
pendente, sem quebrar nada.
"""

import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hiperparametros import HYPER, METRICA, ROTULO  # noqa: E402

HERE = Path(__file__).parent
LOGS = HERE / "logs-semeadas"
DADOS = Path(
    "C:/Users/gabri/OneDrive/Documentos/4A/FL/FL/Design dos Resultados/dados"
)
MANIFESTO = Path(
    "C:/Users/gabri/OneDrive/Documentos/4A/FL/FL/Design dos Resultados"
    "/dashboard/simulacoes.json"
)
SEED = 42

# (task, lr, batch) -- os mesmos hiperparametros das runs anteriores.
RAPIDAS = [
    ("cifar10-pretrained", "1.0", "32"),
    ("stackexchange", "0.1", "32"),
    ("cifar10", "0.1", "32"),
]
LENTA = ("stackexchange-pretrained", "5e-4", "16")
FRACOES = [("1.0", "10"), ("0.7", "7")]
PESOS = ["examples", "equal"]


def fila() -> list[dict]:
    """Build the run list, cheapest and most informative first."""
    itens = []
    for tarefa, lr, batch in RAPIDAS:
        for fracao, minimo in FRACOES:
            for peso in PESOS:
                itens.append(dict(tarefa=tarefa, lr=lr, batch=batch,
                                  fracao=fracao, minimo=minimo, peso=peso))
    # A LoRA e cara: primeiro a fracao 0,7, que e onde existe grupo vazio e
    # portanto onde o peso igual tem mais chance de mudar alguma coisa.
    tarefa, lr, batch = LENTA
    for fracao, minimo in [("0.7", "7"), ("1.0", "10")]:
        for peso in PESOS:
            itens.append(dict(tarefa=tarefa, lr=lr, batch=batch,
                              fracao=fracao, minimo=minimo, peso=peso))
    return itens


def sim_id(fracao: str, peso: str) -> str:
    return f"seed-{'full' if fracao == '1.0' else 'partial07'}-{peso}"


def ajustar(**valores: str) -> None:
    caminho = HERE / "pyproject.toml"
    texto = caminho.read_text(encoding="utf-8")
    for chave, valor in valores.items():
        chave = chave.replace("_", "-")
        texto, n = re.subn(
            rf"(?m)^(\s*{re.escape(chave)}\s*=\s*).*$", rf"\g<1>{valor}", texto
        )
        if n != 1:
            raise SystemExit(f"chave {chave!r} apareceu {n} vezes no pyproject.toml")
    caminho.write_text(texto, encoding="utf-8")


def simulacao(fracao: str, peso: str) -> dict:
    igual = peso == "equal"
    parcial = fracao != "1.0"
    return {
        "id": sim_id(fracao, peso),
        "grupo": "estrategia" if igual else "participacao",
        "label": ("Peso igual por grupo" if igual else "Peso por exemplos")
                 + (" — parcial 0,7" if parcial else " — total"),
        "compare_to": sim_id(fracao, "examples" if igual else "equal"),
        "config": {
            "clients": 10, "rounds": 20, "groups": 4,
            "fraction_train": float(fracao), "fraction_evaluate": float(fracao),
            "strategy": "Grouped FedAvg", "seed": SEED,
            "extra": [
                {"k": "peso grupo→global", "v": "igual" if igual else "proporcional aos exemplos"},
                {"k": "min-train-nodes", "v": "7" if parcial else "10"},
                {"k": "determinismo", "v": f"semeado (seed {SEED})"},
            ],
        },
        "summary": (
            ("Cada grupo pesa o mesmo na agregação grupo→global, independentemente "
             "de quantos dados tem. É a configuração em que o agrupamento deixa de "
             "ser contabilidade e passa a mudar o modelo.")
            if igual else
            ("Cada grupo pesa proporcionalmente aos exemplos que trouxe. Esta "
             "média de dois níveis telescopa: o resultado é idêntico ao FedAvg "
             "puro, e serve aqui como linha de base contra o peso igual.")
        ),
        "notes": [],
        "runs": [],
    }


def registrar(item: dict, csv: str) -> None:
    """Add this run to the dashboard manifest, creating its simulation."""
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))
    por_id = {s["id"]: s for s in manifesto["simulacoes"]}
    alvo = sim_id(item["fracao"], item["peso"])
    if alvo not in por_id:
        nova = simulacao(item["fracao"], item["peso"])
        manifesto["simulacoes"].append(nova)
        por_id[alvo] = nova

    sim = por_id[alvo]
    if any(r["id"] == item["tarefa"] for r in sim["runs"]):
        return
    metrica, chance, eval_set = METRICA[item["tarefa"]]
    rotulo, origem = ROTULO[item["tarefa"]]
    sim["runs"].append({
        "id": item["tarefa"], "origin": origem, "label": rotulo,
        "metric": metrica, "chance": chance, "csv": csv,
        "eval_set": eval_set, "hyper": dict(HYPER[item["tarefa"]]),
        "queued": True,
    })
    MANIFESTO.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"       registrada no painel: {alvo} / {item['tarefa']}", flush=True)


def rodar(item: dict, indice: int, total: int) -> bool:
    nome = f"{item['tarefa']}--frac{item['fracao'].replace('.', '')}--{item['peso']}"
    LOGS.mkdir(exist_ok=True)
    destino = LOGS / f"{nome}.log"

    ajustar(task_name=f'"{item["tarefa"]}"', learning_rate=item["lr"],
            batch_size=item["batch"], fraction_train=item["fracao"],
            fraction_evaluate=item["fracao"], min_train_nodes=item["minimo"],
            min_evaluate_nodes=item["minimo"], num_server_rounds="20",
            seed=str(SEED), strategy='"grouped"',
            group_weight=f'"{item["peso"]}"')

    antes = {p.name for p in HERE.glob("fl_results_*.csv")}
    inicio = time.time()
    print(f"[{datetime.now():%H:%M}] ({indice}/{total}) {nome}", flush=True)

    with destino.open("w", encoding="utf-8", errors="replace") as saida:
        processo = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(HERE / "run_local.ps1")],
            stdout=saida, stderr=subprocess.STDOUT, cwd=HERE,
        )
        csv = None
        while processo.poll() is None:
            if csv is None:
                novos = {p.name for p in HERE.glob("fl_results_*.csv")} - antes
                if novos:
                    csv = sorted(novos)[-1]
                    registrar(item, csv)
            if csv:
                origem, alvo = HERE / csv, DADOS / csv
                if origem.exists():
                    shutil.copy2(origem, alvo)   # so copia; nunca edita
            time.sleep(4)

    if csv and (HERE / csv).exists():
        shutil.copy2(HERE / csv, DADOS / csv)

    texto = destino.read_text(encoding="utf-8", errors="replace")
    rodadas = re.findall(r"Round (\d+): accuracy=([0-9.]+)", texto)
    minutos = (time.time() - inicio) / 60
    if rodadas and int(rodadas[-1][0]) >= 20:
        print(f"       OK  round {rodadas[-1][0]}, acc "
              f"{float(rodadas[-1][1]) * 100:.2f}%, {minutos:.0f} min", flush=True)
        return True
    print(f"       FALHOU ou incompleta ({minutos:.0f} min) — ver {destino.name}",
          flush=True)
    return False


def main() -> int:
    itens = fila()
    print(f"{len(itens)} runs semeadas na fila\n", flush=True)
    for indice, item in enumerate(itens, start=1):
        try:
            rodar(item, indice, len(itens))
        except Exception as erro:  # noqa: BLE001 - uma run ruim nao para a fila
            print(f"       ERRO: {erro!r}", flush=True)
    print("\nfila concluida", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
