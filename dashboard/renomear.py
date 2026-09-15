"""Renomeia as simulacoes para o rotulo dizer o que de fato esta na tela.

Dois problemas que os nomes antigos escondiam:

1. "Participação total", "FedAvg puro — participação total" e "Peso por
   exemplos — total" sao **o mesmo algoritmo**. Ja provamos que a agregacao
   agrupada com peso proporcional aos exemplos telescopa e da exatamente o
   FedAvg puro. Tres nomes diferentes para a mesma conta faziam a barra lateral
   parecer ter tres metodos quando tem um.

2. Nada no rotulo dizia se a run era semeada. Essa e a distincao que decide se
   uma diferenca pode ser lida como efeito ou nao, e estava invisivel.

O esquema novo poe os tres eixos que realmente variam no proprio rotulo:
metodo · participacao · determinismo. E acrescenta um chip `equivale a` nas
simulacoes que sao FedAvg puro por construcao, para a duplicacao ficar dita em
vez de escondida.
"""

import io
import json
import sys
from pathlib import Path

MANIFESTO = Path(__file__).parent / "simulacoes.json"

GRUPOS = {
    "experimento": (
        "Experimento semeado — o peso do grupo",
        "Seeded experiment — the group weight",
    ),
    "regua": (
        "Régua do ruído — sem semente, não conclui nada",
        "Noise ruler — unseeded, concludes nothing",
    ),
}

# id -> (rotulo pt, rotulo en, equivalencia ou None)
NOMES = {
    "seed-full-examples": (
        "FedAvg, peso por exemplos — total · semeado",
        "FedAvg, example-weighted — full · seeded",
        "FedAvg puro",
    ),
    "seed-full-equal": (
        "Grupos com peso igual — total · semeado",
        "Equal-weight groups — full · seeded",
        None,
    ),
    "seed-partial07-examples": (
        "FedAvg, peso por exemplos — parcial 0,7 · semeado",
        "FedAvg, example-weighted — partial 0.7 · seeded",
        "FedAvg puro",
    ),
    "seed-partial07-equal": (
        "Grupos com peso igual — parcial 0,7 · semeado",
        "Equal-weight groups — partial 0.7 · seeded",
        None,
    ),
    "full-10c-20r": (
        "FedAvg agrupado — total · SEM semente",
        "Grouped FedAvg — full · NO seed",
        "FedAvg puro",
    ),
    "partial07-10c-20r": (
        "FedAvg agrupado — parcial 0,7 · SEM semente",
        "Grouped FedAvg — partial 0.7 · NO seed",
        "FedAvg puro",
    ),
    "baseline-full-10c-20r": (
        "FedAvg sem grupos — total · SEM semente",
        "Ungrouped FedAvg — full · NO seed",
        None,
    ),
    "baseline-partial07-10c-20r": (
        "FedAvg sem grupos — parcial 0,7 · SEM semente",
        "Ungrouped FedAvg — partial 0.7 · NO seed",
        None,
    ),
}

EQUIV_PT = "equivale a"
EQUIV_EN = "equivalent to"
EQUIV_VAL_EN = "plain FedAvg"

NOTA_EQUIV = {
    "title": "Esta simulação é FedAvg puro, com outro nome",
    "title_en": "This simulation is plain FedAvg under another name",
    "body":
        "A agregação em dois níveis com peso proporcional aos exemplos "
        "<b>telescopa</b>: o <code>N_g</code> que pondera o grupo cancela com o "
        "<code>N_g</code> que normaliza a média dentro dele, e sobra a média "
        "ponderada de todos os clients — que é o FedAvg puro. Agrupar aqui é "
        "apenas colocar parênteses numa soma. Verificado numericamente "
        "(<code>1e-16</code>) e rodando o treino inteiro com a mesma semente "
        "(idêntico até a última casa). Então <b>não espere</b> que esta "
        "simulação difira das outras marcadas com o mesmo <i>equivale a</i>: "
        "quando diferem, a diferença é ruído de execução, não método.",
    "body_en":
        "Two-level aggregation with example-proportional weights "
        "<b>telescopes</b>: the <code>N_g</code> weighting the group cancels the "
        "<code>N_g</code> normalizing the mean inside it, leaving the "
        "example-weighted mean over all clients — which is plain FedAvg. "
        "Grouping here is just putting parentheses around a sum. Verified "
        "numerically (<code>1e-16</code>) and by running the full training with "
        "the same seed (identical to the last digit). So <b>do not expect</b> "
        "this simulation to differ from the others tagged with the same "
        "<i>equivalent to</i>: when they differ, the difference is run-to-run "
        "noise, not method.",
}


def main() -> int:
    manifesto = json.loads(MANIFESTO.read_text(encoding="utf-8"))

    for grupo in manifesto.get("grupos", []):
        par = GRUPOS.get(grupo["id"])
        if par:
            grupo["label"], grupo["label_en"] = par

    renomeadas = marcadas = 0
    for sim in manifesto["simulacoes"]:
        entrada = NOMES.get(sim["id"])
        if not entrada:
            continue
        sim["label"], sim["label_en"], equivalente = entrada
        renomeadas += 1

        extra = sim.setdefault("config", {}).setdefault("extra", [])
        extra[:] = [e for e in extra if e["k"] != EQUIV_PT]
        if equivalente:
            # O chip vem primeiro: e a informacao que muda como ler o resto.
            extra.insert(0, {
                "k": EQUIV_PT, "k_en": EQUIV_EN,
                "v": equivalente, "v_en": EQUIV_VAL_EN,
            })
            notas = sim.setdefault("notes", [])
            if not any(n.get("title") == NOTA_EQUIV["title"] for n in notas):
                notas.insert(0, json.loads(json.dumps(NOTA_EQUIV)))
            marcadas += 1

    MANIFESTO.write_text(
        json.dumps(manifesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{renomeadas} simulacoes renomeadas, {marcadas} marcadas como FedAvg puro")
    for sim in manifesto["simulacoes"]:
        print(f"  [{sim.get('grupo','—'):<11}] {sim['label']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
