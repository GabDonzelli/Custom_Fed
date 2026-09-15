"""Congela o painel ao vivo num HTML autocontido, dentro do vault.

O painel de :8771 desenha o que busca em /api/data a cada 3s. Um arquivo salvo
nao pode depender disso, entao este script chama o mesmo build_payload() que
alimenta o endpoint, embute o resultado como uma constante no lugar do fetch, e
escreve o HTML ao lado das notas.

Reaproveita o index.html em vez de reescrever a pagina: a barra lateral de
simulacoes, o seletor de referencia, os graficos e as tabelas ja estao prontos
ali. As unicas mudancas sao a fonte dos dados e o texto do carimbo, que num
snapshot nao pode dizer "atualizado agora".

Os CSVs da simulacao sao apenas lidos. Rodar de novo depois que uma run
terminar simplesmente regrava o HTML.

Rodar:  python congelar.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import servidor

HERE = Path(__file__).parent
PROJETO = HERE.parent          # quickstart-pytorch
VAULT = Path(
    "C:/Users/gabri/OneDrive/Documentos/4A/FL/FL/Design dos Resultados"
)

# (arquivo, idioma inicial). A pagina continua bilingue nos quatro -- o botao de
# idioma funciona em qualquer um; muda so com qual idioma abre.
SAIDAS = [
    (PROJETO / "resultados.html", "pt"),
    (PROJETO / "results.html", "en"),
]
# O vault e o lugar de onde as notas do Obsidian linkam o painel, mas ele vive
# fora do repositorio. Escrever la so se a pasta existir mantem o script
# funcionando num clone que nao tem vault nenhum.
if VAULT.is_dir():
    SAIDAS += [(VAULT / "Resultados.html", "pt"), (VAULT / "Results.html", "en")]

# O texto exato que o index.html usa para buscar e para carimbar. Se o painel ao
# vivo mudar, estas ancoras deixam de casar e o script falha em vez de gerar uma
# pagina quebrada em silencio.
FETCH_ANCHOR = """    const response = await fetch("/api/data", { cache: "no-store" });
    const payload = await response.json();
    if (payload.error) throw new Error(payload.error);
    adopt(payload);
"""
FETCH_REPLACEMENT = """    adopt(SNAPSHOT);
"""

STAMP_ANCHOR = """    const text = t("atualizado") + " " +
      new Date().toLocaleTimeString(LANG === "en" ? "en-GB" : "pt-BR") + " · " +
      (running ? running + " " + t("runEmAndamento") : t("nenhumaRun"));
"""
# O carimbo do snapshot tambem segue o idioma: o mesmo arquivo serve os dois.
STAMP_REPLACEMENT = """    const text = t("snapshotDe") + " " +
      (LANG === "en" ? SNAPSHOT_STAMP_EN : SNAPSHOT_STAMP_PT) +
      (running ? " · " + running + " " + t("aindaRodava") : " · " + t("todasConcluidas"));
"""

DECL_ANCHOR = "let LAST = null;"
POLL_ANCHOR = "setInterval(tick, 3000);"

# Num arquivo salvo nao existe servidor local, entao a mensagem de falha
# herdada apontaria para a causa errada se o render quebrasse.
ERROR_ANCHOR = """    document.getElementById("stamp").textContent = t("semConexao");
"""
ERROR_REPLACEMENT = """    document.getElementById("stamp").textContent = t("falhaSnapshot") + error.message;
"""


def embed(html: str, payload: dict, stamp: str, stamp_en: str) -> str:
    """Swap the live fetch for a baked-in constant."""
    for anchor in (FETCH_ANCHOR, STAMP_ANCHOR, DECL_ANCHOR, POLL_ANCHOR, ERROR_ANCHOR):
        if anchor not in html:
            raise SystemExit(
                f"ancora nao encontrada no index.html:\n{anchor!r}\n"
                "O painel mudou; ajuste congelar.py antes de gerar."
            )

    # `</script>` dentro de uma string JSON encerraria a tag do navegador.
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")

    html = html.replace(FETCH_ANCHOR, FETCH_REPLACEMENT)
    html = html.replace(STAMP_ANCHOR, STAMP_REPLACEMENT)
    html = html.replace(ERROR_ANCHOR, ERROR_REPLACEMENT)
    html = html.replace(
        DECL_ANCHOR,
        f'const SNAPSHOT = {blob};\n'
        f'const SNAPSHOT_STAMP_PT = "{stamp}";\n'
        f'const SNAPSHOT_STAMP_EN = "{stamp_en}";\n{DECL_ANCHOR}',
    )
    # Sem servidor para consultar, repetir o tick so redesenharia os mesmos dados.
    html = html.replace(POLL_ANCHOR, "")
    return html


def main() -> int:
    payload = servidor.build_payload()
    # Dois carimbos: o formato de data e uma das coisas que mais denuncia
    # traducao malfeita, e "15/09/2026" e ambiguo para quem le em ingles.
    agora = datetime.now()
    stamp = agora.strftime("%d/%m/%Y às %H:%M")
    stamp_en = agora.strftime("%d %b %Y, %H:%M")

    html = (HERE / "index.html").read_text(encoding="utf-8")
    for destino, idioma in SAIDAS:
        congelado = embed(html, payload, stamp, stamp_en)
        if idioma != "pt":
            congelado = congelado.replace('let LANG = "pt";', f'let LANG = "{idioma}";', 1)
        destino.write_text(congelado, encoding="utf-8")
        print(f"escrito: {destino.name}  ({destino.stat().st_size / 1024:.0f} KB)")
    for sim in payload["simulacoes"]:
        config = sim.get("config", {})
        line = " · ".join(
            part
            for part in (
                f"{config.get('clients')} clients" if config.get("clients") else "",
                f"{config.get('rounds')} rounds" if config.get("rounds") else "",
                f"{config.get('groups')} grupos" if config.get("groups") else "",
            )
            if part
        )
        done = sim["counts"]["done"]
        print(f"  {sim['id']}  ({line})  {done}/{len(sim['runs'])} runs concluidas")
        for run in sim["runs"]:
            final = run["final"]
            shown = (
                "sem dados"
                if final is None
                else f"round {run['last_round']}, acc {final * 100:.2f}%"
            )
            print(f"    [{run['status']:>7}] {run['id']:<24} {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
