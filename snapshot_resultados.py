"""Congela o dashboard ao vivo num HTML autocontido.

O dashboard de :8771 desenha `LAST`, que ele busca em /api/data a cada 3s. Um
arquivo salvo nao pode depender disso: o server.py e um processo temporario e o
diretorio dele fica em %TEMP%. Entao este script chama o mesmo build_payload()
que alimenta o endpoint, embute o resultado como uma constante no lugar do
fetch, e escreve o HTML no diretorio do projeto -- que fica no OneDrive e
sobrevive a qualquer limpeza de temporarios.

Reaproveita o index.html existente em vez de reescrever a pagina: o seletor
lateral de modos, os graficos e as tabelas ja estao prontos ali. As unicas
mudancas sao a fonte dos dados e o texto do carimbo, que num snapshot nao pode
dizer "atualizado agora".

Rodar de novo depois que uma run terminar simplesmente regrava o arquivo.
"""

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

LIVE = Path(
    "C:/Users/gabri/AppData/Local/Temp/claude"
    "/c--Users-gabri-OneDrive-Documentos-4A-FL"
    "/d729ee18-e2c3-4e39-8ef4-a8467e8dab75/scratchpad/dashboard"
)
OUTPUT = Path(
    "C:/Users/gabri/OneDrive/Documentos/4A/FL/quickstart-pytorch/resultados.html"
)

# O texto exato que o index.html usa para buscar e para carimbar. Se o dashboard
# ao vivo mudar, estas ancoras deixam de casar e o script falha em vez de gerar
# uma pagina quebrada em silencio.
FETCH_ANCHOR = """    const response = await fetch("/api/data", { cache: "no-store" });
    LAST = await response.json();
    if (LAST.error) throw new Error(LAST.error);
"""
FETCH_REPLACEMENT = """    LAST = SNAPSHOT;
"""

STAMP_ANCHOR = """    const text = "atualizado " + new Date().toLocaleTimeString("pt-BR") + " · " +
      (running ? running + " run em andamento" : "nenhuma run em andamento");
"""
STAMP_REPLACEMENT = """    const text = "snapshot de " + SNAPSHOT_STAMP +
      (running ? " · " + running + " run ainda rodava neste momento" : " · todas as runs concluídas");
"""

DECL_ANCHOR = "let LAST = null;"
POLL_ANCHOR = "setInterval(tick, 3000);"

# Num arquivo salvo nao existe servidor local, entao a mensagem de falha herdada
# apontaria para a causa errada se o render quebrasse.
ERROR_ANCHOR = """    document.getElementById("stamp").textContent = "sem conexão com o servidor local";
"""
ERROR_REPLACEMENT = """    document.getElementById("stamp").textContent = "falha ao desenhar o snapshot: " + error.message;
"""


def load_live_server():
    """Import the running dashboard's server.py by path."""
    spec = importlib.util.spec_from_file_location("live_server", LIVE / "server.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def embed(html: str, payload: dict, stamp: str) -> str:
    """Swap the live fetch for a baked-in constant."""
    for anchor in (FETCH_ANCHOR, STAMP_ANCHOR, DECL_ANCHOR, POLL_ANCHOR, ERROR_ANCHOR):
        if anchor not in html:
            raise SystemExit(
                f"ancora nao encontrada no index.html ao vivo:\n{anchor!r}\n"
                "O dashboard mudou; ajuste snapshot.py antes de gerar."
            )

    # `</script>` dentro de uma string JSON encerraria a tag do navegador.
    blob = json.dumps(payload, ensure_ascii=False).replace("</", "<\/")

    html = html.replace(FETCH_ANCHOR, FETCH_REPLACEMENT)
    html = html.replace(STAMP_ANCHOR, STAMP_REPLACEMENT)
    html = html.replace(ERROR_ANCHOR, ERROR_REPLACEMENT)
    html = html.replace(
        DECL_ANCHOR,
        f'const SNAPSHOT = {blob};\nconst SNAPSHOT_STAMP = "{stamp}";\n{DECL_ANCHOR}',
    )
    # Sem servidor para consultar, repetir o tick so redesenharia os mesmos dados.
    html = html.replace(POLL_ANCHOR, "")
    return html


def main() -> int:
    server = load_live_server()
    payload = server.build_payload()
    stamp = datetime.now().strftime("%d/%m/%Y às %H:%M")

    html = (LIVE / "index.html").read_text(encoding="utf-8")
    OUTPUT.write_text(embed(html, payload, stamp), encoding="utf-8")

    print(f"escrito: {OUTPUT}  ({OUTPUT.stat().st_size / 1024:.0f} KB)")
    for mode in payload["modes"]:
        done = sum(1 for run in mode["runs"] if run["status"] == "done")
        print(f"  {mode['id']}: {done}/{len(mode['runs'])} runs concluidas")
        for run in mode["runs"]:
            last = run["last_round"]
            final = run["final"]
            shown = "sem dados" if final is None else f"round {last}, acc {final * 100:.2f}%"
            print(f"    [{run['status']:>7}] {run['id']:<24} {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
