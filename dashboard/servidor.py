"""Servidor local do painel de simulacoes federadas.

Le o manifesto simulacoes.json, resolve os CSVs escritos pelo ResultsLogger do
projeto e expoe tudo em /api/data. A pagina faz polling desse endpoint, entao o
acompanhamento por round vem de graca: o ResultsLogger da flush a cada round, e
o servidor releo arquivo a cada requisicao.

Diferenca em relacao a versao anterior: o eixo de navegacao agora e a
*simulacao*, e o numero de rounds, de clients e de grupos passou a viver dentro
de cada simulacao em vez de no topo do manifesto. Duas simulacoes com 20 e 30
rounds podem conviver no mesmo painel; antes o manifesto so sabia dizer um
numero para todas.

Os CSVs sao abertos em modo leitura e nunca reescritos. Este processo nao
produz resultado nenhum -- so desenha o que a simulacao ja produziu.

Rodar:  python servidor.py     ->  http://127.0.0.1:8771/
"""

import csv
import json
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
MANIFEST = HERE / "simulacoes.json"
INDEX = HERE / "index.html"
PORT = 8771


def _parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_rows(csv_path: Path) -> list[dict]:
    """Read one results CSV, tolerating a partially written trailing line."""
    rows = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for record in csv.DictReader(handle):
                if record.get("round") in (None, ""):
                    continue
                try:
                    round_num = int(record["round"])
                except ValueError:
                    continue
                clients = record.get("num_clients_trained")
                rows.append(
                    {
                        "round": round_num,
                        "accuracy": _parse_float(record.get("accuracy")),
                        "loss": _parse_float(record.get("loss")),
                        "clients": (
                            None if clients in ("N/A", "", None) else int(clients)
                        ),
                        "timestamp": record.get("timestamp"),
                    }
                )
    except FileNotFoundError:
        return []
    rows.sort(key=lambda row: row["round"])
    return [row for row in rows if row["accuracy"] is not None]


def _elapsed_seconds(rows: list[dict]):
    if len(rows) < 2:
        return None
    try:
        start = datetime.fromisoformat(rows[0]["timestamp"])
        end = datetime.fromisoformat(rows[-1]["timestamp"])
    except (TypeError, ValueError):
        return None
    return (end - start).total_seconds()


def _describe_run(entry: dict, project_dir: Path, expected: int) -> dict:
    csv_name = entry.get("csv")
    rows = _read_rows(project_dir / csv_name) if csv_name else []

    if not csv_name:
        # "na fila" e "nunca rodada nesta simulacao" se pareceriam iguais aqui --
        # as duas comecam sem CSV -- e confundir uma com a outra faria o painel
        # anunciar como ausente uma run que esta so esperando a vez.
        status = "pending" if entry.get("queued") else "planned"
    elif not rows:
        status = "pending"
    elif rows[-1]["round"] >= expected:
        status = "done"
    elif entry.get("interrupted"):
        # Um CSV que parou antes do ultimo round e ambiguo: pode estar sendo
        # escrito agora ou pode ter morrido no meio. O arquivo sozinho nao
        # distingue os dois casos, entao quem sabe que a run morreu marca
        # "interrupted" no manifesto -- sem isso ela ficaria "rodando" para
        # sempre e o painel anunciaria uma run em andamento que nao existe.
        status = "interrupted"
    else:
        status = "running"

    # O round 0 e a avaliacao do modelo inicial, antes de qualquer treino.
    trained = [row for row in rows if row["round"] > 0]
    best = max(rows, key=lambda row: row["accuracy"]) if rows else None

    # Quantos clients treinaram em cada round -- a evidencia de que a
    # configuracao de participacao de fato valeu, e nao ficou so no arquivo.
    client_counts = sorted({row["clients"] for row in trained if row["clients"]})

    return {
        **entry,
        "status": status,
        "rows": rows,
        "last_round": rows[-1]["round"] if rows else None,
        "baseline": rows[0]["accuracy"] if rows and rows[0]["round"] == 0 else None,
        "final": rows[-1]["accuracy"] if rows else None,
        "final_loss": rows[-1]["loss"] if rows else None,
        "best": best,
        "clients_last": trained[-1]["clients"] if trained else None,
        "client_counts": client_counts,
        "elapsed": _elapsed_seconds(rows),
    }


def _describe_simulation(sim: dict, project_dir: Path) -> dict:
    config = sim.get("config", {})
    expected = int(config.get("rounds", 0))
    runs = [_describe_run(entry, project_dir, expected) for entry in sim.get("runs", [])]

    counts = {"done": 0, "running": 0, "interrupted": 0, "pending": 0, "planned": 0}
    for run in runs:
        counts[run["status"]] = counts.get(run["status"], 0) + 1

    # O estado da simulacao e derivado das runs, nao declarado no manifesto:
    # um campo escrito a mao envelhece assim que uma run termina.
    if counts["running"]:
        state = "running"
    elif counts["done"] == len(runs) and runs:
        state = "done"
    elif counts["done"] or counts["interrupted"]:
        # Uma run interrompida e um desfecho, nao uma espera: a simulacao que
        # so tem essa run ja terminou o que ia terminar, e chamar isso de "na
        # fila" sugeriria que ainda vai acontecer alguma coisa.
        state = "partial"
    else:
        state = "pending"

    return {
        **{key: value for key, value in sim.items() if key != "runs"},
        "runs": runs,
        "counts": counts,
        "state": state,
    }


def build_payload() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    # Caminho relativo e resolvido contra a pasta do manifesto: assim o painel
    # funciona em qualquer clone, sem ninguem editar um caminho absoluto.
    project_dir = Path(manifest["project_dir"])
    if not project_dir.is_absolute():
        project_dir = (MANIFEST.parent / project_dir).resolve()

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "titulo": manifest.get("titulo", "Aprendizado federado"),
        "subtitulo": manifest.get("subtitulo", "Painel de simulações"),
        "grupos": manifest.get("grupos", []),
        "simulacoes": [
            _describe_simulation(sim, project_dir) for sim in manifest["simulacoes"]
        ],
        "planejadas": manifest.get("planejadas", []),
        # Texto que explica o desenho do plano -- um eixo por vez, e os avisos
        # que mudam como ler os resultados. Vive no manifesto e nao na pagina
        # para ser editavel sem mexer no HTML.
        "planejamento": manifest.get("planejamento", {}),
        # Sintese dos achados do projeto, e os campos `_en` de tudo: a pagina
        # e bilingue e escolhe entre `campo` e `campo_en` no cliente.
        "achados": manifest.get("achados", {}),
        "titulo_en": manifest.get("titulo_en"),
        "subtitulo_en": manifest.get("subtitulo_en"),
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - nome exigido por BaseHTTPRequestHandler
        if self.path.startswith("/api/data"):
            try:
                body = json.dumps(build_payload()).encode("utf-8")
            except Exception as error:  # noqa: BLE001 - o painel mostra o erro
                body = json.dumps({"error": repr(error)}).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
            return
        if self.path.split("?")[0] in ("/", "/index.html"):
            self._send(INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        self.send_error(404)

    def log_message(self, *args) -> None:
        """Silencia o log por requisicao: o polling geraria uma linha a cada 3s."""


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Painel em http://127.0.0.1:{PORT}/  (Ctrl+C para parar)", flush=True)
    server.serve_forever()
