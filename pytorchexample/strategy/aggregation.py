"""Pure aggregation helpers for the two-level FedAvg calculation."""

from collections.abc import Callable
from dataclasses import dataclass

from flwr.app import ArrayRecord, MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords


@dataclass(frozen=True)
class GroupAggregation:
    """Store one temporary group model and its aggregation metadata."""

    group_id: int
    arrays: ArrayRecord
    metrics: MetricRecord
    num_examples: float
    partition_ids: tuple[int, ...]


def _get_record_weight(record: RecordDict, weighted_by_key: str) -> float:
    """Read the aggregation weight from the only MetricRecord in a reply."""
    metric_record = next(iter(record.metric_records.values()))
    return float(metric_record[weighted_by_key])


def aggregate_records_in_group(
    group_id: int,
    records: list[RecordDict],
    partition_ids: list[int],
    weighted_by_key: str,
    metrics_aggregation_fn: Callable[[list[RecordDict], str], MetricRecord],
) -> GroupAggregation:
    """Aggregate client models and metrics inside one group."""
    if not records:
        raise ValueError(f"Group {group_id} has no successful client replies.")

    num_examples = sum(
        _get_record_weight(record, weighted_by_key) for record in records
    )
    if num_examples <= 0:
        raise ValueError(f"Group {group_id} has a non-positive aggregation weight.")

    return GroupAggregation(
        group_id=group_id,
        arrays=aggregate_arrayrecords(records, weighted_by_key),
        metrics=metrics_aggregation_fn(records, weighted_by_key),
        num_examples=num_examples,
        partition_ids=tuple(sorted(partition_ids)),
    )


def aggregate_group_models(
    group_aggregations: list[GroupAggregation],
    weighted_by_key: str,
    arrayrecord_key: str,
    group_weight_mode: str = "examples",
) -> ArrayRecord:
    """Combine one model per group into the global model.

    Dois modos, e a escolha entre eles decide se o agrupamento e metodo ou
    contabilidade:

    "examples" -- cada grupo pesa o total de exemplos que trouxe. Composta com a
    media interna do grupo, essa escolha *telescopa*: o N_g que pondera o grupo
    cancela com o N_g que normaliza a media dentro dele, e o resultado e
    exatamente a media ponderada de todos os clients. Ou seja, agrupar vira
    apenas colocar parenteses numa soma, e o modelo global e identico ao do
    FedAvg puro (verificado ponta a ponta em 07/09/2026).

    "equal" -- todo grupo pesa 1, independentemente de quantos dados tem. Agora o
    N_g nao cancela e o agrupamento passa a mudar o modelo. E o que da sentido a
    grupo como unidade real (regiao, hospital, operadora) em vez de rotulo: um
    grupo com 500 exemplos influencia tanto quanto um com 12.000. Tambem limita
    o cliente dominante -- no Stack Exchange um unico autor detem 54% dos dados,
    e sob peso por exemplos ele arrasta o modelo global sozinho.

    Um grupo sem client sorteado nao chega aqui: a estrategia o pula antes. Isso
    importa mais no modo "equal", onde os grupos restantes passam a dividir todo
    o peso entre si -- com 4 grupos e um vazio, cada sobrevivente sobe de 1/4
    para 1/3.
    """
    if group_weight_mode not in ("examples", "equal"):
        raise ValueError(
            f"group_weight_mode={group_weight_mode!r} desconhecido; "
            "use 'examples' ou 'equal'."
        )

    group_records = [
        RecordDict(
            {
                arrayrecord_key: group.arrays,
                "group-weight": MetricRecord(
                    {
                        weighted_by_key: (
                            1.0
                            if group_weight_mode == "equal"
                            else group.num_examples
                        )
                    }
                ),
            }
        )
        for group in group_aggregations
    ]
    return aggregate_arrayrecords(group_records, weighted_by_key)
