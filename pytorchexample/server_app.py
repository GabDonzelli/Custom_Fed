"""ServerApp: escolhe a estrategia por configuracao e semeia a run.

Duas mudancas em relacao ao original:

1. A estrategia deixou de estar fixa no codigo e passou a vir de
   `strategy = "grouped" | "plain"` no pyproject.toml. As duas convivem no mesmo
   codigo, com as mesmas tarefas, o mesmo ResultsLogger e os mesmos
   hiperparametros -- entao comparar uma com a outra isola de fato a agregacao,
   em vez de comparar duas branches que podem ter divergido em outra coisa.

2. A run e semeada. Ver pytorchexample/seeding.py para as quatro fontes de
   aleatoriedade e o que cada uma faz. `seed = 0` desliga, para reproduzir o
   comportamento nao-deterministico antigo se algum dia for preciso.
"""
from logging import INFO

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid, ServerApp

from pytorchexample.results_logger import ResultsLogger
from pytorchexample.seeding import DEFAULT_SEED, seed_everything
from pytorchexample.strategy.grouped_fedavg import GroupedFedAvg
from pytorchexample.strategy.grouping import build_partition_groups
from pytorchexample.strategy.tracked_fedavg import TrackedFedAvg
from pytorchexample.tasks.registry import get_task

app = ServerApp()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_strategy(context: Context, num_partitions: int):
    """Pick the aggregation strategy named in the run config."""
    nome = str(context.run_config.get("strategy", "grouped")).lower()
    comum = {
        "fraction_train": float(context.run_config.get("fraction-train", 1.0)),
        "fraction_evaluate": float(context.run_config["fraction-evaluate"]),
        "min_train_nodes": int(context.run_config.get("min-train-nodes", 2)),
        # Flower defaults this to 2 as well. Left unset, a run with a single
        # client trains fine and then hangs forever on "Waiting for nodes to
        # connect: 1 connected (minimum required: 2)" in the evaluate phase.
        "min_evaluate_nodes": int(context.run_config.get("min-evaluate-nodes", 2)),
        "min_available_nodes": num_partitions,
        "weighted_by_key": "num-examples",
    }

    if nome == "plain":
        log(INFO, "strategy=plain: FedAvg puro, sem agrupamento de particoes")
        return TrackedFedAvg(**comum)

    if nome != "grouped":
        raise ValueError(f"strategy={nome!r} desconhecida; use 'grouped' ou 'plain'.")

    manual = context.run_config.get("partition-groups")
    group_seed = context.run_config.get("group-seed")
    partition_groups = build_partition_groups(
        group_mode=str(context.run_config.get("group-mode", "sequential")),
        num_partitions=num_partitions,
        num_groups=int(context.run_config["num-groups"]),
        manual_specification=str(manual) if manual is not None else None,
        seed=int(group_seed) if group_seed is not None else None,
    )
    modo_peso = str(context.run_config.get("group-weight", "examples")).lower()
    log(
        INFO,
        "strategy=grouped: %d grupos, peso grupo->global = %s",
        len(partition_groups),
        modo_peso,
    )
    for group_id, partition_ids in partition_groups.items():
        log(
            INFO,
            "configured group %d with partitions %s",
            group_id,
            list(partition_ids),
        )
    return GroupedFedAvg(
        partition_groups=partition_groups,
        group_weight_mode=modo_peso,
        **comum,
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run federated training with the configured strategy."""
    seed = int(context.run_config.get("seed", DEFAULT_SEED))
    if seed:
        # Antes de create_model(): e esta chamada que fixa os pesos iniciais, e
        # portanto o round 0. Tambem alcanca o `random` global do Python, de
        # onde o sample_nodes do Flower sorteia os clients.
        seed_everything(seed)
        log(INFO, "run semeada com seed=%d (determinismo ligado)", seed)
    else:
        log(INFO, "seed=0: run NAO semeada, resultados nao sao reproduziveis")

    task_name = str(context.run_config["task-name"])
    num_partitions = int(context.run_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])

    task = get_task(task_name)
    global_model = task.create_model()
    initial_arrays = ArrayRecord(task.get_federated_arrays(global_model))
    centralized_testloader = task.load_centralized_data(
        num_partitions=num_partitions,
        batch_size=batch_size,
    )

    strategy = _build_strategy(context, num_partitions)
    results_logger = ResultsLogger()

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """Evaluate the current global model on centralized test data."""
        model = task.create_model()
        task.load_federated_arrays(model, arrays.to_torch_state_dict())
        evaluation_metrics = task.evaluate(model, centralized_testloader, DEVICE)

        results_logger.log_round(
            round_num=server_round,
            accuracy=evaluation_metrics.get("accuracy", 0.0),
            loss=evaluation_metrics.get("loss", 0.0),
            num_clients=strategy.last_train_client_count,
        )
        return MetricRecord(evaluation_metrics)

    num_rounds = int(context.run_config["num-server-rounds"])
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        # A semente viaja no config: e assim que o ClientApp sabe de que run faz
        # parte e consegue derivar a propria semente, por particao e por round.
        train_config=ConfigRecord(
            {"lr": float(context.run_config["learning-rate"]), "seed": seed}
        ),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    strategy.log_participation_summary(
        num_rounds=num_rounds, num_partitions=num_partitions
    )
    results_logger.close()

    if bool(context.run_config["save-model"]):
        # Whatever the task chose to federate. For a from-scratch task that is
        # the full model; for a LoRA task it is only the adapters, which are
        # useless without reloading the pretrained base they were trained on.
        torch.save(result.arrays.to_torch_state_dict(), "final_model.pt")
