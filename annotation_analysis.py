import json
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import typer
from nltk.metrics import agreement

app = typer.Typer()

partly_distances: set[tuple[str, str]] = {
    # correct
    ("correct", "partly correct"),
    ("incorrect", "partly correct"),
    # valid
    ("valid", "partly valid"),
    ("invalid", "partly valid"),
}


def ordinal_distance(label1: str, label2: str) -> float:
    if label1 == label2:
        return 0.0

    if (label1, label2) in partly_distances or (label2, label1) in partly_distances:
        return 1.0

    return 3.0


def parse_export(path: Path) -> dict[str, Any]:
    with path.open("r") as file:
        data = json.load(file)

    output = {}

    for task in data:
        task_key = ":".join(
            [
                task["data"]["corpus_id"],
                task["data"]["premise"],
                task["data"]["trait"],
                task["data"]["operator"],
                task["data"]["entity_1"],
                task["data"]["entity_2"],
                task["data"]["quantity"],
            ]
        ).lower()

        assert task_key not in output, f"Duplicate task key {task_key} in {path}"
        assert len(task["annotations"]) == 1

        annotation = task["annotations"][0]

        task_value = {
            "data": task["data"],
            "annotation": {
                entry["from_name"]: next(iter(entry["value"].values()))[0]
                for entry in annotation["result"]
            },
        }
        output[task_key] = task_value

    return output


@app.command()
def convert(input_path: Path, output_path: Path) -> None:
    data = parse_export(input_path)

    with output_path.open("w") as file:
        json.dump(data, file, indent=2)


@app.command()
def statistics(
    field: str,
    paths: list[Path],
) -> None:
    label_occurences = defaultdict(int)
    total_labels = 0
    labels_per_prediction = defaultdict(lambda: defaultdict(int))

    for path in paths:
        data = parse_export(path)

        for task in data.values():
            label = task["annotation"][field]

            label_occurences[label] += 1
            total_labels += 1

            if field == "validation-rating":
                prediction = task["data"]["validation"]
                labels_per_prediction[prediction][label] += 1

    typer.echo(f"Total labels: {total_labels}")

    for label, occurences in label_occurences.items():
        typer.echo(f"{label}: {occurences} ({occurences / total_labels * 100:.2f}%)")

    if field == "validation-rating":
        typer.echo()
        typer.echo("Labels per prediction:")
        for prediction, labels in labels_per_prediction.items():
            typer.echo(f"Prediction: {prediction}")
            for label, occurences in labels.items():
                typer.echo(f"  {label}: {occurences}")


@app.command()
def iaa(
    coder1_path: Path,
    coder2_path: Path,
    fields: Annotated[list[str], typer.Option(..., "--field", default_factory=list)],
) -> None:
    coder1_data = parse_export(coder1_path)
    coder2_data = parse_export(coder2_path)

    tasks = set(coder1_data.keys()).intersection(coder2_data.keys())
    all_labels = []
    known_labels = []
    concordant_classes = defaultdict(int)
    disconcordant_classes = defaultdict(int)
    concordant_labels = 0
    discordant_labels = 0

    if not fields:
        fields = ["statement-rating", "validation-rating"]

    for task_key in tasks:
        coder1_task = coder1_data[task_key]
        coder2_task = coder2_data[task_key]

        for field in fields:
            label_id = f"{task_key}:{field}"
            coder1_label = coder1_task["annotation"][field].lower()
            coder2_label = coder2_task["annotation"][field].lower()

            all_labels.append(("coder1", label_id, coder1_label))
            all_labels.append(("coder2", label_id, coder2_label))

            if coder1_label != "unknown" and coder2_label != "unknown":
                known_labels.append(("coder1", label_id, coder1_label))
                known_labels.append(("coder2", label_id, coder2_label))

            if coder1_label == coder2_label:
                concordant_classes[coder1_label] += 1
                concordant_labels += 1
            else:
                disconcordant_classes[(coder1_label, coder2_label)] += 1
                discordant_labels += 1

    total_labels = concordant_labels + discordant_labels

    _echo_task("All", all_labels)
    _echo_task("Known", known_labels)
    typer.echo(
        f"Concordant labels: {concordant_labels} ({concordant_labels / total_labels * 100:.2f}%)"
    )
    typer.echo(
        f"Discordant labels: {discordant_labels} ({discordant_labels / total_labels * 100:.2f}%)"
    )
    typer.echo(f"Concordant classes: {concordant_classes}")
    typer.echo(f"Disconcordant classes: {disconcordant_classes}")


def _echo_task(title: str, labels: list[tuple[str, str, str]]) -> None:
    typer.echo(f"{title} labels ({len(labels) / 2}):")
    assert len(labels) > 0
    task = agreement.AnnotationTask(labels, distance=ordinal_distance)

    # typer.echo(f"Bennett's S: {task.S()}")
    # typer.echo(f"Scott's Pi: {task.pi()}")
    # typer.echo(f"Fleiss's Kappa: {task.multi_kappa()}")
    # typer.echo(f"Cohen's Kappa: {task.kappa()}")
    typer.echo(f"Cohen's Weighted Kappa: {task.weighted_kappa()}")
    typer.echo(f"Krippendorff's Alpha: {task.alpha()}")

    typer.echo()


if __name__ == "__main__":
    app()

# poetry run python -m quantigpt.iaa quantigpt/data/kilian-$CORPUS.json quantigpt/data/martin-$CORPUS.json
