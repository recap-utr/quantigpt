import asyncio
import random
from pathlib import Path
from typing import Annotated, Any, Mapping

import openai
import orjson
import typer
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam as ChatMessage
from openai.types.chat.completion_create_params import Function, FunctionCall

random.seed(0)

app = typer.Typer()

client = openai.AsyncOpenAI()

with Path("./schema.json").open("rb") as fp:
    schema = orjson.loads(fp.read())

Dataset = Mapping[str, Any]
Datasets = Mapping[str, Dataset]
Prediction = dict[str, Any]
Predictions = list[Prediction]
PredictionsMap = Mapping[str, Predictions]


@app.command()
def prettify(dataset_path: Path, predictions_path: Path) -> None:
    raw_data = orjson.loads(dataset_path.read_bytes())
    datasets: Datasets = {}

    for dataset in raw_data.values():
        datasets |= dataset

    predictions_map: PredictionsMap = orjson.loads(predictions_path.read_bytes())

    # TODO: print predictions with the dataset in a nice way


@app.command()
def run(
    input_path: Path,
    output_path: Path,
    ids: Annotated[list[str], typer.Option(..., "--id", default_factory=list)],
    sample_size: float = 1.0,
    model: str = "gpt-3.5-turbo",
):
    assert not (ids and sample_size < 1.0)
    assert input_path.suffix == ".json"
    assert output_path.suffix == ".json"

    raw_data = orjson.loads(input_path.read_bytes())
    datasets: Datasets = {}

    for dataset in raw_data.values():
        datasets |= dataset

    if ids:
        datasets = {k: v for k, v in datasets.items() if k in ids}
    elif sample_size < 1.0:
        sample = random.sample(list(datasets), int(len(datasets) * sample_size))
        datasets = {k: v for k, v in datasets.items() if k in sample}

    predictions = asyncio.run(run_async(datasets, model))

    with output_path.open("wb") as fp:
        fp.write(orjson.dumps(predictions))


async def run_async(datasets: Datasets, model: str) -> PredictionsMap:
    return dict(
        await asyncio.gather(
            *(process_dataset(id, dataset, model) for id, dataset in datasets.items())
        )
    )


async def process_dataset(
    id: str, dataset: Dataset, model: str
) -> tuple[str, Predictions]:
    user_prompt = orjson.dumps(
        {
            "premise": " ".join(dataset["premise_sentences"]),
            "claim": dataset["claim"],
            "stance": dataset["stance"].lower(),
            "pattern_matches": [
                {
                    "premise_sentence": entry["sentence_text"],
                    "pattern_name": entry["pattern_name"],
                    "pattern": entry["pattern_string"],
                    "operator": entry["operator"],
                }
                for entry in dataset["matching_sentences"]
            ],
        }
    ).decode()
    system_prompt = """
You will be provided with a claim, its premise, and the stance between them.
The goal is to extract quantity statements from the premise.
To make it easier for you, I used pattern to extract sentences containing some operator from the premise.
Based on this information, you should be able to extract the quantity statements for each provided match.
"""

    res = await fetch_openai(
        model,
        user_prompt,
        system_prompt,
        [{"name": "predict_quantity_statements", "parameters": schema}],
        {"name": "predict_quantity_statements"},
    )

    assert res.function_call is not None

    statements: list[dict[str, Any]] = orjson.loads(res.function_call.arguments)[
        "statements"
    ]

    return id, statements


async def fetch_openai(
    model: str,
    user_prompt: str,
    system_prompt: str,
    functions: list[Function] | NotGiven = NOT_GIVEN,
    function_call: FunctionCall | NotGiven = NOT_GIVEN,
) -> ChatCompletionMessage:
    system_message: ChatMessage = {
        "role": "system",
        "content": system_prompt,
    }
    user_message: ChatMessage = {
        "role": "user",
        "content": user_prompt,
    }

    response = await client.chat.completions.create(
        model=model,
        messages=[system_message, user_message],
        functions=functions,
        function_call=function_call,
    )

    return response.choices[0].message


if __name__ == "__main__":
    app()
