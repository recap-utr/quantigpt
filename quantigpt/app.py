import asyncio
import random
from pathlib import Path
from typing import Annotated, Any, Mapping, Optional

import openai
import orjson
import typer
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam as ChatMessage
from openai.types.chat.completion_create_params import Function, FunctionCall

random.seed(42)

app = typer.Typer()

client = openai.AsyncOpenAI()

with Path("./schema.json").open("rb") as fp:
    schema = orjson.loads(fp.read())

Dataset = Mapping[str, Any]
Datasets = Mapping[str, Dataset]
Prediction = dict[str, Any]
Predictions = list[Prediction]
PredictionsMap = Mapping[str, Predictions]

operator_map: dict[str, str] = {
    "greater": ">",
    "less": "<",
    "equal": "=",
    "approx": "≈",
    "greater_or_equal": ">=",
    "less_or_equal": "<=",
}


@app.command()
def prettify(
    input_path: Path,
    output_path: Path,
) -> None:
    loaded_output = orjson.loads(output_path.read_bytes())
    assert len(loaded_output) == 1
    corpus: str = next(iter(loaded_output))

    predictions_map: PredictionsMap = loaded_output[corpus]
    datasets: Datasets = orjson.loads(input_path.read_bytes())[corpus]

    for id, predictions in predictions_map.items():
        dataset = datasets[id]
        print(f"Claim: {dataset['claim'].strip()}")
        print(f"Stance: {dataset['stance'].lower()}")

        for input_match in dataset["matching_sentences"]:
            prediction = next(
                (
                    x
                    for x in predictions
                    if x["premise_id"] == input_match["sentence_id"]
                ),
                None,
            )
            formatted_prediction = "n/a"

            if prediction is not None:
                operator = operator_map.get(prediction.get("operator", ""), "n/a")
                quantity = prediction.get("quantity", "n/a")
                entity1 = prediction.get("entity_1", "n/a")
                entity2 = prediction.get("entity_2", "n/a")
                trait = prediction.get("trait", "n/a")
                formatted_prediction = (
                    f"'{entity1}' {operator} '{entity2}': {quantity}x {trait}".lower()
                )

            print(f"- Premise: {input_match['sentence_text']}")
            print(f"  Prediction: {formatted_prediction}")

        print()


@app.command()
def predict(
    input_path: Path,
    output_path: Path,
    ids: Annotated[list[str], typer.Option(..., "--id", default_factory=list)],
    corpus: Annotated[str, typer.Option(...)],
    sample_size: Optional[int] = None,
    skip_first: Optional[int] = None,
    model: str = "gpt-4-turbo-preview",
):
    assert not (ids and sample_size)
    assert not (ids and skip_first)
    assert input_path.suffix == ".json"
    assert output_path.suffix == ".json"

    corpora = orjson.loads(input_path.read_bytes())
    datasets: Datasets = corpora[corpus]

    dataset_ids = list(datasets.keys())
    random.shuffle(dataset_ids)

    if sample_size:
        ids = random.sample(dataset_ids, sample_size)

    if ids:
        datasets = {k: v for k, v in datasets.items() if k in ids}

    predictions = asyncio.run(run_async(datasets, model))

    with output_path.open("wb") as fp:
        fp.write(orjson.dumps({corpus: predictions}))


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
            "premises": dataset["premise_sentences"],
            "claim": dataset["claim"],
            "stance": dataset["stance"].lower(),
            "pattern_matches": [
                {
                    "premise_id": entry["sentence_id"],
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
You are an assistant that extracts quantitative statements from arguments.
Each argument consists of a claim (a statement that is being argued) and a premise (a statement that supports or attacks the claim).
The stance indicates whether the premise supports or attacks the claim.
You will be provided with a claim, its premise, and the stance between them.
Your goal is to extract the quantity statements from the premise that are relevant to the claim.
As a starting point, a pattern-based approach has been used to identify sentences in the premise that contain some free-form operator.
The operator indicates the relationship between two currently unknown entities in the sentence.
As additional context, you are provided the entire regex pattern that matched the sentence together with the operator.
Your goal is to extract all relevant information to call the function `predict_quantity_statements`.
The premise id will later be used to match the extracted quantity statements with the provided premise, so make sure to keep it.
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

    print(f"Processed {id}")

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
