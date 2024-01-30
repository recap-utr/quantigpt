import asyncio
from pathlib import Path
from typing import Annotated, Any, Mapping, Optional

import openai
import orjson
import typer
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam as ChatMessage
from openai.types.chat.completion_create_params import Function, FunctionCall

app = typer.Typer()

client = openai.AsyncOpenAI()

with Path("./schema.json").open("rb") as fp:
    schema = orjson.loads(fp.read())

Dataset = Mapping[str, Any]
Datasets = Mapping[str, Dataset]


@app.command()
def run(
    path: Path,
    ids: Annotated[
        Optional[list[str]], typer.Option(..., "--id", default_factory=list)
    ],
    model: str = "gpt-3.5-turbo",
):
    raw_data = orjson.loads(path.read_bytes())
    data = {}

    for dataset in raw_data.values():
        data |= dataset

    asyncio.run(run_async(data, ids or [], model))


async def run_async(datasets: Datasets, ids: list[str], model: str):
    if ids:
        datasets = {k: v for k, v in datasets.items() if k in ids}

    return await asyncio.gather(
        *(process_dataset(id, dataset, model) for id, dataset in datasets.items())
    )


async def process_dataset(id: str, dataset: Dataset, model: str):
    user_prompt = orjson.dumps(
        {
            "claim": dataset["claim"],
            "premise": " ".join(dataset["premise_sentences"]),
            "stance": dataset["stance"].lower(),
            "premise_matches": [
                {
                    "premise_sentence": entry["sentence_text"],
                    "pattern_name": entry["pattern_name"],
                    "operator": entry["operator"],
                }
                for entry in dataset["matching_sentences"]
            ],
        }
    ).decode()
    system_prompt = """
You will be provided with a claim, its premise, and the stance between them.
The goal is to extract quantity statements from the premise.
To make it easier for you, I already extracted sentences from the premise along with the operator.
Based on this information, you should be able to extract the quantity statements for each premise match.
"""

    res = await fetch_openai(
        model,
        user_prompt,
        system_prompt,
        [{"name": "predict_quantity_statements", "parameters": schema}],
        {"name": "predict_quantity_statements"},
    )

    assert res.function_call is not None

    args = orjson.loads(res.function_call.arguments)

    print(f"{id}: {args}")


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
