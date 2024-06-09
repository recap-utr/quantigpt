import asyncio
import json
import random
import time
from pathlib import Path
from typing import Annotated, Any, Mapping, Optional, cast

import openai
import orjson
import requests
import tiktoken
import typer
from bs4 import BeautifulSoup
from googlesearch import search
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam as ChatMessage
from openai.types.chat.completion_create_params import Function, FunctionCall
from rank_bm25 import BM25Okapi
from tqdm import tqdm

encoder = tiktoken.get_encoding("cl100k_base")


def token_length(text: str) -> int:
    return len(encoder.encode(text))


random.seed(42)

app = typer.Typer(pretty_exceptions_enable=False)

Dataset = Mapping[str, Any]
Datasets = Mapping[str, Dataset]
AugmentedDataset = Mapping[str, Any]
AugmentedDatasets = Mapping[str, AugmentedDataset]
Prediction = dict[str, Any]
Predictions = list[Prediction]

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
    pattern_matches_path: Path,
    predicted_statements_path: Path,
) -> None:
    predicted_statements = orjson.loads(predicted_statements_path.read_bytes())

    predictions_map: Mapping[str, Predictions] = predicted_statements
    datasets: Datasets = orjson.loads(pattern_matches_path.read_bytes())

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
def validate(
    predicted_statements_path: Path,
    output_path: Path,
    checkpoints_path: Path = Path("data/validate-checkpoints.log"),
) -> None:
    # init
    count_tableId = 0
    map_argId_premise = {}
    checkpoints_set = set()

    if checkpoints_path.exists():
        with checkpoints_path.open("r", encoding="utf-8") as fp:
            checkpoints_set = set(fp.readlines())

    with predicted_statements_path.open("r", encoding="utf-8") as fp:
        data = orjson.loads(fp.read())

    # iterate each argument
    for arg_id in tqdm(data, desc="arg_id"):
        # update checkpoints
        if arg_id in checkpoints_set:
            continue

        map_argId_premise[arg_id] = []

        for premise in data[arg_id]:
            # extract data
            premise_id = premise["premise_id"]
            entity_1 = premise["entity_1"]
            entity_2 = premise["entity_2"]
            trait = premise["trait"]
            operator = premise["operator"]
            quantity = premise["quantity"]

            # init extended premise node
            node_premise_with_source = {}
            node_premise_with_source.update(premise)
            node_premise_with_source["results"] = []

            # sleep
            time.sleep(60)
            random_sleep_interval = random.randint(50, 60)

            # identify Wikipedia pages by Google search
            google_search_string = f"{entity_1} {trait} {quantity} times {operator} than {entity_2} site:en.wikipedia.org"
            node_premise_with_source["google_search_string"] = google_search_string

            for result in search(
                google_search_string,
                lang="en",
                sleep_interval=random_sleep_interval,
                timeout=120,
                advanced=True,
                num_results=10,
            ):
                url = result.url
                url = cast(str, url)

                snippet_title = result.title
                snippet_description = result.description

                # assign wiki results
                premise_result = {
                    "url": url,
                    "title": snippet_title,
                    "google_snippet_description": snippet_description,
                }

                thepage = requests.get(url)
                soup = BeautifulSoup(thepage.text, "html.parser")

                # more context identified by google search snippet in Wikipedia article
                paragraphs = soup.text.split("\n\n")
                tokenized_corpus = [doc.split(" ") for doc in paragraphs]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = snippet_description.split(" ")
                doc_scores = bm25.get_scores(tokenized_query)
                best_position = -1
                best_score = -1.0
                current_position = 0

                for score in doc_scores:
                    if best_position == -1 or float(score) > best_score:
                        best_position = current_position
                        best_score = float(score)

                    current_position += 1

                context_found_by_snippet = paragraphs[best_position]
                premise_result["context_found_by_snippet"] = context_found_by_snippet

                # short summary of Wikipedia article (text before first headline)
                content_div_abstract = ""

                for paragraph in soup.select(
                    "#mw-content-text > .mw-parser-output > p"
                ):
                    content_div_abstract += paragraph.get_text() + "\n"
                    next_sibling = paragraph.find_next_sibling()

                    if next_sibling is not None and next_sibling.name == "h2":
                        break

                premise_result["summary"] = content_div_abstract

                # Short description in html
                html_short_description = soup.find(
                    "div",
                    {"class": "shortdescription nomobile noexcerpt noprint searchaux"},
                )
                premise_result["short_description"] = (
                    html_short_description.text if html_short_description else ""
                )

                # extract tables from Wikipedia
                wiki_tables = soup.find_all("table", {"class": "wikitable"})
                premise_result["wiki_tables"] = []
                for wiki_table in wiki_tables:
                    count_tableId += 1
                    final_table_id = (
                        str(arg_id) + "_" + str(premise_id) + "_" + str(count_tableId)
                    )

                    premise_wiki_table = {
                        "final_table_id": final_table_id,
                        "wiki_table": str(_remove_all_attrs(wiki_table)),
                    }

                    premise_result["wiki_tables"].append(premise_wiki_table)

                node_premise_with_source["results"].append(premise_result)

            map_argId_premise[arg_id].append(node_premise_with_source)

            # intermediate storage and update checkpoints
            with output_path.open("w", encoding="utf-8") as fp:
                json.dump(
                    map_argId_premise,
                    fp,
                    cls=ComplexEncoder,
                    default=str,
                )

            if arg_id not in checkpoints_set:
                with checkpoints_path.open("a", encoding="utf-8") as fp:
                    fp.write(arg_id + "\n")


# https://stackoverflow.com/a/57128498
def _remove_all_attrs(soup):
    for tag in soup.find_all(True):
        if tag.text == "a" and "href" in tag.attrs:
            continue
        else:
            tag.attrs = {}
    return soup


# https://docs.python.org/3/library/json.html#encoders-and-decoders
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return [obj.real, obj.imag]
        # Let the base class default method raise the TypeError
        return super().default(obj)


@app.command()
def predict_statements(
    pattern_matches_path: Path,
    output_path: Path,
    ids: Annotated[list[str], typer.Option(..., "--id", default_factory=list)],
    sample_size: Optional[int] = None,
    skip_first: Optional[int] = None,
    model: str = "gpt-4o-2024-05-13",
):
    assert not (ids and sample_size)
    assert not (ids and skip_first)
    assert pattern_matches_path.suffix == ".json"
    assert output_path.suffix == ".json"

    datasets: Datasets = orjson.loads(pattern_matches_path.read_bytes())

    dataset_ids = list(datasets.keys())
    random.shuffle(dataset_ids)

    if sample_size:
        ids = random.sample(dataset_ids, sample_size)

    if ids:
        datasets = {k: v for k, v in datasets.items() if k in ids}

    predictions = asyncio.run(_predict_statements_wrapper(datasets, model))

    with output_path.open("wb", encoding="utf-8") as fp:
        fp.write(orjson.dumps(predictions))


async def _predict_statements_wrapper(
    datasets: Datasets, model: str
) -> Mapping[str, Predictions]:
    client = openai.AsyncOpenAI()

    with Path("./predict_statements.json").open("r", encoding="utf-8") as fp:
        schema = orjson.loads(fp.read())

    return dict(
        await asyncio.gather(
            *(
                _predict_statements(id, dataset, client, model, schema)
                for id, dataset in datasets.items()
            )
        )
    )


async def _predict_statements(
    id: str, dataset: Dataset, client: openai.AsyncClient, model: str, schema: Any
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

## Task Description

An argument consists of a claim (a statement that is being argued) and a premise (a statement that supports or attacks the claim).
The stance indicates whether the premise supports or attacks the claim.
The goal is to extract quantity statements from the premise that are relevant to the claim.
A quantity statement consists of two entities (e.g., "computers" and "consoles"), a trait between them (e.g., "cost"), an operator (e.g., "greater"), and a quantity (e.g., "2").
The operator indicates the relationship between the two entities, and the quantity specifies the amount of the trait that one entity has compared to the other.

## Input

You will be provided with a claim, its premise, and the stance between them.
As a starting point, a pattern-based approach has been used to identify sentences in the premise that contain some free-form operator.
The operator indicates the relationship between two currently unknown entities in the sentence.
As additional context, you are provided the entire regex pattern that matched the sentence together with the operator.

## Output

You shall extract all relevant information to call the function `predict_statements`.

## Constraints

If `quantity == 1.0`, the operator `equal` or `approx` must be used.
If `quantity` is any other value, the operator must be one of the other four options.
The `premise_id` will later be used to match the extracted quantity statements with the provided premise, so make sure to keep it.
"""

    res = await fetch_openai(
        client,
        model,
        user_prompt,
        system_prompt,
        [{"name": "predict_statements", "parameters": schema}],
        {"name": "predict_statements"},
    )

    assert res.function_call is not None

    statements: list[dict[str, Any]] = orjson.loads(res.function_call.arguments)[
        "statements"
    ]

    print(f"Processed {id}")

    return id, statements


@app.command()
def predict_validations(
    pattern_matches_path: Path,
    augmented_statements_path: Path,
    output_path: Path,
    ids: Annotated[list[str], typer.Option(..., "--id", default_factory=list)],
    model: str = "gpt-4o-2024-05-13",
):
    assert augmented_statements_path.suffix == ".json"
    assert output_path.suffix == ".json"

    augmented_statements = orjson.loads(augmented_statements_path.read_bytes())
    pattern_matches = orjson.loads(pattern_matches_path.read_bytes())

    if ids:
        augmented_statements = {
            k: v for k, v in augmented_statements.items() if k in ids
        }

    predictions = asyncio.run(
        _predict_validations_wrapper(pattern_matches, augmented_statements, model)
    )

    with output_path.open("wb", encoding="utf-8") as fp:
        fp.write(orjson.dumps(predictions))


async def _predict_validations_wrapper(
    pattern_matches: Datasets, augmented_statements: AugmentedDatasets, model: str
) -> Mapping[str, Prediction]:
    client = openai.AsyncOpenAI()

    with Path("./predict_validation.json").open("r", encoding="utf-8") as fp:
        schema = orjson.loads(fp.read())

    return dict(
        await asyncio.gather(
            *(
                _predict_validation(
                    id,
                    pattern_matches[id],
                    augmented_dataset,
                    client,
                    model,
                    schema,
                )
                for id, augmented_dataset in augmented_statements.items()
            )
        )
    )


async def _predict_validation(
    id: str,
    original_dataset: Dataset,
    augmented_dataset: AugmentedDataset,
    client: openai.AsyncClient,
    model: str,
    schema: Any,
) -> tuple[str, Prediction]:
    wiki_results = []

    for result in augmented_dataset["results"]:
        if "en.wikipedia.org" in result["url"]:
            new_wiki_results = [
                *wiki_results,
                {
                    "url": result["url"],
                    "title": result["title"],
                    "google_snippet_description": result["google_snippet_description"],
                    "summary": result["summary"],
                    "short_description": result["short_description"],
                    "context_found_by_snippet": result["context_found_by_snippet"],
                    "tables": [table["wiki_table"] for table in result["wiki_tables"]],
                },
            ]

            # TODO: Document the limitation of wiki tables to 50000 tokens
            if token_length(orjson.dumps(new_wiki_results).decode()) < 50000:
                wiki_results = new_wiki_results

    user_prompt = orjson.dumps(
        {
            "claim_text": original_dataset["claim"],
            "premise_text": original_dataset["premise_sentences"][
                augmented_dataset["premise_id"]
            ],
            "stance": original_dataset["stance"].lower(),
            "entity_1": augmented_dataset["entity_1"],
            "entity_2": augmented_dataset["entity_2"],
            "trait": augmented_dataset["trait"],
            "operator": augmented_dataset["operator"],
            "quantity": augmented_dataset["quantity"],
            "google_search_string": augmented_dataset["google_search_string"],
            "wikipedia_search_results": wiki_results,
        }
    ).decode()
    system_prompt = """
You are an assistant that verifies quantitative statements via provided retrieval results.

## Task Description

In the previous step, you extracted a quantitative statement from an argument.
A quantity statement consists of two entities (e.g., "computers" and "consoles"), a trait between them (e.g., "cost"), an operator (e.g., "greater"), and a quantity (e.g., "2").
Via a web search, we identified relevant Wikipedia pages that contain additional context such as tables and summaries.
The goal is to validate the extracted quantity statement based on the provided context.

## Input

You will be provided with the extracted quantity statement, the claim, the premise, their stance, the web search string, and the Wikipedia search results.
The tables have been extracted in their HTML representation.
Only the given information shall be used to validate the quantity statement.

## Output

You shall extract all relevant information to call the function `predict_validation`.

## Constraints

Do not use any external information beyond the provided context.
If no data is available for the queried validation, respond with `unknown`.
"""

    res = await fetch_openai(
        client,
        model,
        user_prompt,
        system_prompt,
        [{"name": "predict_validation", "parameters": schema}],
        {"name": "predict_validation"},
    )

    assert res.function_call is not None

    prediction: dict[str, Any] = orjson.loads(res.function_call.arguments)

    print(f"Processed {id}")

    return id, prediction


async def fetch_openai(
    client: openai.AsyncClient,
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


def export_labelstudio(
    pattern_matches_path: Path,
    augmented_statements_path: Path,
    validated_statements_path: Path,
    output_path: Path,
):
    assert pattern_matches_path.suffix == ".json"
    assert output_path.suffix == ".json"

    pattern_matches: Datasets = orjson.loads(pattern_matches_path.read_bytes())
    augmented_statements = orjson.loads(augmented_statements_path.read_bytes())
    validated_statements = orjson.loads(validated_statements_path.read_bytes())
    export_data: list[dict[str, Any]] = []

    for id, validated_statement in validated_statements.items():
        pattern_match = pattern_matches[id]
        augmented_statement = augmented_statements[id]

        export_data.append(
            {
                # "id": id,
                "data": {
                    "formatted_data": f"""
<p><strong>Claim:</strong> {pattern_match['claim']}</p>
<p><strong>Premise:</strong> {pattern_match['premise_sentences'][augmented_statement['premise_id']]}</p>
<p><strong>Stance:</strong> {pattern_match['stance']}</p>
""".strip(),
                    "formatted_statement": f"""
<p><strong>Entity 1:</strong> {augmented_statement['entity_1']}</p>
<p><strong>Entity 2:</strong> {augmented_statement['entity_2']}</p>
<p><strong>Trait:</strong> {augmented_statement['trait']}</p>
<p><strong>Operator:</strong> {augmented_statement['operator']}</p>
<p><strong>Quantity:</strong> {augmented_statement['quantity']}</p>
""".strip(),
                    "formatted_validation": f"""
<p><strong>Validation:</strong> {validated_statement['validation']}</p>
<p><strong>Reasoning:</strong> {validated_statement['reasoning']}</p>
""".strip(),
                    "entity_1": augmented_statement["entity_1"],
                    "entity_2": augmented_statement["entity_2"],
                    "trait": augmented_statement["trait"],
                    "operator": augmented_statement["operator"],
                    "quantity": augmented_statement["quantity"],
                    "claim": pattern_match["claim"],
                    "premise": pattern_match["premise_sentences"][
                        augmented_statement["premise_id"]
                    ],
                    "stance": pattern_match["stance"],
                    "validation": validated_statement["validation"],
                    "reasoning": validated_statement["reasoning"],
                },
            }
        )

    with output_path.open("wb", encoding="utf-8") as fp:
        fp.write(orjson.dumps(export_data))


if __name__ == "__main__":
    app()
