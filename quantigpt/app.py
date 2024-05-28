import asyncio
import json
import random
import time
from pathlib import Path
from typing import Annotated, Any, Mapping, Optional, cast

import openai
import orjson
import requests
import typer
from bs4 import BeautifulSoup
from googlesearch import search
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageParam as ChatMessage
from openai.types.chat.completion_create_params import Function, FunctionCall
from rank_bm25 import BM25Okapi
from tqdm import tqdm

random.seed(42)

app = typer.Typer()

with Path("./schema.json").open("r", encoding="utf-8") as fp:
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
def validate(
    input_path: Path,
    output_path: Path,
    checkpoints_path: Path = Path("data/validate-checkpoints.log"),
) -> None:
    # init
    count_tableId = 0
    map_argId_premise = {}

    with checkpoints_path.open("r", encoding="utf-8") as fp:
        checkpoints_set = set(fp.readlines())

    with input_path.open("r", encoding="utf-8") as fp:
        data = orjson.loads(fp.read())

    # iterate each argument
    for arg_id in tqdm(data["args"], desc="arg_id"):
        # update checkpoints
        if arg_id in checkpoints_set:
            continue

        map_argId_premise[arg_id] = []

        for premise in data["args"][arg_id]:
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
                    indent=4,
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
def predict(
    input_path: Path,
    output_path: Path,
    ids: Annotated[list[str], typer.Option(..., "--id", default_factory=list)],
    corpus: Annotated[str, typer.Option(...)],
    sample_size: Optional[int] = None,
    skip_first: Optional[int] = None,
    model: str = "gpt-4-turbo-preview",
):
    client = openai.AsyncOpenAI()

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

    predictions = asyncio.run(run_async(datasets, client, model))

    with output_path.open("wb", encoding="utf-8") as fp:
        fp.write(orjson.dumps({corpus: predictions}))


async def run_async(
    datasets: Datasets, client: openai.AsyncClient, model: str
) -> PredictionsMap:
    return dict(
        await asyncio.gather(
            *(
                process_dataset(id, dataset, client, model)
                for id, dataset in datasets.items()
            )
        )
    )


async def process_dataset(
    id: str, dataset: Dataset, client: openai.AsyncClient, model: str
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

If `quantity == 1.0`, the operator `equal` or `approx` must be used.
If `quantity` is any other value, the operator must be one of the other four options.
The `premise_id` will later be used to match the extracted quantity statements with the provided premise, so make sure to keep it.
"""

    res = await fetch_openai(
        client,
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


if __name__ == "__main__":
    app()
