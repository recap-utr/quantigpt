# QuantiGPT

## Predicting

```shell
poetry run python -m quantigpt predict data/input.json data/OUTPUT_FILE.json --corpus args --sample-size 100 --skip-first 0
```

## Pretty Printing

```shell
poetry run python -m quantigpt prettify data/input.json data/OUTPUT_FILE.json
```

## Validation

```shell
poetry run python -m quantigpt validate data/output-args.json data/output-args-with-wiki-tables.csv
poetry run python -m quantigpt validate data/output-kialo.json data/output-kialo-with-wiki-tables.csv
```


