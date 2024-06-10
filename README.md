# QuantiGPT

```shell
CORPUS=args # or CORPUS=kialo
```

## Predicting

```shell
poetry run python -m quantigpt predict-statements data/pattern-matches-$CORPUS.json data/predicted-statements-$CORPUS.json --sample-size 50 --skip-first 0
```

## Pretty Printing

```shell
poetry run python -m quantigpt prettify pattern-matches-$CORPUS.json data/predicted-statements-$CORPUS.json
```

## Augmentation

```shell
poetry run python -m quantigpt augment-statements data/predicted-statements-$CORPUS.json data/augmented-statements-$CORPUS.json
```

## Validation

```shell
poetry run python -m quantigpt predict-validations data/pattern-matches-$CORPUS.json data/augmented-statements-$CORPUS.json data/predicted-validations-$CORPUS.json
```

## Label Studio Export

```shell
poetry run python -m quantigpt export-labelstudio data/pattern-matches-$CORPUS.json data/augmented-statements-$CORPUS.json data/predicted-validations-$CORPUS.json data/labelstudio-$CORPUS.json
```
