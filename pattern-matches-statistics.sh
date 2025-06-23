#!/bin/bash

# Extract pattern names from patterns.json
patterns=$(jq -r '.[].pattern_name' data/patterns.json)

# Define corpora
corpora=("kialo" "args")

# Loop over each corpus
for corpus in "${corpora[@]}"; do
    echo "=== Statistics for corpus: $corpus ==="
    
    # Total unique claims for this corpus
    total_claims=$(jq '[.[] | .claim] | unique | length' "data/pattern-matches-$corpus.json")
    echo "Total unique claims: $total_claims"
    echo
    
    # Statistics for each pattern
    for pattern in $patterns; do
        count=$(jq "[.[] | select(.matching_sentences[0].pattern_name == \"$pattern\") | .claim] | unique | length" "data/pattern-matches-$corpus.json")
        echo "$pattern: $count"
    done
    echo
done