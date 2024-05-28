import json


def main():
    restore_from_file()


def restore_from_file():

    map_dataset_claim_stance_premiseSentencesList_match = {}
    pattern_hub = set()


    with open('input.json', encoding='utf8') as file:
        content = json.load(file)

        for dataset in content:
            map_dataset_claim_stance_premiseSentencesList_match[dataset] = {}

            for arg_id in content[dataset]:

                claim = content[dataset][arg_id]['claim']
                stance = content[dataset][arg_id]['stance']
                premise_sentences_list = str(content[dataset][arg_id]['premise_sentences'])

                map_dataset_claim_stance_premiseSentencesList_match[dataset][claim] = {}
                map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance] = {}

                map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list] = {}

                matching_sentences = content[dataset][arg_id]['matching_sentences']
                for matching_sentence in matching_sentences:
                    sentence_id = matching_sentence['sentence_id']
                    sentence_text = matching_sentence['sentence_text']
                    pattern_name = matching_sentence['pattern_name']
                    pattern_string = matching_sentence['pattern_string']
                    operator = matching_sentence['operator']

                    map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id] = {}
                    map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id]['sentence_text'] = sentence_text
                    map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id]['pattern_name'] = pattern_name
                    map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id]['pattern_string'] = pattern_string
                    map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id]['operator'] = operator

                    pattern_hub.add(pattern_name + '\n' + pattern_string + '\n')

        # all patterns
        for entry in pattern_hub:
            print(entry)

        for dataset in map_dataset_claim_stance_premiseSentencesList_match:
            print('dataset: ' + str(dataset))

            # statistics (general)
            print('- found claims: ' + str(len(map_dataset_claim_stance_premiseSentencesList_match[dataset])))

            pro_stance_premises = 0
            con_stance_premises = 0
            for claim in map_dataset_claim_stance_premiseSentencesList_match[dataset]:
                for stance in map_dataset_claim_stance_premiseSentencesList_match[dataset][claim]:
                    if stance == 'PRO':
                        pro_stance_premises += 1
                    elif stance == 'CON':
                        con_stance_premises += 1
            print('-- with stance PRO: ' + str(pro_stance_premises))
            print('-- with stance CON: ' + str(con_stance_premises))
            print()

            # statistics (matches)
            matches_found = 0
            used_patterns = {}
            used_operators = {}
            for claim in map_dataset_claim_stance_premiseSentencesList_match[dataset]:
                for stance in map_dataset_claim_stance_premiseSentencesList_match[dataset][claim]:
                    for premise_sentences_list in map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance]:
                        matches_found += 1

                        for sentence_id in map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list]:

                            pattern_name = map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id]['pattern_name']
                            if pattern_name not in used_patterns:
                                used_patterns[pattern_name] = 1
                            else:
                                used_patterns[pattern_name] += 1

                            operator_name = map_dataset_claim_stance_premiseSentencesList_match[dataset][claim][stance][premise_sentences_list][sentence_id]['operator']
                            if operator_name not in used_operators:
                                used_operators[operator_name] = 1
                            else:
                                used_operators[operator_name] += 1

            print('used patterns:')
            for pattern in used_patterns:
                print(' ---- ' + str(pattern) + ': ' + str(used_patterns[pattern]))
            print()
            print('used operators:')
            for operator in used_operators:
                print(str(operator) + ';' + str(used_operators[operator]))
            print()
            print('---------------')


if __name__ == '__main__':
    main()
