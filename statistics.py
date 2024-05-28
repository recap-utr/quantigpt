import json
import os
import re

from nltk import sent_tokenize
from tqdm import tqdm

from candidate_sentence_finder import path_to_kialo_merged, pattern_kialo_premise


def main():
    statistics()
    statistics_about_preliminary_findings()


def statistics():

    # sentence counter for args.me
    with open(os.path.join('data', 'argument_data', 'args-me-1.0-cleaned.json')) as args_me_json:
        args_me_json_content = json.load(args_me_json)

        number_of_args_me_arguments = 0
        number_of_args_me_sentences = 0
        for argument in tqdm(args_me_json_content["arguments"]):
            for premise in argument['premises']:
                number_of_args_me_arguments += 1
                number_of_args_me_sentences += len(sent_tokenize(premise["text"]))

        print('number_of_args_me_arguments: ' + str(number_of_args_me_arguments))
        print('number_of_args_me_sentences: ' + str(number_of_args_me_sentences))


    # sentence counter for kialo
    number_of_kialo_arguments = 0
    number_of_kialo_sentences = 0

    for root, dirs, files in tqdm(os.walk(path_to_kialo_merged)):
        for file in files:
            if file.endswith(".txt"):

                with open(os.path.join(path_to_kialo_merged, file), 'r', encoding="utf8") as kialo_file:
                    kialo_file_lines = kialo_file.readlines()

                    counter = 0
                    for line in kialo_file_lines:
                        if "Discussion Title: " in line:
                            pass
                        else:
                            number_of_kialo_arguments += 1

                            premise_matcher = re.search(pattern_kialo_premise, line)
                            if premise_matcher:
                                number_of_kialo_sentences += len(sent_tokenize(line.replace(premise_matcher.group(0), "")))

    print('number_of_kialo_arguments: ' + str(number_of_kialo_arguments))
    print('number_of_kialo_sentences: ' + str(number_of_kialo_sentences))


    map_dataset_claim_stance_premiseSentencesList_match = {}
    pattern_hub = set()

    with open(os.path.join('data', 'input.json'), encoding='utf8') as file:
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
                    if stance == 'PRO' or stance == 'Pro':
                        pro_stance_premises += 1
                    elif stance == 'CON' or stance == 'Con':
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
        file.close()

def statistics_about_preliminary_findings():

    path_to_preliminary_output = os.path.join('data', 'statistics', 'preliminary_output.txt')
    with open(path_to_preliminary_output, encoding='utf8') as f:
        lines = f.readlines()

        counts = []

        for line in lines:
            match = re.match('count:(\d+)', line)
            if match and int(match.group(1)) != 6192:
                counts.append(int(match.group(1)))

        print(counts)
        print('count: ' + str(len(counts)))
        print('sum: ' + str(sum(counts)))
        print('mean: ' + str(statistics.mean(counts)))
        print('median: ' + str(statistics.median(counts)))