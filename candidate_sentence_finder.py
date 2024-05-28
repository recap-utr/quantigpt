import json
import os
import re

from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def main():

    map_patternName_patternString = extract_patterns_from_json_file()
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences = {}
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences = extract_arguments_from_args_me_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString)
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences = extract_arguments_from_kialo_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString)
    write_json_file_with_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences)

    statistics()

def extract_patterns_from_json_file():
    map_patternName_patternString = {}
    with open(os.path.join('data', 'patterns.json')) as patterns_json:
        patterns_json_content = json.load(patterns_json)
        for pattern in patterns_json_content:
            pattern_name = pattern['pattern_name']
            pattern_string = pattern['pattern_string']
            map_patternName_patternString[pattern_name] = pattern_string
    return map_patternName_patternString


def extract_arguments_from_args_me_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString):
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'] = {}
    with open(os.path.join('data', 'args-me-1.0-cleaned.json')) as args_me_json:
        args_me_json_content = json.load(args_me_json)

        for argument in tqdm(args_me_json_content["arguments"]):

            id = argument["id"]
            conclusion = argument["conclusion"]

            for premise in argument['premises']:
                premise_sentences = sent_tokenize(premise["text"])
                stance = premise["stance"]

                if id not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences:
                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id] = {}

                if conclusion not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]:
                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]['claim'] = conclusion

                if str(premise_sentences) not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]:
                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]['premise_sentences'] = premise_sentences

                if stance not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]:
                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]['stance'] = stance

                if 'matching_sentences' not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]:
                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]['matching_sentences'] = []
                map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['args'][id]['matching_sentences'] += find_matches(
                    map_patternName_patternString,
                    premise_sentences
                )

    return map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences


pattern_kialo_premise = "(\d+\.)+ (Pro|Con): "

def extract_arguments_from_kialo_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString):
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'] = {}

    path_to_kialo_merged = os.path.join("data", "argument_data", "kialo-merged")

    for root, dirs, files in tqdm(os.walk(path_to_kialo_merged)):
        for file in files:
            if file.endswith(".txt"):

                with open(os.path.join(path_to_kialo_merged, file), 'r', encoding="utf8") as kialo_file:
                    kialo_file_lines = kialo_file.readlines()
                    claim = None

                    counter = 0
                    for line in kialo_file_lines:
                        if "Discussion Title: " in line:
                            claim = line.replace("Discussion Title: ", "")
                        else:
                            counter += 1

                            premise_matcher = re.search(pattern_kialo_premise, line)
                            if premise_matcher:
                                argument_id = os.path.join(path_to_kialo_merged, file) + "_" + str(counter)
                                premise_id = premise_matcher.group(1)
                                premise_stance = premise_matcher.group(2)
                                premise_sentences = sent_tokenize(line.replace(premise_matcher.group(0), ""))

                                if argument_id not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences:
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id] = {}

                                if claim not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]:
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['claim'] = claim

                                if str(premise_sentences) not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]:
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['premise_sentences'] = premise_sentences

                                if premise_stance not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]:
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['premise_stance'] = premise_stance

                                if claim not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]:
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['claim'] = claim

                                if 'matching_sentences' not in map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]:
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['matching_sentences'] = []
                                map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['matching_sentences'] += find_matches(
                                    map_patternName_patternString,
                                    premise_sentences
                                )

                    kialo_file.close()

    return map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences


def find_matches(map_patternName_patternString, premise_sentences):

    nodes = []

    sentence_id_counter = 0
    for premise_sentence in premise_sentences:

        for pattern_name in map_patternName_patternString:
            pattern_string = map_patternName_patternString[pattern_name]
            matcher = re.search(pattern_string, premise_sentence)

            if matcher:
                nodes.append(
                    {'sentence_id': sentence_id_counter,
                     'sentence_text': premise_sentence,
                     'patternName': pattern_name,
                     'pattern_string': pattern_string,
                     'operator': matcher.group()}
                )
        sentence_id_counter += 1

    return nodes


def write_json_file_with_matches(json_content):

    json_content_to_print = {}

    for dataset in json_content:
        json_content_to_print[dataset] = {}
        for id in json_content[dataset]:
            if len(json_content[dataset][id]['matching_sentences']) > 0:
                json_content_to_print[dataset][id] = json_content[dataset][id]

    with open(os.path.join('data', 'input.json'), 'w') as file:
        json.dump(json_content_to_print, file, indent=4)


def statistics():

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
