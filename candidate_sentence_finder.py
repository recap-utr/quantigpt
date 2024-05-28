import json
import os
import re

from nltk.tokenize import sent_tokenize
from tqdm import tqdm


pattern_kialo_premise = "(\d+\.)+ (Pro|Con): "
path_to_kialo_merged = os.path.join("data", "argument_data", "kialo-merged")


def main():

    map_patternName_patternString = extract_patterns_from_json_file()
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences = {}
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences = extract_arguments_from_args_me_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString)
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences = extract_arguments_from_kialo_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString)
    write_json_file_with_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences)


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
    with open(os.path.join('data', 'argument_data', 'args-me-1.0-cleaned.json')) as args_me_json:
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


def extract_arguments_from_kialo_and_find_matches(map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences, map_patternName_patternString):
    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'] = {}

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
                                    map_dataset_argumentId_claimPremisesentencesStanceMatchingSentences['kialo'][argument_id]['stance'] = premise_stance

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
                     'pattern_name': pattern_name,
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
        file.close()


if __name__ == '__main__':
    main()
