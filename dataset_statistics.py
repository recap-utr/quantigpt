import csv
import json
import os
import re
from statistics import mean, median

from nltk import sent_tokenize
from tqdm import tqdm

from candidate_sentence_finder import path_to_kialo_merged, pattern_kialo_premise


def main():
    # statistics_about_datasets()
    # statistics_about_preliminary_findings()
    statistics_about_methods()


def statistics_about_datasets():

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
        print('mean: ' + str(mean(counts)))
        print('median: ' + str(median(counts)))


def statistics_about_methods():

    for corpus_name in ['args', 'kialo']:

        statistics_about_predicted_statements(corpus_name, False)
        statistics_about_predicted_statements(corpus_name, True)
        statistics_about_augmented_statements(corpus_name)
        statistics_about_predicted_validations(corpus_name)


def statistics_about_predicted_statements(corpus_name, use_extended):
    map_entities = {}
    map_traits = {}
    map_operators = {}
    map_quantities = {}

    dataset_name_for_output = 'predicted_statements_with_dataset_' + corpus_name + ('-extended' if use_extended else '')
    corpus_name = 'predicted-statements-' + corpus_name + ('-extended' if use_extended else '')

    with open(os.path.join('data', corpus_name + '.json')) as predicted_statements_file:
        json_content = json.load(predicted_statements_file)
        for arg_id in json_content:
            for premise in json_content[arg_id]:
                increase_number_of_appearance(map_entities, premise['entity_1'])
                increase_number_of_appearance(map_entities, premise['entity_2'])
                increase_number_of_appearance(map_traits, premise['trait'])
                increase_number_of_appearance(map_operators, premise['operator'])
                increase_number_of_appearance(map_quantities, premise['quantity'])

        predicted_statements_file.flush()
        predicted_statements_file.close()

    header = ['dataset_name_for_output', 'map_name', 'key', 'number_of_appearances']
    output = [header]
    output += (print_map_in_csv_format(dataset_name_for_output, 'map_entities', map_entities))
    output += (print_map_in_csv_format(dataset_name_for_output, 'map_traits', map_traits))
    output += (print_map_in_csv_format(dataset_name_for_output, 'map_operators', map_operators))
    output += (print_map_in_csv_format(dataset_name_for_output, 'map_quantities', map_quantities))

    with open(os.path.join("data", "statistics", "statistics_about_" + str(dataset_name_for_output) + "_.csv"), 'w', newline='\n') as csv_output_file:
        csv_writer = csv.writer(csv_output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(output)
        csv_output_file.flush()
        csv_output_file.close()


def increase_number_of_appearance(map_with_number_of_appearances, attribute_name):
    if attribute_name not in map_with_number_of_appearances:
        map_with_number_of_appearances[attribute_name] = 1
    else:
        map_with_number_of_appearances[attribute_name] += 1


def print_map_in_csv_format(dataset_name_for_output, map_name, map_to_print):
    output = []
    for key in map_to_print:
        output.append([str(dataset_name_for_output), str(map_name), str(key), str(map_to_print[key])])

    return output


def statistics_about_augmented_statements(corpus_name):

    dataset_name_for_output = 'augmented_statements_with_dataset ' + corpus_name
    corpus_name = 'augmented-statements-' + corpus_name

    map_augmented_statements = {}
    map_augmented_statements['title_lengths'] = []
    map_augmented_statements['google_snippet_descpription_lengths'] = []
    map_augmented_statements['context_found_by_snippet_lengths'] = []
    map_augmented_statements['summary_lengths'] = []
    map_augmented_statements['short_description_lengths'] = []
    map_augmented_statements['number_of_wiki_tables'] = []
    map_augmented_statements['lengths_of_wiki_tables'] = []

    with open(os.path.join('data', corpus_name + '.json')) as predicted_statements_file:
        json_content = json.load(predicted_statements_file)
        for arg_id in json_content:
            for premise in json_content[arg_id]:

                number_of_wikipedia_results = len(premise['results'])
                title_lengths = []
                google_snippet_descpription_lengths = []
                context_found_by_snippet_lengths = []
                summary_lengths = []
                short_description_lengths = []
                number_of_wiki_tables = []
                lengths_of_wiki_tables = []

                for result in premise['results']:
                    title_lengths.append(len(result['title']))
                    google_snippet_descpription_lengths.append(len(result['google_snippet_description']))
                    context_found_by_snippet_lengths.append(len(result['context_found_by_snippet']))
                    summary_lengths.append(len(result['summary']))
                    short_description_lengths.append(len(result['short_description']))
                    number_of_wiki_tables.append(len(result['wiki_tables']))
                    for wiki_table in result['wiki_tables']:
                        lengths_of_wiki_tables.append(len(wiki_table['wiki_table']))

                map_augmented_statements['title_lengths'].append(title_lengths)
                map_augmented_statements['google_snippet_descpription_lengths'].append(google_snippet_descpription_lengths)
                map_augmented_statements['context_found_by_snippet_lengths'].append(context_found_by_snippet_lengths)
                map_augmented_statements['summary_lengths'].append(summary_lengths)
                map_augmented_statements['short_description_lengths'].append(short_description_lengths)
                map_augmented_statements['number_of_wiki_tables'].append(number_of_wiki_tables)
                map_augmented_statements['lengths_of_wiki_tables'].append(lengths_of_wiki_tables)

        predicted_statements_file.flush()
        predicted_statements_file.close()

    header = ['dataset_name_for_output', 'measure', 'results_found_for_this_measure', 'number_of_results_found_for_this_measure', 'mean_number_of_results_found_for_this_measure', 'median_number_of_results_found_for_this_measure']
    output = [header]
    for measure in map_augmented_statements:
        for numbers in map_augmented_statements[measure]:
            output.append([str(dataset_name_for_output),
                             str(measure),
                             str(numbers),
                             str(len(numbers)),
                             str(mean(numbers) if len(numbers) > 0 else ''),
                             str(median(numbers) if len(numbers) > 0 else '')])

    with open(os.path.join("data", "statistics", "statistics_about_" + str(dataset_name_for_output) + ".csv"), 'w', newline='\n') as csv_output_file:
        csv_writer = csv.writer(csv_output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(output)
        csv_output_file.flush()
        csv_output_file.close()


def statistics_about_predicted_validations(corpus_name):

    map_reasoning = {}
    map_validation = {}

    dataset_name_for_output = 'predicted_validations_with_dataset ' + corpus_name
    corpus_name = 'predicted-validations-' + corpus_name

    with open(os.path.join('data', corpus_name + '.json')) as predicted_validations_file:
        json_content = json.load(predicted_validations_file)
        for arg_id in json_content:
            for entry in json_content[arg_id]:
                increase_number_of_appearance(map_validation, entry['validation'])
                increase_number_of_appearance(map_reasoning, entry['reasoning'])

        predicted_validations_file.flush()
        predicted_validations_file.close()

    header = ['dataset_name_for_output', 'map_name', 'key', 'number_of_appearances']
    output = [header]
    output += (print_map_in_csv_format(dataset_name_for_output, 'map_validation', map_validation))
    output += (print_map_in_csv_format(dataset_name_for_output, 'map_reasoning', map_reasoning))

    with open(os.path.join("data", "statistics", "statistics_about_" + str(dataset_name_for_output) + "_.csv"), 'w', newline='\n') as csv_output_file:
        csv_writer = csv.writer(csv_output_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(output)
        csv_output_file.flush()
        csv_output_file.close()


if __name__ == '__main__':
    main()
