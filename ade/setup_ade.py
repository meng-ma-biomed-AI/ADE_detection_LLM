import itertools
import json
import os
import numpy as np
import pandas as pd
import requests
from collections import defaultdict
from statistics import mean, median

from spacy.lang.en import English
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tokenizers import BertWordPieceTokenizer


class Medication:
    def __init__(self):
        self.rxnorm_api_url = "https://rxnav.nlm.nih.gov/"

    def get_ingredients_for_drug(self, drug_name):
        """
        Given a drug name, get the ingredients for the drug
        :param drug_name: Drug name, preferrably brand name, as str
        :return: set of ingredients in the drug
        """
        get_request = "REST/rxclass/class/byDrugName.json?drugName={}&relaSource=ATC".format(drug_name)

        response = requests.get(self.rxnorm_api_url + get_request)
        response = response.json()

        ingredients = set()
        if 'rxclassDrugInfoList' in response:
            for ingredient in response['rxclassDrugInfoList']['rxclassDrugInfo']:
                ingredients.add(ingredient['minConcept']['name'])

        return ingredients

    def get_drug_brand_names_for_ingredient(self, ingredient_name):
        """
        Get brand name of drugs given ingredient name.
        At this point we only support querying one ingredient.
        It can however be extended to query multiple ingredients using the same API.
        :param ingredient_name: Str, ingredient name for a drug.
        :return: Set of brand names of drugs containing the ingredient
        """
        brand_names = set()

        get_cui_request = "REST/rxcui.json?name={}".format(ingredient_name)

        response = requests.get(self.rxnorm_api_url + get_cui_request)
        response = response.json()

        if len(response['idGroup']):
            rx_cui = response['idGroup']['rxnormId'][0]
            # print("Ingredient: ", ingredient_name, "RX-CUI: ", rx_cui)

            get_bn_request = 'REST/brands.json?ingredientids={}'.format(str(rx_cui))

            response = requests.get(self.rxnorm_api_url + get_bn_request)
            response = response.json()

            if 'conceptProperties' in response['brandGroup']:
                for brand in response['brandGroup']['conceptProperties']:
                    brand_names.add(brand['name'])

        return brand_names

    def get_equivalent_meds(self, meds):
        """
        Given a list of medication names, get equivalent medications using the RxNorm API.
        Note that this function relies on access to internet, since it calls the RxNorm APIs online.
        :param meds: Iterable of medication strings
        :return: Dictionary, {lowercased_med: [equivalent_meds]}.
                Note that the list of equivalent meds may contain med in a different casing scheme.
        """
        equiv_meds_dict = defaultdict(set)

        for med in meds:
            synonyms = set()
            synonyms.update(self.get_ingredients_for_drug(med))
            synonyms.update(self.get_drug_brand_names_for_ingredient(med))
            synonyms.remove(med)  # Remove the medication itself from its synonym list if it got added by mistake.

            if len(synonyms):
                equiv_meds_dict[med.lower()].update(synonyms)

        equiv_meds_dict.default_factory = None
        equiv_meds_dict = {key: list(value) for key, value in equiv_meds_dict.items()}

        return equiv_meds_dict


def read_tsv(fname, dir_name):
    """Reads a TSV file and returns a pandas dataframe"""
    df = pd.read_csv(os.path.join(dir_name, fname), sep='\t')
    return df


def get_data(fname_notes, fname_annots, dir_data):
    """
    Reads the annotated dataset and joins it with the notes data to obtain the annotated text, i.e., HPI section.
    :param fname_notes: TSV file with note text and corresponding deidentified note key.
    :param fname_annots: TSV file with annotated data.
                        It contains a deidentified note key along with concept and relation descriptions.
                        conc_relat is set to 1 if it represents a potential relation instead of an individual concept.
                        In relations, conc1 is either med/ae concept, and conc2 is hosp concept.
                        For both conc1 and conc2, we have start offsets, end offsets and phrases.
                        The field 'used' is set to 1 if the potential relation is a true relation added by the
                        annotators, 0 otherwise.
    :param dir_data: Directory containing both the data files.
    :return: Pandas dataframe containing the annotated data along with HPI text of that data.
    """
    annots_df = read_tsv(fname_annots, dir_data)
    # print(annots_df.columns)
    notes_df = read_tsv(fname_notes, dir_data)

    annotated_data = annots_df.join(notes_df.set_index('deid_note_key'), on='deid_note_key')
    annotated_data['HPI'] = annotated_data.apply(lambda row: row.note[row.hpi_beg: row.hpi_end], axis=1)

    return annotated_data


def get_med_synonym_dict(annotated_data, fname="med_synonyms.json", data_dir='../data'):
    """
    Returns the list of equivalent medications for every medication annotated in the dataset.
    If this list has been curated earlier, the code returns the list by loading it from a file.
    Else, it curates the list and saves it before learning it.
    :param annotated_data: Pandas dataframe of the annotated data
    :param fname: Filename for saving dictionary of equivalent medications.
    :param data_dir: Output directory name
    :return: Dictionary of equivalent medications
    """
    if os.path.exists(os.path.join(data_dir, fname)):
        with open(os.path.join(data_dir, fname)) as f:
            med_syn_dict = json.load(f)
            # med_syn_dict = {key: set(value) for key, value in med_syn_dict.items()}
            return med_syn_dict

    if annotated_data is None:
        raise ValueError("Please enter a valid dataset")

    meds = annotated_data[annotated_data['conc1_type'] == 'med']['conc1_phrase'].unique()
    med_syn_dict = Medication().get_equivalent_meds(meds)

    print(med_syn_dict)

    with open(os.path.join(data_dir, fname), "w") as outfile:
        json.dump(med_syn_dict, outfile)

    return med_syn_dict


def create_ade_triplets(annotated_data, fname_triplets, dir_data):
    """
    For all hospitalization mentions across all notes, retrieve all potential triplet relations between
    AEs, MEDs and HOSP events. If relation between (HOSP_i, AE) as well as (HOSP_i, MED) exists, then
    (HOSP_i, AE, MED) is considered to be a valid relation triple.
    :param annotated_data: pandas dataframe for annotated data
    :param fname_triplets: name of triplets file to save
    :param dir_data: output data directory
    :return: pandas dataframe containing all potential relation triples
    """
    # Retain only relations (the space of all possible relations + annotated relations) from the annotated data
    relations_data = annotated_data[annotated_data['conc_relat'] == 'relat']

    triplets_df = list()

    # Iterating over all the unique notes
    for note_id in relations_data['deid_note_id'].unique():
        df = relations_data[relations_data['deid_note_id'] == note_id]
        hpi = df['HPI'].unique()[0]
        patient_id = df['pat_id'].unique()[0]  # This works since we have only one note per patient in the dataset.

        # Iterating over all hospitalization mentions in the note
        for cur_hosp_start in df['conc2_start'].unique():
            hosp_df = df[df['conc2_start'] == cur_hosp_start]
            cur_hosp_end = int(hosp_df['conc2_end'].unique()[0])
            cur_hosp_str = hosp_df['conc2_phrase'].unique()[0]

            # Obtaining all possible meds and aes that could be related to the given hospitalization
            meds = hosp_df[hosp_df['conc1_type'] == 'med']
            aes = hosp_df[hosp_df['conc1_type'] == 'ae']

            # For all combinations of meds and aes that could be related to hosp, check whether they are a valid triple
            for (med_start, ae_start) in itertools.product(meds['conc1_start'].tolist(), aes['conc1_start'].tolist()):
                med_end = int(meds[meds['conc1_start'] == med_start]['conc1_end'].item())
                # ae_end = int(aes[aes['conc1_start'] == ae_start]['conc1_end'].item())
                ae_end = max(aes[aes['conc1_start'] == ae_start]['conc1_end'].unique()) # selecting longest span in case of overlapping spans

                med_str = meds[meds['conc1_start'] == med_start]['conc1_phrase'].item()
                ae_str = max(aes[aes['conc1_start'] == ae_start]['conc1_phrase'].unique(), key=len)

                ae_cui = aes[(aes['conc1_start'] == ae_start) & (aes['conc1_end'] == ae_end)]['conc1_cui'].item()

                # Whether the patient was on med before hosp
                is_valid_meds_hosp_rel = meds[meds['conc1_start'] == med_start]['used'].item()

                # Whether the ae was a cause of hosp
                is_valid_ae_hosp_rel = aes[(aes['conc1_start'] == ae_start) & (aes['conc1_end'] == ae_end)]['used'].item()

                # is a valid related triplet if both relations intersect on the hosp
                is_valid_triplet = is_valid_meds_hosp_rel and is_valid_ae_hosp_rel

                triplets_df.append({
                        'deid_note_id': note_id,
                        'PatientDurableKey': patient_id,
                        'HPI': hpi,
                        'hosp_start': int(cur_hosp_start),
                        'hosp_end': int(cur_hosp_end),
                        'hosp_str': cur_hosp_str,
                        'med_start': int(med_start),
                        'med_end': int(med_end),
                        'med_str': med_str,
                        'ae_start': int(ae_start),
                        'ae_end': int(ae_end),
                        'ae_str': ae_str,
                        'ae_cui': ae_cui,
                        'is_triplet': is_valid_triplet,

                })

                # TODO: SQL: group by MED (gen), AE to remove duplicate triples if needed.

    triplets_df = pd.DataFrame(triplets_df)
    triplets_df = triplets_df.loc[:, ~triplets_df.columns.str.contains('^Unnamed')]
    triplets_df.to_csv(os.path.join(dir_data, fname_triplets))


def compute_token_stats(df, vocab_file='/Users/madhumita.sushil/BERT_models/ucsf_bert_model/512/500k/vocab.txt',
                        is_lower=False):
    texts = df['HPI'].unique()

    tok = BertWordPieceTokenizer(vocab_file, lowercase=is_lower)
    tokens = [tok.encode(i).tokens for i in texts]
    token_lens = [len(i) for i in tokens]

    print("Min: ", min(token_lens),
          "Mean: ", mean(token_lens),
          "Median: ", median(token_lens),
          "Max: ", max(token_lens)
          )

    import plotly.express as px
    fig = px.histogram(pd.DataFrame(token_lens, columns=['n_tokens']), x="n_tokens")
    fig.show()


def analyze_triplets(fname_triplets, dir_data):
    df = pd.read_csv(os.path.join(dir_data, fname_triplets))
    print("Total number of patients/notes: ", len(df['PatientDurableKey'].unique()) )
    print("Number of positive triplets: ", len(df[df['is_triplet'] == 1]))
    print("Total number of negative triplets: ", len(df[df['is_triplet'] == 0]))

    compute_token_stats(df)


def analyze_ade_length_bias(fname_is_ade_in_note, dir_data,
                            vocab_file='/Users/madhumita.sushil/BERT_models/ucsf_bert_model/512/500k/vocab.txt',
                            is_lower=False):
    df = pd.read_csv(os.path.join(dir_data, fname_is_ade_in_note))

    df_pos = df[df['label'] == 1]
    df_neg = df[df['label'] == 0]

    texts_pos = df_pos['sentence'].to_list()
    texts_neg = df_neg['sentence'].to_list()

    tok = BertWordPieceTokenizer(vocab_file, lowercase=is_lower)

    token_lens_pos = [len(tok.encode(text).tokens) for text in texts_pos]
    token_lens_neg = [len(tok.encode(text).tokens) for text in texts_neg]

    df_pos = pd.DataFrame({
        'n_tokens': token_lens_pos,
        'label': ['ADE'] * len(token_lens_pos)
    })

    df_neg = pd.DataFrame({
        'n_tokens': token_lens_neg,
        'label': ['No_ADE'] * len(token_lens_neg)
    })
    import plotly.express as px
    fig = px.scatter(pd.concat([df_pos, df_neg], ignore_index=True), x='label', y="n_tokens",
                     width=500,
                     labels={
                         "label": "Label",
                         "n_tokens": "Number of tokens",
                     },
                     )
    fig.update_layout(font_size=16)
    fig.show()


def get_split_pt_idx(pt_idx, seed=1234):
    train_pts, other_pts = train_test_split(pt_idx, test_size=0.2,
                                            random_state=seed)

    dev_pts, test_pts = train_test_split(other_pts, test_size=0.5, random_state=seed)

    print("Num of patients in train, dev, test splits: ", len(train_pts), len(dev_pts), len(test_pts))

    return train_pts, dev_pts, test_pts


def get_splits_triplets(fname_triplets, med_syn_dict, replace_all_med_mentions, dir_data, fname_task='ade.csv'):
    df = pd.read_csv(os.path.join(dir_data, fname_triplets))
    annotated_pts = df[df['is_triplet'] == 1]["PatientDurableKey"].unique()

    train_pts, dev_pts, test_pts = get_split_pt_idx(annotated_pts)

    df_train = df[df["PatientDurableKey"].isin(train_pts)]
    df_dev = df[df["PatientDurableKey"].isin(dev_pts)]
    df_test = df[df["PatientDurableKey"].isin(test_pts)]

    df_train = _get_relation_triples_df(df_train, med_syn_dict, replace_all_med_mentions)
    df_dev = _get_relation_triples_df(df_dev, med_syn_dict, replace_all_med_mentions)
    df_test = _get_relation_triples_df(df_test, med_syn_dict, replace_all_med_mentions)

    df_train.to_csv(os.path.join(dir_data, 'train_'+fname_task))
    df_dev.to_csv(os.path.join(dir_data, 'dev_'+fname_task))
    df_test.to_csv(os.path.join(dir_data, 'test_'+fname_task))

    return df_train, df_dev, df_test


def _is_ae_near_hosp(text, hosp_start, hosp_end, ae_start, ae_end, nlp):
    """
    Identifies whether the ae is within a 2 sentence window of the hospitalization.
    :param text: Text that corresponds to hosp and AE events.
    :param hosp_start: Start character offset of hosp.
    :param hosp_end:  End character offset of hosp.
    :param ae_start: Start character offset of AE.
    :param ae_end: End character offset of AE.
    :param nlp: Spacy sentence detector pipeline
    :return: True if AE is within 2 sentences of hosp, False otherwise.
    """
    doc = nlp(text)

    hosp_sent_idx, ae_sent_idx = None, None

    for i, sent in enumerate(doc.sents):
        sent_start = sent.start_char
        sent_end = sent.end_char

        if hosp_start >= sent_start and hosp_end <= sent_end:
            hosp_sent_idx = i

        if ae_start >= sent_start and ae_end <= sent_end:
            ae_sent_idx = i

        if ae_sent_idx is not None and hosp_sent_idx is not None:
            break

    if ae_sent_idx is not None and hosp_sent_idx is not None and abs(ae_sent_idx - hosp_sent_idx) > 2:
        return False
    else:
        return True


def _replace_med_synonyms(synonym_dict, med_str, text):
    """
    Replace equivalent medication mentions in text with the current medications.
    :param synonym_dict: Dictionary containing a medication and all its equivalent medications
    :param med_str: medication name to replace equivalents with
    :param text: base text for replacement
    :return: text with equivalent meds replaced with the same name
    """

    if not med_str.lower() in synonym_dict:
        return text

    for syn in synonym_dict[med_str.lower()]:
        text = text.replace(syn, med_str)
        text = text.replace(syn.capitalize(), med_str)

    return text


def _get_replacements(text, start_offsets, end_offsets, med_syn_dict, replace_all_med_mentions):
    """
    :param text: Text to replace concepts with placeholders
    :param start_offsets: Dict {concept_type: start_offset_concept_type}
    :param end_offsets: Dict {concept_type: end_offset_concept_type}
    :param med_syn_dict: Dictionary of all equivalent meds
    :param replace_all_med_mentions: True if all the mentions of a medication string should be replaced with placeholder
    :return: text with concepts replaced with placeholders
    """
    replacement_dict = {
        'med': '@MED$',
        'ae': '@AE$',
        'hosp': '@HOSP$',
    }

    # Iteratively replacing the mentions to avoid conflicts related to changed offsets later.
    prev_phrase_len = 0
    prev_type_len = 0

    if replace_all_med_mentions:
        med_str = text[start_offsets['med']: end_offsets['med']]

    for (type, start) in sorted(start_offsets.items(), key=lambda item: item[1]):
        start = start - prev_phrase_len + prev_type_len
        end = end_offsets[type] - prev_phrase_len + prev_type_len

        text = text[:start] + replacement_dict[type] + text[end:]

        prev_phrase_len += (end - start)
        prev_type_len += len(replacement_dict[type])

    if replace_all_med_mentions:
        if med_syn_dict is not None:
            text = _replace_med_synonyms(med_syn_dict, med_str, text)
        text = text.replace(med_str, replacement_dict['med'])

    return text


def _get_relation_triples_df(df, med_syn_dict, replace_all_med_mentions=True):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    new_data = list()

    n_pos, n_neg = 0, 0

    nlp = English()
    nlp.add_pipe("sentencizer")

    for index, row in df.iterrows():
        text = row["HPI"]

        start_offsets = {
            'med': row["med_start"],
            'ae': row["ae_start"],
            'hosp': row["hosp_start"],
        }

        end_offsets = {
            'med': row["med_end"],
            'ae': row["ae_end"],
            'hosp': row["hosp_end"],
        }

        if not row["is_triplet"] and not _is_ae_near_hosp(text,
                                                          start_offsets['hosp'], end_offsets['hosp'],
                                                          start_offsets['ae'], end_offsets['ae'],
                                                          nlp):
            continue

        text = _get_replacements(text, start_offsets, end_offsets, med_syn_dict, replace_all_med_mentions)

        if str(row["is_triplet"]) == '0':
            n_neg += 1
        else:
            n_pos += 1

        new_data.append({
            "idx": row['PatientDurableKey']+'_'+row["deid_note_id"]+"_"+str(index),
            "sentence": text,
            "label": row["is_triplet"],
        })

    model_df = pd.DataFrame(new_data)
    model_df = model_df.drop_duplicates()

    model_df['label'].value_counts()

    print("Number of positive instances: ", n_pos)
    print("Number of negative instances: ", n_neg)

    return model_df


def get_df_is_ade_note(fname_triplets, data_dir, fname='ade_in_note.csv'):
    df = pd.read_csv(os.path.join(data_dir, fname_triplets))
    annotated_pts = df["PatientDurableKey"].unique()

    train_pts, dev_pts, test_pts = get_split_pt_idx(annotated_pts)

    df_train = df[df["PatientDurableKey"].isin(train_pts)]
    df_dev = df[df["PatientDurableKey"].isin(dev_pts)]
    df_test = df[df["PatientDurableKey"].isin(test_pts)]

    for df, set_type in zip((df_train, df_dev, df_test), ('train', 'dev', 'test')):
        is_ade_in_note = list()
        n_pos, n_neg = 0, 0

        for cur_note_idx in df.deid_note_id.unique():
            cur_df = df[df['deid_note_id'] == cur_note_idx]
            unique_triplets = cur_df['is_triplet'].unique()
            unique_triplets = [str(i) for i in unique_triplets]

            if '1' in unique_triplets:
                is_ade = 1
                n_pos += 1
            else:
                is_ade = 0
                n_neg += 1

            is_ade_in_note.append({
                'idx': cur_note_idx,
                'sentence': cur_df['HPI'].unique()[0],
                'label': is_ade,
            })

        print("Subset: ", set_type)
        print("Total num of samples: ", len(is_ade_in_note))
        print("Num of positive samples: ", n_pos)
        print("Num of negative samples: ", n_neg)
        is_ade_in_note = pd.DataFrame(is_ade_in_note)
        is_ade_in_note.to_csv(os.path.join(data_dir, set_type+'_'+fname))

        if set_type == 'train':
            analyze_ade_length_bias(set_type + '_' + fname, data_dir)


def _get_splits_relations(relations_data):
    train_pts, dev_pts, test_pts = get_split_pt_idx(relations_data['pat_id'].unique())
    df_train = relations_data[relations_data["pat_id"].isin(train_pts)]
    df_dev = relations_data[relations_data["pat_id"].isin(dev_pts)]
    df_test = relations_data[relations_data["pat_id"].isin(test_pts)]

    return df_train, df_dev, df_test


def get_binary_tasks(annotated_data, med_syn_dict, dir_data, task_name, replace_all_med_mentions=True):
    # Retain only relations (the space of all possible relations + annotated relations) from the annotated data
    relations_data = annotated_data[annotated_data['conc_relat'] == 'relat']

    task_name = task_name.lower()

    if task_name not in ['med', 'ae']:
        raise ValueError("Please input a valid task (med|ae). You entered: ", task_name)

    assert relations_data['conc2_type'].unique() == ['hospitalized'], "Other types than hosp present as conc2"
    assert 'hospitalized' not in relations_data['conc1_type'].unique(), "Hosp present as conc1"

    fname_dict = {
        'med': 'med_before_hosp.csv',
        'ae': 'hosp_for_ae.csv'
    }

    replacement_dict = {
        'med': '@MED$',
        'ae': '@AE$'
    }

    df_train, df_dev, df_test = _get_splits_relations(relations_data)

    nlp = English()
    nlp.add_pipe("sentencizer")

    for (cur_rel_df, set_type) in zip((df_train, df_dev, df_test), ("train", "dev", "test")):
        data = list()

        for note_id in cur_rel_df['deid_note_id'].unique():
            df = cur_rel_df[cur_rel_df['deid_note_id'] == note_id]

            # Only HPI is annotated
            text = df['HPI'].unique()[0]

            for cur_hosp_start in df['conc2_start'].unique():
                hosp_df = df[df['conc2_start'] == cur_hosp_start]
                cur_hosp_end = int(hosp_df['conc2_end'].unique()[0])

                # Getting meds/aes
                concepts = hosp_df[hosp_df['conc1_type'] == task_name]

                for conc_start in concepts['conc1_start'].tolist():
                    # selecting longest span in case of overlapping spans; mainly needed for AEs
                    conc_end = max(concepts[concepts['conc1_start'] == conc_start][
                                     'conc1_end'].unique())

                    # Selecting the longest concept in case of overlapping concepts
                    concept_str = max(concepts[concepts['conc1_start'] == conc_start]['conc1_phrase'].unique(), key=len)

                    if task_name == 'med':
                        # Replacing hospitalizations with placeholder
                        new_text = text[:int(cur_hosp_start)] + '@HOSP$' + text[cur_hosp_end:]

                        if replace_all_med_mentions:
                            if med_syn_dict is not None:
                                # Replacing equivalent medications with medication string
                                new_text = _replace_med_synonyms(med_syn_dict, str(concept_str), new_text)
                            # Replacing all mentions of the same medication with their placeholder
                            new_text = new_text.replace(concept_str, replacement_dict[task_name])
                        else:
                            new_text = text[:int(conc_start)] + '@MED$' + text[conc_end:]

                    else:
                        # Replacing specific mentions of hospitalization and AEs with their placeholder
                        new_text = _get_replacements(text,
                                                     {'hosp': int(cur_hosp_start), task_name: int(conc_start)},
                                                     {'hosp': cur_hosp_end, task_name: conc_end},
                                                     None,
                                                     replace_all_med_mentions
                                                    )

                    # Whether the binary relation is used or not
                    is_related = concepts[(concepts['conc1_start'] == conc_start) &
                                          (concepts['conc1_end'] == conc_end)
                                          ]['used'].item()

                    if not is_related and \
                        task_name == 'ae' and not _is_ae_near_hosp(text, int(cur_hosp_start),
                                                               cur_hosp_end, conc_start, conc_end,
                                                               nlp):
                        continue

                    assert len(concepts[concepts['conc1_start'] == conc_start]['pat_id'].unique()) == 1, "More than one Pt IDX"
                    pt_idx = concepts[concepts['conc1_start'] == conc_start]['pat_id'].unique()[0]

                    assert len(
                        concepts[concepts['conc1_start'] == conc_start]['pat_id'].unique()) == 1, "More than one note IDX"
                    note_idx = concepts[concepts['conc1_start'] == conc_start]['deid_note_id'].unique()[0]

                    data.append({
                        "idx": pt_idx + '_' + note_idx + "_" + str(len(data)),
                        "sentence": new_text,
                        'label': is_related,
                    })

        data = pd.DataFrame(data)
        data.drop_duplicates(inplace=True)
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

        n_pos = len(data[data['label'] == 1])
        n_neg = len(data[data['label'] == 0])
        print("Num pos examples for task ", task_name, " in set ", set_type, " is: ", n_pos)
        print("Num neg examples for task ", task_name, " in set ", set_type, " is: ", n_neg)
        data.to_csv(os.path.join(dir_data, set_type + '_' + fname_dict[task_name]))

        if set_type == 'train':
            undersample_majority_class(data, majority_prop=0.8,
                                       fname_undersampled='train_'
                                                          + os.path.splitext(fname_dict[task_name])[0]
                                                          + '_us_0.8.csv',
                                       dir_data=dir_data)


def get_class_weights(df_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=df_train['label'].unique(), y=df_train['label'].to_list())
    print("Class weights: ", class_weights)
    return class_weights


def undersample_majority_class(df_train, dir_data, majority_prop=0.5, fname_undersampled='train_undersampled.csv'):
    nmin = df_train['label'].value_counts().min()
    nmaj = int(np.ceil((nmin * majority_prop) / (1-majority_prop)))
    print("Minority class samples: ", nmin, "Majority class samples: ", nmaj)

    df_train_pos = df_train[df_train['label'] == 1].sample(nmin)
    df_train_neg = df_train[df_train['label'] == 0].sample(nmaj)
    df_train = pd.concat([df_train_neg, df_train_pos], ignore_index=True)
    # df_train = df_train.groupby('label').apply(lambda x: x.sample(nmin)).reset_index(drop=True)
    df_train.to_csv(os.path.join(dir_data, fname_undersampled))
    return df_train


def get_initial_bias(df_train):
    pos = len(df_train[df_train['is_triplet'] == 1])
    neg = len(df_train[df_train['is_triplet'] == 0])
    initial_bias = np.log([pos/neg])
    return initial_bias


def main(task):
    fname_annots = 'annotations_prod_2022-04-18.tsv'
    fname_notes = 'IBD_note_regex.tsv'
    fname_triplets = 'triplets.csv'
    dir_data = '../data/'

    # Get annotated data along with corresponding HPI
    data = get_data(fname_notes, fname_annots, dir_data)

    # Get equivalent medications from RXNorm; gets brand names for generics and generics for brand names
    med_syn_dict = get_med_synonym_dict(data)

    if task.lower() in ['binary_ae']:
        replace_all_med_mentions = False
        get_binary_tasks(data, med_syn_dict, dir_data, task[7:], replace_all_med_mentions)
    elif task.lower() in ['binary_med']:
        replace_all_med_mentions = True
        get_binary_tasks(data, med_syn_dict, dir_data, task[7:], replace_all_med_mentions)
    else:
        # Create triplets file if it has not been created already
        if not os.path.exists(os.path.join(dir_data, fname_triplets)):
            create_ade_triplets(data, fname_triplets, dir_data)
        analyze_triplets(fname_triplets, dir_data)

        if task == 'is_ade_in_note':
            get_df_is_ade_note(fname_triplets, dir_data)
        elif task == 'rel_triples':
            replace_all_med_mentions = True
            df_train, df_dev, df_test = get_splits_triplets(fname_triplets,
                                                            med_syn_dict, replace_all_med_mentions,
                                                            dir_data,
                                                            fname_task='ade_triples_nearby.csv')

            undersample_majority_class(df_train, majority_prop=0.8,
                                       fname_undersampled='train_ade_triples_nearby_us_0.8.csv',
                                       dir_data=dir_data)


if __name__ == '__main__':
    task = 'binary_ae'
    main(task)
