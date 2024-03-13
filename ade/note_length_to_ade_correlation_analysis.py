import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from tokenizers import BertWordPieceTokenizer


def compute_note_lengths(fname_is_ade_in_note, dir_data,
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

    return df_pos, df_neg


def test_association(df_pos, df_neg):
    shapiro_stat, shapiro_p = stats.shapiro(df_pos['n_tokens'].tolist() +
                                            df_neg['n_tokens'].tolist())
    print("Shapiro stats: ", shapiro_stat, "P-value: ", shapiro_p)
    if shapiro_p < 0.05:
        print("Tokens length distribution is not normal")

    print("Using MANN WHITNEY U TEST due to absence of normal distribution")
    bins = np.linspace(100, 3500, 100)
    plt.hist(df_pos['n_tokens'].tolist(), bins, alpha=0.5, label='ADE')
    plt.hist(df_neg['n_tokens'].tolist(), bins, alpha=0.5, label='No_ADE')
    plt.legend(loc='upper right')
    plt.show()

    statistic, p_val = stats.mannwhitneyu(df_pos['n_tokens'].tolist(),
                                          df_neg['n_tokens'].tolist(),
                                          alternative='greater')

    print("Mann-Whitney statistic: ", statistic, 'P-value: ', p_val)
    if p_val < 0.05:
        print("Rejected null hypothesis in favor of the alternative hypothesis that length distribution of positive labels is more than that of negative labels")


def plot_note_length_against_label(df_pos, df_neg):
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


if __name__ == '__main__':
    df_pos, df_neg = compute_note_lengths('train_ade_in_note.csv', '../data/')
    test_association(df_pos, df_neg)