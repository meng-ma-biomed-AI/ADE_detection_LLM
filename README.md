# Serious adverse event detection from IBD clinical notes

This repository contains source code for the analysis to infer adverse drug events from clinical notes, accompanying the paper [Algorithmic Identification of Treatment-Emergent Adverse Events From Clinical Notes Using Large Language Models: A Pilot Study in Inflammatory Bowel Disease](https://ascpt.onlinelibrary.wiley.com/doi/10.1002/cpt.3226).

Please cite the paper using the following bibtex:

```
@article{https://doi.org/10.1002/cpt.3226,
author = {Silverman, Anna L. and Sushil, Madhumita and Bhasuran, Balu and Ludwig, Dana and Buchanan, James and Racz, Rebecca and Parakala, Mahalakshmi and El-Kamary, Samer and Ahima, Ohenewaa and Belov, Artur and Choi, Lauren and Billings, Monisha and Li, Yan and Habal, Nadia and Liu, Qi and Tiwari, Jawahar and Butte, Atul J. and Rudrapatna, Vivek A.},
title = {Algorithmic Identification of Treatment-Emergent Adverse Events From Clinical Notes Using Large Language Models: A Pilot Study in Inflammatory Bowel Disease},
journal = {Clinical Pharmacology \& Therapeutics},
volume = {n/a},
number = {n/a},
pages = {},
doi = {https://doi.org/10.1002/cpt.3226},
url = {https://ascpt.onlinelibrary.wiley.com/doi/abs/10.1002/cpt.3226},
eprint = {https://ascpt.onlinelibrary.wiley.com/doi/pdf/10.1002/cpt.3226},
abstract = {Outpatient clinical notes are a rich source of information regarding drug safety. However, data in these notes are currently underutilized for pharmacovigilance due to methodological limitations in text mining. Large language models (LLMs) like Bidirectional Encoder Representations from Transformers (BERT) have shown progress in a range of natural language processing tasks but have not yet been evaluated on adverse event (AE) detection. We adapted a new clinical LLM, University of California – San Francisco (UCSF)-BERT, to identify serious AEs (SAEs) occurring after treatment with a non-steroid immunosuppressant for inflammatory bowel disease (IBD). We compared this model to other language models that have previously been applied to AE detection. We annotated 928 outpatient IBD notes corresponding to 928 individual patients with IBD for all SAE-associated hospitalizations occurring after treatment with a non-steroid immunosuppressant. These notes contained 703 SAEs in total, the most common of which was failure of intended efficacy. Out of eight candidate models, UCSF-BERT achieved the highest numerical performance on identifying drug-SAE pairs from this corpus (accuracy 88–92\%, macro F1 61–68\%), with 5–10\% greater accuracy than previously published models. UCSF-BERT was significantly superior at identifying hospitalization events emergent to medication use (P < 0.01). LLMs like UCSF-BERT achieve numerically superior accuracy on the challenging task of SAE detection from clinical notes compared with prior methods. Future work is needed to adapt this methodology to improve model performance and evaluation using multicenter data and newer architectures like Generative pre-trained transformer (GPT). Our findings support the potential value of using large language models to enhance pharmacovigilance.}
}
```
