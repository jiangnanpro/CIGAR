# CIGAR: Contrastive learning for GHA Recommendation

This repository contains our dataset and the notebook we used to build the dataset.

- **Build dataset.ipynb** is used to build our dataset. 
- **Extract action repositories.ipynb** is used to download the YAML and Readme.md files from action repositories.
- **action_user_assigned_names.csv.gz** is the dataframe d_1 consisting of (action_id, name_u).
- **action_official_names_descriptions.csv.gz** is the dataframe d_2 consisting of (action_id, name_o, desc_o).
- **train.csv.gz**, **valid.csv.gz**, **test.csv.gz** are the datasets we built to train, valid and test our approach CIGAR.

