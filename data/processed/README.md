# data/processed

- raw file: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz`
- raw md5: `2882d69e70bba0fec14995f26787cc25`
- raw rows: 7831, columns: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53, mol_id, smiles
- unparseable smiles dropped: 8 (see dropped.csv)
- rows after standardization: 7823
- duplicate canonical smiles groups: 72 (114 rows merged)
- conflicting labels set to NaN: 44 (task, molecule) pairs (see conflicts.csv)
- clean rows: 7709, unique scaffolds: 2285
- standardization: largest fragment only, charges and stereo untouched
