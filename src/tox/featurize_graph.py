"""B1 - standardized SMILES -> PyTorch Geometric molecular graphs.

Graph schema (one `torch_geometric.data.Data` per molecule):
    x           [N, ATOM_DIM]  float32  atom features (one-hot / flags, see below)
    edge_index  [2, 2E]        int64    every undirected bond appears in BOTH directions
    edge_attr   [2E, BOND_DIM] float32  bond features, identical for the two directions
    y           [1, 12]        float32  labels in `TASKS` order, NaN = not measured

`y` has a leading dimension of 1 on purpose: PyG concatenates it along dim 0 when
batching, so a batch of B molecules gives y of shape [B, 12] (not [B*12]).
Missing labels stay NaN here; they are masked in the loss, never filled.

Input SMILES are expected to be the already-standardized canonical SMILES of contract C1
(largest fragment, charges and stereo kept), so no further cleaning is done here.
"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

from tox.constants import TASKS

RDLogger.DisableLog("rdApp.*")

# --- feature vocabularies. The last slot of every one-hot is an "other / unknown" bucket.
ATOMS = [6, 7, 8, 9, 15, 16, 17, 35, 53, 5, 14, 11, 19, 3, 12, 20, 26, 29, 30, 34, 50, 33, 80, 82]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDIZATIONS = [Chem.HybridizationType.S, Chem.HybridizationType.SP,
                  Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
                  Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
CHIRALITY = [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
             Chem.ChiralType.CHI_TETRAHEDRAL_CCW]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE,
              Chem.BondType.AROMATIC]
BOND_STEREO = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE,
               Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]


def _onehot(value, choices) -> list[float]:
    """One-hot over `choices` plus a trailing 'other' slot for unseen values."""
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> list[float]:
    return (_onehot(atom.GetAtomicNum(), ATOMS)
            + _onehot(atom.GetDegree(), DEGREES)
            + _onehot(atom.GetFormalCharge(), CHARGES)
            + _onehot(atom.GetTotalNumHs(), NUM_HS)
            + _onehot(atom.GetHybridization(), HYBRIDIZATIONS)
            + _onehot(atom.GetChiralTag(), CHIRALITY)
            + [float(atom.GetIsAromatic()), float(atom.IsInRing())])


def bond_features(bond: Chem.Bond) -> list[float]:
    return (_onehot(bond.GetBondType(), BOND_TYPES)
            + _onehot(bond.GetStereo(), BOND_STEREO)
            + [float(bond.GetIsConjugated()), float(bond.IsInRing())])


_ref = Chem.MolFromSmiles("CC")
ATOM_DIM = len(atom_features(_ref.GetAtomWithIdx(0)))
BOND_DIM = len(bond_features(_ref.GetBondWithIdx(0)))
NUM_TASKS = len(TASKS)


def smiles_to_graph(smiles: str, y=None) -> Data | None:
    """Convert one SMILES to a graph. Returns None if RDKit cannot parse it.

    y: optional array-like of length 12 (TASKS order, NaN = missing). If omitted, y is
    all-NaN so the graph can still be used for inference on unlabeled molecules.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)

    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]                    # both directions, adjacent, same features
        dst += [j, i]
        attr += [f, f]
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(attr, dtype=torch.float32).reshape(-1, BOND_DIM)

    if y is None:
        y = np.full(NUM_TASKS, np.nan)
    y = torch.as_tensor(np.asarray(y, dtype=np.float32)).reshape(1, NUM_TASKS)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)


def build_graphs(smiles, labels=None, skip_invalid: bool = False) -> list[Data]:
    """Graphs for a list of SMILES, in order.

    labels: optional [n, 12] array in TASKS order (NaN = missing).
    Unparseable SMILES raise ValueError unless skip_invalid=True. C1 SMILES were already
    validated, so an invalid one here means something upstream is wrong and should be loud.
    """
    smiles = list(smiles)
    if labels is not None:
        labels = np.asarray(labels, dtype=np.float32)
        if labels.shape != (len(smiles), NUM_TASKS):
            raise ValueError(f"labels must be [{len(smiles)}, {NUM_TASKS}], got {labels.shape}")
    graphs = []
    for k, smi in enumerate(smiles):
        g = smiles_to_graph(smi, None if labels is None else labels[k])
        if g is None:
            if skip_invalid:
                continue
            raise ValueError(f"cannot parse SMILES: {smi!r}")
        graphs.append(g)
    return graphs
