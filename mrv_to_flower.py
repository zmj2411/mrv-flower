#!/usr/bin/env python3
"""Convert ChemAxon MRV snapshots into FlowER-ready reaction lines.

The script walks through the "Mechanism and Catalytic Site Atlas" folder,
orders all Step_macie.*.mrv files, treats consecutive snapshots as a
reactant/product pair, and writes FlowER-compatible SMILES lines that include
all catalytic residues and substrates. Component-level SMILES are also listed
with human-readable labels so catalysts and substrates can be distinguished.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

from rdkit import Chem
from rdkit.Chem import rdchem

# ---- Metadata helpers -----------------------------------------------------
ENTRY_TITLES: Dict[str, str] = {
    "185": "8-oxoguanine DNA-glycosylase (type-1 OGG1 family)",
}

CHEBI_LABELS: Dict[str, str] = {
    "chebi:137052": "DNA_fragment",
    "chebi:15377": "water",
}

AMINO_ACID_PATTERN = re.compile(
    r"(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)",
    re.IGNORECASE,
)

STEP_PATTERN = re.compile(r"Step_macie\.(\d+)\.(\d+)\.(\d+)")

BOND_TYPE_MAP: Dict[str, rdchem.BondType] = {
    "1": rdchem.BondType.SINGLE,
    "2": rdchem.BondType.DOUBLE,
    "3": rdchem.BondType.TRIPLE,
    "4": rdchem.BondType.QUADRUPLE,
    "A": rdchem.BondType.AROMATIC,
    "a": rdchem.BondType.AROMATIC,
    "ar": rdchem.BondType.AROMATIC,
}

ROLE_PRIORITY = {"catalyst": 0, "substrate": 1, "untyped": 2}

BRACKET_ATOM_RE = re.compile(r"\[([^\]]+)\]")


@dataclass
class Component:
    smiles: str
    role: str
    label: str


def pairwise(seq: Sequence[Path]) -> Iterable[Tuple[Path, Path]]:
    for left, right in itertools.zip_longest(seq, seq[1:]):
        if right is None:
            return
        yield left, right


def extract_entry_id(path: Path) -> str:
    match = STEP_PATTERN.search(path.name)
    if match:
        return match.group(1)
    entry_match = re.search(r"entry_(\d+)", path.name)
    return entry_match.group(1) if entry_match else "unknown"


def infer_entry_title(path: Path) -> str:
    entry_id = extract_entry_id(path)
    return ENTRY_TITLES.get(entry_id, f"Entry {entry_id}")


def extract_step_index(path: Path) -> Tuple[int, int, int]:
    match = STEP_PATTERN.search(path.name)
    if not match:
        return (float("inf"), float("inf"), float("inf"))
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def parse_mrv(file_path: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    tree = ET.parse(file_path)
    root = tree.getroot()
    atoms = []
    for atom in root.findall('.//{*}atom'):
        atoms.append(atom.attrib.copy())
    bonds = []
    for bond in root.findall('.//{*}bond'):
        refs = bond.get('atomRefs2', '').split()
        if len(refs) != 2:
            continue
        payload = bond.attrib.copy()
        payload['atomRefs'] = refs
        bonds.append(payload)
    return atoms, bonds


def _map_number_from_atom_id(atom_id: str, fallback: int) -> int:
    match = re.search(r"(\d+)$", atom_id or "")
    return int(match.group(1)) if match else fallback


def strip_implicit_h_counts(smiles: str) -> str:
    """Drop RDKit's implicit hydrogen markers (e.g., CH3) from bracket atoms."""

    def _replace(match: re.Match[str]) -> str:
        body = match.group(1)
        if not body:
            return match.group(0)

        idx = 0
        while idx < len(body) and body[idx].isdigit():
            idx += 1
        isotope = body[:idx]
        if idx >= len(body):
            return match.group(0)

        symbol = body[idx]
        idx += 1
        if idx < len(body) and body[idx].islower():
            symbol += body[idx]
            idx += 1

        if symbol == "H" and not isotope:
            return match.group(0)

        chirality = []
        while idx < len(body) and body[idx] == "@":
            chirality.append("@")
            idx += 1
        remainder = body[idx:]

        if remainder.startswith("H"):
            j = 1
            while j < len(remainder) and remainder[j].isdigit():
                j += 1
            remainder = remainder[j:]

        new_body = f"{isotope}{symbol}{''.join(chirality)}{remainder}"
        return f"[{new_body}]" if new_body else match.group(0)

    return BRACKET_ATOM_RE.sub(_replace, smiles)


def component_to_smiles(atom_list: List[Dict[str, str]], bond_list: List[Dict[str, str]]) -> str:
    rw_mol = Chem.RWMol()
    index_map: Dict[str, int] = {}
    for idx, atom in enumerate(atom_list, start=1):
        symbol = atom.get('elementType', '*')
        if symbol == '*':
            rd_atom = Chem.Atom(0)
            rd_atom.SetAtomicNum(0)
        else:
            rd_atom = Chem.Atom(symbol)
        charge = int(atom.get('formalCharge', '0'))
        rd_atom.SetFormalCharge(charge)
        rd_atom.SetNoImplicit(True)
        rd_atom.SetNumExplicitHs(0)
        map_num = _map_number_from_atom_id(atom.get('id', ''), idx)
        rd_atom.SetAtomMapNum(map_num)
        new_idx = rw_mol.AddAtom(rd_atom)
        index_map[atom['id']] = new_idx
    for bond in bond_list:
        a1, a2 = bond['atomRefs']
        if a1 not in index_map or a2 not in index_map:
            continue
        order = BOND_TYPE_MAP.get(bond.get('order', '1'), rdchem.BondType.SINGLE)
        rw_mol.AddBond(index_map[a1], index_map[a2], order)
        if order == rdchem.BondType.AROMATIC:
            rd_bond = rw_mol.GetBondBetweenAtoms(index_map[a1], index_map[a2])
            rd_bond.SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(index_map[a1]).SetIsAromatic(True)
            rw_mol.GetAtomWithIdx(index_map[a2]).SetIsAromatic(True)
    mol = rw_mol.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)

    raw_smiles = Chem.MolToSmiles(
        mol,
        canonical=False,
        allBondsExplicit=False,
        allHsExplicit=False,
    )
    return strip_implicit_h_counts(raw_smiles)


def infer_component_role(labels: Sequence[str]) -> Tuple[str, str]:
    clean_labels = [label for label in labels if label]
    for label in clean_labels:
        match = AMINO_ACID_PATTERN.search(label)
        if match:
            return "catalyst", match.group(1).capitalize()
    for label in clean_labels:
        lowered = label.lower()
        if lowered.startswith('chebi:'):
            return "substrate", CHEBI_LABELS.get(lowered, lowered)
    if clean_labels:
        return "substrate", clean_labels[0]
    return "substrate", "unlabeled"


def extract_components(file_path: Path) -> List[Component]:
    atoms, bonds = parse_mrv(file_path)
    atoms_by_id = {atom['id']: atom for atom in atoms}
    adjacency: Dict[str, set] = defaultdict(set)
    for bond in bonds:
        a1, a2 = bond['atomRefs']
        adjacency[a1].add(a2)
        adjacency[a2].add(a1)
    for atom_id in atoms_by_id:
        adjacency.setdefault(atom_id, set())

    visited = set()
    components: List[Component] = []
    for atom_id in atoms_by_id:
        if atom_id in visited:
            continue
        stack = [atom_id]
        current_ids: List[str] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            current_ids.append(cur)
            stack.extend(adjacency[cur] - visited)
        id_set = set(current_ids)
        comp_atoms = [atoms_by_id[idx] for idx in current_ids]
        comp_bonds = [bond for bond in bonds if bond['atomRefs'][0] in id_set and bond['atomRefs'][1] in id_set]
        labels = []
        for atom in comp_atoms:
            label = atom.get('mrvExtraLabel') or atom.get('mrvAlias') or ''
            if label:
                labels.append(label)
        role, readable = infer_component_role(labels)
        smiles = component_to_smiles(comp_atoms, comp_bonds)
        components.append(Component(smiles=smiles, role=role, label=readable))

    components.sort(key=lambda comp: (ROLE_PRIORITY.get(comp.role, 3), comp.label, comp.smiles))
    return components


def components_to_string(components: Sequence[Component]) -> str:
    return '.'.join(comp.smiles for comp in components if comp.smiles)


def format_component(comp: Component) -> str:
    kind = "Catalyst" if comp.role == "catalyst" else "Substrate"
    return f"{kind}[{comp.label}]"


def main() -> None:
    base_dir = Path(__file__).parent
    step_dir = base_dir / "Mechanism and Catalytic Site Atlas"
    step_files = sorted(step_dir.glob("Step_macie*.mrv"), key=extract_step_index)
    if len(step_files) < 2:
        raise SystemExit("Need at least two step files to build reactions.")

    reaction_name = infer_entry_title(step_files[0])
    output_lines: List[str] = []
    reaction_counter = 0
    for react_path, prod_path in pairwise(step_files):
        react_components = extract_components(react_path)
        prod_components = extract_components(prod_path)
        output_lines.append(
            f"# {reaction_name} | {react_path.name} -> {prod_path.name}"
        )
        output_lines.append("# Reactant components:")
        for comp in react_components:
            output_lines.append(f"#   {format_component(comp)}: {comp.smiles}")
        output_lines.append("# Product components:")
        for comp in prod_components:
            output_lines.append(f"#   {format_component(comp)}: {comp.smiles}")
        reactant_line = components_to_string(react_components)
        product_line = components_to_string(prod_components)
        output_lines.append(f"{reactant_line}>>{product_line}")
        output_lines.append("")
        reaction_counter += 1

    target_file = base_dir / "flower_reactions.txt"
    target_file.write_text('\n'.join(output_lines), encoding='utf-8')
    print(f"Wrote {reaction_counter} reactions to {target_file}")


if __name__ == "__main__":
    main()
