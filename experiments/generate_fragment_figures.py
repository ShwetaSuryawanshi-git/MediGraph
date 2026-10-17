"""
experiments/generate_fragment_figures.py
-----------------------------------------
Regenerates RDKit Morgan bit fragment images using DrawMorganBit with verified bitInfo.
Annotates each image with:
- Drug Name
- Bit ID
- Morgan Radius
- Drug SMILES
- Substructure Environment SMILES
- Number of training molecules in which bit is on
Outputs:
- figures/shap_fragments/imatinib_bit_207_radius_1.png (and imatinib_bit_207.png)
- figures/shap_fragments/gefitinib_bit_841_radius_1.png (and gefitinib_bit_841.png)
- figures/shap_fragments/erlotinib_bit_974_radius_1.png (and erlotinib_bit_974.png)
- figures/shap_fragments/dasatinib_bit_798_radius_2.png (and dasatinib_bit_798.png)
- results/fragments_check.csv
"""

import os
import io
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"
FRAGMENTS_DIR = FIGURES_DIR / "shap_fragments"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FRAGMENTS_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load DAVIS dataset to count train molecules with bit on
df_davis = pd.read_csv(DATA_DIR / "davis.tab", sep="\t")
unique_smiles = df_davis["X1"].unique()
n_unique_drugs = len(unique_smiles)

DRUGS = [
    {
        "name": "Imatinib",
        "target": "ABL1",
        "smiles": "Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(C)CC5",
        "bit_id": 207,
        "radius": 1,
        "filename": "imatinib_bit_207_radius_1.png",
        "alt_filename": "imatinib_bit_207.png"
    },
    {
        "name": "Gefitinib",
        "target": "EGFR",
        "smiles": "COc1cc2c(ncnc2cc1OCCCN3CCOCC3)Nc4ccc(c(c4)Cl)F",
        "bit_id": 841,
        "radius": 1,
        "filename": "gefitinib_bit_841_radius_1.png",
        "alt_filename": "gefitinib_bit_841.png"
    },
    {
        "name": "Erlotinib",
        "target": "EGFR",
        "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
        "bit_id": 974,
        "radius": 1,
        "filename": "erlotinib_bit_974_radius_1.png",
        "alt_filename": "erlotinib_bit_974.png"
    },
    {
        "name": "Dasatinib",
        "target": "SRC/ABL",
        "smiles": "Cc1cccc(c1Cl)NC(=O)c2cnc(s2)Nc3cc(nc(n3)C)N4CCN(CC4)CCO",
        "bit_id": 798,
        "radius": 2,
        "filename": "dasatinib_bit_798_radius_2.png",
        "alt_filename": "dasatinib_bit_798.png"
    }
]

# Calculate molecule frequencies across unique DAVIS drugs
bit_counts = {}
for drug_info in DRUGS:
    b = drug_info["bit_id"]
    count = 0
    for s in unique_smiles:
        m = Chem.MolFromSmiles(s)
        if m:
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024)
            if fp[b]:
                count += 1
    bit_counts[b] = count

check_records = []

for item in DRUGS:
    name = item["name"]
    smiles = item["smiles"]
    bit_id = item["bit_id"]
    radius = item["radius"]
    mol = Chem.MolFromSmiles(smiles)

    bit_info = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024, bitInfo=bit_info)
    atom_idx, rad = bit_info[bit_id][0]

    # Extract substructure SMILES of the environment
    env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom_idx)
    amap = {}
    submol = Chem.PathToSubmol(mol, env, atomMap=amap)
    sub_smiles = Chem.MolToSmiles(submol) if submol.GetNumAtoms() > 0 else mol.GetAtomWithIdx(atom_idx).GetSymbol()

    n_on = bit_counts[bit_id]
    pct_on = (n_on / n_unique_drugs) * 100.0

    check_records.append({
        "drug": name,
        "target": item["target"],
        "bit_id": bit_id,
        "radius": rad,
        "central_atom_idx": atom_idx,
        "substructure_smiles": sub_smiles,
        "drug_smiles": smiles,
        "train_molecules_on": n_on,
        "total_train_molecules": n_unique_drugs,
        "train_prevalence_pct": round(pct_on, 1)
    })

    # Render base Morgan bit image (400x350)
    img_bytes = Draw.DrawMorganBit(mol, bit_id, bit_info, useSVG=False, molSize=(450, 380))
    base_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Add text annotation banner
    canvas_w = 600
    canvas_h = 520
    annotated = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # Paste base image centered
    base_w, base_h = base_img.size
    offset_x = (canvas_w - base_w) // 2
    annotated.paste(base_img, (offset_x, 60))

    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()

    # Top header text
    header_text = f"{name} | Bit ID: {bit_id} | Morgan Radius: {rad} (Center Atom: {atom_idx})"
    draw.text((20, 15), header_text, fill=(15, 23, 42), font=font)
    draw.text((20, 32), f"Environment Substructure: {sub_smiles} | In {n_on}/{n_unique_drugs} ({pct_on:.1f}%) DAVIS Drugs", fill=(71, 85, 105), font=font)

    # Bottom footer text (SMILES)
    smiles_line1 = smiles[:75]
    smiles_line2 = smiles[75:] if len(smiles) > 75 else ""
    draw.text((20, 465), f"Drug SMILES: {smiles_line1}", fill=(100, 116, 139), font=font)
    if smiles_line2:
        draw.text((20, 482), f"             {smiles_line2}", fill=(100, 116, 139), font=font)
    draw.text((20, 500), "Note: Local circular environment; does not assert full pharmacophore.", fill=(148, 163, 184), font=font)

    # Save to both target filenames
    path1 = FRAGMENTS_DIR / item["filename"]
    path2 = FRAGMENTS_DIR / item["alt_filename"]
    annotated.save(str(path1))
    annotated.save(str(path2))
    print(f"Generated {path1.name} and {path2.name}")

df_check = pd.DataFrame(check_records)
df_check.to_csv(RESULTS_DIR / "fragments_check.csv", index=False)
print(f"Saved {RESULTS_DIR / 'fragments_check.csv'}")
print(df_check[["drug", "bit_id", "radius", "substructure_smiles", "train_molecules_on", "train_prevalence_pct"]])
