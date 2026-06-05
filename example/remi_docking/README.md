# Semi-Flexible Docking Example (1MQ6)

This example demonstrates semi-flexible docking using TorchDock with the 1MQ6 protein-ligand complex.

## Files

| File | Description |
|------|-------------|
| `1mq6_protein.pdb` | Raw receptor PDB (unprocessed) |
| `1mq6_ligand.smi` | Ligand SMILES (for docking input) |
| `1mq6_ligand.mol2` | Native ligand conformation (ground truth reference) |

## Workflow

### 1. Prepare Receptor

```bash
torchdock prepare_receptor -i 1mq6_protein.pdb -o receptor.pdbqt
```

This will clean the protein (remove water, add hydrogens) and convert to PDBQT format.

### 2. Prepare Ligand (from SMILES)

```bash
# Read SMILES from file and generate 3D conformation
torchdock prepare_ligand -smi "COc1cc(Cl)cc(C(=O)[N-]c2ccc(Cl)cn2)c1[N-]C(=O)C1=C(Cl)C(CN(C)C2=NCCO2)=CS1" -o ligand.pdbqt
```

The ligand for docking is generated from SMILES, with 3D coordinates created automatically.

### 3. Define Docking Box

Use the native ligand (mol2) to calculate the binding site center:

```bash
torchdock define_box -l 1mq6_ligand.mol2 -o box.json
```

This computes the box center from the native ligand's center of mass.

### 4. Run Semi-Flexible Docking

```bash
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt -v
```

The receptor remains rigid while the ligand is flexible during docking.

### 5. Convert Results

```bash
torchdock convert_result -i result.pdbqt -o ./output
```

This generates `output/result.sdf` and `output/result.pdb` for visualization.

### 6. Calculate RMSD (Optional)

Compare the docking result against the native ligand conformation:

```bash
# Convert native mol2 to pdbqt for RMSD calculation
torchdock prepare_ligand -i 1mq6_ligand.mol2 -o reference.pdbqt

# Calculate RMSD
torchdock rmsd -p result.pdbqt -r reference.pdbqt
```

## Expected Output

- `receptor.pdbqt` - Prepared receptor
- `ligand.pdbqt` - Prepared ligand (from SMILES)
- `box.json` - Docking box configuration
- `result.pdbqt` - Docking poses (multiple conformers)
- `output/` - Converted SDF and PDB files
