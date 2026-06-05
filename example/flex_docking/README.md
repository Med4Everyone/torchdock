# Flexible Docking Example (4FKV)

This example demonstrates flexible docking using TorchDock with the 4FKV protein-ligand complex. We use 4EK3, a homologous apo protein structure (aligned to 4FKV), as the receptor, and dock the native ligand from 4FKV. Flexible residues near the binding site are allowed to move during docking.

## Files

| File | Description |
|------|-------------|
| `4ek3.pdb` | Homologous apo receptor PDB (aligned to 4FKV, without ligand) |
| `4fkv_holo.pdb` | Holo structure PDB (protein-ligand complex, for reference) |
| `4fkv_ligand.pdb` | Native ligand conformation extracted from 4FKV holo (reference) |
| `4fkv_ligand.smi` | Ligand SMILES (for docking input) |

## Workflow

### 1. Prepare Receptor

```bash
torchdock prepare_receptor -i 4ek3.pdb -o receptor.pdbqt
```

### 2. Prepare Ligand (from SMILES)

```bash
torchdock prepare_ligand -smi "[N-2]S(=O)(=O)c1ccc([N-]N=C2C(=O)[N-]c3ccc(C(=O)[N-]CCC4=NC=NC4)cc32)cc1" -o ligand.pdbqt
```

### 3. Define Docking Box

Specify the binding site center manually:

```bash
torchdock define_box -c 25.8 27.6 27.5 -s 20 20 20 -o box.json
```

### 4. Run Flexible Docking

```bash
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt -f -v
```

The `-f` flag enables flexible docking. Residues within the binding site are automatically detected and treated as flexible.

You can also specify flexible residues manually:

```bash
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt -f --flex_residues "A:10,A:18,A:33,A:64,A:80,A:82,A:86,A:89,A:134,A:145" -v -d cuda
```

> `-d cuda` enables GPU acceleration. Remove it to use CPU.

### 5. Convert Results

```bash
torchdock convert_result -i result.pdbqt -o ./output
```

### 6. Calculate RMSD (Optional)

Compare the docking result against the native ligand conformation:

```bash
# Convert native ligand PDB to PDBQT for RMSD calculation
torchdock prepare_ligand -i 4fkv_ligand.pdb -o reference.pdbqt

# Calculate RMSD
torchdock rmsd -p result.pdbqt -r reference.pdbqt
```

## Expected Output

- `receptor.pdbqt` - Prepared receptor
- `ligand.pdbqt` - Prepared ligand (from SMILES)
- `box.json` - Docking box configuration
- `result.pdbqt` - Docking poses with flexible side-chain conformations
- `output/` - Converted SDF and PDB files
