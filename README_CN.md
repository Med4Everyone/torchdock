# TorchDock

<!-- Badges -->

[![PyPI version](https://badge.fury.io/py/torchdock.svg)](https://pypi.org/project/torchdock/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()

> 基于 PyTorch 的可微分子对接框架

**[English](README.md)** | 中文

TorchDock 将经典经验性评分函数（Vina、Vinardo）重构为完全可微的 PyTorch 版本，通过端到端梯度优化进行构象搜索。相比传统离散采样方法，TorchDock 在复杂柔性对接（含蛋白质侧链柔性）场景下可实现近百倍的加速，并提供基于早停策略的虚拟筛选流程，在保持高召回率的前提下实现约 5 倍加速，支持大规模化合物库筛选。

<p align="center">
  <img src="assets/torchdock_concept.png" alt="TorchDock 方法概念图" width="900">
</p>

---

## 特性

- 可微的 Vina / Vinardo 评分函数（纯 PyTorch 实现）
- 基于梯度的端到端构象搜索，支持 SMAC 全局初始化
- 柔性对接：支持蛋白质侧链柔性，高维场景下加速近百倍
- 虚拟筛选早停策略：基于 XGBoost 预评分，加速约 5 倍
- 完整的 CLI 工具链：配体/受体制备 + 对接 + 结果转换
- 支持 CPU / GPU 异构计算
- 插件式架构，支持自定义评分函数

---

## 安装

TorchDock 需要 **Python ≥ 3.10** 和 **OpenBabel**。

> **平台说明：** TorchDock 在 Linux（Ubuntu 22.04）上开发和测试。macOS 可能兼容但未经测试。Windows 用户可通过 [WSL](https://learn.microsoft.com/zh-cn/windows/wsl/install) 运行。

### 第一步：创建 conda 环境并安装 OpenBabel

```bash
conda create -n torchdock python=3.12 -y
conda activate torchdock
conda install -c conda-forge openbabel -y
```

### 第二步：安装 TorchDock

```bash
pip install torchdock
```

### 仅 CPU 安装（可选，节省磁盘空间）

默认安装包含 CUDA 运行时库（约 2GB）。如果确定不需要 GPU，可先安装 CPU 版 PyTorch 以节省空间：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torchdock
```

### 从源码安装

```bash
git clone https://github.com/Med4Everyone/torchdock.git
cd torchdock
pip install -e .
# 国内用户：pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 验证安装

```bash
torchdock --help
```

预期输出：

```
TorchDock v0.1.0 — Differentiable molecular docking framework

Usage: torchdock <command> [options]

Commands:
  dock                 Run molecular docking.
  prepare_ligand       Convert SMILES or file input to PDBQT ligand.
  prepare_receptor     Prepare receptor PDBQT from a PDB file.
  define_box           Define a docking box from a ligand or manual coordinates.
  convert_result       Convert TorchDock PDBQT results to SDF and PDB.
  rmsd                 Calculate RMSD between docking poses and a reference.
```

---

## 快速入门

```bash
# 1. 准备受体
torchdock prepare_receptor -i protein.pdb -o receptor.pdbqt

# 2. 准备配体
torchdock prepare_ligand -smi "CC(=O)O" -o ligand.pdbqt  # 从 SMILES
# 或
torchdock prepare_ligand -i ligand.mol2 -o ligand.pdbqt   # 从文件（mol2/sdf/pdb/mol）

# 3. 定义对接盒子
torchdock define_box -l ligand.pdbqt -o box.json           # 从配体自动计算中心
# 或
torchdock define_box -c 10.0 20.0 30.0 -s 25 25 25 -o box.json  # 手动指定中心和尺寸

# 4. 运行对接（半柔性 / 柔性）
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt -f  # 柔性

# 5. 转换结果
torchdock convert_result -i result.pdbqt -o ./output
```

> 完整可运行示例见 [example/](example/) 目录。

---

## 快速对接

TorchDock 提供基于早停策略的快速对接模式（`--early_stop`），适用于大规模化合物库的初筛：

- **截断式梯度优化**：对候选分子执行有限步数优化，快速评估结合潜力
- **XGBoost 预评分筛选**：仅对预测得分最优的候选分子启动完整对接优化
- **效率提升**：在保持头部高分分子召回率（80%）的前提下，相比全库对接整体加速约 5 倍

> ⚠️ 注意：快速对接模式下，大部分候选分子仅获得预测评分，未经历完整构象优化。建议将结果作为初筛参考，对感兴趣的高分分子再进行标准对接。

```bash
# 启用快速对接模式
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt --early_stop
```

---

## CLI 命令参考

### `torchdock dock`

运行分子对接。

```bash
torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt
```

| 参数                         | 说明                                           | 必填 |
| ---------------------------- | ---------------------------------------------- | :--: |
| `-r, --protein_pdbqt_path` | 受体 PDBQT 文件                                |  ✅  |
| `-l, --ligand_pdbqt_path`  | 配体 PDBQT 文件                                |  ✅  |
| `-o, --output_path`        | 输出结果文件                                   |  ✅  |
| `-b, --box_file_path`      | 盒子配置 JSON 文件                             |  *  |
| `-bc, --box_center X Y Z`  | 盒子中心坐标                                   |  *  |
| `-bs, --box_size DX DY DZ` | 盒子尺寸                                       |  *  |
| `-c, --config_file_path`   | 自定义配置文件                                 |      |
| `-f, --flex`               | 启用柔性对接                                   |      |
| `--flex_residues`          | 柔性残基（如 `A:123,A:125`），不填则自动检测 |      |
| `-sc, --score_only`        | 仅评分，不搜索                                 |      |
| `-es, --early_stop`        | 启用 early stopping                            |      |
| `-d, --device`             | 计算设备（`cpu` / `cuda` / `cuda:0`）    |      |
| `-nw, --num_workers`       | CPU 工作进程数（默认 4）                       |      |
| `-v, --verbose`            | 详细输出                                       |      |

> `*` 盒子定义：使用 `-b` 指定 JSON 文件，或使用 `-bc` + `-bs` 手动指定中心和尺寸，二选一。

### `torchdock prepare_ligand`

将 SMILES 或分子文件转换为 PDBQT 配体。

```bash
# 从 SMILES
torchdock prepare_ligand -smi "CC(=O)Oc1ccccc1C(=O)O" -o ligand.pdbqt

# 从文件（推荐 mol2）
torchdock prepare_ligand -i molecule.mol2 -o ligand.pdbqt

# 批量转换
torchdock prepare_ligand -b ligands.csv -o ./output_dir
```

| 参数               | 说明                                          | 必填 |
| ------------------ | --------------------------------------------- | :--: |
| `-smi, --smiles` | SMILES 字符串（单分子）                       |  *  |
| `-i, --input`    | 输入文件（推荐 .mol2，也支持 .pdb/.mol/.sdf） |  *  |
| `-b, --batch`    | 批量 CSV 文件（含 ID 和 SMILES 列）           |  *  |
| `-o, --output`   | 输出文件或目录                                |  ✅  |
| `-s, --seed`     | 随机种子（3D 坐标生成）                       |      |
| `-d, --remove-h` | 去除氢原子                                    |      |

> `*` 表示三种输入方式三选一。

### `torchdock prepare_receptor`

从 PDB 文件制备受体 PDBQT。

```bash
torchdock prepare_receptor -i protein.pdb -o receptor.pdbqt
```

| 参数                | 说明            | 必填 |
| ------------------- | --------------- | :--: |
| `-i, --input`     | 输入 PDB 文件   |  ✅  |
| `-o, --output`    | 输出 PDBQT 文件 |  ✅  |
| `-d, --remove-h`  | 去除氢原子      |      |
| `-nc, --no-clean` | 跳过蛋白质清洗  |      |

### `torchdock define_box`

定义对接盒子。

```bash
# 从配体自动计算中心
torchdock define_box -l ligand.pdbqt -o box.json

# 手动指定中心
torchdock define_box -c 10.0 20.0 30.0 -s 25 25 25 -o box.json
```

| 参数                    | 说明                                             | 必填 |
| ----------------------- | ------------------------------------------------ | :--: |
| `-l, --ligand`        | 配体文件（.mol2/.sdf/.pdb/.pdbqt，自动计算中心） |  *  |
| `-c, --center X Y Z`  | 手动盒子中心                                     |  *  |
| `-s, --size SX SY SZ` | 盒子尺寸（默认 20 20 20）                        |      |
| `-o, --output`        | 输出 JSON 文件                                   |  ✅  |
| `-v, --visualize`     | 生成盒子可视化 PDB                               |      |

> `*` 输入方式二选一：`-l` 从配体文件自动计算中心，或 `-c` 手动指定中心坐标。

### `torchdock convert_result`

将对接结果转换为 SDF 和 PDB 格式。

```bash
torchdock convert_result -i result_remi.pdbqt -o ./output
```

| 参数             | 说明              | 必填 |
| ---------------- | ----------------- | :--: |
| `-i, --input`  | 结果 PDBQT 文件   |  ✅  |
| `-o, --output` | 输出目录          |  ✅  |
| `-t, --top-k`  | 仅转换前 k 个构象 |      |

### `torchdock rmsd`

计算对接结果与参考构象的 RMSD。

```bash
torchdock rmsd -p result.pdbqt -r reference.pdbqt
```

| 参数                | 说明                       | 必填 |
| ------------------- | -------------------------- | :--: |
| `-p, --predicted` | 预测结果 PDBQT             |  ✅  |
| `-r, --reference` | 参考结构 PDBQT             |  ✅  |
| `-t, --top-k`     | 仅计算前 k 个构象          |      |
| `-q, --quiet`     | 静默模式（只输出 RMSD 值） |      |

---

## Python API

TorchDock 支持通过 Python 代码调用对接：

```python
from torchdock.pipeline.docking_runner import docking

# 基本对接
result = docking(
    protein_pdbqt_path="receptor.pdbqt",
    ligand_pdbqt_path="ligand.pdbqt",
    box_center=[15.0, 20.0, 25.0],
    box_size=[20.0, 20.0, 20.0],
    output_path="result.pdbqt",
)

# 返回值: [torchdock_score, total_score, inter_score, intra_score, unbound_score]
print(f"TorchDock Score: {result[0]:.3f}")
```

使用配置文件：

```python
result = docking(
    protein_pdbqt_path="receptor.pdbqt",
    ligand_pdbqt_path="ligand.pdbqt",
    box_file_path="box.json",
    output_path="result.pdbqt",
    config_file_path="config.yaml",
    device="cuda",  # 使用 GPU
)
```

柔性对接：

```python
result = docking(
    protein_pdbqt_path="receptor.pdbqt",
    ligand_pdbqt_path="ligand.pdbqt",
    box_center=[15.0, 20.0, 25.0],
    box_size=[20.0, 20.0, 20.0],
    output_path="result.pdbqt",
    flex=True,                          # 启用柔性对接
    flex_residues="A:123,A:125,B:45",   # 指定柔性残基（不填则自动检测）
)
```

---

## 引用

如果 TorchDock 对您的研究有帮助，请引用：

```bibtex
@software{torchdock,
  title={Coming Soon},
  author={Coming Soon},
  year={2026},
  url={https://github.com/Med4Everyone/torchdock}
}
```

---

## 致谢

TorchDock 由阿里巴巴通义实验室 AI4S 团队与中国药科大学吴建盛教授团队联合开发，旨在以开源工具推动计算制药领域的发展。项目主要贡献者包括胡靖琨、刘俊龙、丁季。

## 许可证

[Apache License 2.0](LICENSE)

