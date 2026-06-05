# Contributing to TorchDock

Thank you for contributing to TorchDock! This project is in **early v0.1.0 release**, and we especially need help with:

---

## How to Contribute

### 🐞 Reporting Bugs
**Please open an issue with**:
- **Description**: Clear summary of the bug.
- **Steps to Reproduce**: How to trigger it (e.g., `torchdock dock -r receptor.pdbqt -l ligand.pdbqt -b box.json -o result.pdbqt`).
- **Expected vs Actual**: What you expected vs what happened.
- **Environment**: OS, Python, PyTorch versions.

> *Note for v0.1.0:* No test code needed—just provide steps to manually reproduce.

---

### ✨ Suggesting Features
**Please include**:
- **The Problem**: Your specific use case (e.g., *"Need to dock 1,000+ ligands in batch"*).
- **Your Solution**: How it should work (e.g., *"Add `--batch` option to `torchdock dock`"*).
- **Alternatives**: Other methods you considered.

---

## Submitting Pull Requests
1. Fork the repo and create a branch.
2. Make your changes:
   - Follow PEP 8 style (use `ruff` for linting: `ruff check torchdock/`).
   - **Add documentation** for new features (docstrings or README updates).
3. **Manual Verification**:
   - New features: Provide a **1-line command** to test (e.g., `python examples/test_gpu.py`).
   - Bug fixes: Explain how to confirm the fix.
4. Write clear commit messages (e.g., `fix: resolve GPU memory leak`).
5. Push your branch and open a PR to `main`.

> **Roadmap**: Unit tests will be **required once the project enters stable development**, not yet for now.

---

## License
By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

## Code of Conduct
This project adheres to the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you agree to uphold its terms.

---

## Need Help?
Ask questions in [Discussions](https://github.com/Med4Everyone/torchdock/discussions) or open a new issue.  
**New to open source?** Check out [this guide](https://opensource.guide/how-to-contribute/) for first-timers.

---

**Thank you!** ✨  
The TorchDock Team