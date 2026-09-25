# Contributing to Block-to-Panchayat Weather Downscaling Engine

Thank you for your interest in contributing to this Smart India Hackathon (SIH) project! We welcome contributions to improve our meteorological downscaling pipelines, machine learning models, and agro-advisory rule engines.

---

## Code of Conduct

1. **Precision & Scientific Rigor:** All modeling changes must be validated against spatial holdouts (e.g. Leave-One-Station-Out Cross Validation) to prevent spatial data leakage.
2. **Standardization:** Maintain compatibility with official Local Government Directory (LGD) spatial codes and India Meteorological Department (IMD) binary data conventions.
3. **Reproducibility:** Every feature modification or new model should include accompanying unit/integration tests in `tests/`.

---

## Development Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Uddip07/sih26074.git
   cd sih26074
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Test Suite:**
   ```bash
   pytest -v
   ```

---

## Contribution Workflow

1. Fork the repo and create a new feature branch (`git checkout -b feature/your-feature-name`).
2. Commit your modifications with descriptive commit messages (`git commit -m "feat: enhance orographic lapse rate modeling"`).
3. Ensure all tests pass (`pytest -v`).
4. Push to your branch and submit a Pull Request.
