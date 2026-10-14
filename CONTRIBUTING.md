# Contributing & Git Workflow Guide

Welcome! This document outlines our collaborative Git workflow, branching conventions, and quality standards for the **Airbnb Market Analysis** project.

---

## 1. Branching Strategy

We follow a structured Git feature-branch workflow:

- **`main`**: The single source of truth for production-ready code, documentation, and reports. Direct commits to `main` are restricted.
- **`feature/<phase-name>`**: Used for building distinct analysis phases (e.g., `feature/phase-1-setup`, `feature/phase-2-data-cleaning`, `feature/phase-3-sql-layer`, etc.).
- **`fix/<bug-description>`**: Used for addressing data issues or bug fixes discovered during testing.

### Workflow Steps:
1. **Create and switch to a feature branch**:
   ```bash
   git checkout -b feature/<feature-name>
   ```
2. **Make focused, atomic commits** with clear messages.
3. **Run tests** before merging:
   ```bash
   pytest tests/
   ```
4. **Merge into `main`** via pull request or fast-forward merge:
   ```bash
   git checkout main
   git merge feature/<feature-name>
   ```

---

## 2. Commit Message Standards

We use the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` A new feature or analysis module (e.g., `feat(cleaning): add currency parsing and amenity cleaner`)
- `fix:` A bug fix or data error correction (e.g., `fix(features): handle zero division in occupancy rate`)
- `docs:` Documentation only changes (e.g., `docs(assumptions): log outlier threshold justification`)
- `test:` Adding or updating automated tests (e.g., `test(cleaning): add unit test for price sanitizer`)
- `refactor:` Code change that neither fixes a bug nor adds a feature
- `style:` Formatting, missing semi-colons, etc.

---

## 3. Pull Request Checklist

Before submitting a Pull Request (PR) or merging a feature:
- [ ] Code adheres to PEP 8 standards with descriptive variable and function names.
- [ ] Every non-trivial function includes type annotations and docstrings.
- [ ] Automated tests pass (`pytest tests/`).
- [ ] Documentation (`docs/data_dictionary.md`, `docs/assumptions_log.md`) has been updated.
- [ ] No raw data or sensitive credentials are committed.
