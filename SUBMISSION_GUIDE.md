# SIH 2026 Submission Guide

Checklist before sharing this repository link.

## Required repository content

- [x] Actual source code is present (`src/`, `config/`, `tests/`).
- [x] `README.md` explains the project clearly.
- [x] PS ID is filled in `README.md` (section 1); PS title is included.
- [x] Problem statement and proposed solution are explained.
- [x] Key features are listed.
- [x] Technology stack is listed.
- [x] Setup and run instructions work (`README.md` section 12).
- [x] Team members and roles are filled in `README.md` (section 1).
- [x] Screenshots are in `assets/screenshots/`.
- [ ] Final PPT is placed in `submission/`, or its viewer link is in `submission/PRESENTATION.md`.
- [ ] Demo video link is in `submission/DEMO.md` (optional).
- [ ] Repository is accessible to reviewers (check in a private/incognito window).

## Do not upload

- Passwords, API keys, access tokens
- `.env` files (only `.env.example`, which holds no secrets, is committed)
- Private credentials or other confidential information

Large regenerable data (`data/*/raw/`, `data/_shared/`, model binaries, daily forecasts) is git-ignored; the pipeline
rebuilds it (`python -m src.pipeline.run`).

## README answers

1. What problem are we solving? — section 2
2. What is the proposed solution? — section 3
3. How does it work? — section 6 and `docs/architecture.md`
4. Which technologies? — section 5
5. How can a reviewer run it? — section 12
6. What does the output look like? — section 11
7. Features and impact? — sections 4 and 8
