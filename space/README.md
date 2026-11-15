---
title: Commit Risk Predictor
emoji: 🐞
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Estimate how likely a commit is to introduce a bug
---

# Commit Risk Predictor

Interactive demo of [commit-risk-predictor](https://github.com/marjorymell/commit-risk-predictor):
a Random Forest trained on the git history of [vtex/faststore](https://github.com/vtex/faststore)
to estimate how likely a commit is to introduce a bug (Just-in-Time defect prediction, SZZ labels).

Source code, tests and evaluation live in the GitHub repository.
