---
title: Commit Risk Predictor
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Estimate how likely a commit is to introduce a bug
---

# Commit Risk Predictor

Interactive demo of [commit-risk-predictor](https://github.com/marjorymell/commit-risk-predictor):
estimates how likely a commit is to introduce a bug (Just-in-Time defect prediction), trained on the
git history of [vtex/faststore](https://github.com/vtex/faststore) with SZZ labels.

The model runs in the browser: a logistic regression exported from scikit-learn to `model.json`
and evaluated by `risk.js`. Source code, tests and evaluation live in the GitHub repository.
