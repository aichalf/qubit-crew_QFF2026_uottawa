# Q-Regime — Current Project Status

## Main question

Can a Qiskit quantum-kernel classifier identify upcoming high-volatility
S&P 500 regimes competitively with a classical RBF-SVM when labeled
training data is scarce?

## Current target

Binary classification of the next 5-trading-day volatility regime.

The target is based on Bloomberg 5-day Garman-Klass historical volatility.
We verified the Bloomberg series independently using Yahoo OHLC data:

- Same-day correlation: ~0.999965
- Best alignment: shift = 0

The future target is therefore created by shifting the trailing
5-day Garman-Klass series by 5 trading days.

## Data period

2010-01-04 to 2024-12-31

Chronological split:

- Train: 2010–2020
- Validation: 2021–2022
- Test: 2023–2024

The test set has not been used for model selection.

## Final main features

1. return_1d
2. momentum_5d
3. gk_vol_5d
4. gk_vol_20d

## Features tested but excluded from main model

- VIX
- 3-month Treasury yield
- 2Y–10Y Treasury spread

VIX increased high-volatility recall but slightly reduced balanced
accuracy and ROC-AUC.

The 7-feature macro model performed worse overall than the 5-feature model.

## Classical results

Full validation, 4-feature RBF-SVM:

- Balanced Accuracy: 0.7957
- ROC-AUC: 0.8908
- F1: 0.8425
- High-vol Recall: 0.8819

Simple 5-day Garman-Klass persistence baseline:

- ROC-AUC: 0.8591

## Small-data experiment

Training sizes:

- N = 20
- N = 50
- N = 100
- N = 200

Three chronological training windows are used for each N.

Classical mean ROC-AUC:

- N=20: 0.8698
- N=50: 0.8896
- N=100: 0.8982
- N=200: 0.8949

## Next task

Run the Qiskit quantum-kernel model on exactly the same:

- 4 features
- chronological windows
- N values
- validation observations
- MinMax scaling to [0, pi]

Start with:

- ZZ feature map
- reps=1
- linear entanglement
- 4 qubits
- FidelityQuantumKernel
- QSVC
- class_weight="balanced"

Then compare quantum vs classical using:

- Balanced Accuracy
- ROC-AUC
- F1
- High-vol Recall

Do not use the 2023–2024 test set yet.