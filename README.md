# Q-Regime

**Small-data quantum kernel learning for S&P 500 volatility-regime classification**

Qubit Crew · Qiskit Fall Fest 2026 · University of Ottawa

---

## Financial Motivation and Problem Framing

Volatility is one of the most important quantities in quantitative finance. It shapes risk, option prices, position sizing, leverage, and how aggressively a book must be protected when markets turn rough. Even outside finance, firms such as Citadel, Jane Street, and Two Sigma are familiar names. Whatever their exact internal methods, they operate in markets where the volatility environment is a core input, not a side detail.

Take options as a concrete example. An option’s value depends heavily on expected movement, not only on whether the S&P 500 goes up or down. When expected movement rises, option prices change and desks often need more offsetting trades to manage the risk of the position. Those offsetting trades are commonly called hedges: positions taken to reduce damage if the market jumps. In a calm week, a desk may not need as much protection. In a stressed week, waiting too long can get expensive. That is why a short-horizon volatility call matters even if you never try to predict the market’s direction.

Our project studies a simple version of that problem:

> **Given only information available today, is the S&P 500 likely to enter a high-volatility regime over the next five trading days?**

We chose a five-day horizon because it roughly represents one trading week. That is short enough to matter for tactical risk management, and more meaningful than trying to call a single day. We also frame the task as classification rather than predicting an exact volatility number. A risk process usually needs a regime call: treat the next week as stressed or not. The model’s job is to make that call.

The upside of getting this right is operational, not academic. A reliable five-day high-volatility flag is a signal that could support decisions such as reducing exposure, buying protection earlier, tightening risk limits, or adjusting option inventory before stress is obvious in the rear-view mirror. This project does not claim to replace a trading desk’s full volatility stack. It asks whether that regime flag can be learned from a few carefully chosen market features, under the same data limits that make quantum kernel methods practical to run and interesting to compare.

### How we chose the volatility measure

The project did not begin with the final volatility definition. Our first implementation used five-day volatility calculated from daily close-to-close returns. That gave us a simple starting point and let us build the initial classical and quantum learning pipeline. While reviewing the finance side of the problem, we identified an important weakness: a trading day can swing sharply during the session and still finish close to where it started. In that case, a close-to-close measure can make the day look much calmer than it actually was.

To address this, we moved to range-based volatility estimators that use more of the information available during the trading day. We considered Parkinson, Garman–Klass, and Rogers–Satchell volatility, and selected **Garman–Klass historical volatility** because it uses Open, High, Low, and Close prices. That lets it capture intraday movement that a close-to-close measure can miss.

We obtained 5-day and 20-day Garman–Klass historical volatility from Bloomberg. Before using that data in the final pipeline, we independently reconstructed the 5-day Garman–Klass series from Yahoo Finance OHLC prices. The reconstructed series matched the Bloomberg series almost perfectly on the same dates, which confirmed both the calculation and the time alignment.

### Short-term and medium-term volatility

We use two volatility horizons because they describe different aspects of the market state. The **5-day** Garman–Klass measure captures the recent short-term environment. The **20-day** measure provides a broader view over roughly one trading month. That distinction matters: a sudden spike in short-term volatility can mean one thing in an otherwise calm market, and something different when longer-term volatility is already elevated.

The prediction target is the **future** 5-day Garman–Klass volatility. At each date, the model only receives information that would have been available at that point in time. The outcome is based on volatility realized over the following five trading sessions. A period is labeled high volatility when that future 5-day value sits above the **67th percentile** of the training-period distribution — roughly the top third of historical outcomes the model was allowed to see. The threshold is fit on training data only, so validation and test periods never help define the target.

### Why this becomes a small-data problem

The S&P 500 history contains thousands of trading days. That can create a false sense of abundance. For a five-day volatility regime call, those rows are not thousands of independent lessons.

Nearby labels overlap. Monday’s “next five days” question and Tuesday’s question share most of the same future window, so treating every calendar day as a fresh example double-counts the same market episode. Volatility also arrives in clusters: calm periods and stressed periods come in stretches, not as isolated coin flips. And market regimes change. A model trained on a quiet decade is not automatically prepared for the next stress cycle.

We treat that seriously in the experimental design. First, we thin the data by keeping every fifth observation, so neighboring five-day targets overlap much less. That shrinks the training history from thousands of daily rows to a few hundred more separated examples. Second, we deliberately restrict training further to **20, 50, 100, and 200** labeled observations, repeated across multiple chronological windows in the 2010–2020 history. Each model is then scored on the same later validation period.

This matches a practical reality as well as a statistical one. When a new shock arrives, a desk may have only a thin recent window of comparable labeled days before it must act. The question is not only “can the model learn from a decade of data?” It is “can it still flag an upcoming high-volatility week when the labeled history is scarce?”

That scarce-label setting is also where quantum kernel methods are most practical to compare with classical kernels: the kernel matrix grows with the number of training points, so small \(N\) is both a financial constraint and a fair test bed.

### Why quantum kernels

A classical RBF-SVM decides whether two market days are similar using a mathematical kernel. A quantum-kernel classifier keeps the same learning idea, but changes the notion of similarity: each day is encoded into a quantum state, and the overlap between two states becomes the similarity score. A classical SVM then learns the high-volatility vs calm decision boundary from that similarity matrix.

That matters here for three reasons.

First, the finance problem is naturally nonlinear and regime-dependent. Calm weeks and stressed weeks can look different in ways that are not a simple linear cut on returns and volatility.

Second, near-term quantum kernel methods are most practical on small labeled sets with few features. Our experiment is built around exactly that setting: four market features and training sizes of 20, 50, 100, and 200 observations. This is not a limitation we are hiding. It is the regime where a quantum kernel can be run and compared honestly.

Third, we give the quantum model a matched classical rival. The baseline is an RBF-SVM trained on the same features, the same chronological windows, the same validation days, and the same scaling. The comparison is therefore about the similarity measure itself: classical RBF versus quantum-state overlap, under the same data limits.

Volatility is persistent but not perfectly predictable. A classical kernel can already capture much of that persistence. Any quantum method has to clear that bar, not a random guess.

The core question is therefore:

> **Can a quantum-kernel classifier identify upcoming high-volatility S&P 500 regimes competitively with a classical RBF-SVM when labeled training data is scarce?**

The finance problem is not decoration for a quantum demo. It supplies nonlinear structure, regime shifts, temporal dependence, limited effective sample size, and a fair classical baseline. If quantum kernels help in near-term quantitative finance, small-data regime classification is a plausible place to look. If the classical kernel wins cleanly, that result still matters: it shows where this market decision already lives in classical geometry, and where quantum methods still need a better match to the data.
