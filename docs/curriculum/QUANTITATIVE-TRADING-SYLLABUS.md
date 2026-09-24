# Quantitative Trading, Financial Markets & Investment Engineering: detailed syllabus

Generated from the live catalogue by `node scripts/verify-professional-curriculum.mjs --write-syllabi`. Do not edit this derived list independently.

See [scope, role routes, research and authoring rules](PROFESSIONAL-TRADING-SYSTEM-DESIGN-PLAN.md). Listed order is module reading order; specialist branches are deliberate optional depth. Briefs are plans, not completed research/write or implementation checkpoints.

## Markets, Institutions & Financial Accounting

### 1. Professional Trading: Participants, Desks & the Trade Lifecycle

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `professional-trading-participants-desks-the-trade-lifecycle`.
- Prerequisites: No topic-level prerequisites; begin here.
- Named concept coverage (planned): Buy side versus sell side; Agency versus principal; Trade capture; Clearing versus settlement.
- Scope: Follow an investment idea through an order, fill, clearing and settlement; Distinguish asset managers, hedge funds, dealers, brokers, exchanges and proprietary firms; Map research, portfolio, execution, development and risk responsibilities; Explain who holds inventory, owes cash and bears each failure risk.
- Investigation: trade-lifecycle swimlanes — Who owns the risk at each handoff? Move a simulated order between institutions and inspect obligations.
- Practice: Map a pension allocation and a market-maker trade. Success: Identify counterparties, revenue sources, custody and settlement obligations.

### 2. Market Instruments, Returns & Cash-Flow Accounting

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `market-instruments-returns-cash-flow-accounting`.
- Prerequisites: Algebra, Functions, Exponentials & Logarithms
- Named concept coverage (planned): Simple and log returns; Total returns; Dividends and splits; Gross versus net returns.
- Scope: Identify a claim, price, holding and cash flow; Calculate simple and log returns; Account for dividends and share splits; Separate gross returns from costs; Reconcile a small portfolio ledger.
- Investigation: holdings and cash ledger — Does a share split create an investment gain? Apply a split or dividend and trace price, share count and cash separately.
- Practice: Calculate returns from a synthetic corporate-action series. Success: Holdings and cash reconcile without double-counting adjusted prices or dividends.

### 3. Market Efficiency, Behavioral Finance & Sources of Trading Returns

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `market-efficiency-behavioral-finance-sources-of-trading-returns`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting; Professional Trading: Participants, Desks & the Trade Lifecycle
- Named concept coverage (planned): Efficient-market hypothesis; Limits to arbitrage; Risk versus alpha; Behavioral biases; Liquidity provision.
- Scope: Ask why a trade might earn a return and who takes the other side; Distinguish compensation for risk, liquidity, information and behavioral effects; Compare market-efficiency hypotheses with limits to arbitrage; Turn an appealing market story into a falsifiable economic hypothesis.
- Investigation: return-source and counterparty map — Who pays for this apparent edge and why might it persist? Remove a claimed friction and reassess the hypothesis.
- Practice: Classify three proposed strategy rationales. Success: Separate evidence, assumptions and reasons an opportunity may disappear.

### 4. Financial Statements, Valuation & Fundamental Research

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `financial-statements-valuation-fundamental-research`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Income, balance sheet and cash-flow statements; Working capital; Enterprise versus equity value; DCF and multiples; Point-in-time filings.
- Scope: Connect a company business to its financial statements; Reconcile income, balance sheet and cash flow with working capital; Compare enterprise value, equity value, multiples and discounted cash flows; Explain estimates, accounting choices and point-in-time filing availability.
- Investigation: linked statement bridge — How can profit rise while cash falls? Change receivables, capex and debt in a balanced toy company.
- Practice: Value a company under two operating scenarios. Success: Statements reconcile and terminal-value sensitivity is explicit.

### 5. Interest Rates, Compounding, Discounting & Market Conventions

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `interest-rates-compounding-discounting-market-conventions`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Day-count conventions; Business-day adjustment; Discount factors; Continuous compounding; Settlement calendars.
- Scope: Price dated cash flows in a stated currency; Compare simple, periodic and continuous compounding; Apply day counts, business calendars and settlement lags; Reconcile present values across quote conventions.
- Investigation: cash-flow timeline — Does changing a quoted rate convention change value? Switch conventions while retaining the same economic cash flows.
- Practice: Reprice a short cash-flow schedule. Success: Accrual dates, discount factors and units agree.

### 6. Trading P&L, Positions, Cost Basis & Performance Accounting

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `trading-p-l-positions-cost-basis-performance-accounting`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Realized versus unrealized P&L; TWR versus IRR; Financing and borrow costs; FX translation; Lot accounting and jurisdiction-specific tax assumptions.
- Scope: Trace signed fills into holdings and cash; Separate realized, unrealized, trading and financing P&L; Compare time-weighted and money-weighted performance; Reconcile marks, fees, cash movements and base-currency translation.
- Investigation: double-entry position ledger — Why does account cash differ from strategy profit? Replay trades, funding and investor flows separately.
- Practice: Reconcile a multi-day trading statement. Success: Ending equity equals starting equity plus flows and net P&L.

### 7. Macroeconomics, Monetary Policy & Cross-Asset Transmission

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `macroeconomics-monetary-policy-cross-asset-transmission`.
- Prerequisites: Interest Rates, Compounding, Discounting & Market Conventions
- Named concept coverage (planned): Inflation and growth surprises; Policy-rate expectations; Balance-of-payments transmission; Release vintages and revisions.
- Scope: Connect inflation, growth and policy to market expectations; Read economic releases with revisions and release calendars; Trace rates, currencies, equities and credit through competing scenarios; Distinguish an economic narrative from a testable surprise signal.
- Investigation: scenario transmission map — Why can good economic news lower an asset price? Change the surprise relative to expectations and policy response.
- Practice: Write competing scenarios for an inflation release. Success: Specify observables, timing, alternative explanations and invalidation.

### 8. Hedge Fund Structures, Mandates, Fees & Prime Brokerage

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `hedge-fund-structures-mandates-fees-prime-brokerage`.
- Prerequisites: Professional Trading: Participants, Desks & the Trade Lifecycle; Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): CTAs and multi-manager platforms; High-water marks; Hurdles and performance fees; Prime brokerage; Liquidity gates and side pockets.
- Scope: Translate an investor mandate into portfolio constraints; Explain long-short funds, CTAs, multi-manager platforms and proprietary books; Model fees, high-water marks, liquidity terms and operational counterparties; Assess gross-to-net returns, incentives and due-diligence questions.
- Investigation: fund cash and responsibility map — Who receives a gross trading gain? Apply expenses, management fees and a high-water-mark example.
- Practice: Compare two hypothetical fund mandates. Success: Explain fee conventions, leverage limits, redemptions and operational dependencies.

## Instruments, Financing & Asset-Class Mechanics

### 9. Equities, Corporate Actions, Indices & ETF Mechanics

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `equities-corporate-actions-indices-etf-mechanics`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): ETF creation and redemption; NAV and authorized participants; Rights, spin-offs and splits; Index rebalancing; Tracking difference.
- Scope: Trace ownership through dividends, splits, rights and spin-offs; Distinguish price, total-return and index series; Explain ETF creation, redemption, NAV and tracking differences; Model rebalance turnover and corporate-action adjustments.
- Investigation: basket and share ledger — When can an ETF price differ from its basket? Change basket marks, costs and creation constraints.
- Practice: Reconcile a dividend and index rebalance. Success: No adjustment or cash distribution is counted twice.

### 10. Bonds, Yield Curves, Duration, Convexity & Credit Spreads

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `bonds-yield-curves-duration-convexity-credit-spreads`.
- Prerequisites: Interest Rates, Compounding, Discounting & Market Conventions
- Named concept coverage (planned): Clean versus dirty price; DV01 and key-rate duration; Z-spread and OAS; Credit migration; Recovery assumptions.
- Scope: Price a bond from dated coupons and principal; Distinguish clean price, dirty price, yield and discount curve; Compute duration, convexity, DV01 and key-rate exposure; Separate rate moves, credit spreads and default losses.
- Investigation: cash-flow and curve-shock view — Can equal-duration bonds respond differently? Apply parallel and nonparallel yield shocks.
- Practice: Hedge a two-bond portfolio under curve twists. Success: Explain remaining convexity and key-rate risk.

### 11. Futures, Forwards, Basis, Carry & Contract Rolls

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `futures-forwards-basis-carry-contract-rolls`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Cost of carry; Initial and variation margin; Basis convergence; Contango and backwardation; Continuous-contract construction.
- Scope: Compare a future with a forward cash-flow agreement; Price cost of carry and distinguish basis from expected spot return; Trace daily variation margin, expiry and delivery; Construct continuous research series without inventing executable roll prices.
- Investigation: spot-futures and margin timeline — Where does a roll return come from? Change curve shape and track actual contracts and cash.
- Practice: Reconcile a rolled futures position. Success: Contract multipliers, margin cash and roll fills are explicit.

### 12. Foreign Exchange, Cross-Currency Basis & FX Forwards

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `foreign-exchange-cross-currency-basis-fx-forwards`.
- Prerequisites: Interest Rates, Compounding, Discounting & Market Conventions
- Named concept coverage (planned): Covered interest parity; FX swaps; Cross-currency basis; NDFs; Quote and settlement conventions.
- Scope: Read base and quote currencies and spot settlement conventions; Derive covered interest parity using matched cash flows; Explain forward points, currency swaps and cross-currency basis; Translate hedged and unhedged multi-currency P&L.
- Investigation: two-currency cash-flow loop — Is an apparent FX arbitrage self-financing? Add funding spreads and settlement calendars.
- Practice: Price and reconcile a currency hedge. Success: All currency signs, notionals and cash dates agree.

### 13. Commodities, Storage, Seasonality & Physical Delivery

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `commodities-storage-seasonality-physical-delivery`.
- Prerequisites: Futures, Forwards, Basis, Carry & Contract Rolls
- Named concept coverage (planned): Convenience yield; Storage constraints; Power and gas markets; Weather and seasonality; Location and grade basis; Negative prices.
- Scope: Connect a commodity contract to physical quality and location; Explain storage constraints, convenience yield and seasonal curves; Compare energy, metals and agricultural supply chains; Analyze delivery, weather and basis risks in spread trades.
- Investigation: inventory and forward-curve view — Why can neighboring delivery months diverge? Change storage availability and seasonal demand.
- Practice: Stress a calendar-spread hypothesis. Success: Delivery terms, inventory constraints and liquidity are documented.

### 14. Options Contracts, Payoffs, Exercise & Assignment

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `options-contracts-payoffs-exercise-assignment`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): European versus American exercise; Put-call parity; Assignment and pin risk; Cash versus physical settlement; Spread and combination payoffs.
- Scope: Read option rights, obligations and contract multipliers; Construct payoff and profit diagrams for calls, puts and spreads; Distinguish European, American, cash and physical settlement; Analyze exercise, assignment, dividends and expiration risk.
- Investigation: payoff and obligation diagram — How does profit differ from payoff? Change premium, exercise style and underlying settlement.
- Practice: Reconcile an assigned option spread. Success: Premiums, delivered assets, cash and residual exposure balance.

### 15. Securities Lending, Short Selling, Repo & Funding Liquidity

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `securities-lending-short-selling-repo-funding-liquidity`.
- Prerequisites: Equities, Corporate Actions, Indices & ETF Mechanics; Bonds, Yield Curves, Duration, Convexity & Credit Spreads
- Named concept coverage (planned): Repo haircut; Securities borrow and recalls; Hard-to-borrow fees; Locate and settlement constraints; Funding squeezes.
- Scope: Trace the borrowed asset and collateral in a short or repo; Compute borrow fees, haircuts, rebates and financing cash flows; Explain recalls, buy-ins, specialness and rehypothecation boundaries; Stress a profitable trade under funding withdrawal.
- Investigation: collateral and funding network — Can a solvent strategy be forced to liquidate? Raise haircuts or recall borrowed shares.
- Practice: Cost a financed long-short position. Success: Include borrow availability, cash timing and adverse funding scenarios.

### 16. Swaps, Credit Derivatives & Structured Products

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `swaps-credit-derivatives-structured-products`.
- Prerequisites: Bonds, Yield Curves, Duration, Convexity & Credit Spreads; Options Contracts, Payoffs, Exercise & Assignment
- Named concept coverage (planned): Interest-rate swaps; CDS and hazard rates; Total-return swaps; Structured notes; Waterfalls and tranche risk.
- Scope: Map swap legs, reset dates and contingent credit payments; Explain interest-rate swaps, CDS, total-return swaps and securitization; Decompose structured notes and convertible bonds into risk components; Trace documentation, optionality and counterparty dependencies.
- Investigation: contingent cash-flow tree — Which party pays after a rate reset or default? Trigger contract events and inspect settlement obligations.
- Practice: Decompose a structured payoff into simpler claims. Success: State model, credit, liquidity and legal-document assumptions.

### 17. Digital-Asset Market Structure, Custody & Perpetual Futures

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `digital-asset-market-structure-custody-perpetual-futures`.
- Prerequisites: Futures, Forwards, Basis, Carry & Contract Rolls
- Named concept coverage (planned): Perpetual funding; Mark and index prices; Liquidations; Custody and exchange risk; Stablecoin depegs.
- Scope: Compare centralized venues, on-chain settlement and custody models; Trace perpetual funding, margin and liquidation mechanics; Explain stablecoin, bridge, oracle and venue-credit exposure; Reconcile fragmented 24-hour markets and on-chain transaction costs.
- Investigation: collateral and liquidation timeline — Can a hedge fail when one venue liquidates? Change collateral values, funding and withdrawal availability.
- Practice: Stress a cross-venue basis position. Success: Account for liquidation rules, custody and unavailable transfers.

### 18. Credit, Mortgages, Prepayment & Securitized-Product Risk

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `credit-mortgages-prepayment-securitized-product-risk`.
- Prerequisites: Swaps, Credit Derivatives & Structured Products
- Named concept coverage (planned): MBS and ABS; CPR and PSA prepayment; Negative convexity; Tranche waterfalls; Default and recovery.
- Scope: Trace loan cash flows through defaults, recoveries and prepayments; Explain mortgage optionality, tranches, waterfalls and correlation exposure; Compare spread, duration and extension risk under scenarios; Separate contractual cash flows from model-dependent valuation.
- Investigation: cash-flow waterfall and prepayment timeline — How does refinancing change who receives cash and when? Change prepayment and default timing.
- Practice: Stress a small securitized pool and its tranches. Success: Conserve cash and explain model, credit and liquidity limits.

## Quantitative Foundations & Financial Econometrics

### 19. Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks)

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `probability-theory-for-quant-finance-martingales-stopping-times-random-walks`.
- Prerequisites: Probability Distributions & Bayes' Theorem; Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Conditional expectation; Filtrations and stopping times; Martingales; Optional-stopping assumptions; Change of measure foundations.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 20. Linear Algebra for Finance (Covariance Matrices, PCA, Factor Decomposition)

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `linear-algebra-for-finance-covariance-matrices-pca-factor-decomposition`.
- Prerequisites: Vectors, Matrices & Tensor Operations; Random Variables, Expectation & Covariance
- Named concept coverage (planned): Positive-semidefinite covariance; PCA risk factors; Cholesky simulation; Matrix conditioning; Factor exposures.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 21. Python for Quantitative Research (NumPy, Pandas, Vectorized Backtesting)

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `python-for-quantitative-research-numpy-pandas-vectorized-backtesting`.
- Prerequisites: NumPy: Arrays, Broadcasting & Vectorization; Pandas: Data Wrangling, Joins & Grouping; Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Timestamp alignment; Vectorization; Panel data; Numerical precision; Reproducible data transformations.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 22. Econometrics (Cointegration, Granger Causality, VECM, Unit Roots)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `econometrics-cointegration-granger-causality-vecm-unit-roots`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks); Linear & Logistic Regression
- Named concept coverage (planned): ADF and KPSS; Engle-Granger and Johansen tests; VECM; Granger predictability versus causality; Structural breaks.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 23. Convex Optimization & Dynamic Programming for Finance (CVXPY, Bellman Equations)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `convex-optimization-dynamic-programming-for-finance-cvxpy-bellman-equations`.
- Prerequisites: Convex Optimization; MDPs, Bellman Equations & Dynamic Programming
- Named concept coverage (planned): KKT conditions; Duality; CVXPY modeling; Bellman recursion; Constraint sensitivity.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 24. Bayesian Statistics & Inference for Finance (Signal Combination, Hierarchical Models)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `bayesian-statistics-inference-for-finance-signal-combination-hierarchical-models`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks); Bayesian Inference & Conjugate Priors
- Named concept coverage (planned): Hierarchical shrinkage; Posterior predictive checks; Bayesian signal combination; Prior sensitivity.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 25. Stochastic Calculus for Finance (Itô Calculus, SDEs)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `stochastic-calculus-for-finance-it-calculus-sdes`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks); Itô Calculus & Stochastic Differential Equations
- Named concept coverage (planned): Brownian motion; Ito's lemma; Stochastic integration; Girsanov theorem assumptions; Feynman-Kac connection.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 26. ARIMA, GARCH & Classical Time-Series

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `arima-garch-classical-time-series`.
- Prerequisites: Econometrics (Cointegration, Granger Causality, VECM, Unit Roots)
- Named concept coverage (planned): ARMA and ARIMA; ARCH, GARCH and EGARCH; EWMA volatility; Stationarity; Residual diagnostics.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 27. Decision Theory, Utility & Betting Under Uncertainty

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `decision-theory-utility-betting-under-uncertainty`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks)
- Named concept coverage (planned): Expected utility; Risk aversion; Certainty equivalents; Model uncertainty; Decision costs.
- Scope: Represent actions, uncertain outcomes and decision objectives; Compare expected value, utility, risk aversion and information value; Reason about sequential decisions and strategic counterparties; Choose an action while exposing probability and loss assumptions.
- Investigation: decision tree and utility comparison — Can the highest expected payoff be the wrong action for this mandate? Change wealth, loss limits and outcome probabilities.
- Practice: Solve a trading decision under uncertain fill and loss outcomes. Success: Distinguish probability estimates from decision preferences.

### 28. Financial Inference: HAC Errors, Bootstrap & Multiple Testing

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `financial-inference-hac-errors-bootstrap-multiple-testing`.
- Prerequisites: Econometrics (Cointegration, Granger Causality, VECM, Unit Roots); Hypothesis Testing & Confidence Intervals
- Named concept coverage (planned): Newey-West HAC; Block bootstrap; False discovery rate; Multiple comparisons; Dependence-adjusted uncertainty.
- Scope: Identify dependence in overlapping financial observations; Compare IID errors with heteroskedasticity and autocorrelation corrections; Use block bootstrap and multiplicity controls for families of tests; Report uncertainty without treating selected in-sample significance as discovery.
- Investigation: resampling blocks and trial funnel — How many independent observations or trials are present? Change overlap, block length and number of hypotheses.
- Practice: Audit significance for correlated strategy returns. Success: Dependence assumptions and selection history are stated.

### 29. Regime Detection & Hidden Markov Models for Markets

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `regime-detection-hidden-markov-models-for-markets`.
- Prerequisites: Econometrics (Cointegration, Granger Causality, VECM, Unit Roots); Hidden Markov Models (HMM)
- Named concept coverage (planned): Filtered versus smoothed regimes; Transition probabilities; Label switching; Regime instability.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 30. State-Space Filtering, Change Points & Online Financial Estimation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `state-space-filtering-change-points-online-financial-estimation`.
- Prerequisites: Econometrics (Cointegration, Granger Causality, VECM, Unit Roots); Regime Detection & Hidden Markov Models for Markets
- Named concept coverage (planned): Kalman and particle filters; CUSUM; Bayesian online change-point detection; Filtering versus smoothing; Detection delay.
- Scope: Separate latent market state from noisy observations; Compare Kalman, particle filtering and change-point models; Update estimates causally without hindsight smoothing; Measure detection delay, false alarms and parameter drift.
- Investigation: filtered versus smoothed state timeline — Which estimate was knowable when the trade occurred? Reveal future observations only after a prediction.
- Practice: Evaluate an online regime monitor on a synthetic break. Success: Report false alarms, delay and filtered-only decisions.

### 31. Point Processes, Hawkes Models & High-Frequency Econometrics

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `point-processes-hawkes-models-high-frequency-econometrics`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks); Econometrics (Cointegration, Granger Causality, VECM, Unit Roots)
- Named concept coverage (planned): Hawkes intensity; Marked point processes; Microstructure noise; Realized variance and realized kernels; Asynchronous sampling and Epps effect.
- Scope: Model irregularly timed order and trade arrivals; Estimate intensity, excitation and marked-event effects; Explain microstructure noise, realized volatility and asynchronous sampling; Validate stationarity, residuals and event-time forecasting.
- Investigation: event raster and conditional intensity — Does one event raise the modeled chance of another? Change excitation and observe decay and unstable regimes.
- Practice: Fit and diagnose a synthetic event-arrival model. Success: Residual and stability checks accompany parameter estimates.

### 32. Stochastic Control, HJB Equations & Optimal Stopping in Finance

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `stochastic-control-hjb-equations-optimal-stopping-in-finance`.
- Prerequisites: Stochastic Calculus for Finance (Itô Calculus, SDEs); Convex Optimization & Dynamic Programming for Finance (CVXPY, Bellman Equations)
- Named concept coverage (planned): Hamilton-Jacobi-Bellman equation; Verification theorem; Merton portfolio problem; Optimal stopping and free boundaries; Impulse control.
- Scope: Specify state, admissible controls, information and a finite-horizon objective; Connect dynamic programming to Hamilton-Jacobi-Bellman equations and verification; Distinguish continuous control, impulse decisions and optimal stopping with free boundaries; Apply small portfolio and execution models while checking costs, constraints and model assumptions.
- Investigation: state-time value surface and action or stopping regions — When is waiting more valuable than acting now? Change costs and risk preferences and compare a numerical policy with a simple benchmark.
- Practice: Solve and validate a bounded trading-control problem. Success: State admissibility and boundary conditions and check policy value independently of the solver.

## Market Microstructure, Execution & Market Making

### 33. Market Microstructure & Order Book Modeling

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `market-microstructure-order-book-modeling`.
- Prerequisites: Professional Trading: Participants, Desks & the Trade Lifecycle; Futures, Forwards, Basis, Carry & Contract Rolls
- Named concept coverage (planned): Limit order book; Spread and depth; Order-flow imbalance; Microprice; Queue dynamics.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 34. Order Types, Matching Rules, Auctions & Queue Priority

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `order-types-matching-rules-auctions-queue-priority`.
- Prerequisites: Market Microstructure & Order Book Modeling
- Named concept coverage (planned): IOC and FOK; Iceberg orders; Price-time versus pro-rata; Opening and closing auctions; Halts and price bands.
- Scope: Trace market, limit, stop, IOC, FOK and reserve orders; Compare price-time, pro-rata and auction matching rules; Handle cancel-replace priority, halts and price bands; Predict fills from venue-specific rules rather than price touch alone.
- Investigation: matching-engine queue replay — Which order executes first and why? Change priority rules and replay the same arrivals.
- Practice: Manually clear a small continuous book and auction. Success: Preserve quantity, eligibility and stated priority rules.

### 35. RFQ Markets, Dealer Pricing & Electronic OTC Trading

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `rfq-markets-dealer-pricing-electronic-otc-trading`.
- Prerequisites: Market Microstructure & Order Book Modeling; Bonds, Yield Curves, Duration, Convexity & Credit Spreads; Foreign Exchange, Cross-Currency Basis & FX Forwards
- Named concept coverage (planned): Request for quote (RFQ); Dealer axes; All-to-all trading; Streaming quotes; Last-look rules and disclosure; Credit eligibility.
- Scope: Compare a central limit order book with bilateral and multi-dealer request-for-quote workflows; Trace quote requests, responses, expiry, acceptance and credit eligibility; Explain inventory, axes, client flow, information leakage and principal-versus-agency risk; Evaluate quote competition, rejection and execution quality using venue-specific rules.
- Investigation: dealer-client quote lifecycle and inventory trace — Why is the tightest visible quote not necessarily available to this client? Change quote expiry, dealer inventory and credit eligibility.
- Practice: Design an RFQ replay and execution-quality analysis. Success: Preserve timestamped eligibility, responses and rejections without assuming anonymous exchange matching.

### 36. Price Discovery, Adverse Selection & Order-Flow Information

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `price-discovery-adverse-selection-order-flow-information`.
- Prerequisites: Market Microstructure & Order Book Modeling; Econometrics (Cointegration, Granger Causality, VECM, Unit Roots)
- Named concept coverage (planned): Kyle and Glosten-Milgrom models; Trade signing; Adverse selection; Price impact versus predictability; Post-fill markouts.
- Scope: Decompose spreads into inventory, processing and information effects; Explain informed order flow and adverse-selection models; Compare microprice, imbalance, trade sign and quote response; Test predictive relationships without claiming causal identification.
- Investigation: quote and conditional markout view — Why can a successful fill be economically bad? Vary informed flow and measure post-fill price movement.
- Practice: Estimate a bounded order-flow signal with delayed evaluation. Success: Distinguish spread capture from adverse-selection losses.

### 37. Transaction Cost Analysis & Slippage Modeling (Market Impact, Cost-Aware Optimization)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `transaction-cost-analysis-slippage-modeling-market-impact-cost-aware-optimization`.
- Prerequisites: Market Microstructure & Order Book Modeling; Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Implementation shortfall; Effective and realized spread; Temporary versus permanent impact; Opportunity cost; Market-impact calibration.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 38. Execution Algorithms & Optimal Execution (TWAP, VWAP, Almgren-Chriss)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `execution-algorithms-optimal-execution-twap-vwap-almgren-chriss`.
- Prerequisites: Market Microstructure & Order Book Modeling; Convex Optimization & Dynamic Programming for Finance (CVXPY, Bellman Equations)
- Named concept coverage (planned): TWAP and VWAP; Participation of volume (POV); Arrival-price execution; Almgren-Chriss; Volume-curve uncertainty.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 39. Smart Order Routing, Venue Selection & Liquidity Fragmentation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `smart-order-routing-venue-selection-liquidity-fragmentation`.
- Prerequisites: Execution Algorithms & Optimal Execution (TWAP, VWAP, Almgren-Chriss); Order Types, Matching Rules, Auctions & Queue Priority
- Named concept coverage (planned): Maker-taker fees; Dark and lit liquidity; Conditional orders; Best-execution constraints; Venue toxicity.
- Scope: Compare accessible liquidity across venues; Incorporate fees, rebates, latency, fill likelihood and information leakage; Handle partial fills, stale quotes and conditional liquidity; Evaluate routing against a stated benchmark and obligations.
- Investigation: multi-venue order-routing timeline — Is the best displayed price the best attainable fill? Change latency, queue depth and venue charges.
- Practice: Route a synthetic parent order across venues. Success: Reconcile residual quantities and explain execution-quality tradeoffs.

### 40. Market-Making & Liquidity Provision (Avellaneda-Stoikov, Inventory Management)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `market-making-liquidity-provision-avellaneda-stoikov-inventory-management`.
- Prerequisites: Market Microstructure & Order Book Modeling; Stochastic Calculus for Finance (Itô Calculus, SDEs)
- Named concept coverage (planned): Reservation price; Inventory skew; Avellaneda-Stoikov; Spread capture versus adverse selection; Model calibration limits.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 41. Queue Position, Fill Probability & Limit-Order Placement

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `queue-position-fill-probability-limit-order-placement`.
- Prerequisites: Order Types, Matching Rules, Auctions & Queue Priority; Point Processes, Hawkes Models & High-Frequency Econometrics
- Named concept coverage (planned): Queue-reactive models; Cancellation uncertainty; Fill probability; Latency and adverse selection; Hidden liquidity.
- Scope: Estimate volume ahead using observable book data; Model cancellation uncertainty and conditional fill probabilities; Trade off queue priority, adverse selection and inventory; Validate placement decisions with conservative partial-fill assumptions.
- Investigation: queue survival and fill trace — What happens if cancellations occur behind our order? Change hidden queue assumptions and compare bounds.
- Practice: Bound fills from incomplete market-by-price observations. Success: Report uncertainty rather than inventing exact queue positions.

### 42. Multi-Agent RL & Market Simulation

- Level: frontier; retained/shared topic; planned lesson.
- Stable ID: `multi-agent-rl-market-simulation`.
- Prerequisites: Market Microstructure & Order Book Modeling; MDPs, Bellman Equations & Dynamic Programming
- Named concept coverage (planned): Agent-based simulation; Calibration versus realism; Policy interaction; Market-impact feedback; Simulation-to-market gap.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

## Financial Data & Reproducible Research Infrastructure

### 43. Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data)

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `financial-data-pitfalls-survivorship-bias-look-ahead-bias-point-in-time-data`.
- Prerequisites: Python for Quantitative Research (NumPy, Pandas, Vectorized Backtesting); Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Delisting returns; Survivorship bias; As-of joins; Revision and publication timestamps; Look-ahead leakage.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 44. Security Masters, Identifiers, Calendars & Corporate-Action Data

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `security-masters-identifiers-calendars-corporate-action-data`.
- Prerequisites: Equities, Corporate Actions, Indices & ETF Mechanics; Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data)
- Named concept coverage (planned): ISIN, CUSIP and venue identifiers; Symbol reuse; Corporate-action mapping; Trading calendars; Bitemporal data.
- Scope: Resolve instruments independently of changing ticker symbols; Model listing histories, identifiers and effective-date intervals; Apply trading calendars, time zones and corporate-action versions; Join observations without conflating companies, securities and contracts.
- Investigation: bitemporal identifier timeline — Which instrument did this ticker name on that date? Move between effective and knowledge dates.
- Practice: Repair a research universe with renamed and delisted securities. Success: Entity links and adjustment histories remain reproducible.

### 45. Tick Data, Order-Book Reconstruction & Feed Quality

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `tick-data-order-book-reconstruction-feed-quality`.
- Prerequisites: Market Microstructure & Order Book Modeling; Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data)
- Named concept coverage (planned): Sequence gaps; Snapshots and incremental updates; Corrections and trade busts; Crossed-book diagnostics; Exchange versus receive time.
- Scope: Decode trades, quotes and order-level messages; Reconstruct books from snapshots and ordered deltas; Detect gaps, duplicates, corrections, crossed books and auction states; Compare event, receive and exchange timestamps.
- Investigation: message-to-book replay — Does a missing cancel change the apparent liquidity? Inject a gap and recover from a snapshot.
- Practice: Rebuild a small synthetic book from a feed. Success: Sequence checks and book invariants catch deliberate corruptions.

### 46. Financial Time-Series Storage: Columnar Files, kdb+ & Streaming Data

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `financial-time-series-storage-columnar-files-kdb-streaming-data`.
- Prerequisites: Python for Quantitative Research (NumPy, Pandas, Vectorized Backtesting); Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data)
- Named concept coverage (planned): kdb+ and q; Parquet and Arrow; Time partitioning; As-of queries; Append-only event replay.
- Scope: Choose storage around symbol-time access patterns; Compare columnar partitions, compression and time-series databases; Design as-of joins, streaming ingestion and versioned snapshots; Benchmark scans, selective queries and reproducible retrieval costs.
- Investigation: partition and as-of join map — Which partitions and records answer this historical query? Change partition keys and temporal join bounds.
- Practice: Design a tick-to-research storage layout. Success: Point-in-time semantics, retention and measured query plans are explicit.

### 47. Alternative Data Sources (Satellite, Web Traffic, Social)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `alternative-data-sources-satellite-web-traffic-social`.
- Prerequisites: Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data)
- Named concept coverage (planned): Geospatial proxies; Web and social signals; Entity mapping; Coverage bias; Economic mechanism.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 48. Alternative-Data Due Diligence, Entitlements & Research Lineage

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `alternative-data-due-diligence-entitlements-research-lineage`.
- Prerequisites: Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data); Alternative Data Sources (Satellite, Web Traffic, Social)
- Named concept coverage (planned): Data licensing; MNPI and provenance; Consent and privacy; Vendor selection bias; Lineage and deletion.
- Scope: Trace a dataset from collection through permission and delivery; Evaluate coverage drift, revisions, selection and lawful access; Record licenses, entitlements, provenance and reproducible transforms; Distinguish economic information from collection-process artifacts.
- Investigation: data provenance and availability graph — Did this apparent signal exist before the vendor backfilled it? Toggle vintage and collection coverage.
- Practice: Write a data acceptance memo. Success: Include rights, timestamps, revision policy and falsifiable quality checks.

### 49. Reproducible Quant Experiments, Research Compute & GPU Acceleration

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `reproducible-quant-experiments-research-compute-gpu-acceleration`.
- Prerequisites: Python for Quantitative Research (NumPy, Pandas, Vectorized Backtesting); Testing, Debugging & Dependency Management
- Named concept coverage (planned): Dataset and environment versioning; Random seeds; Vectorized CPU versus GPU; Distributed trial accounting; Numerical reproducibility.
- Scope: Separate immutable inputs, experiment configuration and results; Record seeds, software versions and research trial lineage; Profile CPU, vectorized, distributed and GPU workloads; Reproduce a result within numerical tolerance and cost constraints.
- Investigation: experiment dependency graph — Can another researcher recreate this number? Change one input version and trace invalidated outputs.
- Practice: Package a reproducible parameter study. Success: A clean run reproduces data selection and results with declared tolerances.

## Forecasting, Research Design & Backtest Validation

### 50. Feature Engineering & Alpha Research Methodology (IC, IR, Decay, Turnover)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `feature-engineering-alpha-research-methodology-ic-ir-decay-turnover`.
- Prerequisites: Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data); Econometrics (Cointegration, Granger Causality, VECM, Unit Roots)
- Named concept coverage (planned): Information coefficient (IC); Rank IC; Signal decay; Turnover; Neutralization and winsorization.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 51. Backtesting Frameworks & Avoiding Overfitting

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `backtesting-frameworks-avoiding-overfitting`.
- Prerequisites: Python for Quantitative Research (NumPy, Pandas, Vectorized Backtesting); Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data); Market Microstructure & Order Book Modeling
- Named concept coverage (planned): Vectorized versus event-driven backtests; Data leakage; Transaction costs; Reproducible strategy state; Selection bias.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 52. Financial Labels, Event Sampling & Overlapping Outcomes

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `financial-labels-event-sampling-overlapping-outcomes`.
- Prerequisites: Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data); Backtesting Frameworks & Avoiding Overfitting
- Named concept coverage (planned): Triple-barrier labels; Meta-labeling; Event-based bars; Overlapping labels; Sample uniqueness and weights.
- Scope: Define the decision horizon before constructing a target; Compare fixed-horizon, event and barrier-based labels; Track overlapping labels, censoring and sampling weights; Separate executable entry timing from feature and label timestamps.
- Investigation: feature-label interval timeline — Which training outcome overlaps the test decision? Move label horizons and reveal overlap.
- Practice: Construct a labeled event dataset. Success: Every feature is available at entry and every outcome interval is recorded.

### 53. Walk-Forward Testing, Purging, Embargoes & Nested Model Selection

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `walk-forward-testing-purging-embargoes-nested-model-selection`.
- Prerequisites: Backtesting Frameworks & Avoiding Overfitting; Financial Labels, Event Sampling & Overlapping Outcomes
- Named concept coverage (planned): Walk-forward evaluation; Purged cross-validation; Embargo; Nested model selection; Overlapping-horizon leakage.
- Scope: Match evaluation splits to the actual retraining schedule; Remove overlapping information with justified purging and embargo rules; Nest tuning inside temporal validation and retain a final evaluation boundary; Compare stability across regimes and estimate effective evidence.
- Investigation: train-label-test interval bands — Which samples leak into this validation fold? Move train cutoffs and highlight information overlap.
- Practice: Design a temporal evaluation for overlapping holdings. Success: Explain each exclusion and preserve the deployment information set.

### 54. Backtest Overfitting, Deflated Sharpe & Research Trial Accounting

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `backtest-overfitting-deflated-sharpe-research-trial-accounting`.
- Prerequisites: Backtesting Frameworks & Avoiding Overfitting; Financial Inference: HAC Errors, Bootstrap & Multiple Testing
- Named concept coverage (planned): Deflated Sharpe ratio; Probability of backtest overfitting; White reality check and SPA; Trial registry; Research degrees of freedom.
- Scope: Record a research search including discarded variants; Explain selection-adjusted performance and probabilistic Sharpe assumptions; Compare probability-of-overfitting procedures and holdout designs; Decide what evidence remains after repeated reuse of the test set.
- Investigation: null-strategy selection distribution — How impressive is the best of many noisy strategies? Change trial count and dependence between candidates.
- Practice: Audit an attractive selected backtest. Success: Show trial history, assumptions and remaining independent evidence.

### 55. Backtest Selection Bias & Execution Reconciliation

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `backtest-selection-bias-execution-reconciliation`.
- Prerequisites: Backtesting Frameworks & Avoiding Overfitting; Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data); Market Instruments, Returns & Cash-Flow Accounting
- Named concept coverage (planned): Survivor strategy selection; Order-fill reconciliation; Paper-to-live attribution; Negative controls.
- Scope: Record every tested strategy variant; Separate selection and evaluation data; Translate target positions into orders; Model delays, partial fills and fees; Reconcile paper results and explain uncertainty.
- Investigation: signal-to-fill timeline — Which expected trades were never executable at the backtest price? Change latency and fill assumptions while retaining the same signal.
- Practice: Audit a deliberately optimistic synthetic backtest. Success: Selection history, unresolved execution assumptions and net accounting are explicit.

### 56. Event-Driven Backtesting, Fill Models & Paper-to-Live Gaps

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `event-driven-backtesting-fill-models-paper-to-live-gaps`.
- Prerequisites: Backtesting Frameworks & Avoiding Overfitting; Market Microstructure & Order Book Modeling
- Named concept coverage (planned): Event clocks; Partial fills and queue models; Fees and borrow; Latency replay; Exchange rejection paths.
- Scope: Implement the signal-order-acknowledgment-fill-position lifecycle; Model fees, spread, latency, partial fills and queue uncertainty; Reconcile a replay with cash and holdings invariants; Compare vectorized, event-driven and shadow-trading evidence.
- Investigation: event clock and order state machine — Can this fill occur before the order reaches the venue? Change latency and available liquidity.
- Practice: Reject impossible fills in a synthetic replay. Success: Orders, fills, positions and cash reconcile at every event.

### 57. Classical Forecasting (Prophet, N-BEATS, N-HiTS)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `classical-forecasting-prophet-n-beats-n-hits`.
- Prerequisites: ARIMA, GARCH & Classical Time-Series; Walk-Forward Testing, Purging, Embargoes & Nested Model Selection
- Named concept coverage (planned): Prophet decomposition; N-BEATS and N-HiTS neural models; Seasonality; Forecast horizons; Rolling-origin evaluation.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 58. Temporal Fusion Transformers & Neural Forecasting

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `temporal-fusion-transformers-neural-forecasting`.
- Prerequisites: Classical Forecasting (Prophet, N-BEATS, N-HiTS)
- Named concept coverage (planned): Known versus observed covariates; Quantile forecasts; Variable selection; Attention interpretation limits.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 59. Foundation Models for Time-Series (TimesFM, Chronos, Moirai)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `foundation-models-for-time-series-timesfm-chronos-moirai`.
- Prerequisites: Temporal Fusion Transformers & Neural Forecasting
- Named concept coverage (planned): TimesFM, Chronos and Moirai; Zero-shot versus fine-tuned forecasting; Training-data overlap; Forecast calibration; Model-version evaluation.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 60. Financial ML: Nonstationarity, Calibration & Economic Evaluation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `financial-ml-nonstationarity-calibration-economic-evaluation`.
- Prerequisites: Walk-Forward Testing, Purging, Embargoes & Nested Model Selection; Linear & Logistic Regression
- Named concept coverage (planned): Concept drift; Probability calibration; Cost-sensitive objectives; Economic versus statistical significance; Online-learning evaluation.
- Scope: Start from a decision and a simple forecast baseline; Compare ranking, probability, magnitude and uncertainty outputs; Evaluate feature instability, regime shift and calibration; Translate forecasts into cost-aware decisions under risk constraints.
- Investigation: forecast-to-action decision curve — Does a better prediction metric improve net utility? Change trading costs and decision thresholds.
- Practice: Compare a complex model with a simple baseline out of time. Success: Report prediction uncertainty, turnover and net decision consequences.

## Portfolio Construction, Risk & Capital Allocation

### 61. Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `portfolio-optimization-markowitz-black-litterman-risk-parity`.
- Prerequisites: Linear Algebra for Finance (Covariance Matrices, PCA, Factor Decomposition); Convex Optimization & Dynamic Programming for Finance (CVXPY, Bellman Equations)
- Named concept coverage (planned): Mean-variance frontier; Black-Litterman; Risk parity; Robust optimization; Estimation error.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 62. Covariance Estimation, Shrinkage & Multi-Factor Risk Models

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `covariance-estimation-shrinkage-multi-factor-risk-models`.
- Prerequisites: Linear Algebra for Finance (Covariance Matrices, PCA, Factor Decomposition); Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity)
- Named concept coverage (planned): Ledoit-Wolf shrinkage; Factor covariance; Specific risk; Random-matrix denoising; Positive-semidefinite constraints.
- Scope: Estimate exposures, factor covariance and residual risk; Compare sample covariance with shrinkage and structured estimates; Diagnose conditioning, missing observations and unstable correlations; Assess portfolio sensitivity to risk-model error.
- Investigation: covariance spectrum and portfolio weights — Why can small estimation errors create extreme positions? Change sample size and shrinkage strength.
- Practice: Compare risk forecasts on held-out returns. Success: Report forecast error and weight stability under matched constraints.

### 63. Constrained Portfolios, Turnover, Liquidity & Capacity

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `constrained-portfolios-turnover-liquidity-capacity`.
- Prerequisites: Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity); Transaction Cost Analysis & Slippage Modeling (Market Impact, Cost-Aware Optimization)
- Named concept coverage (planned): Transaction-cost-aware rebalancing; Tax-aware lot constraints; Liquidity participation; Capacity; Tracking-error constraints.
- Scope: Translate a mandate into budget, exposure and position constraints; Include borrow, lot size, turnover and nonlinear impact; Model trade scheduling and alpha decay as capital grows; Diagnose binding constraints, infeasibility and capacity limits.
- Investigation: feasible portfolio and cost frontier — When does more capital reduce net opportunity? Increase capital and participation constraints.
- Practice: Optimize a small cost-aware portfolio. Success: Check feasibility, sensitivity and an independent objective calculation.

### 64. Position Sizing, Kelly Criteria & Drawdown-Constrained Allocation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `position-sizing-kelly-criteria-drawdown-constrained-allocation`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks); Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity)
- Named concept coverage (planned): Full and fractional Kelly; Risk of ruin; Drawdown constraints; Leverage; Parameter uncertainty.
- Scope: Compare notional, volatility and loss-budget sizing; Derive log-growth allocation under stated distribution assumptions; Study fractional Kelly, estimation error and correlated bets; Evaluate ruin, drawdown and liquidity constraints.
- Investigation: wealth path and sizing distribution — How does overestimating an edge change survival? Change forecast error and leverage.
- Practice: Compare sizing rules under misspecified probabilities. Success: Report path losses and sensitivity rather than only average growth.

### 65. Risk Models & Tail Risk (VaR, CVaR, Stress Testing, Extreme Value Theory)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `risk-models-tail-risk-var-cvar-stress-testing-extreme-value-theory`.
- Prerequisites: Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity); Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks)
- Named concept coverage (planned): Expected shortfall (ES/CVaR); Historical and parametric VaR; Peaks over threshold; Generalized Pareto distribution; Risk-model backtesting.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 66. Stress Testing, Liquidity Spirals & Crowded-Trade Unwinds

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `stress-testing-liquidity-spirals-crowded-trade-unwinds`.
- Prerequisites: Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity); Securities Lending, Short Selling, Repo & Funding Liquidity
- Named concept coverage (planned): Reverse stress tests; Crowding; Margin spirals; Correlation breakdown; Liquidity-adjusted losses.
- Scope: Build historical, hypothetical and reverse-stress scenarios; Link losses to margin calls, forced sales and market impact; Model crowding, correlation breakdown and liquidations; Identify actions and limits before a risk budget is breached.
- Investigation: funding-market feedback loop — Can forced sales amplify the initial shock? Change haircuts and liquidation depth.
- Practice: Construct a reverse stress for a levered book. Success: Identify a plausible failure path and decision thresholds.

### 67. Counterparty Risk, Collateral, Margin & Valuation Adjustments

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `counterparty-risk-collateral-margin-valuation-adjustments`.
- Prerequisites: Swaps, Credit Derivatives & Structured Products
- Named concept coverage (planned): CVA, DVA, FVA and MVA; Netting and CSA; Potential future exposure; Wrong-way risk; Initial and variation margin.
- Scope: Distinguish current exposure from future exposure distributions; Model netting, collateral, initial and variation margin; Explain CVA, DVA, FVA, MVA and wrong-way risk boundaries; Stress counterparty default and collateral liquidity.
- Investigation: exposure and collateral timelines — What remains at risk after margining? Change netting sets and margin delays.
- Practice: Compare collateralized and uncollateralized exposures. Success: Record closeout, netting, funding and default assumptions.

### 68. Performance Attribution, Skill Evaluation & Manager Selection

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `performance-attribution-skill-evaluation-manager-selection`.
- Prerequisites: Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity); Financial Inference: HAC Errors, Bootstrap & Multiple Testing
- Named concept coverage (planned): Brinson attribution; Factor attribution; Sharpe, Sortino and Calmar ratios; Selection and survivorship; Manager capacity.
- Scope: Separate asset allocation, selection, factor and implementation effects; Compare benchmarks, Sharpe, information ratio and drawdown measures; Assess survivorship, backfill, serial correlation and track-record length; Write a manager evaluation with uncertainty and capacity constraints.
- Investigation: gross-to-net and factor attribution waterfall — Which component explains the reported alpha? Change benchmark and account for fees and smoothing.
- Practice: Audit a synthetic manager track record. Success: Separate evidence of skill from exposure, luck and reporting choices.

### 69. Multi-Strategy Allocation, Risk Budgets & Portfolio Governance

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `multi-strategy-allocation-risk-budgets-portfolio-governance`.
- Prerequisites: Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity); Constrained Portfolios, Turnover, Liquidity & Capacity
- Named concept coverage (planned): Marginal risk contribution; Strategy correlation; Capital allocation; Drawdown escalation; Mandate constraints.
- Scope: Allocate capital across correlated strategy books; Design marginal-risk budgets, limits and drawdown escalation; Model shared execution, financing and crowded exposures; Document promotion, scale-down and retirement decisions.
- Investigation: strategy-to-factor exposure network — Are apparently different strategies sharing one risk? Stress a common exposure and reallocate budgets.
- Practice: Propose a multi-strategy allocation memo. Success: Explain correlation uncertainty, capacity and accountable overrides.

## Derivative Pricing, Numerical Methods & Hedging

### 70. Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `options-pricing-derivatives-black-scholes-monte-carlo-neural-sdes`.
- Prerequisites: Options Contracts, Payoffs, Exercise & Assignment; Stochastic Calculus for Finance (Itô Calculus, SDEs)
- Named concept coverage (planned): Black-Scholes-Merton; Black-76; Bachelier normal model; Garman-Kohlhagen; Risk-neutral valuation.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 71. No-Arbitrage Pricing, Numeraires & Risk-Neutral Measures

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `no-arbitrage-pricing-numeraires-risk-neutral-measures`.
- Prerequisites: Stochastic Calculus for Finance (Itô Calculus, SDEs); Options Contracts, Payoffs, Exercise & Assignment
- Named concept coverage (planned): Fundamental theorems of asset pricing; Equivalent martingale measures; Numeraire changes; Forward measures; Incomplete markets.
- Scope: Construct replication in a small complete market; Distinguish physical probabilities from pricing measures; Change numeraire and identify discounting assumptions; Explain incomplete markets and limits of a unique model price.
- Investigation: replicating cash-flow tree — Which probability prices a payoff and which forecasts it? Change physical drift while preserving replicated prices.
- Practice: Price and hedge a binomial claim two ways. Success: Replication and discounted pricing agree under declared assumptions.

### 72. Greeks, Dynamic Hedging & Options P&L Attribution

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `greeks-dynamic-hedging-options-p-l-attribution`.
- Prerequisites: Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs)
- Named concept coverage (planned): Delta, gamma, vega and theta; Vanna and volga; Discrete hedging error; Volatility P&L; Sticky-strike versus sticky-delta.
- Scope: Compute first and higher-order price sensitivities; Connect delta, gamma, vega, theta and rho to local P&L; Track discrete hedge rebalancing and transaction costs; Diagnose residual P&L from jumps, surfaces and model error.
- Investigation: hedge path and sensitivity surface — Why does a delta-neutral book still move? Shock spot and volatility and vary hedge frequency.
- Practice: Reconcile a hedged option path. Success: Explain sensitivity approximation error and funding costs.

### 73. Volatility Surfaces, Calibration & Static Arbitrage

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `volatility-surfaces-calibration-static-arbitrage`.
- Prerequisites: Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs)
- Named concept coverage (planned): Implied-volatility smile; SVI and SSVI; Butterfly and calendar arbitrage; Surface interpolation; Market quote conventions.
- Scope: Translate option prices into implied volatility quotes; Compare strike, moneyness, delta and maturity conventions; Calibrate a surface while checking calendar and butterfly constraints; Assess interpolation, extrapolation and quote uncertainty.
- Investigation: strike-maturity surface and arbitrage checks — Can a smooth surface still imply an impossible price? Move a quote and inspect price constraints.
- Practice: Repair an inconsistent small option surface. Success: Retain bid-ask uncertainty and demonstrate no-arbitrage checks.

### 74. Local Volatility, Stochastic Volatility, Jumps & Rough Models

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `local-volatility-stochastic-volatility-jumps-rough-models`.
- Prerequisites: Volatility Surfaces, Calibration & Static Arbitrage; Stochastic Calculus for Finance (Itô Calculus, SDEs)
- Named concept coverage (planned): Dupire local volatility; Heston; SABR; Merton and Bates jump models; Rough volatility.
- Scope: Compare diffusion assumptions behind local and stochastic volatility; Explain Heston, SABR, jump and rough-volatility mechanisms; Calibrate with identifiability and numerical constraints; Compare hedging and extrapolation behavior outside calibration quotes.
- Investigation: same-fit different-path comparison — Can two models fit vanillas and disagree on an exotic? Hold vanilla calibration fixed and compare paths.
- Practice: Compare two model families on a bounded pricing task. Success: State parameter uncertainty and out-of-sample limitations.

### 75. Pricing Trees, PDE Solvers & Early-Exercise Boundaries

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `pricing-trees-pde-solvers-early-exercise-boundaries`.
- Prerequisites: Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs); Numerical Methods (Finite Differences, Quadrature, Root Finding)
- Named concept coverage (planned): Binomial and trinomial trees; Crank-Nicolson; Finite-difference stability; American free boundary; Convergence checks.
- Scope: Build a recombining tree and a finite-difference pricing grid; Set terminal and boundary conditions with contract conventions; Handle American exercise as an optimal stopping problem; Check stability, convergence and agreement with analytic limits.
- Investigation: backward pricing lattice and exercise frontier — Where is exercising better than continuing? Change dividend and grid resolution.
- Practice: Price an American put with convergence evidence. Success: Separate truncation, discretization and model error.

### 76. Monte Carlo Pricing, Variance Reduction & Adjoint Greeks

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `monte-carlo-pricing-variance-reduction-adjoint-greeks`.
- Prerequisites: Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs); Stochastic Calculus for Finance (Itô Calculus, SDEs)
- Named concept coverage (planned): Antithetic and control variates; Importance and quasi-Monte Carlo sampling; Longstaff-Schwartz; Pathwise and likelihood-ratio Greeks; Adjoint algorithmic differentiation (AAD).
- Scope: Simulate discounted payoffs with confidence intervals; Compare antithetic, control-variate, quasi-Monte Carlo and importance methods; Estimate sensitivities with bumping, pathwise, likelihood and adjoint methods; Handle bias, discontinuities and least-squares early exercise.
- Investigation: estimator error versus computation — Does a narrower interval hide discretization bias? Change path count and time step independently.
- Practice: Validate a variance-reduced estimator against an analytic case. Success: Separate sampling error, discretization and differentiation assumptions.

### 77. Interest-Rate Curve Construction & Multi-Curve Derivatives

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `interest-rate-curve-construction-multi-curve-derivatives`.
- Prerequisites: Bonds, Yield Curves, Duration, Convexity & Credit Spreads; Swaps, Credit Derivatives & Structured Products
- Named concept coverage (planned): Bootstrapping; OIS discounting; Forward-rate curves; Hull-White and short-rate models; LIBOR-market-model history and RFR conventions.
- Scope: Bootstrap discount and forward curves from quoted instruments; Respect calendars, day counts, collateral and index conventions; Price swaps, caps and swaptions with model-specific assumptions; Measure curve-node risk and calibration residuals.
- Investigation: instrument-to-curve bootstrap — Does the fitted curve reprice its input instruments? Change one quote and inspect propagated nodes.
- Practice: Bootstrap a small curve and reprice its inputs. Success: Date conventions, interpolation and residual tolerance are explicit.

### 78. Financial Calibration, Fourier Pricing & Numerical Sensitivity

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `financial-calibration-fourier-pricing-numerical-sensitivity`.
- Prerequisites: Local Volatility, Stochastic Volatility, Jumps & Rough Models; Monte Carlo Pricing, Variance Reduction & Adjoint Greeks
- Named concept coverage (planned): Characteristic functions; FFT and COS pricing; Calibration identifiability; Regularization; Finite-difference sensitivity.
- Scope: Separate pricing, calibration and sensitivity error budgets; Use characteristic functions, Fourier inversion and quadrature where applicable; Compare constrained local and global calibration with regularization; Diagnose ill-conditioning, nonidentifiability and numerical differentiation error.
- Investigation: calibration loss landscape and price-error plot — Can very different parameters fit the same quotes? Perturb quotes, integration bounds and solver tolerances.
- Practice: Validate a calibration on analytic and perturbed cases. Success: Report residuals, stability and numerical convergence separately.

### 79. Exotic Options, Credit Models & Model Risk Validation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `exotic-options-credit-models-model-risk-validation`.
- Prerequisites: Monte Carlo Pricing, Variance Reduction & Adjoint Greeks; Counterparty Risk, Collateral, Margin & Valuation Adjustments
- Named concept coverage (planned): Barrier, Asian and digital options; Bermudan exercise; Structural and reduced-form credit; Copula dependence; Independent benchmark pricing.
- Scope: Specify barrier, Asian, basket and path-dependent contractual events; Contrast structural and reduced-form default models; Stress model choice, correlations, recovery and calibration ambiguity; Build independent benchmarks and model-use limitations.
- Investigation: path-dependent payoff and default tree — Which path detail changes the payout? Alter a barrier crossing or recovery while keeping the terminal spot fixed.
- Practice: Write a model-validation note for one structured claim. Success: Identify independently checked cases, limitations and reserve rationale.

### 80. Deep Hedging & RL for Trading

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `deep-hedging-rl-for-trading`.
- Prerequisites: Greeks, Dynamic Hedging & Options P&L Attribution; MDPs, Bellman Equations & Dynamic Programming
- Named concept coverage (planned): Transaction-cost-aware hedging; Utility and risk objectives; Policy constraints; Simulator misspecification; Benchmark hedges.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

## Alpha Strategies & Investment Hypotheses

### 81. Alpha Signal Generation & Factor Models

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `alpha-signal-generation-factor-models`.
- Prerequisites: Feature Engineering & Alpha Research Methodology (IC, IR, Decay, Turnover); Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity)
- Named concept coverage (planned): Factor exposures and returns; Risk premium versus alpha; Cross-sectional regression; Signal combination; Orthogonalization.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 82. Equity Factors, Cross-Sectional Signals & Neutralization

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `equity-factors-cross-sectional-signals-neutralization`.
- Prerequisites: Alpha Signal Generation & Factor Models; Econometrics (Cointegration, Granger Causality, VECM, Unit Roots)
- Named concept coverage (planned): Value, momentum, quality and low volatility; Sector and beta neutrality; Fama-MacBeth regression; Factor crowding.
- Scope: Build a point-in-time cross-sectional universe; Construct value, momentum, quality, size and defensive signals; Separate ranking, winsorization and sector or risk neutralization; Measure turnover, factor overlap and net out-of-sample behavior.
- Investigation: rank and exposure decomposition — Is a signal mostly a sector bet? Compare raw and neutralized portfolios.
- Practice: Build and audit a small factor portfolio. Success: Expose membership, rebalance, neutralization and cost assumptions.

### 83. Trend Following, Time-Series Momentum & Managed Futures

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `trend-following-time-series-momentum-managed-futures`.
- Prerequisites: Futures, Forwards, Basis, Carry & Contract Rolls; Backtesting Frameworks & Avoiding Overfitting
- Named concept coverage (planned): Time-series momentum; Volatility targeting; Futures rolls; Whipsaw risk; Crisis behavior.
- Scope: Distinguish trend following from cross-sectional momentum; Compare signal horizons, scaling and entry-exit rules; Construct a diversified futures portfolio with realistic rolls; Diagnose whipsaws, crisis behavior and volatility-target lag.
- Investigation: price trend and position timeline — How does the same rule react to a reversal? Change horizon while holding costs and risk budget fixed.
- Practice: Compare trend horizons on synthetic regimes. Success: Use the same risk budget and report turnover and drawdowns.

### 84. Statistical Arbitrage & Pairs Trading (Cointegration, Ornstein-Uhlenbeck)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `statistical-arbitrage-pairs-trading-cointegration-ornstein-uhlenbeck`.
- Prerequisites: Econometrics (Cointegration, Granger Causality, VECM, Unit Roots); Backtesting Frameworks & Avoiding Overfitting
- Named concept coverage (planned): Ornstein-Uhlenbeck process; Half-life estimation; Hedge-ratio stability; Cointegration versus correlation; Spread breakdown.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 85. Carry, Value & Cross-Asset Risk Premia

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `carry-value-cross-asset-risk-premia`.
- Prerequisites: Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity); Foreign Exchange, Cross-Currency Basis & FX Forwards
- Named concept coverage (planned): Currency and futures carry; Value convergence; Cross-asset signals; Crash exposure; Financing-adjusted returns.
- Scope: Separate expected carry from realized price change; Compare currency, rates, credit, equity and commodity premia; Model financing, hedging and correlated crash exposure; Attribute returns to rewarded risk, implementation and residual alpha.
- Investigation: carry and shock P&L decomposition — What offsets many small carry gains? Introduce a funding or volatility shock.
- Practice: Compare two carry implementations. Success: Include funding, tail dependence and executable hedging costs.

### 86. Event-Driven Trading, Merger Arbitrage & Index Reconstitutions

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `event-driven-trading-merger-arbitrage-index-reconstitutions`.
- Prerequisites: Financial Statements, Valuation & Fundamental Research; Backtesting Frameworks & Avoiding Overfitting
- Named concept coverage (planned): Deal-break risk; Merger spread; Tender and spin-off events; Index inclusion flows; Event-time leakage.
- Scope: Translate an event into dated conditional cash flows; Analyze merger completion, breaks and deal terms; Model earnings surprises, spin-offs and index flows; Separate event probability, crowding and financing effects.
- Investigation: event tree and tradable information timeline — Which outcome explains the observed spread? Change completion probability and break value.
- Practice: Write a scenario-weighted event study. Success: Use available information, conditional losses and explicit execution windows.

### 87. Fixed-Income Relative Value, Curve Trades & Basis Strategies

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `fixed-income-relative-value-curve-trades-basis-strategies`.
- Prerequisites: Bonds, Yield Curves, Duration, Convexity & Credit Spreads; Securities Lending, Short Selling, Repo & Funding Liquidity
- Named concept coverage (planned): Curve steepeners and butterflies; Swap spread; Cash-futures basis; On-the-run versus off-the-run; Convertible and capital-structure arbitrage.
- Scope: Construct curve steepeners, butterflies and spread positions; Hedge DV01 and key-rate exposures with contract conventions; Analyze cash-futures basis and cheapest-to-deliver optionality; Stress convergence under funding and liquidity shocks.
- Investigation: key-rate exposure bars and financing timeline — Is this relative-value position actually duration neutral? Shock each curve node and repo spread.
- Practice: Design a financed curve trade. Success: Separate convergence thesis, hedge error, roll and funding P&L.

### 88. Volatility Trading, Dispersion & Variance Risk Premia

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `volatility-trading-dispersion-variance-risk-premia`.
- Prerequisites: Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs); Greeks, Dynamic Hedging & Options P&L Attribution
- Named concept coverage (planned): Variance swaps; Dispersion and correlation; Volatility risk premium; Gamma scalping; Tail hedging and skew.
- Scope: Separate implied from realized volatility and variance; Analyze delta-hedged options, variance swaps and skew exposure; Explain index-versus-component dispersion and correlation risk; Measure hedging error, costs and jump losses.
- Investigation: variance and hedge P&L attribution — Can implied volatility fall while this position loses? Change spot path, skew and hedge frequency.
- Practice: Stress a toy dispersion portfolio. Success: Reconcile Greeks, correlation exposure, costs and tail scenarios.

### 89. Sentiment Analysis & NLP for Finance (FinBERT)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `sentiment-analysis-nlp-for-finance-finbert`.
- Prerequisites: Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data); Financial ML: Nonstationarity, Calibration & Economic Evaluation
- Named concept coverage (planned): FinBERT; Filing and news timestamps; Entity resolution; Event extraction; Text leakage.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 90. Graph Neural Networks for Financial Markets (Sector Rotation, Systemic Risk)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `graph-neural-networks-for-financial-markets-sector-rotation-systemic-risk`.
- Prerequisites: Alpha Signal Generation & Factor Models; Financial ML: Nonstationarity, Calibration & Economic Evaluation
- Named concept coverage (planned): Supply-chain and ownership graphs; Temporal graph leakage; Systemic risk; Sector propagation.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 91. LLMs for Financial Analysis

- Level: frontier; retained/shared topic; planned lesson.
- Stable ID: `llms-for-financial-analysis`.
- Prerequisites: Sentiment Analysis & NLP for Finance (FinBERT)
- Named concept coverage (planned): Evidence-linked extraction; RAG over filings; Numeric verification; Point-in-time evaluation; Hallucination and prompt-injection controls.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 92. Crypto & DeFi Quantitative Strategies (AMM, MEV, On-Chain Analytics)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `crypto-defi-quantitative-strategies-amm-mev-on-chain-analytics`.
- Prerequisites: Digital-Asset Market Structure, Custody & Perpetual Futures; Backtesting Frameworks & Avoiding Overfitting
- Named concept coverage (planned): AMMs and concentrated liquidity; Impermanent loss; MEV; Gas and settlement risk; Oracle manipulation.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

## HFT, Exchange Connectivity & Low-Latency Engineering

### 93. High-Frequency Trading & Low-Latency Infrastructure

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `high-frequency-trading-low-latency-infrastructure`.
- Prerequisites: Market Microstructure & Order Book Modeling; Event-Driven Backtesting, Fill Models & Paper-to-Live Gaps
- Named concept coverage (planned): Tick-to-trade latency; Colocation; Latency budgets; Jitter; Exchange protocol boundaries.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 94. Market-Making Risk, Markouts & Cross-Instrument Hedging

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `market-making-risk-markouts-cross-instrument-hedging`.
- Prerequisites: Market-Making & Liquidity Provision (Avellaneda-Stoikov, Inventory Management); Greeks, Dynamic Hedging & Options P&L Attribution
- Named concept coverage (planned): Inventory limits; Markout horizons; Hedge basis risk; Toxic flow; Quote pull decisions.
- Scope: Attribute spread capture, inventory movement, fees and hedge costs; Set inventory-skew and exposure limits across related instruments; Measure conditional markouts over multiple horizons; Stress stale quotes, volatility bursts and hedge venue failure.
- Investigation: inventory and markout dashboard — Did the quote earn a spread or acquire unwanted risk? Replay fills and delayed hedges.
- Practice: Diagnose a loss-making market-making session. Success: Reconcile fill-level economics and propose testable control changes.

### 95. Low-Latency Systems & C++ for Quant (Lock-Free, FPGA, Co-Location)

- Level: advanced; retained/shared topic; planned lesson.
- Stable ID: `low-latency-systems-c-for-quant-lock-free-fpga-co-location`.
- Prerequisites: High-Frequency Trading & Low-Latency Infrastructure; C & C++ Foundations for GPU Programming; CPU Architecture (Cores, Caches, SIMD, Pipelining)
- Named concept coverage (planned): C++ hot paths; FPGA offload boundaries; Colocation topology; Deterministic replay.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 96. FIX Sessions, Binary Protocols & Order Gateway State

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `fix-sessions-binary-protocols-order-gateway-state`.
- Prerequisites: Order Types, Matching Rules, Auctions & Queue Priority; Networking Foundations: Packets, Transport, DNS & Sockets
- Named concept coverage (planned): FIX sequence recovery; Cancel-replace races; Drop copy; OUCH and SBE protocol families; Order-state reconciliation.
- Scope: Separate session sequencing from business order state; Parse FIX and binary order-entry messages from specifications; Handle reconnects, resend requests, duplicate execution reports and rejects; Reconcile unknown outcomes without accidentally duplicating an order.
- Investigation: session and order state machines — Did a lost acknowledgment mean the order failed? Drop acknowledgments and replay messages.
- Practice: Build a simulated reconnect-and-reconcile gateway. Success: Orders and fills remain unique across duplicate and delayed reports.

### 97. Market-Data Feed Handlers, Multicast & Gap Recovery

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `market-data-feed-handlers-multicast-gap-recovery`.
- Prerequisites: Tick Data, Order-Book Reconstruction & Feed Quality; Networking Foundations: Packets, Transport, DNS & Sockets
- Named concept coverage (planned): ITCH; Multicast A/B feeds; Packet loss and gap recovery; Snapshot synchronization; Sequence validation.
- Scope: Decode bounded binary messages and maintain sequence state; Compare multicast delivery, redundant feeds and recovery channels; Apply snapshots consistently with buffered incremental updates; Measure stale-book detection and recovery under burst loss.
- Investigation: dual-feed sequence waterfall — When can the reconstructed book be trusted again? Drop, reorder and duplicate packets.
- Practice: Recover a deterministic feed replay after gaps. Success: Do not publish a valid book until synchronization invariants hold.

### 98. Low-Latency C++: Memory Layout, Atomics & Lock-Free Queues

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `low-latency-c-memory-layout-atomics-lock-free-queues`.
- Prerequisites: C & C++ Foundations for GPU Programming; CPU Architecture (Cores, Caches, SIMD, Pipelining)
- Named concept coverage (planned): Cache-line alignment; False sharing; Memory ordering; SPSC and MPSC queues; Allocation avoidance; SIMD.
- Scope: Profile allocations, cache misses and false sharing on a hot path; Apply ownership, bounded storage and cache-aware layouts; Reason about atomics, memory ordering, ABA and reclamation; Compare lock-free and locked designs under tail-latency workloads.
- Investigation: cache-line ownership and happens-before trace — Why can two independent counters interfere? Change padding and producer-consumer synchronization.
- Practice: Benchmark and validate a bounded message queue. Success: Use race checks, invariants and percentile measurements.

### 99. Kernel Bypass, NIC Queues, NUMA & Network Hot Paths

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `kernel-bypass-nic-queues-numa-network-hot-paths`.
- Prerequisites: Low-Latency C++: Memory Layout, Atomics & Lock-Free Queues; Networking Foundations: Packets, Transport, DNS & Sockets
- Named concept coverage (planned): DPDK and AF_XDP; RSS and flow steering; IRQ affinity; NUMA placement; Busy polling.
- Scope: Trace packets through NIC rings, interrupts and application buffers; Compare kernel networking, polling, DPDK and zero-copy approaches; Tune affinity, NUMA placement, batching and power behavior; Measure latency distributions alongside loss and CPU cost.
- Investigation: packet path and CPU placement diagram — Which boundary adds jitter under load? Compare interrupt and poll timelines with bounded queues.
- Practice: Design a reproducible packet-latency experiment. Success: Report hardware, offered load, loss and full latency distribution.

### 100. Clock Synchronization, Hardware Timestamping & Latency Metrology

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `clock-synchronization-hardware-timestamping-latency-metrology`.
- Prerequisites: Market-Data Feed Handlers, Multicast & Gap Recovery
- Named concept coverage (planned): PTP versus NTP; Hardware timestamping; Clock uncertainty; TSC calibration; One-way versus round-trip latency.
- Scope: Distinguish event time, receive time and monotonic duration; Explain PTP, hardware timestamps, drift and asymmetric paths; Build wire-to-wire and tick-to-trade latency budgets; Quantify measurement error, coordinated omission and percentile uncertainty.
- Investigation: clock-offset and packet timeline — Can an apparently negative latency be a measurement error? Change clock offsets and measurement points.
- Practice: Audit a latency benchmark. Success: State clock uncertainty, timestamp locations and load-generation method.

### 101. FPGA Trading Pipelines, Hardware Offload & Verification

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `fpga-trading-pipelines-hardware-offload-verification`.
- Prerequisites: Kernel Bypass, NIC Queues, NUMA & Network Hot Paths; GPU RTL Design, Verification & Hardware Tradeoffs
- Named concept coverage (planned): HDL pipeline design; Fixed-point arithmetic; Clock-domain crossing; Hardware-software co-simulation; Feed-to-order validation.
- Scope: Map packet parsing, filtering and simple decisions into hardware stages; Compare FPGA and CPU boundaries, fixed-point arithmetic and resources; Specify backpressure, timing closure and host control; Verify protocol behavior and compare wire-to-wire measurements.
- Investigation: clocked pipeline and valid-ready waveform — What happens when the output cannot accept a decision? Insert stalls and malformed packets.
- Practice: Specify and simulate a bounded hardware risk filter. Success: Assert timing-independent correctness and resource limits.

### 102. Colocation, Exchange Certification & Capacity Engineering

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `colocation-exchange-certification-capacity-engineering`.
- Prerequisites: FIX Sessions, Binary Protocols & Order Gateway State; Clock Synchronization, Hardware Timestamping & Latency Metrology
- Named concept coverage (planned): Venue conformance testing; Session throttles; Network redundancy; Rack and cross-connect constraints; Failover drills.
- Scope: Map racks, cross-connects, market-access and recovery dependencies; Plan bandwidth, message rates, throttles and redundant routes; Exercise exchange conformance and disaster-recovery scenarios; Compare total access cost, capacity and venue-specific obligations.
- Investigation: physical connectivity and failure-domain map — Which common dependency defeats redundant links? Fail power, carrier and exchange sessions separately.
- Practice: Produce an exchange-connectivity acceptance plan. Success: Include certification evidence, monitoring and rollback criteria.

## Production Trading, Controls & Fund Operations

### 103. OMS, EMS, Portfolio State & Real-Time Risk Architecture

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `oms-ems-portfolio-state-real-time-risk-architecture`.
- Prerequisites: FIX Sessions, Binary Protocols & Order Gateway State; Trading P&L, Positions, Cost Basis & Performance Accounting
- Named concept coverage (planned): Order and execution management; Event-sourced positions; Drop-copy reconciliation; Real-time limits; Recovery checkpoints.
- Scope: Separate target portfolios, parent orders, child orders and executions; Design durable order state and real-time position reconciliation; Connect research, execution, risk, accounting and control planes; Recover uncertain orders after process or network failure.
- Investigation: order and position authority map — Which component may change the official position? Replay duplicate fills and a restart.
- Practice: Design a simulated multi-strategy trading stack. Success: Identify authoritative state, reconciliation and failure boundaries.

### 104. Pre-Trade Limits, Kill Switches & Fat-Finger Protection

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `pre-trade-limits-kill-switches-fat-finger-protection`.
- Prerequisites: OMS, EMS, Portfolio State & Real-Time Risk Architecture
- Named concept coverage (planned): Price collars; Order-size and credit limits; Cancel-on-disconnect; Self-trade prevention; Kill-switch scope.
- Scope: Specify credit, notional, price, quantity and message-rate controls; Model exposure including outstanding orders and concurrent strategies; Design independent kill, cancel-on-disconnect and permission paths; Test controls under stale data, reconnects and partial outages.
- Investigation: order admission and cancellation state machine — Can two simultaneous orders exceed a shared limit? Race requests and interrupt the cancel channel.
- Practice: Test a fail-safe simulated risk gateway. Success: Demonstrate rejected breaches and reconcile surviving exposure.

### 105. ML Model Lifecycle in Production Trading (Drift, Monitoring, Feature Stores)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `ml-model-lifecycle-in-production-trading-drift-monitoring-feature-stores`.
- Prerequisites: Financial ML: Nonstationarity, Calibration & Economic Evaluation; OMS, EMS, Portfolio State & Real-Time Risk Architecture
- Named concept coverage (planned): Feature lineage; Training-serving parity; Drift monitoring; Model rollback; Shadow deployment.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 106. Trading Releases, Shadow Runs, Incident Response & Recovery

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `trading-releases-shadow-runs-incident-response-recovery`.
- Prerequisites: OMS, EMS, Portfolio State & Real-Time Risk Architecture; Pre-Trade Limits, Kill Switches & Fat-Finger Protection
- Named concept coverage (planned): Paper versus shadow modes; Canary capital limits; Replay recovery; Change control; Incident command.
- Scope: Promote a version from replay to shadow to bounded production; Define drift, stale-signal, P&L and position monitoring; Practice rollback, kill, reconciliation and post-incident review; Separate research improvement from operational authorization.
- Investigation: deployment and recovery timeline — What must be reconciled before a restart may trade? Inject stale data or an order-state mismatch.
- Practice: Run a tabletop trading incident. Success: Preserve audit evidence and prove known positions before resumption.

### 107. Clearing, Settlement, Reconciliation & Treasury Operations

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `clearing-settlement-reconciliation-treasury-operations`.
- Prerequisites: Market Instruments, Returns & Cash-Flow Accounting; Securities Lending, Short Selling, Repo & Funding Liquidity
- Named concept coverage (planned): CCP novation; DvP and PvP; Settlement fails; Cash and position breaks; Corporate-action reconciliation; Treasury liquidity.
- Scope: Trace allocations, confirmations, clearing and settlement; Reconcile broker, custodian, administrator and internal records; Plan margin, collateral substitutions, cash forecasts and fails; Distinguish operational breaks from economic P&L.
- Investigation: post-trade obligation ledger — Which unsettled obligation causes tomorrow's cash shortfall? Delay a settlement or change collateral eligibility.
- Practice: Resolve a synthetic broker reconciliation break. Success: Keep cash, positions, margin and settlement dates consistent.

### 108. Trading Regulation, Market Conduct & Surveillance

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `trading-regulation-market-conduct-surveillance`.
- Prerequisites: Professional Trading: Participants, Desks & the Trade Lifecycle; Order Types, Matching Rules, Auctions & Queue Priority
- Named concept coverage (planned): Market access controls; Best execution; Spoofing and wash-trade surveillance; Short-sale obligations; Jurisdiction-specific reporting.
- Scope: Map venue, participant and jurisdiction before applying rules; Explain market access, best execution, reporting and recordkeeping; Identify spoofing, wash trading, manipulation and MNPI concerns through compliance cases; Compare US, EU, UK and India sources with effective dates and applicability.
- Investigation: control and evidence matrix — What evidence distinguishes a legitimate cancellation from abusive conduct? Inspect contrasting annotated order histories.
- Practice: Create a jurisdiction-specific compliance checklist for a hypothetical desk. Success: Cite current primary rules and distinguish law, guidance and firm policy.

### 109. Explainable AI & Model Governance in Finance (SHAP, SR 11-7)

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `explainable-ai-model-governance-in-finance-shap-sr-11-7`.
- Prerequisites: Financial ML: Nonstationarity, Calibration & Economic Evaluation; Trading Regulation, Market Conduct & Surveillance
- Named concept coverage (planned): SR 26-2 and historical SR 11-7; Independent validation; Model inventory; Use limitations; SHAP versus model validity.
- Scope: Separate a model explanation from validation and governance evidence; Compare development, independent challenge, monitoring and model-use limitations; Explain the historical SR 11-7 reference and the April 2026 SR 26-2 replacement with its applicability; Build a risk-proportionate governance record without treating supervisory guidance as universal law.
- Investigation: model claim and governance evidence map — Does an explanation establish that the model is valid for this use? Trace a changed use through assumptions, independent review and monitoring.
- Practice: Write a model-use and validation memo. Success: Identify applicable dated guidance, independent evidence and unresolved limitations.

### 110. Credit Scoring & Fraud Detection

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `credit-scoring-fraud-detection`.
- Prerequisites: Financial ML: Nonstationarity, Calibration & Economic Evaluation; Explainable AI & Model Governance in Finance (SHAP, SR 11-7)
- Named concept coverage (planned): PD, LGD and EAD; Imbalanced outcomes; Calibration; Fairness and adverse-action explanations; Adversarial drift.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 111. Trading Security, Access Controls & Business Continuity

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `trading-security-access-controls-business-continuity`.
- Prerequisites: OMS, EMS, Portfolio State & Real-Time Risk Architecture
- Named concept coverage (planned): Least privilege; Secrets and signing keys; Disaster recovery; Segregation of duties; Operational resilience.
- Scope: Threat-model credentials, order channels, data feeds and research artifacts; Separate development, production, approvals and secret access; Design immutable audit trails, key rotation and recovery exercises; Validate continuity under cyber, vendor, facility and personnel failures.
- Investigation: trust boundary and recovery map — Can a research credential place a live order? Trace permissions and disable a compromised identity.
- Practice: Write a desk security and continuity design. Success: Show least privilege, recovery objectives and rehearsed controls.

## Professional Practice, Interviews & Integrated Capstones

### 112. Quant Interview Mathematics (Brainteasers, Expected Value, Game Theory)

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `quant-interview-mathematics-brainteasers-expected-value-game-theory`.
- Prerequisites: Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks); Arrays, Strings & Hash Maps
- Named concept coverage (planned): Conditional probability; Expected stopping time; Combinatorics; Game-theoretic reasoning; Estimation and simulation checks.
- Scope: retain the mechanisms and named subtopics in this existing title. Its individual concept-level brief still needs design before a rewrite; this expansion does not claim the older lesson has been re-researched.

### 113. Quant Research Communication, Replication & Investment Memos

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `quant-research-communication-replication-investment-memos`.
- Prerequisites: Backtest Overfitting, Deflated Sharpe & Research Trial Accounting
- Named concept coverage (planned): Falsifiable hypothesis; Replication package; Negative results; Investment versus rejection memo.
- Scope: Turn an economic hypothesis into a falsifiable research proposal; Document data, trials, costs, uncertainty and rejected alternatives; Reproduce a result and communicate failure evidence; Defend deployment, further study or rejection in an investment memo.
- Investigation: claim-to-evidence map — Which conclusion depends on an untested assumption? Remove an evidence source and inspect weakened claims.
- Practice: Write a reproducible investment memo with a negative result. Success: Another researcher can reconstruct the decision and its limits.

### 114. Quant Developer & Trader Interviews: Coding, Markets & Design

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `quant-developer-trader-interviews-coding-markets-design`.
- Prerequisites: Quant Interview Mathematics (Brainteasers, Expected Value, Game Theory); OMS, EMS, Portfolio State & Real-Time Risk Architecture
- Named concept coverage (planned): Coding under constraints; Market arithmetic; Concurrency and systems; Trade-lifecycle debugging; Explaining uncertainty.
- Scope: Map research, developer, trader and risk interview expectations; Solve probability, estimation, algorithms and market-making cases; Defend a bounded system design and diagnose an execution failure; Explain assumptions and tradeoffs under changed constraints.
- Investigation: decision and constraint board — Which answer changes when latency or inventory constraints change? Add one new interview constraint at a time.
- Practice: Complete a timed mock loop and technical debrief. Success: Justify reasoning, code correctness, risks and uncertainty.

### 115. Quant Capstone: Reproducible Cross-Asset Research & Portfolio

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `quant-capstone-reproducible-cross-asset-research-portfolio`.
- Prerequisites: Constrained Portfolios, Turnover, Liquidity & Capacity; Walk-Forward Testing, Purging, Embargoes & Nested Model Selection; Quant Research Communication, Replication & Investment Memos
- Named concept coverage (planned): Point-in-time universe; Research trial registry; Cost and capacity model; Independent portfolio reconciliation.
- Scope: Choose one falsifiable cross-asset hypothesis and licensed dataset; Build point-in-time inputs, baselines and a frozen evaluation; Construct a cost-aware portfolio with risk and capacity analysis; Deliver reproducible artifacts and a deployment-or-rejection memo.
- Investigation: research-to-portfolio evidence dashboard — Does the conclusion survive a plausible cost or data change? Inspect sensitivity using frozen evaluation rules.
- Practice: Complete an independently reproducible research package. Success: Cash reconciles, leakage tests pass and conclusions state uncertainty.

### 116. Quant Capstone: Exchange Replay, Market Maker & Risk Gateway

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `quant-capstone-exchange-replay-market-maker-risk-gateway`.
- Prerequisites: Market-Data Feed Handlers, Multicast & Gap Recovery; Market-Making Risk, Markouts & Cross-Instrument Hedging; Pre-Trade Limits, Kill Switches & Fat-Finger Protection
- Named concept coverage (planned): Order-book replay; Fill and queue assumptions; Pre-trade limits; Fault recovery; Markout and P&L reconciliation.
- Scope: Specify a bounded synthetic exchange and order protocol; Implement deterministic feed replay, order state and a market-making policy; Inject gaps, latency, partial fills and risk breaches; Report reconciled economics, correctness and reproducible latency measurements.
- Investigation: synchronized book, order and risk replay — Can the system recover without unknown exposure? Pause at any event and inspect all authoritative states.
- Practice: Build an offline exchange simulator and review its failures. Success: Conserve quantity and cash and separate simulated from hardware performance.

### 117. Quant Capstone: Derivatives Library & Independent Model Validation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `quant-capstone-derivatives-library-independent-model-validation`.
- Prerequisites: Exotic Options, Credit Models & Model Risk Validation; Interest-Rate Curve Construction & Multi-Curve Derivatives
- Named concept coverage (planned): Contract conventions; Pricing benchmarks; Convergence and Greeks; Calibration stability; Validation memo.
- Scope: Specify contracts, market conventions and supported model boundaries; Implement prices, sensitivities and calibration with independent references; Test no-arbitrage limits, convergence and stressed inputs; Deliver model documentation and explicit unsupported-use cases.
- Investigation: calibration, convergence and risk comparison — Which disagreement is numerical and which is a modeling choice? Compare analytic, tree and Monte Carlo cases.
- Practice: Deliver a small validated pricing and risk library. Success: Independent benchmarks and convergence tests cover claimed behavior.
