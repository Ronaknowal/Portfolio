# Curriculum coverage inventory

Generated from the live catalogue by `node scripts/build-curriculum-inventory.mjs`. Scope review: 9 September 2026. Regenerate after catalogue changes; this is a status report, not teaching policy.

**1218 unique topics · 28 modules · 213 registered published lessons · 335 topic-specific briefs · 7 guided paths.**

**883 older topics still need individual design.** Every module has domain guidance; this does not make those older topics fully planned or fact-checked. 365 topics have recorded prerequisite reviews; the remaining edges need individual review. Linux is the user-approved teaching reference.

See [the research and scope plan](../../LEARNING-CURRICULUM-PLAN.md), [authoring handoff](../../LESSON-AUTHORING-HANDOFF.md), and [full machine-readable inventory](curriculum-inventory.json).

## Module coverage

| Module | Topics | Published | Individual briefs | Prerequisite reviews recorded |
| --- | ---: | ---: | ---: | ---: |
| Mathematical & Statistical Foundations | 57 | 42 | 51 | 52 |
| Classical Machine Learning | 39 | 36 | 3 | 6 |
| Deep Learning Fundamentals & Architectures | 42 | 39 | 2 | 7 |
| Large Language Models — Architecture, Training & Inference | 61 | 57 | 4 | 6 |
| Reinforcement Learning | 33 | 0 | 2 | 3 |
| Generative Models | 24 | 0 | 2 | 4 |
| NLP, Computer Vision & Multimodal AI | 49 | 0 | 3 | 3 |
| AI for Quantitative Finance | 39 | 0 | 2 | 2 |
| Neural Engineering & Computational Neuroscience | 92 | 0 | 92 | 92 |
| Evolutionary & Bio-Inspired Algorithms | 17 | 0 | 1 | 1 |
| GPU Engineering, CUDA & Large-Scale Systems | 97 | 0 | 97 | 97 |
| Model Optimization & Efficiency | 23 | 0 | 1 | 1 |
| MLOps & Infrastructure | 48 | 0 | 4 | 5 |
| AI Safety, Alignment & Evaluation | 53 | 0 | 2 | 2 |
| Agentic AI & Tool-Using Systems | 26 | 0 | 2 | 2 |
| Frontier Research Areas | 40 | 0 | 2 | 7 |
| Self-Supervised & Contrastive Learning | 12 | 0 | 1 | 1 |
| Meta-Learning (Learning to Learn) | 46 | 0 | 1 | 1 |
| Quantum AI & Quantum Computing for ML | 48 | 0 | 1 | 2 |
| Landmark Models & What Makes Them Notable | 22 | 0 | 1 | 1 |
| Core Frameworks & Tool Ecosystem | 19 | 0 | 2 | 3 |
| Key Terminology Glossary | 89 | 1 | 0 | 0 |
| LLM Evaluation & Assessment | 158 | 0 | 2 | 3 |
| Programming & Scientific Computing | 17 | 17 | 17 | 17 |
| Data Structures & Algorithms | 22 | 22 | 22 | 22 |
| JAX & Functional ML | 13 | 0 | 2 | 3 |
| Robotics, Embodied AI & Simulation | 24 | 0 | 14 | 20 |
| Drosophila & Fly Embodiment | 12 | 0 | 2 | 2 |

Counts within modules may include shared topics. The headline counts each stable topic ID once.

## Individual authoring inventory

`brief` means a topic-specific starting plan. `design needed` means use the domain playbook, research the topic and complete its individual design before writing. Published content also needs an individual quality review unless the handoff records one.

### Mathematical & Statistical Foundations

**Module anchor:** Small vectors, measurements and uncertainty; connect exact operations to an interpreted decision.

**Teaching strategy:** Pose a concrete question and identify known/unknown quantities → Link a small numerical case, a representation and defined notation → Derive the mechanism with justified intermediate steps → Interpret a complete worked result → Test a changed case, assumption and counterexample → Connect to a formal or applied deeper branch.

**Practice:** Hand calculation, explanation of a representation, flawed reasoning to repair, and independent transfer with an explained solution.

**Verification:** Check assumptions, conventions, proof steps, units, numerical approximations and independent reference values.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Vectors, Matrices & Tensor Operations | foundation | published | brief |
| Matrix Decompositions (SVD, QR, Cholesky, LU) | foundation | published | brief |
| Eigenvalues & Eigenvectors | foundation | published | brief |
| Matrix Calculus & Jacobians | intermediate | published | brief |
| Tensor Algebra & Einsum Notation | intermediate | published | brief |
| Randomized Linear Algebra | advanced | published | brief |
| Multivariate Calculus & Gradients | foundation | published | brief |
| Convex Optimization | foundation | published | brief |
| Gradient Descent Variants (SGD, Adam, AdaGrad, RMSProp, LAMB, LARS) | intermediate | published | brief |
| Learning Rate Schedules (Cosine, Warmup, OneCycleLR) | intermediate | published | brief |
| Convex Duality & Lagrangian Methods (KKT Conditions) | intermediate | published | brief |
| Second-Order Methods (L-BFGS, K-FAC, Shampoo, Natural Gradient) | advanced | published | brief |
| Non-Convex Optimization Landscape | advanced | published | brief |
| Constrained & Multi-Objective Optimization | advanced | published | brief |
| Probability Distributions & Bayes' Theorem | foundation | published | brief |
| Maximum Likelihood & MAP Estimation | foundation | published | brief |
| Hypothesis Testing & Confidence Intervals | foundation | published | brief |
| Bayesian Inference & Conjugate Priors | intermediate | published | brief |
| Concentration Inequalities (Hoeffding, Bernstein, Chernoff) | intermediate | published | brief |
| Monte Carlo Methods & MCMC (Metropolis-Hastings, HMC, NUTS) | intermediate | published | brief |
| Variational Inference | intermediate | published | brief |
| Exponential Families & Sufficient Statistics | advanced | published | brief |
| Measure Theory & Probability Spaces | advanced | published | brief |
| Optimal Transport (Wasserstein Distance, Sinkhorn) | advanced | published | brief |
| Causal Inference & Do-Calculus | advanced | published | brief |
| Entropy, Cross-Entropy & KL Divergence | foundation | published | brief |
| Mutual Information & Information Bottleneck | intermediate | published | brief |
| Rate-Distortion Theory | advanced | published | brief |
| f-Divergences & Integral Probability Metrics | advanced | published | brief |
| Graph Fundamentals (Adjacency, Laplacian, Connectivity) | foundation | published | brief |
| Spectral Graph Theory | intermediate | published | brief |
| Combinatorial Optimization & Approximation Algorithms | advanced | published | brief |
| Stochastic Processes (Markov Chains, Brownian Motion, Poisson) | intermediate | published | brief |
| Random Matrix Theory | intermediate | published | brief |
| Queueing Theory (M/M/1, M/G/1, Little's Law) | intermediate | published | brief |
| Dynamical Systems Theory & Chaos | advanced | published | brief |
| Itô Calculus & Stochastic Differential Equations | advanced | published | design needed |
| Numerical Methods (Finite Differences, Quadrature, Root Finding) | foundation | published | design needed |
| Functional Analysis & RKHS | advanced | published | design needed |
| Topology & Topological Data Analysis (TDA) | advanced | published | design needed |
| Category Theory (Emerging Use in ML) | advanced | published | design needed |
| Differential Geometry & Riemannian Manifolds | advanced | published | design needed |
| Algebra, Functions, Exponentials & Logarithms | foundation | planned | brief |
| Sets, Logic, Relations & Proof Techniques | foundation | planned | brief |
| Geometry, Trigonometry & Coordinate Reasoning | foundation | planned | brief |
| Counting, Combinatorics & Mathematical Induction | foundation | planned | brief |
| Single-Variable Calculus: Limits, Derivatives & Integrals | foundation | planned | brief |
| Random Variables, Expectation & Covariance | foundation | planned | brief |
| Sampling, Measurement & Experimental Design | foundation | planned | brief |
| Ordinary Differential Equations & Linear Systems | intermediate | planned | brief |
| Complex Numbers, Fourier & Laplace Transforms | intermediate | planned | brief |
| Conditioning, Stability & Numerical Analysis | intermediate | planned | brief |
| Decision Theory, Risk & Cost-Sensitive Decisions | intermediate | planned | brief |
| Real Analysis, Sequences & Modes of Convergence | intermediate | planned | brief |
| Abstract Algebra, Groups & Symmetry Actions | intermediate | planned | brief |
| Partial Differential Equations, Conservation & Boundary Conditions | intermediate | planned | brief |
| Numerical PDEs: Grids, Finite Elements & Stability | advanced | planned | brief |

### Classical Machine Learning

**Module anchor:** A small tabular prediction problem with a baseline, held-out evaluation and inspectable errors.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Linear & Logistic Regression | foundation | published | design needed |
| Decision Trees & Random Forests | foundation | published | design needed |
| K-Nearest Neighbors (KNN) | foundation | published | design needed |
| Gradient Boosted Trees (XGBoost, LightGBM, CatBoost) | intermediate | published | design needed |
| Support Vector Machines (SVM) | intermediate | published | design needed |
| Naive Bayes & Probabilistic Classifiers | intermediate | published | design needed |
| Ensemble Methods & Stacking | intermediate | published | design needed |
| Recommender Systems (Collaborative Filtering, Matrix Factorization) | intermediate | published | design needed |
| Multi-Label & Multi-Output Learning | intermediate | published | design needed |
| Survival Analysis (Cox Regression, Kaplan-Meier, Hazard Models) | advanced | published | design needed |
| K-Means & Hierarchical Clustering | foundation | published | design needed |
| PCA & Dimensionality Reduction | foundation | published | design needed |
| Clustering Evaluation & Validation (Silhouette, ARI, NMI) | foundation | published | design needed |
| DBSCAN & Density-Based Clustering | intermediate | published | design needed |
| Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF) | intermediate | published | design needed |
| Gaussian Mixture Models (GMM) & EM Algorithm | intermediate | published | design needed |
| t-SNE, UMAP & Manifold Learning | intermediate | published | design needed |
| Independent Component Analysis (ICA) | advanced | published | design needed |
| Non-Negative Matrix Factorization (NMF) | advanced | published | design needed |
| Feature Scaling, Encoding & Imputation | foundation | published | design needed |
| Cross-Validation & Hyperparameter Tuning | foundation | published | design needed |
| Regularization (L1, L2, Elastic Net, Dropout) | intermediate | published | design needed |
| Feature Selection & Importance (SHAP, Permutation, Mutual Info) | intermediate | published | design needed |
| Bias-Variance Tradeoff & Learning Curves | intermediate | published | design needed |
| Imbalanced Learning (SMOTE, Cost-Sensitive Learning) | intermediate | published | design needed |
| AutoML & Neural Architecture Search (NAS) | advanced | published | design needed |
| Hidden Markov Models (HMM) | intermediate | published | design needed |
| Bayesian Networks & Causal Graphical Models | intermediate | published | design needed |
| Conditional Random Fields (CRF) | advanced | published | design needed |
| Gaussian Processes (GP) | advanced | published | design needed |
| Semi-Supervised Learning (Label Propagation, Self-Training, Co-Training) | intermediate | published | design needed |
| Active Learning | advanced | published | design needed |
| Evaluation Metrics (Precision, Recall, F1, AUC-ROC, AP, R², MAE) | foundation | published | design needed |
| PAC Learning & VC Dimension | intermediate | published | design needed |
| Calibration & Conformal Prediction | intermediate | published | design needed |
| Rademacher Complexity & Generalization Bounds | advanced | published | design needed |
| ML Problem Formulation, Baselines & Data Leakage | foundation | planned | brief |
| Time-Series Validation & Forecasting Baselines | intermediate | planned | brief |
| End-to-End Supervised Learning & Error Analysis | intermediate | planned | brief |

### Deep Learning Fundamentals & Architectures

**Module anchor:** One example and one parameter update before scaling to a trained neural model.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Perceptrons, Neurons & Activation Functions | foundation | published | design needed |
| Backpropagation & Automatic Differentiation | foundation | published | design needed |
| Loss Functions (CE, MSE, Focal, Contrastive, Triplet) | foundation | published | design needed |
| Batch/Layer/Group/RMS Normalization | foundation | published | design needed |
| Transfer Learning & Fine-Tuning Strategies | foundation | published | design needed |
| Weight Initialization (Xavier, Kaiming, μP) | intermediate | published | design needed |
| Residual Connections & Skip Connections | intermediate | published | design needed |
| Dropout, DropPath & Stochastic Depth | intermediate | published | design needed |
| Convolution, Pooling & Receptive Fields | foundation | published | design needed |
| Landmark Architectures (LeNet → AlexNet → VGG → ResNet → EfficientNet) | intermediate | published | design needed |
| Depthwise Separable & Dilated Convolutions | intermediate | published | design needed |
| ConvNeXt & Modern CNN Designs | advanced | published | design needed |
| Capsule Networks | advanced | published | design needed |
| RNNs, LSTMs & GRUs | foundation | published | design needed |
| Sequence-to-Sequence & Encoder-Decoder | intermediate | published | design needed |
| Attention Mechanism (Bahdanau, Luong) | intermediate | published | design needed |
| Long-Context Sequence Models (Transformer-XL, Griffin, Perceiver) | advanced | planned | design needed |
| State Space Models (S4, Mamba, Mamba-2) | advanced | published | design needed |
| RWKV & Linear Attention Models | frontier | published | design needed |
| Self-Attention & Multi-Head Attention | foundation | published | design needed |
| Transformer Block Architecture | foundation | published | design needed |
| Positional Encodings (Sinusoidal, Learned, RoPE, ALiBi) | intermediate | published | design needed |
| Grouped-Query Attention (GQA) & Multi-Query Attention (MQA) | intermediate | published | design needed |
| Multi-Head Latent Attention (MLA) | advanced | published | design needed |
| Sparse & Linear Attention Variants | advanced | published | design needed |
| Vision Transformers (ViT, DeiT, Swin, DiNOv2) | advanced | published | design needed |
| Mixture-of-Experts Transformers (MoE) | frontier | published | design needed |
| Interleaved / Cross-Attention Architectures | frontier | published | design needed |
| Message Passing & Graph Convolutions (GCN, GAT, GraphSAGE) | intermediate | published | design needed |
| Graph Transformers & Geometric Deep Learning | advanced | published | design needed |
| Boltzmann Machines & Restricted Boltzmann Machines (RBM) | intermediate | published | design needed |
| Spectral Normalization & Gradient Penalty | intermediate | published | design needed |
| Modern Hopfield Networks | advanced | published | design needed |
| xLSTM (Extended LSTM) | advanced | published | design needed |
| Hyena & Long Convolution Models | advanced | published | design needed |
| Ring Attention & Sequence Parallelism | advanced | published | design needed |
| Advanced Optimizers (Lion, Sophia, Prodigy, Schedule-Free) | advanced | published | design needed |
| Neural ODE & Continuous-Depth Models | advanced | published | design needed |
| Hybrid SSM-Transformer Architectures (Jamba) | frontier | published | design needed |
| Titans (Multi-Memory Architecture) | frontier | published | design needed |
| Mini-Batches, Training Loops & Gradient Accumulation | foundation | planned | brief |
| Neural Training Diagnostics & Reproducible Experiments | intermediate | planned | brief |

### Large Language Models — Architecture, Training & Inference

**Module anchor:** One request from tokens and context through computation, output, evidence and serving behavior.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Byte-Pair Encoding (BPE), WordPiece, SentencePiece, Unigram | foundation | published | design needed |
| Byte-Level Tokenization & Token-Free Models | intermediate | published | design needed |
| Vocabulary Design & Multilingual Tokenization | intermediate | published | design needed |
| Multimodal Tokenization (Visual, Audio, Video) | advanced | published | design needed |
| Dynamic Tokenization (ADAT, BoundlessBPE, LiteToken) | advanced | published | design needed |
| Causal Language Modeling (Next-Token Prediction) | foundation | published | design needed |
| Masked Language Modeling (BERT-style) | foundation | published | design needed |
| Data Curation & Deduplication (MinHash, Bloom Filters) | intermediate | published | design needed |
| Scaling Laws (Kaplan, Chinchilla, Beyond) | intermediate | published | design needed |
| Curriculum Learning & Data Mixing Strategies | advanced | published | design needed |
| FP8 Training & Low-Precision Pre-Training | advanced | published | design needed |
| MoE Training & Expert Load Balancing | advanced | published | design needed |
| Multimodal Pre-Training (Vision Encoders, Cross-Modal Alignment) | advanced | published | design needed |
| Data Curation Pipelines (Curator Models, Quality Filtering) | advanced | published | design needed |
| Synthetic Data Generation for Pre-Training | frontier | published | design needed |
| Supervised Fine-Tuning (SFT) | intermediate | published | design needed |
| RLHF (Reinforcement Learning from Human Feedback) | intermediate | published | design needed |
| DPO (Direct Preference Optimization) | intermediate | published | design needed |
| SimPO (Simple Preference Optimization) | intermediate | published | design needed |
| P-Tuning & Soft Prompt Methods | intermediate | published | design needed |
| GRPO, RLOO, KTO & Advanced Preference Methods | advanced | published | design needed |
| Constitutional AI (CAI) | advanced | published | design needed |
| Process Reward Models (PRM) vs Outcome Reward Models (ORM) | advanced | published | design needed |
| RLAIF, IPO, ORPO & Emerging Alignment Methods | advanced | published | design needed |
| RLVR (Reinforcement Learning with Verifiable Rewards) | advanced | published | design needed |
| DAPO (Dynamic Adaptive Policy Optimization) | advanced | published | design needed |
| Knowledge Distillation for LLMs (DeepSeek-R1-Distill, CoT Distillation) | advanced | published | design needed |
| RL for Reasoning (DeepSeek-R1 Style) | frontier | published | design needed |
| KV-Cache & Memory Management | intermediate | published | design needed |
| Decoding Strategies (Greedy, Beam, Top-k, Top-p, Temperature) | intermediate | published | design needed |
| Structured Output & Constrained Decoding (Outlines, XGrammar) | intermediate | published | design needed |
| Continuous Batching & PagedAttention | advanced | published | design needed |
| Queueing Theory for LLM Serving | advanced | published | design needed |
| Speculative Decoding | advanced | published | design needed |
| Prefix Caching & Prompt Caching | advanced | published | design needed |
| Inference Cost Economics & Compute Scaling | advanced | published | design needed |
| Test-Time Compute Scaling | frontier | published | design needed |
| Inference Engines & Serving | frontier | published | design needed |
| Inference System Architecture (End-to-End) | intermediate | published | design needed |
| Request Routing & Load Balancing | intermediate | published | design needed |
| Autoscaling & GPU Resource Management | intermediate | published | design needed |
| Disaggregated Prefill & Decode | advanced | published | design needed |
| Caching Strategies (Semantic, Exact, KV-Cache Sharing) | advanced | published | design needed |
| Multi-Model Serving & Model Routing | advanced | published | design needed |
| Rate Limiting, Quota Management & Fairness | advanced | published | design needed |
| Guardrails, Input/Output Filtering & Safety Layers | advanced | published | design needed |
| Observability & LLM Monitoring | advanced | published | design needed |
| Streaming & Server-Sent Events (SSE) | advanced | published | design needed |
| Cost Optimization & TCO Analysis | advanced | published | design needed |
| Edge & On-Premise Deployment Architectures | advanced | published | design needed |
| Multi-Region & Global Inference Infrastructure | frontier | published | design needed |
| Context Window Extension (RoPE Scaling, YaRN, NTK-Aware) | intermediate | published | design needed |
| Retrieval-Augmented Generation (RAG) | intermediate | published | design needed |
| Embedding Models & Vector Databases | advanced | published | design needed |
| GraphRAG & Agentic RAG | advanced | published | design needed |
| Model Merging (TIES, DARE, Model Soups, SLERP) | advanced | published | design needed |
| Hybrid Search (Dense + Sparse + Reranking) | advanced | published | design needed |
| Language-Model Batches, Attention Masks & Loss Alignment | foundation | planned | brief |
| Pretraining Corpus Provenance, Rights & Dataset Governance | intermediate | planned | brief |
| Train, Evaluate & Document a Small Language Model | intermediate | planned | brief |
| RAG Corpus Ingestion, Citation Lineage & Freshness | intermediate | planned | brief |

### Reinforcement Learning

**Module anchor:** A small state/action environment with visible reward, return, exploration and termination.

**Teaching strategy:** Specify task, environment and observable success → Introduce frames, units, state, observations and actions → Trace sensing through estimation and decision to actuation → Explain feedback, delay, noise and constraints → Compare trajectories and diagnose a failure → Validate in changed environments and bound physical claims.

**Practice:** Predict response to a change, repair a frame/delay error, reproduce a bounded simulation and evaluate transfer.

**Verification:** Record seeds, timestep, frames, units, actuator limits and termination rules; distinguish simulation from physical evidence.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| MDPs, Bellman Equations & Dynamic Programming | foundation | planned | design needed |
| Value Iteration & Policy Iteration | foundation | planned | design needed |
| Monte Carlo & Temporal Difference (TD) Methods | foundation | planned | design needed |
| On-Policy, Off-Policy & Importance Sampling | foundation | planned | design needed |
| Q-Learning & SARSA | foundation | planned | design needed |
| Exploration vs Exploitation (ε-greedy, UCB, Thompson Sampling) | foundation | planned | design needed |
| DQN & Rainbow Extensions | intermediate | planned | design needed |
| Policy Gradient Methods (REINFORCE, A2C) | intermediate | planned | design needed |
| Policy Gradients, Baselines & Generalized Advantage Estimation (GAE) | intermediate | planned | design needed |
| PPO (Proximal Policy Optimization) | intermediate | planned | design needed |
| TRPO (Trust Region Policy Optimization) | intermediate | planned | design needed |
| Representation Learning for RL (CURL, DrQ, Data Augmentation) | intermediate | planned | design needed |
| A3C, IMPALA & Scalable Distributed RL | advanced | planned | design needed |
| R2D2, Agent57 & Atari-Scale RL | advanced | planned | design needed |
| SAC (Soft Actor-Critic) & Maximum Entropy RL | advanced | planned | design needed |
| DDPG, TD3 & Continuous Action Spaces | advanced | planned | design needed |
| Distributional RL (C51, QR-DQN, IQN) | advanced | planned | design needed |
| Model-Based RL (Dreamer, MuZero, World Models) | advanced | planned | design needed |
| Landmark RL Systems (AlphaGo, AlphaZero & MuZero) | advanced | planned | design needed |
| Hierarchical RL (Options, Goal-Conditioned, HAM) | advanced | planned | design needed |
| Multi-Agent RL (MARL) | advanced | planned | design needed |
| Offline RL & Conservative Q-Learning (CQL) | advanced | planned | design needed |
| Decision Transformer & Sequence Modeling for RL | advanced | planned | design needed |
| IQL (Implicit Q-Learning) | advanced | planned | design needed |
| Inverse RL & Imitation Learning (GAIL, DAgger, BC) | advanced | planned | design needed |
| Reward Shaping & Intrinsic Motivation (Curiosity, RND) | advanced | planned | design needed |
| Sim-to-Real Transfer & Domain Randomization | advanced | planned | design needed |
| Safe RL & Constrained MDPs | advanced | planned | design needed |
| Reward Modeling & RLHF for LLMs | frontier | planned | design needed |
| Self-Play & Population-Based Training | frontier | planned | design needed |
| Simulation Environments | foundation | planned | design needed |
| RL Environment Contracts, Termination & Evaluation | foundation | planned | brief |
| Partial Observability, Belief States & Recurrent Policies | intermediate | planned | brief |

### Generative Models

**Module anchor:** A tiny data distribution and comparable samples under explicit training and sampling rules.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Generative Adversarial Networks (GAN Fundamentals) | intermediate | planned | design needed |
| GAN Variants (DCGAN, WGAN, StyleGAN, CycleGAN, Pix2Pix) | intermediate | planned | design needed |
| Mode Collapse & Training Stability | advanced | planned | design needed |
| VAE & ELBO | intermediate | planned | design needed |
| VQ-VAE & Discrete Latent Spaces | advanced | planned | design needed |
| Hierarchical & Conditional VAEs | advanced | planned | design needed |
| Denoising Diffusion Probabilistic Models (DDPM) | intermediate | planned | design needed |
| Diffusion Forward/Reverse Processes & DDIM Sampling | intermediate | planned | design needed |
| Score-Based Models & SDEs | intermediate | planned | design needed |
| Latent Diffusion & Stable Diffusion | advanced | planned | design needed |
| Classifier-Free Guidance (CFG) | advanced | planned | design needed |
| Rectified Flow & Flow Matching | advanced | planned | design needed |
| Consistency Models & Distillation | advanced | planned | design needed |
| Controllable Generation (ControlNet, IP-Adapter, T2I-Adapter) | advanced | planned | design needed |
| DiT (Diffusion Transformers) | frontier | planned | design needed |
| Evaluation Metrics for Generative Models (FID, IS, CLIP Score) | intermediate | planned | design needed |
| Image & Video Editing with Generative Models (Inpainting, InstructPix2Pix) | intermediate | planned | design needed |
| Normalizing Flows (RealNVP, Glow, Neural ODE) | advanced | planned | design needed |
| Autoregressive Image Models (PixelCNN, ImageGPT, VAR) | advanced | planned | design needed |
| Energy-Based Models (EBMs) | advanced | planned | design needed |
| 3D Generation (Text-to-3D, Score Distillation Sampling) | frontier | planned | design needed |
| Video Generation Architectures & Training | frontier | planned | design needed |
| Generative Modelling: Density, Likelihood & Sampling | foundation | planned | brief |
| Diffusion Sampler Error, Schedules & Numerical Solvers | intermediate | planned | brief |

### NLP, Computer Vision & Multimodal AI

**Module anchor:** Trace a labeled text, image or audio sample through representation and task-specific evaluation.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Text Classification, NER & Sequence Labeling | foundation | planned | design needed |
| Text Preprocessing & Cleaning (Tokenization, Normalization, Deduplication) | foundation | planned | design needed |
| Machine Translation & Cross-Lingual Transfer | intermediate | planned | design needed |
| Question Answering & Reading Comprehension | intermediate | planned | design needed |
| Text Summarization (Extractive & Abstractive) | intermediate | planned | design needed |
| Multilingual & Cross-Lingual NLP (XLM-R, mBERT, Low-Resource Transfer) | intermediate | planned | design needed |
| Entity Linking & Coreference Resolution Pipelines | intermediate | planned | design needed |
| Relation Extraction & Open Information Extraction | intermediate | planned | design needed |
| Speech Recognition (ASR) & Text-to-Speech (TTS) | advanced | planned | design needed |
| Information Extraction & Knowledge Graphs | advanced | planned | design needed |
| Structured Prediction & Document AI (LayoutLM, Donut) | advanced | planned | design needed |
| Word Embeddings (Word2Vec, GloVe, FastText) | foundation | planned | design needed |
| Sentence Transformers & Dense Retrieval (E5, GTE, BGE) | intermediate | planned | design needed |
| Contextual Embeddings (ELMo, BERT Variants) | intermediate | planned | design needed |
| Syntactic Parsing & Coreference Resolution | intermediate | planned | design needed |
| Dialogue Systems & Conversational AI | intermediate | planned | design needed |
| Image Classification | foundation | planned | design needed |
| Camera Geometry, Calibration & Stereo Vision | foundation | planned | design needed |
| Object Detection (YOLO, DETR, RT-DETR) | intermediate | planned | design needed |
| Semantic, Instance & Panoptic Segmentation | intermediate | planned | design needed |
| Pose Estimation & Action Recognition | intermediate | planned | design needed |
| 3D Vision (NeRF, 3D Gaussian Splatting, Depth Estimation) | advanced | planned | design needed |
| Visual SLAM & Visual Odometry | advanced | planned | design needed |
| Depth Estimation & Monocular 3D Understanding | advanced | planned | design needed |
| Self-Supervised Visual Learning (MAE, DINO, SimCLR, CLIP) | advanced | planned | design needed |
| Optical Flow & Video Understanding | advanced | planned | design needed |
| OCR & Document Understanding | advanced | planned | design needed |
| Super-Resolution & Image Restoration | advanced | planned | design needed |
| Medical Imaging (Segmentation, Detection, Classification) | advanced | planned | design needed |
| Foundation Models for Segmentation (SAM, SAM 2) | advanced | planned | design needed |
| Synthetic Data Generation for Training | advanced | planned | design needed |
| Vision-Language Models (CLIP, SigLIP, BLIP-2) | intermediate | planned | design needed |
| Multimodal LLMs (GPT-4V, Gemini, LLaVA, Qwen-VL) | advanced | planned | design needed |
| Text-to-Image Generation (DALL-E, Stable Diffusion, Flux, Midjourney) | advanced | planned | design needed |
| Video Generation (Sora, Runway Gen-3, Kling, Wan) | frontier | planned | design needed |
| Audio/Music Generation (AudioLM, MusicGen, Udio, Suno) | frontier | planned | design needed |
| Early Fusion Multimodality (Llama 4) | frontier | planned | design needed |
| Any-to-Any Models (Unified Multimodal Generation) | frontier | planned | design needed |
| Speaker Diarization & Verification | intermediate | planned | design needed |
| Speech Emotion Recognition & Paralinguistics | intermediate | planned | design needed |
| Audio Classification & Sound Event Detection | intermediate | planned | design needed |
| Music Information Retrieval (Beat Tracking, Chord Recognition, Genre) | intermediate | planned | design needed |
| Voice Cloning & Speaker Adaptation (VALL-E, OpenVoice, XTTS) | advanced | planned | design needed |
| Source Separation & Audio Denoising (Demucs, Band-Split RNN) | advanced | planned | design needed |
| Audio Synthesis Fundamentals (Vocoders, WaveNet, Neural Audio Codecs) | advanced | planned | design needed |
| Spatial Audio & 3D Sound Processing | advanced | planned | design needed |
| Digital Images, Sampling, Color & Geometric Transforms | foundation | planned | brief |
| Audio Sampling, Spectrograms & Speech Data Pipelines | foundation | planned | brief |
| Multimodal Dataset Alignment, Missing Modalities & Evaluation | intermediate | planned | brief |

### AI for Quantitative Finance

**Module anchor:** A temporally separated financial experiment with costs, risk, data provenance and leakage checks.

**Teaching strategy:** Ask a precise research question and establish prerequisites → Explain the baseline and proposed mechanism → Reconstruct a small example or experiment → Locate evidence and check assumptions → Attempt a bounded reproduction and comparison → Identify uncertainty, limitations and a concrete next investigation.

**Practice:** Reconstruct a result, evaluate an alternative explanation and produce a bounded reproducibility report.

**Verification:** Use original papers/data/docs; distinguish hypothesis, demonstration, generalization and unsettled claims; date moving information.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Probability Theory for Quant Finance (Martingales, Stopping Times, Random Walks) | foundation | planned | design needed |
| Linear Algebra for Finance (Covariance Matrices, PCA, Factor Decomposition) | foundation | planned | design needed |
| Python for Quantitative Research (NumPy, Pandas, Vectorized Backtesting) | foundation | planned | design needed |
| Econometrics (Cointegration, Granger Causality, VECM, Unit Roots) | intermediate | planned | design needed |
| Convex Optimization & Dynamic Programming for Finance (CVXPY, Bellman Equations) | intermediate | planned | design needed |
| Bayesian Statistics & Inference for Finance (Signal Combination, Hierarchical Models) | intermediate | planned | design needed |
| ARIMA, GARCH & Classical Time-Series | foundation | planned | design needed |
| Temporal Fusion Transformers & Neural Forecasting | intermediate | planned | design needed |
| Classical Forecasting (Prophet, N-BEATS, N-HiTS) | intermediate | planned | design needed |
| Foundation Models for Time-Series (TimesFM, Chronos, Moirai) | advanced | planned | design needed |
| Regime Detection & Hidden Markov Models for Markets | advanced | planned | design needed |
| Portfolio Optimization (Markowitz, Black-Litterman, Risk Parity) | intermediate | planned | design needed |
| Risk Models & Tail Risk (VaR, CVaR, Stress Testing, Extreme Value Theory) | intermediate | planned | design needed |
| Transaction Cost Analysis & Slippage Modeling (Market Impact, Cost-Aware Optimization) | intermediate | planned | design needed |
| Options Pricing & Derivatives (Black-Scholes, Monte Carlo, Neural SDEs) | intermediate | planned | design needed |
| Deep Hedging & RL for Trading | advanced | planned | design needed |
| Feature Engineering & Alpha Research Methodology (IC, IR, Decay, Turnover) | intermediate | planned | design needed |
| Statistical Arbitrage & Pairs Trading (Cointegration, Ornstein-Uhlenbeck) | intermediate | planned | design needed |
| Sentiment Analysis & NLP for Finance (FinBERT) | intermediate | planned | design needed |
| Alpha Signal Generation & Factor Models | advanced | planned | design needed |
| Alternative Data Sources (Satellite, Web Traffic, Social) | advanced | planned | design needed |
| Graph Neural Networks for Financial Markets (Sector Rotation, Systemic Risk) | advanced | planned | design needed |
| LLMs for Financial Analysis | frontier | planned | design needed |
| Market Microstructure & Order Book Modeling | advanced | planned | design needed |
| Execution Algorithms & Optimal Execution (TWAP, VWAP, Almgren-Chriss) | advanced | planned | design needed |
| Market-Making & Liquidity Provision (Avellaneda-Stoikov, Inventory Management) | advanced | planned | design needed |
| High-Frequency Trading & Low-Latency Infrastructure | advanced | planned | design needed |
| Credit Scoring & Fraud Detection | intermediate | planned | design needed |
| Stochastic Calculus for Finance (Itô Calculus, SDEs) | intermediate | planned | design needed |
| Explainable AI & Model Governance in Finance (SHAP, SR 11-7) | intermediate | planned | design needed |
| Backtesting Frameworks & Avoiding Overfitting | advanced | planned | design needed |
| Crypto & DeFi Quantitative Strategies (AMM, MEV, On-Chain Analytics) | advanced | planned | design needed |
| Multi-Agent RL & Market Simulation | frontier | planned | design needed |
| Financial Data Pitfalls (Survivorship Bias, Look-Ahead Bias, Point-in-Time Data) | foundation | planned | design needed |
| ML Model Lifecycle in Production Trading (Drift, Monitoring, Feature Stores) | intermediate | planned | design needed |
| Low-Latency Systems & C++ for Quant (Lock-Free, FPGA, Co-Location) | advanced | planned | design needed |
| Quant Interview Mathematics (Brainteasers, Expected Value, Game Theory) | foundation | planned | design needed |
| Market Instruments, Returns & Cash-Flow Accounting | foundation | planned | brief |
| Backtest Selection Bias & Execution Reconciliation | intermediate | planned | brief |

### Neural Engineering & Computational Neuroscience

**Module anchor:** A biological signal passes through measurement, analysis and a bounded closed-loop interface.

**Teaching strategy:** Start from a biological question and measurable quantity → Connect biological mechanism to sensor/interface and units → Trace acquisition, reference, sampling, noise and artifacts → Build and validate an analysis, decoder or controller → Interpret behavioral/clinical evidence and uncertainty → Compare conditions and bound translation and safety claims.

**Practice:** Synthetic signals, appropriate open de-identified data, bounded simulations or bench phantoms; independent analysis and failure diagnosis.

**Verification:** Check measurement assumptions, participant/session splits, confounds, nonstationarity, reproducibility and jurisdiction-specific translational evidence.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Neurons, Synapses & Neural Signaling | foundation | planned | brief |
| Plasticity, Sensory Systems & Motor Systems | foundation | planned | brief |
| Internally Generated States & Neural Correlates of Behavior | foundation | planned | brief |
| Hodgkin-Huxley & Leaky Integrate-and-Fire Models | intermediate | planned | brief |
| Brian2, Nengo & Neural Simulation Workflows | intermediate | planned | brief |
| Spike Trains, Local Field Potentials & Neural Population Activity | foundation | planned | brief |
| Dimensionality Reduction & Manifold Analysis for Neural Data | intermediate | planned | brief |
| Neural Population Dynamics & Latent-State Models | advanced | planned | brief |
| Movement, Speech & Sensory-State Decoding | advanced | planned | brief |
| Multi-Dimensional Output Prediction & Closed-Loop Decoding | advanced | planned | brief |
| Human fMRI: Experimental Design, Preprocessing & GLM | intermediate | planned | brief |
| Functional Connectivity & Representational Similarity Analysis | advanced | planned | brief |
| In-Vivo & Awake-Behaving Electrophysiology | advanced | planned | brief |
| Non-Human Primate Electrophysiology | advanced | planned | brief |
| Closed-Loop Brain-Computer Interfaces | frontier | planned | brief |
| Neural Population Coding & Tuning Curves | foundation | planned | brief |
| Spiking Neural Networks (SNNs) | intermediate | planned | brief |
| Biologically Plausible Learning Rules (Feedback Alignment, Equilibrium Propagation) | intermediate | planned | brief |
| Hebbian Learning & Synaptic Plasticity | intermediate | planned | brief |
| Predictive Coding & Free Energy Principle | advanced | planned | brief |
| Computational Models of Memory (Working, Episodic, Semantic) | advanced | planned | brief |
| Neural Decoding & Representation Similarity Analysis (RSA) | advanced | planned | brief |
| Neural Oscillations & Attention in the Brain | advanced | planned | brief |
| Brain-Computer Interfaces (BCI) | advanced | planned | brief |
| Neuromorphic Computing | frontier | planned | brief |
| Whole-Brain Emulation & Connectomics | frontier | planned | brief |
| Neural Engineering: From Biological Questions to Engineered Systems | foundation | planned | brief |
| Functional Neuroanatomy: Brain, Spinal Cord and Peripheral Pathways | foundation | planned | brief |
| Neural Cell Biology: Glia, Myelin, Metabolism and the Blood-Brain Barrier | foundation | planned | brief |
| Physical Units and Circuit Foundations for Neural Engineering | foundation | planned | brief |
| Membrane Biophysics: Ion Gradients, Reversal Potentials and Conductance | intermediate | planned | brief |
| Cable Theory and Multicompartment Neuron Models | intermediate | planned | brief |
| Neural Development, Injury and Repair as Engineering Constraints | intermediate | planned | brief |
| Neurovascular Coupling and Indirect Measures of Neural Activity | intermediate | planned | brief |
| Comparative Neural Systems, Model Organisms and Translation | intermediate | planned | brief |
| Psychophysics and Signal Detection for Neural Engineering | intermediate | planned | brief |
| Signals and Systems for Neuroengineering | foundation | planned | brief |
| Sampling, Aliasing and Quantization in Neural Recordings | foundation | planned | brief |
| Spectral and Time-Frequency Analysis of Neural Signals | intermediate | planned | brief |
| State Estimation and System Identification for Neural Dynamics | advanced | planned | brief |
| Neuroscience Experimental Design, Pseudoreplication and Statistical Power | intermediate | planned | brief |
| Neural Information Measures and Communication Metrics | advanced | planned | brief |
| Electrode-Tissue Interfaces, Impedance and Electrochemistry | intermediate | planned | brief |
| Neural Interface Materials, Mechanics and Foreign-Body Response | advanced | planned | brief |
| Biopotential Amplifiers, Analog Front Ends and ADC Design | intermediate | planned | brief |
| Neural Recording Noise, Referencing, Grounding and Interference | intermediate | planned | brief |
| Embedded Processing, FPGA Pipelines and Real-Time Neural Firmware | advanced | planned | brief |
| Wireless Neural Telemetry, Power Budgets and Thermal Constraints | advanced | planned | brief |
| Neural Device Packaging, Reliability and Design Verification | advanced | planned | brief |
| Chemical Neural Sensing, Biosensors and Targeted Delivery Concepts | advanced | planned | brief |
| Neural Tissue Engineering, Organoids and Biohybrid Interfaces | frontier | planned | brief |
| Intracellular Electrophysiology: Current Clamp, Voltage Clamp and Model Validation | intermediate | planned | brief |
| EEG, Event-Related Potentials and MEG Measurement | intermediate | planned | brief |
| EEG and MEG Forward Models, Source Localization and Uncertainty | advanced | planned | brief |
| Intracranial Recording: ECoG, Depth Electrodes and High-Density Probes | advanced | planned | brief |
| EMG, Peripheral Nerve Recordings and Motor-Unit Signals | intermediate | planned | brief |
| Calcium and Voltage Imaging: Indicators, Optics and Measurement Limits | intermediate | planned | brief |
| fNIRS and Wearable Hemodynamic Neuroimaging | intermediate | planned | brief |
| Structural MRI, Diffusion Imaging and Multimodal Neural Measurement | advanced | planned | brief |
| Optogenetics, Chemogenetics and Cell-Type-Specific Perturbation | advanced | planned | brief |
| Neural Data Synchronization, Event Timing and Lab Streaming Layer | intermediate | planned | brief |
| Neural Preprocessing, Artifact Rejection and Leakage-Safe Pipelines | intermediate | planned | brief |
| Spike Sorting, Unit Quality, Drift and Ground-Truth Validation | advanced | planned | brief |
| Calcium Imaging Pipelines: Motion, Segmentation, Neuropil and Deconvolution | advanced | planned | brief |
| NWB, BIDS and Neural Metadata: Interoperable Data by Design | intermediate | planned | brief |
| Open Neural Data, Reproducibility and Dataset Provenance | intermediate | planned | brief |
| Behavioral Tracking, Stimulus Annotation and Neural-Behavior Alignment | intermediate | planned | brief |
| Neural Encoding Models: GLMs, Point Processes and Receptive Fields | intermediate | planned | brief |
| Neural Decoding Evaluation: Subject Splits, Leakage and Operational Metrics | advanced | planned | brief |
| Nonstationarity, Decoder Recalibration and Longitudinal Adaptation | advanced | planned | brief |
| Causal Neural Inference, Confounding and Perturbation Evidence | advanced | planned | brief |
| NeuroAI and Foundation Models for Neural Data | frontier | planned | brief |
| Neuromorphic Neural Interfaces: Event Processing and Deployment Tradeoffs | advanced | planned | brief |
| Electrical Neural Stimulation: Recruitment, Charge and Model Limits | advanced | planned | brief |
| Noninvasive Neuromodulation: TMS, Electrical and Ultrasound Concepts | advanced | planned | brief |
| DBS and Responsive Neuromodulation: Biomarkers and Adaptive Control | advanced | planned | brief |
| Peripheral, Spinal and Functional Electrical Stimulation Systems | advanced | planned | brief |
| Sensory Neuroprostheses: Hearing, Vision and Somatosensory Feedback | advanced | planned | brief |
| Communication BCIs: Speech, Language Priors and User Intent | advanced | planned | brief |
| Closed-Loop BCI Stability, Shared Control and Failure Recovery | advanced | planned | brief |
| Neurofeedback, Rehabilitation Learning and Evidence of Transfer | advanced | planned | brief |
| Clinical Evidence for Neurotechnology: Endpoints, Trials and Generalization | advanced | planned | brief |
| Neurotechnology Risk Management, Regulation and Quality Systems | advanced | planned | brief |
| Neural Data Privacy, Consent, Agency and Long-Term Stewardship | intermediate | planned | brief |
| Accessible Neurotechnology, Participatory Design and Human Factors | intermediate | planned | brief |
| Responsible Animal, Human-Tissue and Preclinical Neural Research | intermediate | planned | brief |
| Neural Device Cybersecurity, Software Lifecycle and Postmarket Support | advanced | planned | brief |
| Neural Engineering Capstone: Auditable EEG Analysis | advanced | planned | brief |
| Neural Engineering Capstone: Longitudinal Decoder and Closed-Loop Simulator | advanced | planned | brief |
| Neural Engineering Capstone: Simulated Acquisition Chain and Verification File | advanced | planned | brief |
| Neural Engineering Capstone: Multimodal Neural Dataset and Model Comparison | advanced | planned | brief |
| Neural Engineering Capstone: Translation and Participatory Design Dossier | advanced | planned | brief |

### Evolutionary & Bio-Inspired Algorithms

**Module anchor:** Compare optimization on an inspectable landscape using equal evaluation budgets, seeds and constraints.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Black-Box Optimization Budgets, Constraints & Benchmarking | foundation | planned | brief |
| Genetic Algorithms (GA) | foundation | planned | design needed |
| Evolution Strategies (ES, CMA-ES, OpenAI-ES) | intermediate | planned | design needed |
| Differential Evolution (DE) | intermediate | planned | design needed |
| Particle Swarm Optimization (PSO) & Swarm Intelligence | intermediate | planned | design needed |
| Ant Colony Optimization (ACO) | intermediate | planned | design needed |
| Estimation of Distribution Algorithms (EDA, UMDA, BOA) | intermediate | planned | design needed |
| Neuroevolution (NEAT, HyperNEAT, Weight Agnostic Networks) | advanced | planned | design needed |
| Quality-Diversity (MAP-Elites, CMA-ME) | advanced | planned | design needed |
| Genetic Programming & Symbolic Regression | advanced | planned | design needed |
| Multi-Objective Evolutionary Optimization (NSGA-II, MOEA/D) | advanced | planned | design needed |
| Coevolution & Competitive Dynamics | advanced | planned | design needed |
| Evolutionary Combinatorial Optimization (Scheduling, Routing, Packing) | advanced | planned | design needed |
| Memetic Algorithms (Hybrid EA + Local Search) | advanced | planned | design needed |
| POET & Open-Ended Evolution | advanced | planned | design needed |
| LLM-Guided Evolution & FunSearch | frontier | planned | design needed |
| Artificial Life & Open-Ended Search | frontier | planned | design needed |

### GPU Engineering, CUDA & Large-Scale Systems

**Module anchor:** A correct CPU operation becomes a verified and profiled kernel, then a reliable multi-device workflow.

**Teaching strategy:** Define a computation or service contract → Map components, state and resource boundaries → Trace the normal request/work/data flow → Explain coordination and the relevant failure mode → Measure correctness and a diagnosed bottleneck → Repair, compare and validate under a changed workload.

**Practice:** Independent reference computation, fault diagnosis, measured optimization and a reproducible engineering report.

**Verification:** Separate output correctness, memory/concurrency safety, numerical tolerance, performance and platform support.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| C & C++ Foundations for GPU Programming | foundation | planned | brief |
| CMake, Linking & Reproducible Native Builds | foundation | planned | brief |
| Parallel Algorithm Design: Work, Span & Decomposition | foundation | planned | brief |
| GPU Toolchains, Drivers & Compatibility | foundation | planned | brief |
| CPU–GPU Work Partitioning & Acceleration Economics | foundation | planned | brief |
| Floating-Point Representation & Numerical Error | foundation | planned | brief |
| CPU Architecture (Cores, Caches, SIMD, Pipelining) | foundation | planned | brief |
| GPU Architecture (SMs, Warps, Threads, Memory Hierarchy) | foundation | planned | brief |
| Memory Hierarchy (HBM, L2, SRAM, Registers, Bandwidth) | intermediate | planned | brief |
| Tensor Cores & Specialized AI Hardware | intermediate | planned | brief |
| Interconnects & Network Topology (NVLink, InfiniBand, NVSwitch) | advanced | planned | brief |
| CUDA Programming Model | intermediate | planned | brief |
| Triton (GPU Kernel DSL) | intermediate | planned | brief |
| CUDA Libraries (cuBLAS, cuDNN, cuFFT, NCCL, Thrust, CUB) | intermediate | planned | brief |
| Memory Coalescing, Bank Conflicts & Shared Memory Tiling | advanced | planned | brief |
| Warp-Level Primitives & Cooperative Groups | advanced | planned | brief |
| Custom CUDA Kernel Writing for ML Ops | advanced | planned | brief |
| Flash Attention Implementation | advanced | planned | brief |
| Operator Fusion & Torch Compile | advanced | planned | brief |
| Ring-AllReduce & Communication Primitives | advanced | planned | brief |
| Compiler & Graph Optimization (XLA, TorchInductor, MLIR) | advanced | planned | brief |
| Data Parallelism (DDP) | intermediate | planned | brief |
| FSDP (Fully Sharded Data Parallelism) | advanced | planned | brief |
| Tensor Parallelism (TP) | advanced | planned | brief |
| Pipeline Parallelism (PP) | advanced | planned | brief |
| Expert Parallelism & All-to-All Communication | advanced | planned | brief |
| 3D Parallelism & Hybrid Strategies | advanced | planned | brief |
| Gradient Checkpointing (Activation Recomputation) | advanced | planned | brief |
| ZeRO Optimization Stages (1, 2, 3) | advanced | planned | brief |
| Networking & Communication Optimization (RDMA, GPUDirect, SHARP) | advanced | planned | brief |
| GPU Profiling & Roofline Analysis | intermediate | planned | brief |
| FLOP Utilization (MFU) & Training Efficiency | advanced | planned | brief |
| Google TPUs (Tensor Processing Units) | advanced | planned | brief |
| Cerebras Wafer-Scale Engine | advanced | planned | brief |
| Groq LPU (Language Processing Unit) | advanced | planned | brief |
| Other AI Accelerators (Graphcore IPU, Intel Gaudi, Apple Neural Engine) | advanced | planned | brief |
| AMD ROCm & MI300/MI350 Ecosystem | advanced | planned | brief |
| Power, Cooling & Energy Constraints for AI Clusters | advanced | planned | brief |
| GPU Indexing, Shapes, Strides & Boundary Handling | intermediate | planned | brief |
| GPU Atomics, Memory Ordering & Visibility | intermediate | planned | brief |
| GPU Synchronization, Barriers & Deadlock | intermediate | planned | brief |
| CUDA Streams, Events & Asynchronous Execution | intermediate | planned | brief |
| Host–Device Transfers, Pinned Memory & Buffer Lifetimes | intermediate | planned | brief |
| Unified Memory, Address Spaces & Oversubscription | intermediate | planned | brief |
| GPU Debugging & Compute Sanitizer | intermediate | planned | brief |
| Reliable GPU Benchmarking & Experimental Design | intermediate | planned | brief |
| Nsight Systems: End-to-End Timeline Diagnosis | intermediate | planned | brief |
| Nsight Compute: Counters, Stalls & Kernel Diagnosis | intermediate | planned | brief |
| Occupancy, Register Pressure & Latency Hiding | advanced | planned | brief |
| GPU Cache Behavior, Data Layout & Locality | advanced | planned | brief |
| Parallel Reductions & Stable Softmax | intermediate | planned | brief |
| Parallel Scan, Compaction & Segmented Operations | intermediate | planned | brief |
| Histograms, Scatter–Gather & Atomic Contention | intermediate | planned | brief |
| GEMM from Scalar Loops to Hierarchical Tiling | intermediate | planned | brief |
| GPU Stencils, Convolution & Boundary Conditions | intermediate | planned | brief |
| Sparse GPU Formats & Sparse Matrix Kernels | advanced | planned | brief |
| GPU Sorting, Selection & Top-k | advanced | planned | brief |
| Irregular GPU Algorithms & Graph Traversal | advanced | planned | brief |
| GPU Random Number Generation & Monte Carlo | intermediate | planned | brief |
| Low-Precision GPU Arithmetic & Error Budgets | advanced | planned | brief |
| GPU FFTs, Spectral Methods & Signal Pipelines | intermediate | planned | brief |
| GPU Linear Solvers & Iterative Refinement | advanced | planned | brief |
| Tensor-Core Matrix Instructions & Fragment Layouts | advanced | planned | brief |
| Asynchronous Copy, TMA & Pipeline Staging | advanced | planned | brief |
| Thread-Block Clusters & Distributed Shared Memory | advanced | planned | brief |
| Persistent Kernels, Warp Specialization & Work Queues | advanced | planned | brief |
| CUTLASS & CuTe Layout Algebra | advanced | planned | brief |
| Advanced Triton: Layouts, Pipelines & Persistent Kernels | advanced | planned | brief |
| GPU Kernel Autotuning & Dispatch Design | advanced | planned | brief |
| PTX, SASS & GPU Instruction Analysis | advanced | planned | brief |
| CUDA Runtime, Driver API & Device Code Linking | advanced | planned | brief |
| Framework GPU Operators, Autograd & Interoperability | advanced | planned | brief |
| GPU Memory Pools, Virtual Memory & Allocation Strategy | advanced | planned | brief |
| CUDA Graphs, Capture & Dynamic Workloads | advanced | planned | brief |
| Multi-GPU Topology, Peer Access & NUMA Placement | advanced | planned | brief |
| NCCL Collectives: Correctness, Tuning & Diagnosis | advanced | planned | brief |
| Communication–Compute Overlap & Distributed Critical Paths | advanced | planned | brief |
| CUDA-Aware MPI, NVSHMEM & GPU-Initiated Communication | advanced | planned | brief |
| Distributed GPU Checkpointing & Failure Recovery | advanced | planned | brief |
| GPU Containers, Packaging & Deployment Compatibility | intermediate | planned | brief |
| GPU Scheduling, Slurm, Kubernetes, MIG & MPS | advanced | planned | brief |
| GPU Health, ECC, Xid Errors & Reliability Diagnosis | advanced | planned | brief |
| GPU Correctness CI & Performance Regression Testing | advanced | planned | brief |
| GPU Capacity Planning, Energy Measurement & Cost | advanced | planned | brief |
| GPU Multi-Tenancy, Isolation & Secure Operations | advanced | planned | brief |
| GPU Input Pipelines, Storage & Data-Movement Bottlenecks | intermediate | planned | brief |
| Portable GPU Kernels with HIP & ROCm | intermediate | planned | brief |
| AMD GPU Profiling & Architecture-Aware Tuning | advanced | planned | brief |
| SYCL, OpenMP Offload & OpenACC | intermediate | planned | brief |
| WebGPU Compute & WGSL | intermediate | planned | brief |
| Graphics Pipelines, Compute Shaders & GPU Interoperability | intermediate | planned | brief |
| GPU Scientific Simulation & Domain Decomposition | advanced | planned | brief |
| GPU Architecture Modeling & Simulator Validation | advanced | planned | brief |
| GPU RTL Design, Verification & Hardware Tradeoffs | advanced | planned | brief |
| GPU Engineering Capstone: Verified Kernel Library | advanced | planned | brief |
| GPU Engineering Capstone: Multi-GPU Application | advanced | planned | brief |
| GPU Engineering Capstone: Profiled Inference Engine | advanced | planned | brief |

### Model Optimization & Efficiency

**Module anchor:** Compare the same model/task before and after compression using quality, memory and measured latency.

**Teaching strategy:** Define a computation or service contract → Map components, state and resource boundaries → Trace the normal request/work/data flow → Explain coordination and the relevant failure mode → Measure correctness and a diagnosed bottleneck → Repair, compare and validate under a changed workload.

**Practice:** Independent reference computation, fault diagnosis, measured optimization and a reproducible engineering report.

**Verification:** Separate output correctness, memory/concurrency safety, numerical tolerance, performance and platform support.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Post-Training Quantization (PTQ) | intermediate | planned | design needed |
| Mixed Precision Training (FP16, BF16, TF32) | intermediate | planned | design needed |
| Quantization-Aware Training (QAT) | advanced | planned | design needed |
| Weight-Only vs. Weight-Activation Quantization | advanced | planned | design needed |
| GPTQ, AWQ & Advanced PTQ Methods for LLMs | advanced | planned | design needed |
| Activation-Aware Quantization & Outlier Handling (SmoothQuant, LLM.int8()) | advanced | planned | design needed |
| Sub-4-Bit & 1-Bit Quantization (BitNet, AQLM) | frontier | planned | design needed |
| LoRA (Low-Rank Adaptation) | intermediate | planned | design needed |
| QLoRA | intermediate | planned | design needed |
| Adapters, Prefix Tuning, Prompt Tuning & IA3 | advanced | planned | design needed |
| DoRA, rsLoRA & Advanced LoRA Variants | advanced | planned | design needed |
| Multi-LoRA Serving & LoRA Merging (S-LoRA, Punica) | advanced | planned | design needed |
| Knowledge Distillation | intermediate | planned | design needed |
| Structured & Unstructured Pruning | advanced | planned | design needed |
| Layer Pruning & Depth Reduction | advanced | planned | design needed |
| Neural Architecture Search (NAS) | advanced | planned | design needed |
| Sparse Models & Activation Sparsity (Mixture-of-Depths) | advanced | planned | design needed |
| Token Merging & Reduction (ToMe, Dynamic Token Pruning) | advanced | planned | design needed |
| ONNX & Model Export | intermediate | planned | design needed |
| Edge Inference (TensorRT, CoreML, TFLite) | intermediate | planned | design needed |
| On-Device SLMs (Small Language Models) | advanced | planned | design needed |
| Speculative Decoding Variants (Medusa, EAGLE, Self-Speculative) | advanced | planned | design needed |
| Compression Validation & Deployment Regression Testing | intermediate | planned | brief |

### MLOps & Infrastructure

**Module anchor:** Follow data/model/request versions through delivery, observation, failure and recovery.

**Teaching strategy:** Define a computation or service contract → Map components, state and resource boundaries → Trace the normal request/work/data flow → Explain coordination and the relevant failure mode → Measure correctness and a diagnosed bottleneck → Repair, compare and validate under a changed workload.

**Practice:** Independent reference computation, fault diagnosis, measured optimization and a reproducible engineering report.

**Verification:** Separate output correctness, memory/concurrency safety, numerical tolerance, performance and platform support.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Networking Foundations: Packets, Transport, DNS & Sockets | foundation | planned | brief |
| Experiment Tracking & Logging | foundation | planned | design needed |
| Hyperparameter Optimization | intermediate | planned | design needed |
| Data Versioning & Pipeline Management | intermediate | planned | design needed |
| ML Pipeline Orchestration (Airflow, Prefect, Dagster, Kubeflow) | intermediate | planned | design needed |
| Data Quality & Observability (Great Expectations, Monte Carlo, Soda) | intermediate | planned | design needed |
| Model Serving & API Frameworks | intermediate | planned | design needed |
| Containerization & Orchestration | intermediate | planned | design needed |
| LLMOps (Prompt Versioning, Token Cost Management, LLM Monitoring) | intermediate | planned | design needed |
| Feature Stores | advanced | planned | design needed |
| Model Monitoring & Drift Detection | advanced | planned | design needed |
| CI/CD for ML Pipelines | advanced | planned | design needed |
| A/B Testing & Canary Deployments for ML Models | advanced | planned | design needed |
| GPU Cloud & Training Platforms | intermediate | planned | design needed |
| Hugging Face Ecosystem | intermediate | planned | design needed |
| Cost Management & FinOps for AI (Spot Instances, Reserved Capacity) | intermediate | planned | design needed |
| Task Taxonomy & Model Selection Framework | foundation | planned | design needed |
| Build vs. Fine-Tune vs. Prompt vs. API | intermediate | planned | design needed |
| Model Sizing & Latency-Quality Tradeoffs | intermediate | planned | design needed |
| Designing for Classification Tasks | intermediate | planned | design needed |
| Designing for Conversational AI & Chatbots | intermediate | planned | design needed |
| Designing for Information Extraction (NER, RE, IE) | intermediate | planned | design needed |
| Designing for Search & Retrieval Systems | intermediate | planned | design needed |
| Designing for Code Generation & Analysis | advanced | planned | design needed |
| Designing for Reasoning & Mathematical Tasks | advanced | planned | design needed |
| Designing for Multi-Modal Pipelines | advanced | planned | design needed |
| Designing for Real-Time & Streaming Applications | advanced | planned | design needed |
| Designing for Agentic Workflows | advanced | planned | design needed |
| Microservices vs. Monolithic AI Systems | intermediate | planned | design needed |
| Request-Response vs. Batch vs. Streaming Architectures | intermediate | planned | design needed |
| RAG System Architecture (Production-Grade) | advanced | planned | design needed |
| Model Cascade & Router Architectures | advanced | planned | design needed |
| Human-in-the-Loop System Design | advanced | planned | design needed |
| Fallback & Graceful Degradation | advanced | planned | design needed |
| Data Flywheel & Continuous Learning Architectures | advanced | planned | design needed |
| Evaluation-Driven Development for AI Systems | advanced | planned | design needed |
| Compliance, Audit Trails & Explainability in Production | advanced | planned | design needed |
| AgentOps (Agent Monitoring, Trace Debugging, Multi-Step Evaluation) | advanced | planned | design needed |
| Recommendation System Design (Candidate Generation, Ranking, Re-Ranking) | intermediate | planned | design needed |
| Search Ranking & Learning to Rank (BM25, Neural Ranking, NDCG) | intermediate | planned | design needed |
| Fraud Detection System Design (Real-Time, Feature Stores, Rules + ML) | intermediate | planned | design needed |
| Ads Prediction & Click-Through Rate Modeling | advanced | planned | design needed |
| Content Moderation & Trust & Safety Systems | advanced | planned | design needed |
| Notification & Feed Ranking Systems | advanced | planned | design needed |
| Large-Scale ML System Architecture (Netflix, Uber, Spotify Case Studies) | advanced | planned | design needed |
| Networked Services, HTTP Contracts & Identity Boundaries | foundation | planned | brief |
| Distributed Failure Semantics, Retries & Idempotency | intermediate | planned | brief |
| Data Contracts, Lineage & Training-Serving Consistency | intermediate | planned | brief |

### AI Safety, Alignment & Evaluation

**Module anchor:** Test a precise behavioral/safety claim, compare interventions and state evidence limits.

**Teaching strategy:** State the construct and decision the evidence will support → Define sample, unit, rubric and baseline → Trace an item through annotation and scoring → Estimate uncertainty and examine validity threats → Audit a plausible misleading result → Design a changed evaluation and justify its decision limits.

**Practice:** Repair a misleading metric or rubric, compute a small example, audit leakage/bias and design a discriminating held-out test.

**Verification:** Check construct validity, sampling, dependence, calibration, uncertainty, benchmark contamination, cost and scope of conclusions.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| RLHF, DPO & Preference Learning | intermediate | planned | design needed |
| Constitutional AI & Rule-Based Rewards (RBRM) | advanced | planned | design needed |
| Debate, Recursive Reward Modeling & Iterated Amplification | advanced | planned | design needed |
| Scalable Oversight (Process Supervision, Verification Hierarchies) | advanced | planned | design needed |
| Superalignment & Weak-to-Strong Generalization | frontier | planned | design needed |
| Attention Visualization & Probing | intermediate | planned | design needed |
| Mechanistic Interpretability | advanced | planned | design needed |
| Sparse Autoencoders (SAE) for Feature Discovery | advanced | planned | design needed |
| Circuit Analysis & Causal Tracing | advanced | planned | design needed |
| Logit Bias & Output-Level Steering | advanced | planned | design needed |
| Prompt-Based Steering vs. Activation-Based Steering | advanced | planned | design needed |
| Activation Steering & Steering Vectors | intermediate | planned | design needed |
| Contrastive Activation Addition (CAA) | intermediate | planned | design needed |
| Inference-Time Intervention (ITI) | advanced | planned | design needed |
| Representation Engineering (RepE) | advanced | planned | design needed |
| Classifier-Free Guidance for LLMs (CFG) | advanced | planned | design needed |
| Controlled Decoding (PPLM, GeDi, FUDGE) | advanced | planned | design needed |
| Refusal Direction & Refusal Ablation | advanced | planned | design needed |
| SAE-Based Steering & Feature Clamping | advanced | planned | design needed |
| Function Vectors & Task Vectors in Activation Space | advanced | planned | design needed |
| Task Vectors in Weight Space | advanced | planned | design needed |
| Activation Patching & Causal Interventions | advanced | planned | design needed |
| Persona Steering & Character Control | advanced | planned | design needed |
| Concept Erasure & Unlearning | advanced | planned | design needed |
| Model Editing (ROME, MEMIT, MEND) | advanced | planned | design needed |
| Knowledge Editing & Belief Revision | advanced | planned | design needed |
| Steering for Safety, Honesty & Capability Elicitation | frontier | planned | design needed |
| Steering Composition & Interference | frontier | planned | design needed |
| Red-Teaming & Adversarial Testing | intermediate | planned | design needed |
| Jailbreak Taxonomies & Defense Mechanisms | intermediate | planned | design needed |
| Hallucination Detection & Mitigation | intermediate | planned | design needed |
| Sycophancy & Reward Hacking | advanced | planned | design needed |
| Adversarial Robustness & Certified Defenses | advanced | planned | design needed |
| Fairness, Bias & Responsible AI | advanced | planned | design needed |
| AI Control & Containment (Monitoring, Tripwires, Untrusted Model Protocols) | advanced | planned | design needed |
| Faithful Chain-of-Thought & Reasoning Transparency | advanced | planned | design needed |
| LLM Benchmarks (MMLU, HellaSwag, HumanEval, GSM8K, MATH) | intermediate | planned | design needed |
| LLM-as-Judge & Arena-Style Evaluation | advanced | planned | design needed |
| Contamination Detection & Evaluation Integrity | advanced | planned | design needed |
| Elicitation & Dangerous Capability Evaluation | advanced | planned | design needed |
| Deceptive Alignment & Mesa-Optimization | advanced | planned | design needed |
| Goodhart's Law in RLHF & Reward Overoptimization | advanced | planned | design needed |
| Specification Gaming & Reward Misspecification | advanced | planned | design needed |
| Sleeper Agents & Backdoor Attacks | advanced | planned | design needed |
| Lottery Ticket Hypothesis & Sparse Subnetworks | frontier | planned | design needed |
| AI-Generated Content Watermarking | intermediate | planned | design needed |
| Deepfake Detection | intermediate | planned | design needed |
| Copyright, Training Data Legality & Data Rights | advanced | planned | design needed |
| Environmental Impact of Training | advanced | planned | design needed |
| Existential Risk & AI Governance Frameworks | advanced | planned | design needed |
| Model Welfare & Digital Minds | frontier | planned | design needed |
| AI Threat Modelling, Trust Boundaries & Defense in Depth | foundation | planned | brief |
| Safety Cases, Residual Risk & Deployment Decisions | intermediate | planned | brief |

### Agentic AI & Tool-Using Systems

**Module anchor:** A bounded task with explicit state, tool contracts, permissions, retries, stop conditions and held-out evaluation.

**Teaching strategy:** Define a computation or service contract → Map components, state and resource boundaries → Trace the normal request/work/data flow → Explain coordination and the relevant failure mode → Measure correctness and a diagnosed bottleneck → Repair, compare and validate under a changed workload.

**Practice:** Independent reference computation, fault diagnosis, measured optimization and a reproducible engineering report.

**Verification:** Separate output correctness, memory/concurrency safety, numerical tolerance, performance and platform support.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Prompt Design Fundamentals (Zero/Few-Shot, System Prompts) | foundation | planned | design needed |
| Chain-of-Thought (CoT) & Step-by-Step Reasoning | intermediate | planned | design needed |
| Self-Consistency & Majority Voting | intermediate | planned | design needed |
| Tree-of-Thought, Graph-of-Thought & Search-Based Prompting | advanced | planned | design needed |
| Prompt Optimization & DSPy | advanced | planned | design needed |
| Function Calling & Tool Use | intermediate | planned | design needed |
| ReAct (Reasoning + Acting) | intermediate | planned | design needed |
| Plan-and-Execute Workflows & Task Decomposition | intermediate | planned | design needed |
| Reflection, Self-Critique & Recovery Loops | intermediate | planned | design needed |
| Agentic Frameworks | intermediate | planned | design needed |
| Agent Orchestration Patterns (Sequential, Parallel, Hierarchical, Handoff) | intermediate | planned | design needed |
| Multi-Agent Systems & Agent Communication | advanced | planned | design needed |
| Retrieval & Context Management for Agents | advanced | planned | design needed |
| Human-Agent Interaction Design (Approval Flows, Escalation) | advanced | planned | design needed |
| Agent Memory Architectures | advanced | planned | design needed |
| Code Generation Agents & Computer Use | advanced | planned | design needed |
| Model Context Protocol (MCP) | advanced | planned | design needed |
| Agent Evaluation & Benchmarks (SWE-bench, WebArena, GAIA) | advanced | planned | design needed |
| Agent Safety & Sandboxing (Permission Systems, Action Boundaries) | advanced | planned | design needed |
| Agentic RAG & Deep Research (Multi-Step Retrieval) | advanced | planned | design needed |
| Long-Horizon Agents, Planning & Reliable Execution | frontier | planned | design needed |
| Agent-to-Agent Communication Protocols (A2A, MCP Extensions) | frontier | planned | design needed |
| Structured Output (JSON Mode, Outlines, Grammar Constraints) | intermediate | planned | design needed |
| Grounding & Factuality Verification | advanced | planned | design needed |
| Agent State Machines, Durable Execution & Recovery | intermediate | planned | brief |
| Agent Evidence, Untrusted Inputs & Permission Enforcement | intermediate | planned | brief |

### Frontier Research Areas

**Module anchor:** Reconstruct one narrow claim with a baseline and distinguish published evidence from speculation.

**Teaching strategy:** Ask a precise research question and establish prerequisites → Explain the baseline and proposed mechanism → Reconstruct a small example or experiment → Locate evidence and check assumptions → Attempt a bounded reproduction and comparison → Identify uncertainty, limitations and a concrete next investigation.

**Practice:** Reconstruct a result, evaluate an alternative explanation and produce a bounded reproducibility report.

**Verification:** Use original papers/data/docs; distinguish hypothesis, demonstration, generalization and unsettled claims; date moving information.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Federated Learning & Privacy-Preserving ML | advanced | planned | design needed |
| Differential Privacy in ML | advanced | planned | design needed |
| Neuro-Symbolic AI | advanced | planned | design needed |
| Dataset Distillation & Data-Centric AI | advanced | planned | design needed |
| Model Merging & Weight Space Methods | advanced | planned | design needed |
| In-Context Learning Theory | frontier | planned | design needed |
| World Models & Predictive Learning | frontier | planned | design needed |
| Causal Representation Learning | frontier | planned | design needed |
| Continual / Lifelong Learning | frontier | planned | design needed |
| Grokking & Delayed Generalization | frontier | planned | design needed |
| Emergent Abilities & Phase Transitions | frontier | planned | design needed |
| Synthetic Data & Self-Improvement Loops | frontier | planned | design needed |
| Test-Time Training (TTT) & Adaptive Inference | frontier | planned | design needed |
| Reasoning Scaling & Extended Thinking | frontier | planned | design needed |
| Neural Scaling Laws Beyond Chinchilla | frontier | planned | design needed |
| Geometric & Topological Deep Learning | frontier | planned | design needed |
| Memory-Augmented Architectures (Titans, Memorizing Transformers) | frontier | planned | design needed |
| Mechanistic Interpretability at Scale (Full-Model Maps, Feature Universality) | frontier | planned | design needed |
| Tokenizer-Free & Byte-Level Models (MegaByte, SpaceByte) | frontier | planned | design needed |
| Mixture-of-Agents & Collective Intelligence | frontier | planned | design needed |
| Protein Structure Prediction (AlphaFold, ESMFold) | advanced | planned | design needed |
| Drug Discovery & Molecular Generation | advanced | planned | design needed |
| Physics-Informed Neural Networks (PINNs) | advanced | planned | design needed |
| Neural Operators (FNO, DeepONet) | advanced | planned | design needed |
| AI-Driven Materials Discovery (GNoME, Autonomous Labs) | advanced | planned | design needed |
| Biological Sequence Modeling (ESM-3, RNA, Genomics Foundation Models) | advanced | planned | design needed |
| Weather & Climate Prediction (GraphCast, Pangu-Weather, GenCast) | frontier | planned | design needed |
| AI for Mathematics (AlphaProof, FunSearch) | frontier | planned | design needed |
| Autonomous Scientific Laboratories | frontier | planned | design needed |
| Robot Perception (LiDAR, Depth Sensing, Sensor Fusion) | intermediate | planned | design needed |
| Motion Planning & Path Planning (RRT, A*, Trajectory Optimization) | intermediate | planned | design needed |
| Control Theory for Robotics (PID, MPC, Adaptive Control) | intermediate | planned | design needed |
| SLAM (Simultaneous Localization and Mapping) | advanced | planned | design needed |
| Robot Manipulation & Grasping (Dexterous Hands, Contact-Rich) | advanced | planned | design needed |
| Imitation Learning & Learning from Demonstrations | advanced | planned | design needed |
| Sim-to-Real Transfer & Domain Randomization for Robotics | advanced | planned | design needed |
| Vision-Language-Action Models (RT-2, Octo, π₀) | frontier | planned | design needed |
| Humanoid Robotics & Whole-Body Control | frontier | planned | design needed |
| Research Reproduction, Ablations & Evidence Quality | intermediate | planned | brief |
| Scientific Surrogates, Conservation Laws & Out-of-Distribution Validation | intermediate | planned | brief |

### Self-Supervised & Contrastive Learning

**Module anchor:** Compare views, representation objectives, collapse risks and downstream transfer under controlled data.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| SSL Augmentations, Invariance & Representation Evaluation | foundation | planned | brief |
| SSL Pretext Tasks & Historical Methods (Jigsaw, Rotation, Colorization) | foundation | planned | design needed |
| Contrastive Learning (SimCLR, MoCo, InfoNCE Loss) | intermediate | planned | design needed |
| Barlow Twins & VICReg (Redundancy Reduction Methods) | intermediate | planned | design needed |
| Self-Distillation (BYOL, DINO, DINOv2) | intermediate | planned | design needed |
| Masked Autoencoders (MAE, BEiT, data2vec) | intermediate | planned | design needed |
| CLIP & Language-Supervised Visual Learning | advanced | planned | design needed |
| Joint Embedding Predictive Architectures (JEPA) | advanced | planned | design needed |
| V-JEPA Extensions & Cross-Modal JEPA | advanced | planned | design needed |
| Self-Supervised Learning for Video (VideoMAE, V-JEPA 2) | advanced | planned | design needed |
| SSL for Audio & Speech (wav2vec 2.0, HuBERT, AudioMAE) | advanced | planned | design needed |
| Theoretical Foundations of SSL (LeJEPA, Information Theory) | advanced | planned | design needed |

### Meta-Learning (Learning to Learn)

**Module anchor:** Separate tasks, support/query data, inner adaptation and outer learning before comparing baselines.

**Teaching strategy:** Define task, data unit and a baseline → Establish splits and what information is available → Trace a tiny example through representations and computation → Connect objective, parameter update and inference → Evaluate a failure and a controlled comparison → Apply independently and explain assumptions and limits.

**Practice:** Manual calculation, reproducible small implementation, diagnosed failure, and an ablation or changed-data experiment.

**Verification:** Check leakage, shapes, objective, baseline, metrics, seeds, uncertainty and realistic compute requirements; no invented measurements.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Meta-Learning Problem Formulation | foundation | planned | design needed |
| Few-Shot Learning & N-Way K-Shot Setup | foundation | planned | design needed |
| Episodic Training | foundation | planned | design needed |
| Meta-Train / Meta-Test Split | intermediate | planned | design needed |
| MAML (Model-Agnostic Meta-Learning) | intermediate | planned | design needed |
| Reptile | intermediate | planned | design needed |
| ANIL (Almost No Inner Loop) & Meta-SGD | advanced | planned | design needed |
| Implicit MAML & iMAML | advanced | planned | design needed |
| Warped Gradient Descent & Task-Specific Parameterization | advanced | planned | design needed |
| Siamese Networks | intermediate | planned | design needed |
| Prototypical Networks | intermediate | planned | design needed |
| Matching Networks | intermediate | planned | design needed |
| Relation Networks | advanced | planned | design needed |
| Graph Neural Network-Based Meta-Learning | advanced | planned | design needed |
| Memory-Augmented Neural Networks (MANN) | intermediate | planned | design needed |
| Neural Turing Machines & Differentiable Memory | advanced | planned | design needed |
| Meta-Networks (MetaNet) & Hypernetworks for Meta-Learning | advanced | planned | design needed |
| Conditional Neural Processes (CNP) & Neural Processes | advanced | planned | design needed |
| Meta-Reinforcement Learning | advanced | planned | design needed |
| Task-Agnostic & Unsupervised Meta-Learning | advanced | planned | design needed |
| Bayesian Meta-Learning (BMAML, VERSA) | advanced | planned | design needed |
| Learned Optimizers & Learning to Optimize | advanced | planned | design needed |
| AutoML as Meta-Learning | advanced | planned | design needed |
| In-Context Learning as Meta-Learning | frontier | planned | design needed |
| Foundation Models as Few-Shot Learners | frontier | planned | design needed |
| Meta-Learning for Continual Learning | frontier | planned | design needed |
| Recurrent Meta-Learners (RL², SNAIL) | advanced | planned | design needed |
| Transformer-Based Meta-Learners (TNP, In-Context Learners) | advanced | planned | design needed |
| Fast-Weight Programmers | advanced | planned | design needed |
| Few-Shot Object Detection & Segmentation | advanced | planned | design needed |
| Meta-RL: PEARL, VariBAD & Context-Based Adaptation | advanced | planned | design needed |
| Meta-Learning for Drug Discovery & Molecular Properties | advanced | planned | design needed |
| Meta-Learning for NLP (Cross-Lingual Transfer, Prompt Learning) | advanced | planned | design needed |
| DARTS & Meta-Learned Architectures | advanced | planned | design needed |
| Meta-Learned Data Augmentation & Curriculum | advanced | planned | design needed |
| LEO, CAVIA & Latent Space Meta-Learning | advanced | planned | design needed |
| Meta-Learned Loss Functions & Regularizers | advanced | planned | design needed |
| PAC-Bayes Bounds for Meta-Learning | advanced | planned | design needed |
| Meta-Overfitting & When Meta-Learning Helps | advanced | planned | design needed |
| Transfer Learning vs. Meta-Learning | advanced | planned | design needed |
| Compositional Meta-Learning | frontier | planned | design needed |
| Meta-Learning at Scale & Scaling Laws | frontier | planned | design needed |
| Task Vectors & Gradient Descent in Transformers | frontier | planned | design needed |
| Meta-Learning Benchmarks | intermediate | planned | design needed |
| Meta-Learning Libraries | intermediate | planned | design needed |
| Meta-Learning Evaluation, Task Shift & Adaptation Baselines | intermediate | planned | brief |

### Quantum AI & Quantum Computing for ML

**Module anchor:** A tiny circuit/state example separates ideal calculation, noisy execution and claims of advantage.

**Teaching strategy:** Ask a precise research question and establish prerequisites → Explain the baseline and proposed mechanism → Reconstruct a small example or experiment → Locate evidence and check assumptions → Attempt a bounded reproduction and comparison → Identify uncertainty, limitations and a concrete next investigation.

**Practice:** Reconstruct a result, evaluate an alternative explanation and produce a bounded reproducibility report.

**Verification:** Use original papers/data/docs; distinguish hypothesis, demonstration, generalization and unsettled claims; date moving information.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Complex Amplitudes, Tensor Products & Quantum Experiment Accounting | foundation | planned | brief |
| Qubits, Superposition & Entanglement | foundation | planned | design needed |
| Quantum Gates & Circuits | foundation | planned | design needed |
| Measurement & Born Rule | foundation | planned | design needed |
| No-Cloning Theorem & Quantum Teleportation | foundation | planned | design needed |
| Quantum State Spaces & Hilbert Space | intermediate | planned | design needed |
| Density Matrices & Mixed States | intermediate | planned | design needed |
| Quantum Complexity Theory & BQP | intermediate | planned | design needed |
| Quantum Annealing & Adiabatic Quantum Computing | intermediate | planned | design needed |
| Grover's Search Algorithm | intermediate | planned | design needed |
| Shor's Factoring Algorithm | intermediate | planned | design needed |
| Quantum Fourier Transform (QFT) | intermediate | planned | design needed |
| Variational Quantum Eigensolver (VQE) | intermediate | planned | design needed |
| Quantum Walks | intermediate | planned | design needed |
| Quantum Approximate Optimization Algorithm (QAOA) | advanced | planned | design needed |
| Quantum Phase Estimation (QPE) | advanced | planned | design needed |
| HHL Algorithm (Quantum Linear Systems) | advanced | planned | design needed |
| Quantum Simulation (Hamiltonian Simulation) | advanced | planned | design needed |
| Variational Quantum Circuits (VQC) / Parameterized Quantum Circuits | intermediate | planned | design needed |
| Quantum Kernel Methods | intermediate | planned | design needed |
| Quantum Data Encoding (Amplitude, Angle, Basis Encoding) | intermediate | planned | design needed |
| Quantum Neural Networks (QNN) & Hybrid Models | advanced | planned | design needed |
| Barren Plateaus & Trainability | advanced | planned | design needed |
| Quantum Generative Models (QGAN, Quantum Boltzmann Machines) | advanced | planned | design needed |
| Quantum Reinforcement Learning | advanced | planned | design needed |
| Quantum Transfer Learning | advanced | planned | design needed |
| Quantum Reservoir Computing | advanced | planned | design needed |
| Quantum Monte Carlo Methods | advanced | planned | design needed |
| Quantum Portfolio Optimization & Risk Analysis | advanced | planned | design needed |
| Combinatorial Optimization on Quantum Hardware (MaxCut, TSP via QAOA) | advanced | planned | design needed |
| Quantum Chemistry & Materials Science Applications | advanced | planned | design needed |
| Quantum Information Theory (von Neumann Entropy, Fidelity, Tomography) | advanced | planned | design needed |
| Quantum Natural Language Processing (QNLP) | frontier | planned | design needed |
| Quantum Foundation Models & Advantage Benchmarks | frontier | planned | design needed |
| Classical ML for Quantum (AlphaQubit, ML-Assisted Error Decoding) | frontier | planned | design needed |
| Post-Quantum Cryptography & ML Security Implications | advanced | planned | design needed |
| Tensor Networks (MPS, PEPS, MERA) | intermediate | planned | design needed |
| Quantum-Inspired Optimization (Simulated Annealing, DMRG-Inspired) | advanced | planned | design needed |
| Dequantization & Classical Simulation of Quantum ML | advanced | planned | design needed |
| NISQ (Noisy Intermediate-Scale Quantum) Devices | intermediate | planned | design needed |
| Quantum Hardware Technologies (Superconducting, Trapped Ion, Photonic, Neutral Atom) | intermediate | planned | design needed |
| Quantum Error Correction (Surface Codes, Logical Qubits) | advanced | planned | design needed |
| Quantum Error Mitigation | advanced | planned | design needed |
| Fault-Tolerant Quantum Computing | frontier | planned | design needed |
| Quantum Advantage for ML — Status & Outlook | frontier | planned | design needed |
| Quantum Programming Frameworks | foundation | planned | design needed |
| Quantum Circuit Languages & Additional Tools | intermediate | planned | design needed |
| Quantum Simulators | intermediate | planned | design needed |

### Landmark Models & What Makes Them Notable

**Module anchor:** Explain the architectural change against its predecessor and inspect the original experimental evidence.

**Teaching strategy:** Ask a precise research question and establish prerequisites → Explain the baseline and proposed mechanism → Reconstruct a small example or experiment → Locate evidence and check assumptions → Attempt a bounded reproduction and comparison → Identify uncertainty, limitations and a concrete next investigation.

**Practice:** Reconstruct a result, evaluate an alternative explanation and produce a bounded reproducibility report.

**Verification:** Use original papers/data/docs; distinguish hypothesis, demonstration, generalization and unsettled claims; date moving information.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Model Papers, Model Cards & Comparative Evidence | foundation | planned | brief |
| DeepSeek-R1 | intermediate | planned | design needed |
| DeepSeek-V3 | intermediate | planned | design needed |
| Meta Llama 4 (Scout & Maverick) | intermediate | planned | design needed |
| OpenAI o-Series (o1, o3, o4-mini) | intermediate | planned | design needed |
| Claude 4.6 (Opus, Sonnet) | intermediate | planned | design needed |
| Google Gemini (1.5 Pro, 2.0, 2.5) | intermediate | planned | design needed |
| Mistral & Mixtral | intermediate | planned | design needed |
| Qwen Series (Qwen 2.5, Qwen3) | intermediate | planned | design needed |
| Stable Diffusion 3 / Flux | intermediate | planned | design needed |
| Sora | intermediate | planned | design needed |
| Mamba / Mamba-2 | intermediate | planned | design needed |
| GPT Series (GPT-1 → GPT-4o) | intermediate | planned | design needed |
| Phi Series (Phi-1 → Phi-4) | intermediate | planned | design needed |
| Grok Series (Grok-1, Grok-2, Grok-3) | intermediate | planned | design needed |
| Command R & Command R+ | intermediate | planned | design needed |
| GPT-5 / GPT-5.3 Codex | intermediate | planned | design needed |
| Gemma Series (Gemma 2, Gemma 3) | intermediate | planned | design needed |
| AlphaFold 3 | intermediate | planned | design needed |
| NVIDIA Cosmos | intermediate | planned | design needed |
| BERT & T5 (Encoder & Encoder-Decoder Landmarks) | intermediate | planned | design needed |
| DALL-E Series (1, 2, 3) | intermediate | planned | design needed |

### Core Frameworks & Tool Ecosystem

**Module anchor:** One complete task from setup and data to output, with concepts mapped to tested APIs.

**Teaching strategy:** Give plain meaning and why the term/tool appears → Disambiguate related concepts → Show a miniature example or coherent workflow → Explain a common misuse → Link prerequisites and the full mechanism lesson.

**Practice:** Interpret a new occurrence, choose the correct term/tool and explain the distinction.

**Verification:** Verify terminology and current interfaces; never treat a glossary definition as a full mechanism lesson.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| PyTorch | foundation | planned | design needed |
| JAX | foundation | planned | design needed |
| TensorFlow & Keras | foundation | planned | design needed |
| Experiment Tracking Tools (Weights & Biases, MLflow, Neptune) | foundation | planned | design needed |
| Training Acceleration | intermediate | planned | design needed |
| LLM Fine-Tuning Platforms | intermediate | planned | design needed |
| Data Processing & Loading | foundation | planned | design needed |
| Data Annotation & Labeling | intermediate | planned | design needed |
| NLP Libraries | foundation | planned | design needed |
| Computer Vision Libraries | foundation | planned | design needed |
| Visualization | foundation | planned | design needed |
| GPU-Accelerated Computing | foundation | planned | design needed |
| Vector Databases | intermediate | planned | design needed |
| Data Annotation & Labeling (Extended) | intermediate | planned | design needed |
| LLM Serving Frameworks (vLLM, TGI, SGLang, TensorRT-LLM) | intermediate | planned | design needed |
| LLM Application Frameworks (LangChain, LlamaIndex, Haystack) | intermediate | planned | design needed |
| Agent & Workflow Frameworks (CrewAI, AutoGen, LangGraph) | intermediate | planned | design needed |
| Tensor Shapes, Autograd Graphs & Device Boundaries | foundation | planned | brief |
| Dataset Loaders, Checkpoints & Exact Resume Contracts | intermediate | planned | brief |

### Key Terminology Glossary

**Module anchor:** Plain meaning, disambiguation, one miniature example and a linked mechanism lesson.

**Teaching strategy:** Give plain meaning and why the term/tool appears → Disambiguate related concepts → Show a miniature example or coherent workflow → Explain a common misuse → Link prerequisites and the full mechanism lesson.

**Practice:** Interpret a new occurrence, choose the correct term/tool and explain the distinction.

**Verification:** Verify terminology and current interfaces; never treat a glossary definition as a full mechanism lesson.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Arithmetic Intensity | foundation | planned | design needed |
| FLOP Utilization (MFU) | foundation | planned | design needed |
| Scaling Laws | foundation | planned | design needed |
| Chinchilla-Optimal | foundation | planned | design needed |
| Emergent Abilities | foundation | planned | design needed |
| Grokking | foundation | planned | design needed |
| Mode Collapse | foundation | planned | design needed |
| Reward Hacking | foundation | planned | design needed |
| Sycophancy | foundation | planned | design needed |
| Catastrophic Forgetting | foundation | planned | design needed |
| KV-Cache | foundation | planned | design needed |
| Perplexity | foundation | planned | design needed |
| Overparameterized | foundation | planned | design needed |
| Double Descent | foundation | planned | design needed |
| Mixture-of-Experts (MoE) | foundation | planned | design needed |
| Token | foundation | planned | design needed |
| Inference-Time Compute | foundation | planned | design needed |
| RLHF | foundation | planned | design needed |
| DPO | foundation | planned | design needed |
| LoRA | foundation | planned | design needed |
| Flash Attention | foundation | planned | design needed |
| Speculative Decoding | advanced | published | design needed |
| Activation Checkpointing | foundation | planned | design needed |
| RoPE | foundation | planned | design needed |
| SwiGLU | foundation | planned | design needed |
| RMSNorm | foundation | planned | design needed |
| Attention Sink | foundation | planned | design needed |
| Hallucination | foundation | planned | design needed |
| Distillation | foundation | planned | design needed |
| Data Contamination | foundation | planned | design needed |
| Alignment Tax | foundation | planned | design needed |
| Loss Spike | foundation | planned | design needed |
| μP (Maximal Update Parameterization) | foundation | planned | design needed |
| Auxiliary Loss | foundation | planned | design needed |
| Soft Labels | foundation | planned | design needed |
| Few-Shot Learning | foundation | planned | design needed |
| MAML | foundation | planned | design needed |
| Episodic Training | foundation | planned | design needed |
| Qubit | foundation | planned | design needed |
| Superposition | foundation | planned | design needed |
| Entanglement | foundation | planned | design needed |
| NISQ | foundation | planned | design needed |
| Barren Plateau | foundation | planned | design needed |
| Variational Quantum Circuit | foundation | planned | design needed |
| Quantum Advantage | foundation | planned | design needed |
| Hypernetwork | foundation | planned | design needed |
| Goodhart's Law (in RLHF) | foundation | planned | design needed |
| Lottery Ticket Hypothesis | foundation | planned | design needed |
| Representational Collapse | foundation | planned | design needed |
| Loss Landscape Geometry | foundation | planned | design needed |
| PAC Learning | foundation | planned | design needed |
| VC Dimension | foundation | planned | design needed |
| Deceptive Alignment | foundation | planned | design needed |
| Mesa-Optimization | foundation | planned | design needed |
| Model Merging | foundation | planned | design needed |
| Conformal Prediction | foundation | planned | design needed |
| Federated Learning | foundation | planned | design needed |
| No-Cloning Theorem | foundation | planned | design needed |
| Quantum Annealing | foundation | planned | design needed |
| QNLP | foundation | planned | design needed |
| Little's Law | foundation | planned | design needed |
| TTFT / TBT | foundation | planned | design needed |
| Disaggregated Serving | foundation | planned | design needed |
| Model Cascade | foundation | planned | design needed |
| Data Flywheel | foundation | planned | design needed |
| Prefill vs Decode | foundation | planned | design needed |
| Activation Steering | foundation | planned | design needed |
| Steering Vector | foundation | planned | design needed |
| Refusal Direction | foundation | planned | design needed |
| Representation Engineering (RepE) | advanced | planned | design needed |
| Activation Patching | foundation | planned | design needed |
| Model Editing | foundation | planned | design needed |
| Function Vector | foundation | planned | design needed |
| Concept Erasure | foundation | planned | design needed |
| RLVR | foundation | planned | design needed |
| DAPO | foundation | planned | design needed |
| World Model | foundation | planned | design needed |
| Faithful Chain-of-Thought | foundation | planned | design needed |
| Curator Model | foundation | planned | design needed |
| Model Collapse | foundation | planned | design needed |
| SLM (Small Language Model) | foundation | planned | design needed |
| Flow Matching | foundation | planned | design needed |
| Agentic RAG | foundation | planned | design needed |
| Optimal Transport | foundation | planned | design needed |
| Attention Head | foundation | planned | design needed |
| Embedding | foundation | planned | design needed |
| Latent Space | foundation | planned | design needed |
| Tokenizer | foundation | planned | design needed |
| Gradient Accumulation | foundation | planned | design needed |

### LLM Evaluation & Assessment

**Module anchor:** A decision-relevant assessment from construct and sample through scoring, uncertainty and an audit.

**Teaching strategy:** State the construct and decision the evidence will support → Define sample, unit, rubric and baseline → Trace an item through annotation and scoring → Estimate uncertainty and examine validity threats → Audit a plausible misleading result → Design a changed evaluation and justify its decision limits.

**Practice:** Repair a misleading metric or rubric, compute a small example, audit leakage/bias and design a discriminating held-out test.

**Verification:** Check construct validity, sampling, dependence, calibration, uncertainty, benchmark contamination, cost and scope of conclusions.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Evaluation Questions, Units of Analysis & Sampling Plans | foundation | planned | brief |
| Evaluation Artifact Provenance & Reproducible Scoring | foundation | planned | brief |
| LLM-as-Judge Paradigm & Core Methodology | foundation | planned | design needed |
| Pointwise vs Pairwise vs Listwise Evaluation | foundation | planned | design needed |
| Cost-Performance Analysis of LLM Judges | foundation | planned | design needed |
| Judge Biases (Position, Verbosity, Self-Enhancement, Style) | intermediate | planned | design needed |
| Prometheus & Open-Source Judge Models | intermediate | planned | design needed |
| Structured Judging Templates & Chain-of-Thought Scoring | intermediate | planned | design needed |
| Reference-Free vs Reference-Based Evaluation | intermediate | planned | design needed |
| Reasoning Traces & Structured Output for Judges | intermediate | planned | design needed |
| Self-Consistency & Council-of-Judges | advanced | planned | design needed |
| Judge Calibration (Platt Scaling, Isotonic Regression, Anchors) | advanced | planned | design needed |
| Agent-as-a-Judge & Multi-Agent Debate Protocols | advanced | planned | design needed |
| Classical Test Theory (CTT) | foundation | planned | design needed |
| Item Response Theory (IRT): 1PL, 2PL, 3PL Models | intermediate | planned | design needed |
| Reliability Coefficients (Cronbach's Alpha, Cohen's & Fleiss' Kappa, Krippendorff's Alpha) | intermediate | planned | design needed |
| Validity Frameworks (Construct, Criterion, Content) | intermediate | planned | design needed |
| Standard Setting Methods (Angoff, Bookmark, Modified Angoff) | advanced | planned | design needed |
| Differential Item Functioning (DIF) & Test Bias | advanced | planned | design needed |
| Computer-Adaptive Testing (CAT) | advanced | planned | design needed |
| Assessment Security & Item Exposure Control | advanced | planned | design needed |
| Emotional Intelligence Evaluation (EQ-Bench) | advanced | planned | design needed |
| Adaptive Precise Boolean Rubrics | advanced | planned | design needed |
| LLM Metacognition & Self-Knowledge Evaluation | frontier | planned | design needed |
| Significance Testing for LLM Outputs (Paired Bootstrap, Permutation Tests) | foundation | planned | design needed |
| Effect Sizes (Cohen's d, Hedges' g) | foundation | planned | design needed |
| Multiple Comparisons (Bonferroni, Holm-Bonferroni, Benjamini-Hochberg FDR) | intermediate | planned | design needed |
| Power Analysis for Model Evaluation | intermediate | planned | design needed |
| Bootstrap Confidence Intervals (BCa) | intermediate | planned | design needed |
| Bradley-Terry Models for LLM Ranking | intermediate | planned | design needed |
| Elo Rating Systems & Bootstrap Stability | intermediate | planned | design needed |
| Bayesian Model Comparison (Bayes Factors, Credible Intervals) | advanced | planned | design needed |
| Conformal Prediction for LLM Uncertainty | advanced | planned | design needed |
| Uncertainty Quantification (Semantic Entropy, LM-Polygraph) | advanced | planned | design needed |
| Variance Decomposition for LLM Evaluation Runs | advanced | planned | design needed |
| Algorithmic Fairness Definitions & Impossibility Results | foundation | planned | design needed |
| Group Fairness vs Individual Fairness | intermediate | planned | design needed |
| Counterfactual Fairness & Paraphrase Testing | intermediate | planned | design needed |
| WEIRD Bias in LLM Evaluation | intermediate | planned | design needed |
| AI-to-AI Bias (Self-Preference in Judging) | intermediate | planned | design needed |
| Observable vs Unobservable Attributes in LLM Assessment | advanced | planned | design needed |
| Cross-Dimensional Bias Trade-offs | advanced | planned | design needed |
| Multi-Agent Cultural Bias Evaluation (CEBiasBench) | advanced | planned | design needed |
| Intersectional Bias Analysis | advanced | planned | design needed |
| Evaluator Audit Methodology (Paired Testing, Parity Testing, Calibration) | advanced | planned | design needed |
| Retrieval Evaluation (Dense, Sparse & Hybrid) | foundation | planned | design needed |
| Re-Ranking Strategies (Cross-Encoders vs Bi-Encoders) | intermediate | planned | design needed |
| Retrieval Metrics (Recall@K, MRR, nDCG) | intermediate | planned | design needed |
| RAG-Specific Evaluation (Context Relevance, Faithfulness, Groundedness) | intermediate | planned | design needed |
| RAGAS & TruLens Evaluation Frameworks | intermediate | planned | design needed |
| Chunking Strategies & Context Length Management for Eval | intermediate | planned | design needed |
| End-to-End RAG Pipeline Evaluation | intermediate | planned | design needed |
| Rubric-Retrieval Judges | advanced | planned | design needed |
| RAG Contradiction Detection | advanced | planned | design needed |
| RAG-RewardBench | advanced | planned | design needed |
| SFT for Judge Models | intermediate | planned | design needed |
| Preference Optimization for Judges (DPO, KTO, ORPO) | intermediate | planned | design needed |
| LoRA & QLoRA for Judge Fine-Tuning | intermediate | planned | design needed |
| Distillation Quality Evaluation | intermediate | planned | design needed |
| Reward Modeling & Reward Hacking in Evaluation | advanced | planned | design needed |
| Process Reward Models (PRMs) vs Outcome Reward Models (ORMs) | advanced | planned | design needed |
| RewardBench & RewardBench 2 | advanced | planned | design needed |
| Constitutional AI Evaluation & Constitution Design | advanced | planned | design needed |
| Preference Data Quality & Inter-Annotator Agreement | advanced | planned | design needed |
| Preference Proxy Evaluations (PPE) | frontier | planned | design needed |
| Evaluation Observability & Logging | intermediate | planned | design needed |
| Caching, Idempotency & Reproducibility for Judge Systems | intermediate | planned | design needed |
| Online vs Offline Evaluation | intermediate | planned | design needed |
| Evaluation as CI/CD (Unit-Test-Style Evals, DeepEval) | intermediate | planned | design needed |
| LLM Observability Tools (Langfuse, LangSmith, Arize) | intermediate | planned | design needed |
| Drift Detection for Evaluators (Population, Concept, Model) | advanced | planned | design needed |
| Cost-Quality-Latency Pareto Optimization | advanced | planned | design needed |
| Quality-Aware Model Routing (SCORE Framework) | advanced | planned | design needed |
| Shadow Deployment & Canary Rollouts for Evaluators | advanced | planned | design needed |
| A/B Testing & Sequential Analysis at Scale | advanced | planned | design needed |
| Tiered Evaluator Architectures (Cost vs Quality Routing) | advanced | planned | design needed |
| SLAs, Rate Limiting & Degradation Strategies | advanced | planned | design needed |
| Rubric Decomposition & Sub-Criteria Design | foundation | planned | design needed |
| Ordinal Scale Design & Anchor Descriptions | foundation | planned | design needed |
| Exemplar Curation & Few-Shot Selection for Judges | intermediate | planned | design needed |
| Prompt Structure for LLM Judges | intermediate | planned | design needed |
| Rubric Refinement Loops & Agreement Tracking | intermediate | planned | design needed |
| IFEval: Verifiable Instruction-Following Rubrics | intermediate | planned | design needed |
| Evaluation Report Writing & Stakeholder Communication | intermediate | planned | design needed |
| Clinical & Domain-Specific Rubrics (CLEVER, R-IDEA) | advanced | planned | design needed |
| Evaluation-Driven Development with DSPy | advanced | planned | design needed |
| Benchmark Families (HELM, BIG-Bench, MT-Bench, Chatbot Arena) | foundation | planned | design needed |
| Frontier Math Benchmarks (AIME, FrontierMath) | foundation | planned | design needed |
| Humanity's Last Exam (HLE) | foundation | planned | design needed |
| GPQA Diamond for Scientific Reasoning | foundation | planned | design needed |
| Benchmark Contamination & Detection Methods | intermediate | planned | design needed |
| ARC-AGI-2 & ARC-AGI-3 for Abstract Reasoning | intermediate | planned | design needed |
| Chatbot Arena / LMArena Methodology | intermediate | planned | design needed |
| LiveCodeBench for Contamination-Free Code Eval | intermediate | planned | design needed |
| BFCL v4 for Tool Use & Function Calling Evaluation | intermediate | planned | design needed |
| SWE-bench Verified & SWE-bench-Live | intermediate | planned | design needed |
| Dynamic & Private Evaluations | intermediate | planned | design needed |
| Benchmark Saturation & When Benchmarks Stop Being Useful | intermediate | planned | design needed |
| Canary Items & Eval Set Versioning | advanced | planned | design needed |
| Regression & Acceptance Testing for Evaluators | advanced | planned | design needed |
| MLE-bench for ML Engineering Agent Evaluation | advanced | planned | design needed |
| Prompt Injection Against Judges | intermediate | planned | design needed |
| Prompt Sensitivity & Brittleness Evaluation (PromptBench) | intermediate | planned | design needed |
| Gaming Detection & Counter-Gaming Strategies | advanced | planned | design needed |
| Self-Contradictory Reasoning Detection | advanced | planned | design needed |
| Leaderboard Integrity & Vote Rigging Detection | advanced | planned | design needed |
| Data Poisoning for Evaluator Systems | advanced | planned | design needed |
| Integrity Signals & Cheating Detection in Assessment | advanced | planned | design needed |
| Code Evaluation Dimensions (Correctness, Efficiency, Style, Robustness) | foundation | planned | design needed |
| AI-Assisted Coding Evaluation & Rubric Design | intermediate | planned | design needed |
| Coding Benchmarks (HumanEval, MBPP, SWE-bench, LiveCodeBench) | intermediate | planned | design needed |
| BigCodeBench (ICLR 2025) | intermediate | planned | design needed |
| Agentic & Repository-Level Code Evaluation | advanced | planned | design needed |
| Code Generation Failure Pattern Analysis | advanced | planned | design needed |
| Paper Reading & Research Methodology for Eval | foundation | planned | design needed |
| Experiment Tracking & Ablation Design | intermediate | planned | design needed |
| Writing Research Memos & Evaluation Reports | intermediate | planned | design needed |
| Evaluation for Scientific Discovery (Scientist-Bench) | frontier | planned | design needed |
| Chain-of-Thought Monitorability & Faithfulness | advanced | planned | design needed |
| CoT Controllability Evaluation | advanced | planned | design needed |
| Test-Time Compute Scaling Evaluation | advanced | planned | design needed |
| Evaluating Reasoning vs. Memorization | advanced | planned | design needed |
| TruthfulQA & SimpleQA Factuality Suites | intermediate | planned | design needed |
| Hallucination Rate Benchmarking (HalluLens) | intermediate | planned | design needed |
| MetaQA: Metamorphic Testing for Hallucination Detection | advanced | planned | design needed |
| Cross-Layer Attention Probing (CLAP) for Real-Time Detection | advanced | planned | design needed |
| Selective Prediction & Abstention (Conformal Methods) | advanced | planned | design needed |
| NeedleBench: Multi-Needle Retrieval & Reasoning | intermediate | planned | design needed |
| Multimodal Needle in a Haystack (MMNeedle) | advanced | planned | design needed |
| Video-MME for Video Understanding Evaluation | advanced | planned | design needed |
| LongGenBench for Long-Form Generation | advanced | planned | design needed |
| MFCL Vision: Tool Use in Multimodal Models | advanced | planned | design needed |
| AI Agent Index & Capability Taxonomy | foundation | planned | design needed |
| GAIA: General AI Assistant Benchmark | intermediate | planned | design needed |
| CLEAR Framework for Enterprise Agent Evaluation | advanced | planned | design needed |
| Evaluating Agent Tool Selection & Multi-Step Coherence | advanced | planned | design needed |
| Levels of AI Autonomy & Risk Assessment | foundation | planned | design needed |
| OWASP GenAI Red Teaming Guide | intermediate | planned | design needed |
| EU AI Act Compliance Evaluation | intermediate | planned | design needed |
| NIST AI RMF & TEVV Process | intermediate | planned | design needed |
| METR Dangerous Capability Evaluations | advanced | planned | design needed |
| Responsible Scaling Policy Evaluation (Anthropic RSP) | advanced | planned | design needed |
| Automated Red Teaming at Scale | advanced | planned | design needed |
| MMLU-ProX: Multilingual Advanced Evaluation | intermediate | planned | design needed |
| M-IFEval: Multilingual Instruction Following | intermediate | planned | design needed |
| Multilingual Consistency (MLC) & Mother-Tongue Effect (MTE) | advanced | planned | design needed |
| Legal LLM Evaluation (LawBench) | intermediate | planned | design needed |
| HealthBench: Medical AI Evaluation | advanced | planned | design needed |
| MedThink-Bench: Clinical Reasoning Evaluation | advanced | planned | design needed |
| EleutherAI LM Evaluation Harness | foundation | planned | design needed |
| OpenCompass Evaluation Platform | foundation | planned | design needed |
| Model Cards & AI Transparency Documentation | intermediate | planned | design needed |
| AI-Generated Text Detection & Watermarking (SynthID) | advanced | planned | design needed |
| Synthetic Data Quality Evaluation (Fidelity, Utility, Privacy) | intermediate | planned | design needed |
| Model Collapse Detection & Prevention | advanced | planned | design needed |
| Human-LLM Collaborative Annotation | intermediate | planned | design needed |
| RocketEval: Scalable Automated Evaluation | advanced | planned | design needed |
| Annotation Quality Verification (Lapras Framework) | advanced | planned | design needed |

### Programming & Scientific Computing

**Module anchor:** A small measurement project from files and records through transformations to a reproducible report.

**Teaching strategy:** Introduce a practical task and the meaning of inputs → Name the language/runtime entities before syntax → Trace execution alongside state and output → Run a complete small example with explained setup → Investigate an edge case and repair a failure → Complete a changed-input task independently.

**Practice:** Prediction, guided variation, diagnosis and a complete independent task with fixtures, expected results, hints and explained solutions.

**Verification:** Execute in the stated language/library/environment; compare browser models against actual behavior and label their limits.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Python Basics: Types, Control Flow, Functions & Modules | foundation | published | brief |
| Object-Oriented Programming in Python | foundation | published | brief |
| Iterators, Iterables & Generators | intermediate | published | brief |
| Decorators & Context Managers | intermediate | published | brief |
| Testing, Debugging & Dependency Management | intermediate | published | brief |
| NumPy: Arrays, Broadcasting & Vectorization | foundation | published | brief |
| Scientific File Formats, Schemas & Reliable Data I/O | foundation | published | brief |
| SQL, Relational Data & Transactions for ML | foundation | published | brief |
| Pandas: Data Wrangling, Joins & Grouping | foundation | published | brief |
| Matplotlib & Scientific Plotting | foundation | published | brief |
| Reproducible Notebooks & Experiment Structure | intermediate | published | brief |
| Code Documentation, Type Hints & API Design | intermediate | published | brief |
| Git, GitHub & Collaborative Version Control | foundation | published | brief |
| Linux Basics, Filesystems & Processes | foundation | published | brief |
| Bash Scripting & Command-Line Automation | intermediate | published | brief |
| OS Processes, Virtual Memory & Isolation | foundation | published | brief |
| Threads, Concurrency, Locks & Deadlocks | intermediate | published | brief |

### Data Structures & Algorithms

**Module anchor:** Scheduling jobs, finding routes and maintaining changing collections under explicit constraints.

**Teaching strategy:** Specify inputs, outputs and a small task → Establish a simple correct method and its cost → Expose the data structure and trace its changing state → State and justify an invariant and termination → Analyze resources under a stated cost model → Implement, diagnose edge cases and transfer to a new constraint.

**Practice:** Predict a next state, implement from an invariant, construct a counterexample and compare time/space tradeoffs.

**Verification:** Correctness argument plus independent/oracle outputs for meaningful boundary and adversarial cases; measurements do not prove asymptotics.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Arrays, Strings & Hash Maps | foundation | published | brief |
| Linked Lists, Stacks & Queues | foundation | published | brief |
| Trees & Binary Search Trees | foundation | published | brief |
| Heaps, Priority Queues & Tries | intermediate | published | brief |
| Graphs: Representations, BFS & DFS | intermediate | published | brief |
| Disjoint Sets & Union-Find | intermediate | published | brief |
| Complexity Analysis & Recursion | foundation | published | brief |
| Binary Search, Sorting & Two-Pointer Patterns | foundation | published | brief |
| Backtracking & Divide-and-Conquer | intermediate | published | brief |
| Greedy Algorithms & Exchange Arguments | intermediate | published | brief |
| Dynamic Programming: States, Transitions & Optimization | intermediate | published | brief |
| Segment Trees, Fenwick Trees & Range Queries | advanced | published | brief |
| Algorithm Correctness, Loop Invariants & Termination | foundation | published | brief |
| Hashing, Collision Resolution & Amortized Analysis | intermediate | published | brief |
| Shortest Paths, Spanning Trees & Topological Ordering | intermediate | published | brief |
| String Matching, Prefix Functions & Rolling Hashes | intermediate | published | brief |
| Reductions, P, NP & Computational Intractability | intermediate | published | brief |
| Randomized Algorithms, Sampling & Error Guarantees | intermediate | published | brief |
| Network Flow, Minimum Cuts & Bipartite Matching | intermediate | published | brief |
| Computational Geometry, Robust Predicates & Convex Hulls | intermediate | published | brief |
| Persistent Data Structures, Structural Sharing & Versioned Queries | intermediate | published | brief |
| External-Memory Algorithms, B-Trees & I/O Complexity | intermediate | published | brief |

### JAX & Functional ML

**Module anchor:** Transform a pure array program, then inspect shapes, compilation, randomness and device placement.

**Teaching strategy:** Introduce a practical task and the meaning of inputs → Name the language/runtime entities before syntax → Trace execution alongside state and output → Run a complete small example with explained setup → Investigate an edge case and repair a failure → Complete a changed-input task independently.

**Practice:** Prediction, guided variation, diagnosis and a complete independent task with fixtures, expected results, hints and explained solutions.

**Verification:** Execute in the stated language/library/environment; compare browser models against actual behavior and label their limits.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| JAX Arrays, Pure Functions & Functional State | foundation | planned | design needed |
| JIT Compilation with jit | foundation | planned | design needed |
| Automatic Differentiation with grad | foundation | planned | design needed |
| Vectorization with vmap | intermediate | planned | design needed |
| Parallelism with pmap, pjit & Sharding | advanced | planned | design needed |
| Flax, Equinox & PyTree-Based Model Design | intermediate | planned | design needed |
| Optax Optimizers & Training Loops | intermediate | planned | design needed |
| Implement an MLP & CNN in JAX | intermediate | planned | design needed |
| Implement a Transformer in JAX | advanced | planned | design needed |
| Implement PPO in JAX | advanced | planned | design needed |
| Implement a Diffusion Model in JAX | advanced | planned | design needed |
| JAX PRNG Keys, PyTrees & Reproducible State | foundation | planned | brief |
| JAX Tracing, Shape Constraints & Honest Benchmarking | intermediate | planned | brief |

### Robotics, Embodied AI & Simulation

**Module anchor:** A robot senses, estimates, plans and acts under frames, delay, noise and physical limits.

**Teaching strategy:** Specify task, environment and observable success → Introduce frames, units, state, observations and actions → Trace sensing through estimation and decision to actuation → Explain feedback, delay, noise and constraints → Compare trajectories and diagnose a failure → Validate in changed environments and bound physical claims.

**Practice:** Predict response to a change, repair a frame/delay error, reproduce a bounded simulation and evaluate transfer.

**Verification:** Record seeds, timestep, frames, units, actuator limits and termination rules; distinguish simulation from physical evidence.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Coordinate Frames, Transformations & Robot State | foundation | planned | design needed |
| Forward & Inverse Kinematics | foundation | planned | design needed |
| Robot Dynamics & State-Space Models | intermediate | planned | design needed |
| PID Control, Kalman Filters & State Estimation | intermediate | planned | design needed |
| Model Predictive Control, CEM & Latent Planning | advanced | planned | design needed |
| Gymnasium API & Reinforcement Learning Environments | foundation | planned | design needed |
| MuJoCo, Isaac Sim & Isaac Lab | intermediate | planned | design needed |
| Navigation, Manipulation & Locomotion | intermediate | planned | design needed |
| Sim-to-Real Transfer & Domain Randomization | advanced | planned | design needed |
| Vision-Language-Action Models & Robot Foundation Models | frontier | planned | design needed |
| Robot Sensor Calibration, Units & Time Synchronization | foundation | planned | brief |
| Robot Actuators, Transmission, Limits & Saturation | foundation | planned | brief |
| Robot Jacobians, Singularities & Differential Kinematics | intermediate | planned | brief |
| Bayesian Robot Localization, Sensor Fusion & Observability | intermediate | planned | brief |
| Graph-Based SLAM, Data Association & Loop Closure | intermediate | planned | brief |
| Configuration Space, Collision Checking & Feasible Trajectories | intermediate | planned | brief |
| Contact Mechanics, Force Control & Grasp Stability | intermediate | planned | brief |
| ROS 2 Nodes, Messages, Transforms & Lifecycle | foundation | planned | brief |
| Robot Integration, Safety Cases & Hardware-in-the-Loop Validation | intermediate | planned | brief |
| Nonlinear Control, Lyapunov Stability & Regions of Attraction | advanced | planned | brief |
| Robust Control, Disturbance Rejection & Uncertainty Models | advanced | planned | brief |
| Legged Locomotion, Hybrid Dynamics & Contact Transitions | advanced | planned | brief |
| Tactile Sensing, Slip Detection & Contact-State Estimation | intermediate | planned | brief |
| Autonomous Navigation & Driving: Closed-Loop Evaluation | advanced | planned | brief |

### Drosophila & Fly Embodiment

**Module anchor:** A bounded fly behavior linked to biomechanics, sensory feedback and biological measurements.

**Teaching strategy:** Specify task, environment and observable success → Introduce frames, units, state, observations and actions → Trace sensing through estimation and decision to actuation → Explain feedback, delay, noise and constraints → Compare trajectories and diagnose a failure → Validate in changed environments and bound physical claims.

**Practice:** Predict response to a change, repair a frame/delay error, reproduce a bounded simulation and evaluate transfer.

**Verification:** Record seeds, timestep, frames, units, actuator limits and termination rules; distinguish simulation from physical evidence.

| Topic | Level | Content | Design |
| --- | --- | --- | --- |
| Drosophila Biology, Nervous System & Biomechanics | foundation | planned | design needed |
| NeuroMechFly, FlyGym & Flybody | intermediate | planned | design needed |
| MuJoCo & Gymnasium-Based Fly Environments | intermediate | planned | design needed |
| Virtual Terrains, Tasks & Compelling Demonstrations | intermediate | planned | design needed |
| Walking, Turning, Escape & Sensory-Guided Behaviour | intermediate | planned | design needed |
| Motor Control, Sensory Feedback & Closed-Loop Behaviour | advanced | planned | design needed |
| Biomechanical Constraints & Comparing Simulated with Real Behaviour | advanced | planned | design needed |
| Experiment Logging, Configuration & Reproducible Workflows | foundation | planned | design needed |
| Scientific Visualization, Video Rendering & Behavioural Trajectories | intermediate | planned | design needed |
| From Fly Embodiment to Virtual Mouse Systems | frontier | planned | design needed |
| Fly Model Calibration, System Identification & Validation | intermediate | planned | brief |
| Neural-to-Mechanical Coupling & Closed-Loop Fly Experiments | intermediate | planned | brief |

