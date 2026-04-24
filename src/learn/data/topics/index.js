import { trackDefinitions } from "../track-definitions";

// --- Custom content imports (topics with rich JSX content + visualizers) ---
import attentionContent from "./attention";
import backpropContent from "./backprop";
import tokenizationContent from "./tokenization";
import byteLevelContent from "./byte-level-tokenization";
import vocabularyMultilingualContent from "./vocabulary-multilingual";
import multimodalContent from "./multimodal-tokenization";
import dynamicContent from "./dynamic-tokenization";
// Pre-Training section
import causalLMContent from "./causal-language-modeling";
import maskedLMContent from "./masked-language-modeling";
import dedupContent from "./data-curation-deduplication";
import scalingLawsContent from "./scaling-laws";
import curriculumContent from "./curriculum-data-mixing";
import fp8Content from "./fp8-training";
import moeContent from "./moe-training";
import multimodalPretrainingContent from "./multimodal-pretraining";
import curationPipelinesContent from "./data-curation-pipelines";
import syntheticDataContent from "./synthetic-data-pretraining";
// Post-Training & Alignment section
import sftContent from "./supervised-fine-tuning";
import rlhfContent from "./rlhf";
import dpoContent from "./dpo";
import simpoContent from "./simpo";
import pTuningContent from "./p-tuning-soft-prompts";
import grpoRlooKtoContent from "./grpo-rloo-kto";
import constitutionalAIContent from "./constitutional-ai";
import prmVsOrmContent from "./prm-vs-orm";
import rlaifIpoOrpoContent from "./rlaif-ipo-orpo";
import rlvrContent from "./rlvr";
import dapoContent from "./dapo";
import knowledgeDistillationContent from "./knowledge-distillation-llms";
import rlForReasoningContent from "./rl-for-reasoning";
// Inference Optimization section
import kvCacheContent from "./kv-cache";
import decodingStrategiesContent from "./decoding-strategies";
import constrainedDecodingContent from "./constrained-decoding";
import continuousBatchingContent from "./continuous-batching";
import queueingTheoryContent from "./queueing-theory-llm-serving";
import speculativeDecodingContent from "./speculative-decoding";
import prefixCachingContent from "./prefix-caching";
import inferenceCostContent from "./inference-cost-economics";
import testTimeComputeContent from "./test-time-compute";
import inferenceEnginesContent from "./inference-engines";
// AI Inference System Design section
import systemArchitectureContent from "./inference-system-architecture";
import routingLBContent from "./request-routing-load-balancing";
import autoscalingGPUContent from "./autoscaling-gpu";
import disaggregatedPDContent from "./disaggregated-prefill-decode";
import cachingStrategiesContent from "./caching-strategies";
import multiModelServingContent from "./multi-model-serving";
import rateLimitingContent from "./rate-limiting";
import guardrailsContent from "./guardrails";
import observabilityContent from "./observability-llm";
import streamingSSEContent from "./streaming-sse";
import costOptimizationContent from "./cost-optimization-tco";
import edgeOnPremiseContent from "./edge-on-premise";
import multiRegionContent from "./multi-region-global";
// Long Context & Retrieval section
import contextWindowExtensionContent from "./context-window-extension";
import ragContent from "./rag";
import embeddingsVectorDBContent from "./embedding-models-vector-db";
import graphragAgenticContent from "./graphrag-agentic";
import modelMergingContent from "./model-merging";
import hybridSearchContent from "./hybrid-search";
// Classical Machine Learning — Supervised Learning section
import linearLogisticRegressionContent from "./linear-logistic-regression";
import decisionTreesRandomForestsContent from "./decision-trees-random-forests";
import kNearestNeighborsContent from "./k-nearest-neighbors-knn";
import gradientBoostedTreesContent from "./gradient-boosted-trees-xgboost-lightgbm-catboost";
import supportVectorMachinesContent from "./support-vector-machines-svm";
import naiveBayesContent from "./naive-bayes-probabilistic-classifiers";
import ensembleMethodsContent from "./ensemble-methods-stacking";
import recommenderSystemsContent from "./recommender-systems-collaborative-filtering-matrix-factorization";
import multiLabelContent from "./multi-label-multi-output-learning";
import survivalAnalysisContent from "./survival-analysis-cox-regression-kaplan-meier-hazard-models";
// Classical Machine Learning — Unsupervised Learning section
import kMeansHierarchicalContent from "./k-means-hierarchical-clustering";
import pcaContent from "./pca-dimensionality-reduction";
import clusteringEvaluationContent from "./clustering-evaluation-validation-silhouette-ari-nmi";
import dbscanContent from "./dbscan-density-based-clustering";
import anomalyDetectionContent from "./anomaly-outlier-detection-isolation-forest-one-class-svm-lof";
import gmmContent from "./gaussian-mixture-models-gmm-em-algorithm";
import tsneUmapContent from "./t-sne-umap-manifold-learning";
import icaContent from "./independent-component-analysis-ica";
import nmfContent from "./non-negative-matrix-factorization-nmf";
// Classical Machine Learning — Feature Engineering & Model Selection section
import featureScalingContent from "./feature-scaling-encoding-imputation";
import crossValidationContent from "./cross-validation-hyperparameter-tuning";
import regularizationContent from "./regularization-l1-l2-elastic-net-dropout";
import featureSelectionContent from "./feature-selection-importance-shap-permutation-mutual-info";
import biasVarianceContent from "./bias-variance-tradeoff-learning-curves";
import imbalancedLearningContent from "./imbalanced-learning-smote-cost-sensitive-learning";
import automlContent from "./automl-neural-architecture-search-nas";
// Classical Machine Learning — Probabilistic & Graphical Models section
import hmmContent from "./hidden-markov-models-hmm";
import crfContent from "./conditional-random-fields-crf";
import gpContent from "./gaussian-processes-gp";
import bayesianNetworksContent from "./bayesian-networks-causal-graphical-models";
// Classical Machine Learning — Semi-Supervised & Self-Training section
import semiSupervisedContent from "./semi-supervised-learning-label-propagation-self-training-co-training";
import activeLearningContent from "./active-learning";
// Classical Machine Learning — Learning Theory & Evaluation section
import evaluationMetricsContent from "./evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae";
import pacVcContent from "./pac-learning-vc-dimension";
import calibrationConformalContent from "./calibration-conformal-prediction";
import rademacherContent from "./rademacher-complexity-generalization-bounds";
// Deep Learning Fundamentals & Architectures — Core Concepts
import normalizationContent from "./batch-layer-group-rms-normalization";
import dropoutDroppathContent from "./dropout-droppath-stochastic-depth";
import transferLearningContent from "./transfer-learning-fine-tuning-strategies";
import weightInitContent from "./weight-initialization-xavier-kaiming-p";
import residualConnectionsContent from "./residual-connections-skip-connections";

// Map custom content by the slugified title they correspond to in track-definitions
const customContent = {
  "attention-mechanism-bahdanau-luong": attentionContent,
  "backpropagation-automatic-differentiation": backpropContent,
  "byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram": tokenizationContent,
  "byte-level-tokenization-token-free-models": byteLevelContent,
  "vocabulary-design-multilingual-tokenization": vocabularyMultilingualContent,
  "multimodal-tokenization-visual-audio-video": multimodalContent,
  "dynamic-tokenization-adat-boundlessbpe-litetoken": dynamicContent,
  // Pre-Training
  "causal-language-modeling-next-token-prediction": causalLMContent,
  "masked-language-modeling-bert-style": maskedLMContent,
  "data-curation-deduplication-minhash-bloom-filters": dedupContent,
  "scaling-laws-kaplan-chinchilla-beyond": scalingLawsContent,
  "curriculum-learning-data-mixing-strategies": curriculumContent,
  "fp8-training-low-precision-pre-training": fp8Content,
  "moe-training-expert-load-balancing": moeContent,
  "multimodal-pre-training-vision-encoders-cross-modal-alignment": multimodalPretrainingContent,
  "data-curation-pipelines-curator-models-quality-filtering": curationPipelinesContent,
  "synthetic-data-generation-for-pre-training": syntheticDataContent,
  // Post-Training & Alignment
  "supervised-fine-tuning-sft": sftContent,
  "rlhf-reinforcement-learning-from-human-feedback": rlhfContent,
  "dpo-direct-preference-optimization": dpoContent,
  "simpo-simple-preference-optimization": simpoContent,
  "p-tuning-soft-prompt-methods": pTuningContent,
  "grpo-rloo-kto-advanced-preference-methods": grpoRlooKtoContent,
  "constitutional-ai-cai": constitutionalAIContent,
  "process-reward-models-prm-vs-outcome-reward-models-orm": prmVsOrmContent,
  "rlaif-ipo-orpo-emerging-alignment-methods": rlaifIpoOrpoContent,
  "rlvr-reinforcement-learning-with-verifiable-rewards": rlvrContent,
  "dapo-dynamic-adaptive-policy-optimization": dapoContent,
  "knowledge-distillation-for-llms-deepseek-r1-distill-cot-distillation": knowledgeDistillationContent,
  "rl-for-reasoning-deepseek-r1-style": rlForReasoningContent,
  // Inference Optimization
  "kv-cache-memory-management": kvCacheContent,
  "decoding-strategies-greedy-beam-top-k-top-p-temperature": decodingStrategiesContent,
  "structured-output-constrained-decoding-outlines-xgrammar": constrainedDecodingContent,
  "continuous-batching-pagedattention": continuousBatchingContent,
  "queueing-theory-for-llm-serving": queueingTheoryContent,
  "speculative-decoding": speculativeDecodingContent,
  "prefix-caching-prompt-caching": prefixCachingContent,
  "inference-cost-economics-compute-scaling": inferenceCostContent,
  "test-time-compute-scaling": testTimeComputeContent,
  "inference-engines-serving": inferenceEnginesContent,
  // AI Inference System Design
  "inference-system-architecture-end-to-end": systemArchitectureContent,
  "request-routing-load-balancing": routingLBContent,
  "autoscaling-gpu-resource-management": autoscalingGPUContent,
  "disaggregated-prefill-decode": disaggregatedPDContent,
  "caching-strategies-semantic-exact-kv-cache-sharing": cachingStrategiesContent,
  "multi-model-serving-model-routing": multiModelServingContent,
  "rate-limiting-quota-management-fairness": rateLimitingContent,
  "guardrails-input-output-filtering-safety-layers": guardrailsContent,
  "observability-llm-monitoring": observabilityContent,
  "streaming-server-sent-events-sse": streamingSSEContent,
  "cost-optimization-tco-analysis": costOptimizationContent,
  "edge-on-premise-deployment-architectures": edgeOnPremiseContent,
  "multi-region-global-inference-infrastructure": multiRegionContent,
  // Long Context & Retrieval
  "context-window-extension-rope-scaling-yarn-ntk-aware": contextWindowExtensionContent,
  "retrieval-augmented-generation-rag": ragContent,
  "embedding-models-vector-databases": embeddingsVectorDBContent,
  "graphrag-agentic-rag": graphragAgenticContent,
  "model-merging-ties-dare-model-soups-slerp": modelMergingContent,
  "hybrid-search-dense-sparse-reranking": hybridSearchContent,
  // Classical Machine Learning — Supervised Learning
  "linear-logistic-regression": linearLogisticRegressionContent,
  "decision-trees-random-forests": decisionTreesRandomForestsContent,
  "k-nearest-neighbors-knn": kNearestNeighborsContent,
  "gradient-boosted-trees-xgboost-lightgbm-catboost": gradientBoostedTreesContent,
  "support-vector-machines-svm": supportVectorMachinesContent,
  "naive-bayes-probabilistic-classifiers": naiveBayesContent,
  "ensemble-methods-stacking": ensembleMethodsContent,
  "recommender-systems-collaborative-filtering-matrix-factorization": recommenderSystemsContent,
  "multi-label-multi-output-learning": multiLabelContent,
  "survival-analysis-cox-regression-kaplan-meier-hazard-models": survivalAnalysisContent,
  // Classical Machine Learning — Unsupervised Learning
  "k-means-hierarchical-clustering": kMeansHierarchicalContent,
  "pca-dimensionality-reduction": pcaContent,
  "clustering-evaluation-validation-silhouette-ari-nmi": clusteringEvaluationContent,
  "dbscan-density-based-clustering": dbscanContent,
  "anomaly-outlier-detection-isolation-forest-one-class-svm-lof": anomalyDetectionContent,
  "gaussian-mixture-models-gmm-em-algorithm": gmmContent,
  "t-sne-umap-manifold-learning": tsneUmapContent,
  "independent-component-analysis-ica": icaContent,
  "non-negative-matrix-factorization-nmf": nmfContent,
  // Classical Machine Learning — Feature Engineering & Model Selection
  "feature-scaling-encoding-imputation": featureScalingContent,
  "cross-validation-hyperparameter-tuning": crossValidationContent,
  "regularization-l1-l2-elastic-net-dropout": regularizationContent,
  "feature-selection-importance-shap-permutation-mutual-info": featureSelectionContent,
  "bias-variance-tradeoff-learning-curves": biasVarianceContent,
  "imbalanced-learning-smote-cost-sensitive-learning": imbalancedLearningContent,
  "automl-neural-architecture-search-nas": automlContent,
  // Classical Machine Learning — Probabilistic & Graphical Models
  "hidden-markov-models-hmm": hmmContent,
  "conditional-random-fields-crf": crfContent,
  "gaussian-processes-gp": gpContent,
  "bayesian-networks-causal-graphical-models": bayesianNetworksContent,
  // Classical Machine Learning — Semi-Supervised & Self-Training
  "semi-supervised-learning-label-propagation-self-training-co-training": semiSupervisedContent,
  "active-learning": activeLearningContent,
  // Classical Machine Learning — Learning Theory & Evaluation
  "evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae": evaluationMetricsContent,
  "pac-learning-vc-dimension": pacVcContent,
  "calibration-conformal-prediction": calibrationConformalContent,
  "rademacher-complexity-generalization-bounds": rademacherContent,
  // Deep Learning Fundamentals & Architectures — Core Concepts
  "batch-layer-group-rms-normalization": normalizationContent,
  "dropout-droppath-stochastic-depth": dropoutDroppathContent,
  "transfer-learning-fine-tuning-strategies": transferLearningContent,
  "weight-initialization-xavier-kaiming-p": weightInitContent,
  "residual-connections-skip-connections": residualConnectionsContent,
};

// --- Slugify: title → URL-safe ID ---
export function slugify(str) {
  return str
    .toLowerCase()
    .replace(/[()]/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

// --- Process all track definitions into flat topic objects ---
const topicMapBuilder = {};
let globalOrder = 0;

for (const track of trackDefinitions) {
  for (const section of track.sections) {
    for (const topicDef of section.topics) {
      const title = typeof topicDef === "string" ? topicDef : topicDef.title;
      const level = typeof topicDef === "string" ? "foundation" : (topicDef.level || "foundation");
      const id = slugify(title);
      if (!topicMapBuilder[id]) {
        const custom = customContent[id];
        topicMapBuilder[id] = {
          id,
          title: custom ? custom.title : title,
          category: track.id,
          level,
          section: section.name,
          readTime: custom ? custom.readTime : "5 min",
          order: globalOrder++,
          content: custom ? custom.content : null, // null = use PlaceholderContent
        };
      }
    }
  }
}

// Map of all topics keyed by ID for O(1) lookup
export const topicMap = topicMapBuilder;

// All topics sorted by global order (tracks in definition order, topics in section order)
export const allTopicsOrdered = Object.values(topicMap).sort((a, b) => a.order - b.order);

// All topic IDs in master sequence
export const allTopicIds = allTopicsOrdered.map((t) => t.id);

// All categories (track IDs with labels)
export const categories = trackDefinitions.map((t) => ({
  id: t.id,
  label: t.title,
}));
