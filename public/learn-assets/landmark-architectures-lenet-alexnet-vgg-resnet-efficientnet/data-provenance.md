# Architecture investigation: data and calculation provenance

Prepared 2026-09-13; implementation replay 2026-09-22. The complete twelve-fit experiment was executed again and every saved value matches the conserved preparation record exactly. The website now exposes the stated computations, recorded results and deferred feature-map/code downloads. `native-verification.json` records the executed environment and bounded implementation checks. The dataset and experimental protocol below remain unchanged.

## Real dataset and transformation

The original dataset is **Optical Recognition of Handwritten Digits**, E. Alpaydin and C. Kaynak,1998, [UCI record](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [DOI10.24432/C50P49](https://doi.org/10.24432/C50P49). The inspected UCI record distributes it under **CC BY4.0**. Preserve attribution and this transformation notice when redistributing the subset.

The supplied digits-400.csv is the unchanged subset already retained in the preceding convolution packet: the first40 occurrences of each digit0–9 in scikit-learn load_digits, concatenated by label. Scikit-learn's1,797-row copy comes from the historical UCI optical-digit test partition, not MNIST. Current dataset/API description rechecked2026-09-13; the unchanged local subset was reused rather than downloading a second copy.

Columns: source_id is the one-based row in that sklearn copy; pixel_0…pixel_63 are row-major8×8 integer intensities in0…16; digit is the label0…9. Source CSV61,444bytes, SHA-256 **a5b50ff0418e2b470140153a399c9200b2bba68468232507fd393ba89c4fc672**.

Fresh pre-fit inspection found400 distinct source IDs and400 distinct complete pixel vectors among400 rows. No exact duplicate row/image crosses the split. These records do not expose writer identities; uniqueness does not prove writer independence or exclude related handwriting. The development question is recognition of held-out images from this selected corpus, not independent-writer/general deployment reliability.

Divide pixels by the known intensity bound16; stratified train_test_split on row indices with test_size.3,random_state22, yielding280 training and120 development rows. The full source-ID arrays are retained in calculated-inputs.json. No learned transform uses development data. This is a teaching split inside a historical benchmark subset; it is not the benchmark's original train/test experiment and has no untouched final test.

## Actually executed training

architecture-experiments.py ran12 CPU fits: seeds1,2,3 × plain,residual,parallel,inverted_gated. Python3.12.14; PyTorch2.14.0+cpu; NumPy2.3.5; scikit-learn1.9.1; one PyTorch thread. Shared scratch/lesson-tools runtime used read-only, with no installation, pretrained download, CUDA execution, timing comparison or production benchmark.

All models share stem Conv1→12 k3p1/ReLU/maxpool2, body12×4×4→12×4×4, tail Conv12→16 k3p1/ReLU/maxpool2, spatial mean, linear16→10. Common stem/tail/head are constructed before variant body so their initial tensors match within seed. Plain/residual also share branch initialization; the extra addition is their only forward-rule difference. Other families change topology, nonlinearity, parameter count and operators together.

Plain/residual body: two12→12 k3p1 convolutions, intermediate ReLU; final ReLU on branch alone or branch+x. Parallel body:1×1;1×1→ReLU→3×3;1×1→ReLU→5×5;maxpool3s1p1→1×1, each3 output channels, concatenate+ReLU. Inverted:12→36 1×1/SiLU, depthwise36 k3p1/SiLU, mean→linear36→9/ReLU→linear9→36/sigmoid gate, multiply maps, linear1×1 36→12, add input. No BatchNorm in these declared teaching bodies; they are not benchmark architecture reproductions.

Adam.003,400 full-batch updates; no weight decay, dropout, augmentation, early stopping or selected checkpoint. Training/development CE and correct/count recorded in eval/no_grad at steps0,1,25,100,200,400. All final training counts280/280. Full12-run results and layer costs preserved in calculated-inputs.json; manuscript reports all seeds rather than selecting a winner's favorable seed.

Parameter totals plain/residual4650,parallel2502,inverted3999. Conv/linear MACs/image76192,41920,54376 respectively. Instrumentation counts each Conv2d/Linear output element times terms in one weight row/filter. This excludes activation, bias addition, pooling, residual addition, SE map multiplication and other work. No runtime ranking follows from the counts.

Seed1 final development correct/CE: plain116/.1179668084,residual112/.2065355033,parallel119/.0367311090,inverted116/.1970604211. Seed2:115/.2325255871,115/.2073657811,117/.1056078598,112/.3330711424. Seed3:112/.3060728908,114/.3737169802,115/.2091468275,117/.0928347632.

## Saved real visual inputs

For each seed1 model, retain the first development row (source251,actual4,predicted4 in all), plus its first misclassification: plain97 actual8/pred5; residual104 actual3/pred7; parallel379 actual8/pred5; inverted310 actual8/pred0. The second observation is explicitly selected to show a failure, not a representative sample.

For these eight observations: original scaled8×8 input,16 final2×2 feature maps,10 logits,10 probabilities,all10 class activation maps. Each seed1 model additionally retains its final10×16 head weights and10 biases. Float32 mean(CAM)+bias versus head(mean(features)) maximum discrepancy across saved cases5.7220458984375e−6.

The **whole trained backbone is not saved**. These inputs support exact head/feature/CAM investigations and recorded observations, not arbitrary edited-image inference, other seeds' feature maps, or a claim that a kernel picture reconstructs every activation. Running the complete program is the route to new fitted outputs. A feature-space edit is a constructed intervention.

## Exact calculations and independent bounded checks

calculated-inputs.json contains parameter/MAC budgets, VGG component sums and storage estimate, explicit dense/pointwise/depthwise counts, ResNet bottleneck/Inception reduction/SE counts, compound coefficient powers, CAM signed-cell edits/permutation/zero cases, context gates, DenseNet channel-growth counts and changed arithmetic exercises.

author-checks.py separately counts Linear parameters using PyTorch meta tensors (no large weight allocation), recomputes every saved class logit via direct Python loops over feature cells, checks a changed class-map problem, cell-based gate edits and permutation/equal-shift nulls, budget eligibility endpoints, and scaling allocations. It does not call the original CAM helper to “independently” verify it.

The independent Python-double CAM route versus saved float32 logits has maximum difference7.099608438920768e−6, under1e−5. This compares evaluation orders/precisions of the same algebraic identity, not a general guarantee about CAM interpretation. Exact direct/GAP head equality at1×1, changed43,911/903 head case, class-map score0→−1, gate mean-difference invariances and four-candidate budget boundaries passed. Results are retained in author-check-results.json.

The twofold scaling budget with d1.5,r1.25 requires w≈.92376, below this investigation's bound1. The specification correctly calls it infeasible instead of clamping. At a threefold budget w=√1.28≈1.13137 is feasible. No optimization or accuracy surface was simulated.

Initial fit outputs were generated once; no fit was repeated for later text/spec repairs. The bounded author-check program runs only calculations/meta tensors and saved-result reconciliation. No disposable data copies, caches, screenshots or source archives are needed beyond the packet.

## Reproduction and current boundaries

The transparent named-family builders are in `landmark_builders.py`. In separate CPU invocations, `--family resnet18 --compare`, `--family efficientnet_b0 --compare`, `--family vgg16 --compare` and `--family alexnet --compare` each matched the corresponding Torchvision model at maximum absolute logit difference 0, after copying all component state. The environment was PyTorch 2.14.0+cpu and Torchvision 0.29.0+cpu. This checks architecture computation with matched random weights, not training quality. LeNet has no Torchvision counterpart and is explicitly the declared modern variant; all five constructors passed shape/count checks. Large comparisons allocate both full models and were run separately.

The reusable mobile block rejects invalid drop probabilities and handles probability 1 without division by zero. The default EfficientNet-B0 schedule remains unchanged. A seven-class GAP head modification was also checked at two image sizes.

The photograph/pretrained ResNet example was not executed: no external photograph or pretrained weights were downloaded. No hardware timing, full historical ImageNet training, or architecture-wide performance claim is supplied. The browser performs only small arithmetic or reads the stated saved observations; it does not train these networks.
