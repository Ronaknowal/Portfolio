// Complete revision-3 manuscript rendered statically; changes recorded in the implementation record.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ConvNeXtGenealogy, ConvNeXtBlockFigure, ConvNeXtNormalizationLab, ConvNeXtHierarchy, ConvNeXtBudgetLab, ConvNeXtResponseLab, ConvNeXtMaskFigure, ConvNeXtReconstructionLab, ConvNeXtRecordedExperiment, ConvNeXtFusionLab, ConvNeXtProgram, convnextAsset } from '../../components/lesson-labs/ConvNeXtLabs.jsx';
export default {
 title: 'ConvNeXt & Modern CNN Designs',
 readTime: '~70 min read + live investigations, implementation and practice',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson convnext-lesson"><LessonIntro prerequisites="Depthwise convolution, channel normalization, residual paths, tensor shapes and supervised training. The relevant axis and masking contracts are refreshed locally." sections={[["1-start-with-the-comparison-not-the-model-name","1. Start with the comparison, not the model name"],["2-read-one-block-from-the-input-outward","2. Read one block from the input outward"],["3-from-the-block-to-a-feature-hierarchy","3. From the block to a feature hierarchy"],["4-global-response-normalization-look-across-the-feature-map","4. Global response normalization: look across the feature map"],["5-learn-from-missing-pixels-without-giving-away-the-answer","5. Learn from missing pixels without giving away the answer"],["6-an-actual-masked-digit-experiment","6. An actual masked-digit experiment"],["7-deeper-routes-deploy-reparameterize-or-combine-mechanisms","7. Deeper routes: deploy, reparameterize, or combine mechanisms"],["match-the-block-you-built-to-the-maintained-implementation","Match the block you built to the maintained implementation"],["8-practice-reason-about-a-changed-design","8. Practice: reason about a changed design"],["9-readiness-connections-and-other-ways-to-learn","9. Readiness, connections and other ways to learn"]]}>Read a modern convolutional block, build its complete hierarchy, and investigate how a masked reconstruction model uses available evidence.</LessonIntro>
<Prose>{""}<strong>{"Explore as you read."}</strong>{" Change stage dimensions, normalization groups, GRN feature cells, valid visible-patch selections and branch-folding coefficients. Show parameter counts, shared GRN denominator, changed feature maps, reconstruction consequences and folded-kernel equality immediately. Preserve image masking as the learning objective, not UI answer hiding. The labs show current results as you work; you do not enter or submit a guess. Use those comparisons to separate architecture from recipe, global channel context from local normalization, and valid reparameterization from a changed function."}</Prose>

<Prose>{"A visual model has two jobs inside each layer: combine nearby evidence and combine different kinds of evidence. A dark curve beside a vertical stroke is a spatial relationship. Combining “curve,” “stroke” and “enclosed region” detectors is a channel relationship. ConvNeXt organizes these jobs into a small repeated block, then asks an equally important question: how much of a model's success comes from its architecture, and how much comes from how it was trained?"}</Prose>

<Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/depthwise-separable-dilated-convolutions?module=deep-learning-fundamentals"}>{"Depthwise Separable & Dilated Convolutions"}</a>{" lesson explained inexpensive spatial filtering and its restrictions. Here we use those operations to read a complete modern network. We will also train a small model to reconstruct hidden parts of real handwritten digits, then test what its representation makes available to a separate classifier."}</Prose>

<Prose>{""}<strong>{"First pass:"}</strong>{" follow §§1–6, run or inspect the small experiment, and attempt practice 1–5. You should be able to trace one block, identify which numbers a normalizer sees, prevent hidden-input leakage, and distinguish reconstruction from recognition. The deployment, kernel-fusion and hybrid-design branches in §7 and practice 6–8 deepen that understanding; they are not prerequisites for the next lesson."}</Prose>

<H2>{"1. Start with the comparison, not the model name"}</H2>

<Prose>{"Suppose one model gets more examples right than another. Before explaining the difference using their diagrams, ask whether they used the same images, labels, input resolution, training duration, augmentation, optimizer, regularization and evaluation procedure. Changing several of these changes the question."}</Prose>

<Prose>{"An "}<strong>{"architecture"}</strong>{" defines the function and its trainable parameters. A "}<strong>{"training recipe"}</strong>{" defines how those parameters are obtained. A "}<strong>{"checkpoint"}</strong>{" is one resulting set of parameter values. ConvNeXt is a useful case study because its authors first strengthened the training procedure for an existing ResNet, then changed the architecture in stages. The old and enhanced ResNet training runs reported 76.1% and 78.8% ImageNet top-1 accuracy, respectively. That improvement happened before introducing the final ConvNeXt block. "}<a href={"https://arxiv.org/pdf/2201.03545"}>{"Original study, §2.1"}</a>{""}</Prose>

<Prose>{"The recipe included longer training, AdamW, image transformations, mixed training examples and regularization. You do not need to memorize their hyperparameters to understand the evidence: a newer architecture compared with an older training recipe does not isolate the value of the architecture."}</Prose>

<Prose>{"Some individual architecture steps also made the intermediate model worse. Replacing spatial convolutions with depthwise ones reduced computation but initially reduced accuracy; widening the channels recovered capacity. Moving the depthwise filter before expansion initially hurt accuracy; increasing its spatial extent helped in that resulting configuration. These are conditional experiments, not universal bonuses that can be added to any network."}</Prose>

<ConvNeXtGenealogy />

<Prose>{"A practical comparison record can be short:"}</Prose>

<NeuralTable caption={"1. Start with the comparison, not the model name"} headers={[<>{"Question"}</>,<>{"What must be written down"}</>]} rows={[[<>{"What is being predicted?"}</>,<>{"Label definition, unit of observation and decision metric"}</>],[<>{"What may differ?"}</>,<>{"The specific block or training change being investigated"}</>],[<>{"What is held fixed?"}</>,<>{"Data partition, preprocessing, training budget and paired seeds where possible"}</>],[<>{"What did it cost?"}</>,<>{"Parameters and defined operation counts; separately measured latency if available"}</>],[<>{"What supports the conclusion?"}</>,<>{"Counts/errors on reserved development data, variability and the actual comparison"}</>]]} />

<Prose>{"Our small experiment will use this structure. It will not use an ImageNet result to predict an accuracy gain on digits."}</Prose>

<H2>{"2. Read one block from the input outward"}</H2>

<Prose>{"A feature tensor contains a batch of images, spatial positions and channels. Write its shape as "}<InlineMath>{"N\\times C\\times H\\times W"}</InlineMath>{": specimens, channels, rows, columns. At one location, its "}<InlineMath>{"C"}</InlineMath>{" numbers summarize different learned features. A "}<strong>{"depthwise"}</strong>{" filter looks at neighbors within each channel; a "}<strong>{"pointwise"}</strong>{" transformation mixes channels at one location."}</Prose>

<Prose>{"ConvNeXt V1 uses the following residual branch:"}</Prose>

<NeuralTable caption={"2. Read one block from the input outward"} headers={[<>{"Operation"}</>,<>{"Logical output shape"}</>,<>{"What changes"}</>]} rows={[[<>{"Input "}<InlineMath>{"x"}</InlineMath>{""}</>,<>{""}<InlineMath>{"N,C,H,W"}</InlineMath>{""}</>,<>{"Starting evidence"}</>],[<>{"Depthwise 7×7, padding 3"}</>,<>{""}<InlineMath>{"N,C,H,W"}</InlineMath>{""}</>,<>{"Each channel gathers its local spatial neighborhood"}</>],[<>{"Channel LayerNorm"}</>,<>{""}<InlineMath>{"N,H,W,C"}</InlineMath>{" in the code"}</>,<>{"The channel vector is standardized separately at each location"}</>],[<>{"Linear "}<InlineMath>{"C\\rightarrow4C"}</InlineMath>{""}</>,<>{""}<InlineMath>{"N,H,W,4C"}</InlineMath>{""}</>,<>{"Features are mixed into a wider set of combinations"}</>],[<>{"GELU"}</>,<>{""}<InlineMath>{"N,H,W,4C"}</InlineMath>{""}</>,<>{"A smooth nonlinear response changes which combinations contribute"}</>],[<>{"Linear "}<InlineMath>{"4C\\rightarrow C"}</InlineMath>{""}</>,<>{""}<InlineMath>{"N,H,W,C"}</InlineMath>{""}</>,<>{"The expanded features are projected back"}</>],[<>{"Learned channel scale, then DropPath"}</>,<>{""}<InlineMath>{"N,C,H,W"}</InlineMath>{""}</>,<>{"Branch contribution is scaled and optionally masked during training"}</>],[<>{"Add "}<InlineMath>{"x"}</InlineMath>{""}</>,<>{""}<InlineMath>{"N,C,H,W"}</InlineMath>{""}</>,<>{"Existing evidence plus the learned correction"}</>]]} />

<Prose>{"The two linear layers are applied independently at every spatial location with shared weights. They are equivalent to 1×1 convolutions with the same parameters. The permutation changes where the channel axis appears in the tensor interface; it does not mix information between locations."}</Prose>

<ConvNeXtBlockFigure />

<Prose>{"Writing the two channel transformations together, the projected vector at one location is"}</Prose>

<div className="neural-equation"><MathBlock>{"v_j=\\sum_{k=1}^{4C} W^{(2)}_{jk}\\,\\operatorname{GELU}\n\\left(\\sum_{i=1}^{C}W^{(1)}_{ki}\\,\\widehat{x}_i+b^{(1)}_k\\right)+b^{(2)}_j."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"\\widehat{x}"}</InlineMath>{" is the normalized output of the spatial filter. The inner sum mixes features; the nonlinearity prevents the two linear transformations from collapsing into a single fixed linear map. GELU is "}<InlineMath>{"z\\Phi(z)"}</InlineMath>{", where "}<InlineMath>{"\\Phi"}</InlineMath>{" is the standard normal cumulative distribution function. Small negative values can contribute, large positive values mostly pass through, and the response is smooth. This does not make GELU a guarantee of better training than ReLU."}</Prose>

<H3>{"Which values does LayerNorm actually normalize?"}</H3>

<Prose>{"For one specimen and one location, calculate the mean and variance across its "}<InlineMath>{"C"}</InlineMath>{" channels:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mu=\\frac1C\\sum_c x_c,\\qquad\n\\sigma^2=\\frac1C\\sum_c(x_c-\\mu)^2,\\qquad\n\\widehat{x}_c=\\gamma_c\\frac{x_c-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}+\\beta_c."}</MathBlock></div>

<Prose>{"Each location has its own "}<InlineMath>{"\\mu,\\sigma^2"}</InlineMath>{". Learned "}<InlineMath>{"\\gamma,\\beta"}</InlineMath>{" are shared across locations. The implementation uses "}<InlineMath>{"\\epsilon=10^{-6}"}</InlineMath>{" to keep constant channel vectors well-defined. These are current-input statistics in both training and evaluation; there is no BatchNorm running average."}</Prose>

<Prose>{"Consider two locations with channel vectors "}<InlineMath>{"[1,3]"}</InlineMath>{" and "}<InlineMath>{"[101,103]"}</InlineMath>{". Channel LayerNorm maps both to approximately "}<InlineMath>{"[-1,1]"}</InlineMath>{" before its learned affine transformation. A single GroupNorm group instead uses all four values from the specimen; it preserves the large offset between locations in its normalized output. Replacing one operation with the other changes the function even if the output shapes agree."}</Prose>

<ConvNeXtNormalizationLab />

<H3>{"Why the spatial filter comes before expansion"}</H3>

<Prose>{"A 7×7 depthwise filter at width "}<InlineMath>{"C"}</InlineMath>{" uses "}<InlineMath>{"49C"}</InlineMath>{" weights. Moving that filter after a fourfold expansion uses "}<InlineMath>{"196C"}</InlineMath>{". Both designs can be useful, but the latter spends more spatial-filter work at the expanded width. ConvNeXt spends most of its arithmetic on the dense channel transformations."}</Prose>

<Prose>{"With biases, channel LayerNorm and one learned LayerScale vector, a V1 block has"}</Prose>

<div className="neural-equation"><MathBlock>{"(49C+C)+(2C)+(4C^2+4C)+(4C^2+C)+C\n=8C^2+58C"}</MathBlock></div>

<Prose>{"parameters. At "}<InlineMath>{"C=96"}</InlineMath>{", that is 79,296; the two pointwise weight matrices contain 73,728 of them, about 93%. Biases and normalization parameters are small, but counting them correctly matters when checking an implementation."}</Prose>

<Prose>{"Ignoring bias additions, normalization, activation and the residual addition, its convolution/linear work is"}</Prose>

<div className="neural-equation"><MathBlock>{"HW(49C+8C^2)\\quad\\text{MACs per specimen}."}</MathBlock></div>

<Prose>{"One MAC here means one product accumulated into a sum. This is not a wall-clock measurement or a count of every floating-point operation."}</Prose>

<H3>{"Residual scaling and dropping have distinct jobs"}</H3>

<Prose>{"LayerScale learns one multiplier per output channel, initially "}<InlineMath>{"10^{-6}"}</InlineMath>{" in the V1 reference implementation. It starts the residual branch contribution very small. The block is therefore close to its input initially; the whole network is not an identity map, because its stem, downsampling and head still change shapes and values. Learned scales need not remain positive or small."}</Prose>

<Prose>{"DropPath draws one branch mask per specimen, shared across channels and positions. With drop probability "}<InlineMath>{"p"}</InlineMath>{", it returns the branch divided by "}<InlineMath>{"1-p"}</InlineMath>{" when retained and zero when dropped. Evaluation uses the whole branch. This preserves the expected branch contribution for fixed inputs, not the expected final prediction of an arbitrary nonlinear network."}</Prose>

<Prose>{"In the supplied code the branch is calculated before applying the mask. Dropping its contribution does "}<strong>{"not"}</strong>{" automatically save its computation. For 18 blocks whose drop probabilities range linearly from 0 to 0.1, the expected number of retained contributions is 17.1, while all 18 branch calculations still execute. "}<a href={"/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals"}>{"Dropout, DropPath & Stochastic Depth"}</a>{" develops the distinction between masking, expectation and execution."}</Prose>

<H2>{"3. From the block to a feature hierarchy"}</H2>

<Prose>{"ConvNeXt starts with a 4×4 stride 4 convolution. Each initial output position reads one non-overlapping 4×4 input patch, then channel LayerNorm is applied. For a 224×224 image, this produces a 56×56 feature grid."}</Prose>

<Prose>{"Four stages progressively reduce spatial resolution while increasing channel width:"}</Prose>

<NeuralTable caption={"3. From the block to a feature hierarchy"} headers={[<>{"Tiny stage"}</>,<>{"Grid"}</>,<>{"Channels"}</>,<>{"Repeated blocks"}</>,<>{"What becomes possible"}</>]} rows={[[<>{"1"}</>,<>{"56×56"}</>,<>{"96"}</>,<>{"3"}</>,<>{"Local detail represented at many positions"}</>],[<>{"2"}</>,<>{"28×28"}</>,<>{"192"}</>,<>{"3"}</>,<>{"Broader combinations at fewer positions"}</>],[<>{"3"}</>,<>{"14×14"}</>,<>{"384"}</>,<>{"9"}</>,<>{"More processing at a wider intermediate representation"}</>],[<>{"4"}</>,<>{"7×7"}</>,<>{"768"}</>,<>{"3"}</>,<>{"A compact, semantically useful feature map"}</>]]} />

<Prose>{"Between stages, channel LayerNorm precedes a 2×2 stride 2 convolution. For classification, average the final map over its two spatial axes, apply LayerNorm to the resulting 768-vector, and use a linear classifier. Averaging before versus after a nonlinear normalization is a real ordering choice; these operations generally do not commute."}</Prose>

<Prose>{"The hierarchy is useful beyond classification. A segmentation head can use fine-grid features to locate boundaries and coarse-grid features for context. A detector can attach heads to multiple scales. This explains why exposing stage outputs matters even when the original model's final head produces only one label."}</Prose>

<Prose>{"Increasing channels while decreasing area also explains stage cost. Doubling "}<InlineMath>{"C"}</InlineMath>{" and dividing "}<InlineMath>{"HW"}</InlineMath>{" by 4 leaves the leading "}<InlineMath>{"8HWC^2"}</InlineMath>{" term unchanged "}<strong>{"per block"}</strong>{". The depthwise term halves. Adding more blocks to the third stage concentrates work there. The Tiny stage depths are 3,3,9,3:18 blocks. Small, Base, Large and XLarge V1 configurations use 3,3,27,3:36 blocks."}</Prose>

<Prose>{"The attached "}<a href={"/learn-assets/convnext-modern-cnn-designs/convnext-blocks.py"}>{"complete architecture program"}</a>{" implements the block, stage transitions, residual masking, initialization, feature outputs and head. It checks large configurations using "}<strong>{"meta tensors"}</strong>{": shapes and parameter counts are represented without allocating full model weights. It also runs small actual forward/backward calculations. Its computed V1 counts, with a 1,000-class head, are:"}</Prose>

<ConvNeXtProgram file="convnext-blocks.py" title="Read the complete V1/V2 block, hierarchy, initialization and shape checks" />

<NeuralTable caption={"3. From the block to a feature hierarchy"} headers={[<>{"V1 configuration"}</>,<>{"Initial width"}</>,<>{"Parameters"}</>,<>{"Conv/linear MACs at 224²"}</>]} rows={[[<>{"Tiny"}</>,<>{"96"}</>,<>{"28,589,128"}</>,<>{"4,455,531,264"}</>],[<>{"Small"}</>,<>{"96"}</>,<>{"50,223,688"}</>,<>{"8,683,712,256"}</>],[<>{"Base"}</>,<>{"128"}</>,<>{"88,591,464"}</>,<>{"15,354,729,472"}</>],[<>{"Large"}</>,<>{"192"}</>,<>{"197,767,336"}</>,<>{"34,361,433,600"}</>],[<>{"XLarge"}</>,<>{"256"}</>,<>{"350,196,968"}</>,<>{"60,921,030,656"}</>]]} />

<Prose>{"These are calculations for the stated topology, not training results. Input resolution changes activation sizes and work; it does not change a convolution's learned kernel count. Strided sampling also means shifting an image by one pixel need not simply shift its final features. Shared convolution weights do not make an entire downsampled classifier exactly translation invariant."}</Prose>

<Prose>{"The block and the hierarchy are also separable choices. An "}<strong>{"isotropic"}</strong>{" variant keeps the same grid size and channel width through its repeated blocks, using an initial projection to establish that grid. It gives up the native four-scale output hierarchy. The original study tested such configurations too; their result asks whether the block remains useful without staged downsampling, not whether all tasks should discard multiple resolutions."}</Prose>

<ConvNeXtHierarchy /><ConvNeXtBudgetLab />

<H2>{"4. Global response normalization: look across the feature map"}</H2>

<Prose>{"Imagine two channels that produce almost the same spatial pattern. Both may vary strongly across an image, so neither is a “dead channel.” Yet they may offer redundant evidence. Conversely, a quiet channel might encode a rare useful pattern. Counting channels with nonzero variance does not measure representation quality."}</Prose>

<Prose>{"ConvNeXt V2 adds "}<strong>{"global response normalization"}</strong>{", or GRN, inside the expanded branch after GELU. It compares the spatial magnitude of each channel with the other channels in the "}<strong>{"same specimen"}</strong>{". This differs from channel LayerNorm, which compares channels separately at each location."}</Prose>

<Prose>{"For "}<InlineMath>{"X"}</InlineMath>{" with logical shape "}<InlineMath>{"N,H,W,C"}</InlineMath>{", define"}</Prose>

<div className="neural-equation"><MathBlock>{"G_{n,c}=\\sqrt{\\sum_{h,w}X_{n,h,w,c}^2},\\qquad\nR_{n,c}=\\frac{G_{n,c}}{\\frac1C\\sum_jG_{n,j}+\\epsilon},"}</MathBlock></div>

<div className="neural-equation"><MathBlock>{"Y_{n,h,w,c}=X_{n,h,w,c}\n+\\gamma_c X_{n,h,w,c}R_{n,c}+\\beta_c."}</MathBlock></div>

<Prose>{"The first reduction summarizes each whole channel map. The second compares these channel magnitudes. The result broadcasts back to every location. GRN does not subtract a spatial mean or force each channel to unit variance. The definition above matches the dense reference implementation. "}<a href={"https://github.com/facebookresearch/ConvNeXt-V2/blob/main/models/utils.py"}>{"Official GRN implementation"}</a>{""}</Prose>

<H3>{"A two-channel example you can calculate"}</H3>

<Prose>{"Take channel A's two locations as "}<InlineMath>{"[3,4]"}</InlineMath>{", channel B's as "}<InlineMath>{"[0,12]"}</InlineMath>{". Their spatial lengths are 5 and 12. Their average length is 8.5, so relative responses are approximately 0.588235 and 1.411765."}</Prose>

<Prose>{"Set "}<InlineMath>{"\\gamma_A=.5,\\gamma_B=-.5,\\beta=0"}</InlineMath>{". Channel A becomes approximately "}<InlineMath>{"[3.882353,5.176470]"}</InlineMath>{", and channel B becomes "}<InlineMath>{"[0,3.529413]"}</InlineMath>{". Now edit only B's second value from 12 to 0. A's original values have not changed, but its relative response becomes almost 2, so its output becomes almost "}<InlineMath>{"[6,8]"}</InlineMath>{"."}</Prose>

<Prose>{"That is global coupling through a statistic. It is not a new spatial convolution. It also shows why “GRN always boosts strong channels and suppresses weak ones” is misleading: the learned signs matter."}</Prose>

<ConvNeXtResponseLab />

<Prose>{"Initial identity does not mean the layer is absent from learning. For the same two maps and loss "}<InlineMath>{"L=\\frac12\\sum Y^2"}</InlineMath>{", at "}<InlineMath>{"\\gamma=\\beta=0"}</InlineMath>{","}</Prose>

<div className="neural-equation"><MathBlock>{"\\frac{\\partial L}{\\partial\\gamma_A}\\approx14.705881,\\quad\n\\frac{\\partial L}{\\partial\\gamma_B}\\approx203.294094,\\quad\n\\frac{\\partial L}{\\partial\\beta}=[7,12]."}</MathBlock></div>

<Prose>{"Those parameters can change on the first optimizer update. The initial input derivative equals "}<InlineMath>{"X"}</InlineMath>{" for this loss; afterward the learned GRN response changes the input derivative too. The "}<a href={"/learn-assets/convnext-modern-cnn-designs/author-checks.py"}>{"author calculations"}</a>{" check the scale gradients against independent central differences."}</Prose>

<Prose>{"V2 removes the V1 LayerScale and adds two GRN vectors at width "}<InlineMath>{"4C"}</InlineMath>{". Thus its block has "}<InlineMath>{"8C^2+65C"}</InlineMath>{" parameters: "}<InlineMath>{"8C"}</InlineMath>{" added and "}<InlineMath>{"C"}</InlineMath>{" removed, a net "}<InlineMath>{"7C"}</InlineMath>{". At "}<InlineMath>{"C=96"}</InlineMath>{", this is 79,968 parameters; 672 more than V1. GRN itself starts as identity; removing LayerScale does not make the "}<strong>{"whole V2 residual branch"}</strong>{" tiny."}</Prose>

<H2>{"5. Learn from missing pixels without giving away the answer"}</H2>

<Prose>{"A supervised digit classifier receives an image and its class during training. A masked reconstruction model receives only selected image regions and learns to predict the missing regions. The original image supplies a training target even when no class label is used for that training stage. This is "}<strong>{"self-supervised learning"}</strong>{": the training signal is constructed from the data."}</Prose>

<Prose>{"The central constraint is informational. If the hidden pixels enter the encoder through a convolution, normalization statistic or another route, a good reconstruction may reflect access to the answer."}</Prose>

<H3>{"Separate what is visible from what is scored"}</H3>

<Prose>{"Let "}<InlineMath>{"M"}</InlineMath>{" be 1 at visible pixels and 0 at hidden pixels. An input-masking operation forms "}<InlineMath>{"M\\odot x"}</InlineMath>{", where "}<InlineMath>{"\\odot"}</InlineMath>{" is elementwise multiplication. A hidden-pixel loss is"}</Prose>

<div className="neural-equation"><MathBlock>{"L=\\frac{\\sum_{i:M_i=0}(\\widehat{x}_i-x_i)^2}\n{\\#\\{i:M_i=0\\}}."}</MathBlock></div>

<Prose>{"The encoder sees the visible values. The loss compares predictions with the original hidden targets. Editing a hidden target therefore can change the loss without changing the prediction. This is correct behavior, not a contradiction."}</Prose>

<Prose>{"In a multistage convolutional encoder, masking the raw input alone is not the same as maintaining a fixed set of active feature locations. Convolution can write features into inactive locations; biases and channel transformations can make zero inputs nonzero. A masked-dense implementation must keep its intended active set masked at the relevant operations. A sparse implementation explicitly represents and computes on active coordinates. Their runtime costs and normalization behavior must be checked separately."}</Prose>

<Prose>{"The published "}<strong>{"fully convolutional masked autoencoder"}</strong>{", FCMAE, masks 60% of 32×32 input patches, matching the final encoder-grid granularity, and propagates that mask through the hierarchy. Its lightweight decoder receives encoded visible features and mask tokens at missing positions. Its loss uses patch-normalized hidden targets. Our small experiment below preserves the visibility/target distinction while deliberately using smaller patches and a simpler loss. "}<a href={"https://arxiv.org/pdf/2301.00808"}>{"FCMAE construction"}</a>{""}</Prose>

<ConvNeXtMaskFigure />

<H3>{"Why a second evaluation is needed"}</H3>

<Prose>{"An encoder might learn local interpolation that reconstructs textures well but does not separate object categories. To ask whether labels are accessible in its features, freeze the encoder, extract representations, and fit a small supervised classifier using training labels. A "}<strong>{"linear probe"}</strong>{" tests what a linear readout can use. It differs from fine-tuning, which updates the encoder too."}</Prose>

<Prose>{"Neither reconstruction error nor a channel-diversity statistic can substitute for that task evaluation. Even a successful probe on a small development split does not establish deployment performance."}</Prose>

<H2>{"6. An actual masked-digit experiment"}</H2>

<Prose>{"The "}<a href={"/learn-assets/convnext-modern-cnn-designs/digits-400.csv"}>{"offline CSV"}</a>{" contains 400 real 8×8 optical digit images,40 per class, drawn from UCI's Optical Recognition of Handwritten Digits dataset through scikit-learn's local copy. They are not MNIST images. Each integer pixel lies in 0–16; divide by 16 using the known scale. Preserve the "}<a href={"/learn-assets/convnext-modern-cnn-designs/data-provenance.md"}>{"dataset attribution, subset construction and split record"}</a>{"."}</Prose>

<Prose>{"The unit here is an image specimen. Before splitting, the program checks 400 unique source IDs and 400 distinct pixel vectors. Writer identifiers are unavailable, so this is not an independent-writer assessment. It reserves 120 stratified development images and trains on 280, using split seed 22. Development images are excluded even from unlabeled reconstruction training."}</Prose>

<H3>{"The small model and the controlled difference"}</H3>

<Prose>{"The input 8×8 image is divided into sixteen 2×2 patches. Exactly six are visible and ten hidden:62.5% hidden, rather than the paper's 60%. A 2×2 stride 2 stem creates a 4×4 grid with 12 channels. Two ConvNeXt-style encoder blocks use 3×3 depthwise filters, channel LayerNorm,12→48→12 channel mixing and residual addition. A one-block decoder plus a pixel head reconstructs 8×8 pixels."}</Prose>

<Prose>{"We compare two variants that differ only in whether the encoder expansion includes GRN. "}<strong>{"Neither has LayerScale"}</strong>{", and both use the same small decoder. These are paired GRN experiments, not miniature reproductions of every difference between official V1 and V2."}</Prose>

<Prose>{"The training program uses AdamW with learning rate 0.002, weight decay 0.01 and 600 full-batch updates. Weight decay applies to all parameters in this teaching experiment; this is not the paper's optimizer grouping. Each update generates a new six-visible-patch mask. Each paired seed uses identical initial shared tensors and the same mask sequence; GRN's additional scale/shift vectors start at zero."}</Prose>

<Prose>{"For evaluation, four fixed masks per specimen allow comparisons on the same missing pixels. The loss is raw normalized-pixel MSE on the 40 hidden pixels per image, not the paper's patch-normalized target loss. No augmentation, DropPath, checkpoint search or model selection is used. Steps 0,1,100,300,600 are recorded; only the declared final step supplies the comparison."}</Prose>

<H3>{"Run the complete program"}</H3>

<Prose>{"Download "}<a href={"/learn-assets/convnext-modern-cnn-designs/masked-reconstruction.py"}>{"masked-reconstruction.py"}</a>{" and "}<a href={"/learn-assets/convnext-modern-cnn-designs/digits-400.csv"}>{"digits-400.csv"}</a>{" into the same directory. In a Python environment with PyTorch, NumPy and scikit-learn installed, run:"}</Prose>

<CodeBlock language={"bash"}>{"python masked-reconstruction.py"}</CodeBlock>

<Prose>{"The script sets one PyTorch CPU thread and requires no network access or pretrained checkpoint. It contains the full model, mask generator, loss, train/development split, optimizer loop, fixed-mask evaluation and frozen-feature probe. It writes "}<a href={"/learn-assets/convnext-modern-cnn-designs/calculated-inputs.json"}>{"calculated-inputs.json"}</a>{"; the supplied copy contains the actual author run on Python 3.12.14, PyTorch 2.14.0+cpu, NumPy 2.3.5 and scikit-learn 1.9.1. Small numerical differences across library/platform versions are possible."}</Prose>

<ConvNeXtProgram file="masked-reconstruction.py" title="Read the complete masked learning, evaluation and probe program" /><Prose>To reproduce the recorded environment in a separate activated Python environment, run <code>python -m pip install torch==2.14.0 numpy==2.3.5 scipy scikit-learn==1.9.1</code>. For the separate native block bridge, also install <code>torchvision==0.29.0 pillow</code>. The large source is loaded only when you open its disclosure; the downloaded CSV runs offline.</Prose>

<Prose>{"The core masking/loss excerpt is worth reading before running the complete file:"}</Prose>

<CodeBlock language={"python"}>{"def masked_mse(predictions, targets, visible):\n    hidden_pixels = (1-visible).repeat_interleave(2,2).repeat_interleave(2,3)\n    return ((predictions-targets).square()*hidden_pixels).sum()/hidden_pixels.sum()"}</CodeBlock>

<Prose>{"Here "}<code>{"visible"}</code>{" has shape "}<InlineMath>{"N,1,4,4"}</InlineMath>{". Repeating each grid location twice along each spatial axis makes its 2×2 pixel patch share the same visibility. The numerator sums error only where the mask is hidden; the denominator is the number of those pixels. An all-visible mask would have denominator zero and is not an allowed reconstruction-loss input. All-visible images are valid for feature extraction, where this loss is not called."}</Prose>

<Prose>{"Inside the encoder, the input is masked before the stem; inactive feature positions are suppressed after spatial filtering, expansion and residual addition. The decoder can fill missing positions, as it must to predict them. "}<a href={"/learn-assets/convnext-modern-cnn-designs/author-checks.py"}>{"Independent scalar-loop checks"}</a>{" reproduce all four saved real examples to within "}<InlineMath>{"3.3\\times10^{-7}"}</InlineMath>{" of the stored outputs."}</Prose>

<H3>{"What actually happened"}</H3>

<Prose>{"A training-mean-image baseline predicts the same image regardless of visible content. Its development masked MSE is 0.071914. The learned models do better on this reconstruction criterion:"}</Prose>

<NeuralTable caption={"What actually happened"} headers={[<>{"Paired seed"}</>,<>{"GRN absent: masked MSE"}</>,<>{"GRN present: masked MSE"}</>,<>{"Frozen probe correct, absent / present"}</>]} rows={[[<>{"1"}</>,<>{".052188"}</>,<>{".052313"}</>,<>{"115/120 /116/120"}</>],[<>{"2"}</>,<>{".052336"}</>,<>{".051906"}</>,<>{"115/120 /116/120"}</>],[<>{"3"}</>,<>{".051670"}</>,<>{".051488"}</>,<>{"115/120 /116/120"}</>]]} />

<Prose>{"The probe standardizes each extracted feature using training statistics, then fits logistic regression with "}<InlineMath>{"C=1"}</InlineMath>{". It sees the clean image through a frozen encoder. A separate standardized logistic regression fitted directly to the 64 raw pixels gets 118/120 correct. All six feature probes fit the 280 training labels perfectly."}</Prose>

<Prose>{"Several conclusions now become possible, and several do not:"}</Prose>

<ul><li>{"The learned reconstructions beat this training-mean-image baseline on the specified missing-pixel task."}</li><li>{"GRN does not improve reconstruction in every seed. Its probe gets one extra image correct in each paired run on this one development split."}</li><li>{"The simpler raw-pixel classifier gets more development labels right than any frozen-feature probe here."}</li><li>{"We did not compare against a random untrained encoder, vary the label budget, fine-tune the encoder or reserve an untouched test. These results do not isolate the benefit of pretraining or establish a universal architecture ranking."}</li></ul>

<Prose>{"Repeated seeds vary initialization and training masks; they are not independent new datasets. The development data are now consumed by interpretation. If you use these findings to choose a model, obtain a separate appropriate final evaluation before making a deployment claim."}</Prose>

<ConvNeXtRecordedExperiment />

<H3>{"Look at a specimen, then intervene"}</H3>

<Prose>{"The packet retains the two seed 1 models' complete weights and the first two development examples, source 251/label 4 and source 40/label 9. Display their original, visible-only input, reconstruction and hidden-pixel squared error side by side. Show the raw reconstruction values in a numeric view; a clipped display palette must not silently clip the values used for MSE."}</Prose>

<Prose>{"For source 251 with GRN, flipping hidden pixel (row 0, column 0) from 0 to 1 leaves every prediction unchanged, while masked MSE changes from approximately 0.054047 to 0.080387. Flipping the visible pixel (0,2) instead changes the reconstruction, with maximum absolute output change about 0.506610. The paired model without GRN has the same hidden-input invariance and a different visible-input response."}</Prose>

<ConvNeXtReconstructionLab />

<H3>{"Inspect features without overinterpreting a diagnostic"}</H3>

<Prose>{"The program measures spatial cosine distance between distinct nonzero channel maps at the final encoder expansion. For maps "}<InlineMath>{"a,b"}</InlineMath>{", it uses "}<InlineMath>{"(1-\\cos(a,b))/2"}</InlineMath>{"; identical positive-direction maps have distance 0. Near-zero maps are counted separately instead of assigning an arbitrary cosine."}</Prose>

<Prose>{"Every run has zero near-zero-channel fraction under the declared threshold. Mean nonself distances with GRN are slightly higher in seed 1 and slightly lower in seeds 2–3. Thus “more active channels” does not explain the small probe difference, and this diagnostic is not a quality score. Inspect what a statistic measures before attaching an architectural story to it."}</Prose>

<H2>{"7. Deeper routes: deploy, reparameterize, or combine mechanisms"}</H2>

<H3>{"Use a checkpoint as a complete input/output contract"}</H3>

<Prose>{"For a practical pretrained model, record the exact library/version, architecture, weight identifier, input transforms, output classes and intended downstream evaluation. A weight file is not useful independently of this contract."}</Prose>

<Prose>{"The inspected "}<a href={"https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html"}>{"Torchvision ConvNeXt Tiny documentation"}</a>{" exposes "}<code>{"ConvNeXt_Tiny_Weights.IMAGENET1K_V1"}</code>{" and its "}<code>{"transforms()"}</code>{". Its reported 82.52% ImageNet result belongs to Torchvision's modified recipe; it is not the original paper's 82.1% result. For those weights, use the provided resize/crop/normalization and category metadata. Replace the classifier and evaluate your task if the target classes differ; the ImageNet head does not acquire new classes by renaming its outputs."}</Prose>

<Prose>{"The attached architecture program offers "}<code>{"return_features=True"}</code>{" for the four stage maps. This is the useful interface for a downstream head: inspect its exact shapes and scale meaning before connecting it. Full transfer-learning training and split design belong to "}<a href={"/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals"}>{"Transfer Learning & Fine-Tuning Strategies"}</a>{"; no pretrained download is required for this lesson's executed experiment."}</Prose>

<Prose>{"Logical layout and memory layout are different. "}<code>{"permute(0,2,3,1)"}</code>{" makes a view whose dimension order is NHWC. "}<code>{"to(memory_format=torch.channels_last)"}</code>{" retains the logical NCHW shape while changing storage strides. A permutation itself does not copy values, but later operations may need to materialize a suitable layout. Measure the complete workload before promising a speedup. "}<a href={"https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html"}>{"PyTorch's channels-last tutorial"}</a>{""}</Prose>

<H3>{"Fold several training branches into one inference kernel"}</H3>

<Prose>{"A useful deployment idea is "}<strong>{"structural reparameterization"}</strong>{": train using several linear branches, then combine them into a simpler equivalent inference operation. This is different from changing the trained function through approximate compression."}</Prose>

<Prose>{"Suppose a convolution "}<InlineMath>{"z=Wx+b"}</InlineMath>{" is followed by BatchNorm using fixed evaluation statistics "}<InlineMath>{"\\mu,v"}</InlineMath>{" and learned "}<InlineMath>{"\\gamma,\\beta"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{BN}(Wx+b)=\n\\left(\\frac{\\gamma}{\\sqrt{v+\\epsilon}}W\\right)x+\n\\left(\\frac{\\gamma(b-\\mu)}{\\sqrt{v+\\epsilon}}+\\beta\\right)."}</MathBlock></div>

<Prose>{"Each output channel gets its own multiplier and bias. Fold each linear branch this way. Pad a smaller odd-sized kernel with zeros so its center aligns with the larger kernel; represent an identity path as a center coefficient 1 for corresponding input/output channels. Add the aligned kernels and biases."}</Prose>

<Prose>{"This works only when the branches have compatible input/output shapes, stride, coordinate alignment and groups, and the normalization statistics are fixed. A shared activation "}<strong>{"after"}</strong>{" the summed branches can remain after the fused convolution. Separate nonlinear activations inside branches generally cannot be folded this way."}</Prose>

<Prose>{"The author calculation folds a 3×3 branch, a 1×1 branch and identity on a 5×5 constructed input. Separate and fused outputs agree within "}<InlineMath>{"2.9\\times10^{-14}"}</InlineMath>{"; the center output is 111.049826. Moving separate ReLUs inside branches creates a different function. The supplied counterexample differs by more than 23 in one output. These are exact-function checks, not latency benchmarks."}</Prose>

<Prose>{"Large-kernel models such as "}<a href={"https://arxiv.org/pdf/2203.06717"}>{"RepLKNet"}</a>{" use this idea to aid training while retaining a large spatial filter at inference. Its main blocks use a parallel 5×5 branch with the large kernel. "}<a href={"https://arxiv.org/pdf/2206.04040"}>{"MobileOne"}</a>{" applies related deployment-oriented reasoning to small blocks. A 31×31 depthwise kernel reads a broader dense stencil, but its coefficients remain shared learned values; they are not automatically input-dependent attention weights."}</Prose>

<ConvNeXtFusionLab />

<H3>{"When a hybrid is a useful hypothesis"}</H3>

<Prose>{"Attention forms a weighted sum of value vectors, with weights obtained from the current input and query. Convolution uses a learned spatial stencil shared across inputs. Both can mix spatial evidence, but attention layers also include channel projections, and their complete block topology differs from a ConvNeXt block."}</Prose>

<Prose>{"A hybrid can use local convolution where the grid is large and more global input-dependent interactions after the grid has shrunk. "}<a href={"https://arxiv.org/pdf/2106.04803"}>{"CoAtNet"}</a>{" studies such stage arrangements. "}<a href={"https://arxiv.org/pdf/2204.01697"}>{"MaxViT"}</a>{" alternates local block attention with a sparse grid arrangement that connects distant positions. These are concrete choices about which positions communicate, not evidence that adding attention anywhere must help."}</Prose>

<Prose>{"For a factory-defect application, local texture may matter alongside long-range alignment between repeated parts. A ConvNeXt feature hierarchy, a larger convolutional receptive field and a hybrid interaction pattern are competing hypotheses. Split by production unit or scene when multiple images share an origin, establish a simple baseline, then examine the errors that distinguish those hypotheses. A smaller-input label classifier and a high-resolution localization system need different evaluation and memory budgets."}</Prose>

<Prose>{"The later attention and vision-transformer lessons develop the weighted-sum mechanism in full. Here the useful connection is to ask "}<strong>{"which evidence can reach this output, through which operation, at what resolution and cost?"}</strong>{""}</Prose>

<H2>{"Match the block you built to the maintained implementation"}</H2>

<Prose>{"The complete "}<a href={"/learn-assets/convnext-modern-cnn-designs/convnext-blocks.py"}>{"convnext-blocks.py"}</a>{" implements both V1 and V2 block/stage/head composition. It exposes spatial filtering, NHWC normalization, expansion, GELU, V2 response normalization or V1 LayerScale, projection, per-example branch masking and residual addition. "}<code>{"masked-reconstruction.py"}</code>{" supplies the task-specific model, loss and actual learning loop. Those are the scratch mechanisms at this topic's level; convolution indexing, loss derivatives and autograd already have named earlier owners."}</Prose>

<Prose>{"The new "}<a href={"/learn-assets/convnext-modern-cnn-designs/convnext_library_bridge.py"}>{"convnext_library_bridge.py"}</a>{" connects the V1 block to Torchvision's "}<code>{"CNBlock"}</code>{". It copies the depthwise convolution, LayerNorm, two linear maps and channel scale before comparing anything. Torchvision stores LayerScale as "}<code>{"[C,1,1]"}</code>{"; our NHWC branch uses "}<code>{"[C]"}</code>{". The values mean the same per-channel factor only after this layout mapping. Both paths disable stochastic depth for the equality check, use float64, and compare outputs, input gradients and every trainable gradient on a rectangular5×7 map. Random masks or mismatched normalization axes would make an otherwise plausible comparison invalid."}</Prose>

<Prose>Run <code>python convnext_library_bridge.py</code> beside <code>convnext-blocks.py</code> with compatible PyTorch/Torchvision. The offline comparison was executed with PyTorch2.14.0+cpu and Torchvision0.29.0: V1 outputs, input gradients and all trainable gradients agree under the program’s1e−12 absolute/relative tolerances. It deliberately targets V1: Torchvision’s <code>CNBlock</code> does not become V2 merely because both are called ConvNeXt. The local <code>ResponseNorm</code> exposes the complete V2 operation, independently differentiated and checked against central differences. <a href="https://github.com/pytorch/vision/blob/v0.29.0/torchvision/models/convnext.py">Torchvision block source</a>.</Prose>

<Prose>{"The same program accepts "}<code>{"--image example.jpg"}</code>{". It selects "}<code>{"ConvNeXt_Tiny_Weights.IMAGENET1K_V1"}</code>{", applies that enum's RGB transform and reads category labels from its metadata. It downloads that checkpoint if absent, runs eval/inference mode, and reports a real photograph's top-five probabilities. This ordinary application is separate from the masked-digit training experiment, and has no prepared accuracy claim. Pillow and a local image are explicit inputs. Feature adaptation follows the implemented "}<a href={"/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies#transfer-section-3"}>{"Transfer Learning section3"}</a>{"; the imported model's training recipe is not re-created by this inference call."}</Prose>

<ConvNeXtProgram file="convnext_library_bridge.py" title="Read the matched-state library bridge and optional pretrained photograph route" /><Prose>The offline matched-state bridge was executed. The optional photograph route was not executed: it requires a learner-supplied image and the specified external checkpoint. Its complete source shows the ordinary transform/metadata/inference contract without assigning it an unmeasured accuracy.</Prose>

<Prose>{""}<strong>{"Modify the block deliberately:"}</strong>{" change expansion4 to expansion2 in the local block, retaining the depthwise width and residual output width. Rebuild both linear layers and, for V2, the response-normalization parameter vectors at2C. The pointwise matrix weights fall from8C² to4C²; depthwise weights stay49C. The original Torchvision block then ceases to be a direct same-shape counterpart, so compare your modified block to a separately assembled reference with the new dimensions rather than weakening the old assertions."}</Prose>

<details>

<summary>Hint</summary>

<Prose>{"The hidden width belongs to every operation between expansion and projection, not only the first Linear."}</Prose>

</details>

<details>

<summary>Solution and success criteria</summary>

<Prose>{"At C8 use "}<code>{"Linear(8,16)"}</code>{", a16-channel GRN if V2, then "}<code>{"Linear(16,8)"}</code>{". Preserve the output shape, test finite input/weight gradients, and check that zero LayerScale or zero projection still gives the expected residual identity. The pointwise weights total256 instead of512; include biases separately. A256-weight saving is an arithmetic result, not proof of better validation accuracy or latency."}</Prose>

</details>

<H2>{"8. Practice: reason about a changed design"}</H2>

<H3>{"1. Catch a shape-correct normalization error"}</H3>

<Prose>{"A tensor has shape "}<InlineMath>{"N,32,7,32"}</InlineMath>{". Someone applies "}<code>{"nn.LayerNorm(32)"}</code>{" directly and says it normalizes channels. Explain what it actually does and give a correct channel-normalization route."}</Prose>

<details><summary>Hint</summary>

<Prose>{"LayerNorm matches its normalized shape to the trailing dimensions; equal dimension sizes can conceal the wrong axis."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"It normalizes the final width axis of length 32, independently for each specimen/channel/row. Permute to "}<InlineMath>{"N,7,32,32"}</InlineMath>{" with the original channel axis last, apply LayerNorm(32), and permute back. Name the axes explicitly: both trailing dimensions happen to be 32 after the permutation, so shape inspection alone is insufficient. The unchanged intended output shape does not prove the operation is correct."}</Prose>

</details>

<H3>{"2. Change the expansion and count what changed"}</H3>

<Prose>{"Use a 5×5 depthwise kernel, input/output width 64 and expansion factor 3 in a V1-style block. Include all biases, channel LayerNorm and LayerScale. How many parameters and convolution/linear MACs does the block use on a 14×14 grid?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write the two pointwise matrices and their different bias lengths before adding the small vectors."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Depthwise has "}<InlineMath>{"25C+C"}</InlineMath>{"; LayerNorm "}<InlineMath>{"2C"}</InlineMath>{"; the two linear layers "}<InlineMath>{"3C^2+3C"}</InlineMath>{" and "}<InlineMath>{"3C^2+C"}</InlineMath>{"; LayerScale "}<InlineMath>{"C"}</InlineMath>{". Total "}<InlineMath>{"6C^2+33C=26,688"}</InlineMath>{". Conv/linear MACs are "}<InlineMath>{"196(25\\cdot64+6\\cdot64^2)=5,130,496"}</InlineMath>{". Normalization, activations and additions are excluded from that declared operation count."}</Prose>

</details>

<H3>{"3. Predict a cross-channel effect"}</H3>

<Prose>{"For the GRN example, keep "}<InlineMath>{"\\gamma_A=.5"}</InlineMath>{", set "}<InlineMath>{"\\gamma_B=0"}</InlineMath>{", and change B's second value 12→24. Does A's first output rise or fall? Does setting both scales to zero make the statistics stop changing?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"A's norm stays 5; the denominator compares it with B's new norm 24."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"A's relative response becomes "}<InlineMath>{"5/(14.5+\\epsilon)"}</InlineMath>{", smaller than before, so its first output falls to approximately "}<InlineMath>{"3(1+.5\\cdot5/14.5)=3.517241"}</InlineMath>{". Zero scales remove the response-dependent contribution from the output; the norms still change internally. GRN's identity initialization and its statistic computation are different facts."}</Prose>

</details>

<H3>{"4. Diagnose suspiciously good reconstruction"}</H3>

<Prose>{"The encoder masks pixel values, but first subtracts each full image's mean, calculated using visible and hidden pixels. A hidden-pixel edit changes the model's prediction. Is this necessarily a defect in the convolution code? Propose a repair and a direct check."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Ask whether the original hidden value can reach a visible input through preprocessing."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The full-image mean carries hidden information into the centered visible pixels. The convolution can be implemented correctly while the input contract leaks. Use a fixed permitted scale, training-set statistics learned without the evaluated image, or a clearly specified visible-only statistic. Change one hidden target while holding mask and visible values fixed; predictions should stay unchanged under the repaired contract. The hidden-target loss can still change. Target-only patch normalization is a separate path and must not be reused to normalize the encoder input."}</Prose>

</details>

<H3>{"5. Choose the conclusion supported by the experiment"}</H3>

<Prose>{"A colleague reports, “GRN learns more diverse channels, therefore it improves digit recognition and should replace the raw baseline.” Use the saved outcomes to rewrite the conclusion and name one additional experiment that would answer a genuinely missing question."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compare paired reconstruction results, paired diversity results and the raw-pixel classifier separately."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"On this development split the GRN variants' probes get 116/120 versus 115/120, while the raw-pixel probe gets 118/120. GRN does not increase the measured diversity in all seeds and does not improve masked MSE in seed 1. This supports a small paired probe difference in this setting, not the proposed causal explanation or replacement decision. For a pretraining question, predeclare matched random-encoder and pretrained-encoder probes with the same architecture and label budget. For a deployment choice, select using development data and evaluate once on a new appropriately grouped final set."}</Prose>

</details>

<H3>{"6. Check a branch-fusion boundary"}</H3>

<Prose>{"For scalar input "}<InlineMath>{"x=-2"}</InlineMath>{", compare "}<InlineMath>{"\\operatorname{ReLU}(x)+\\operatorname{ReLU}(-x)"}</InlineMath>{" with "}<InlineMath>{"\\operatorname{ReLU}(x-x)"}</InlineMath>{". Can the separate branch activations be removed while preserving the function?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Apply each activation before summing in the first expression."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The first gives "}<InlineMath>{"0+2=2"}</InlineMath>{"; the second gives 0. Linear branch kernels can be added only where the intermediate operations permit the algebra. Keeping one shared activation after an equivalent linear sum is valid; replacing separate nonlinear branches with that shared activation is a different model."}</Prose>

</details>

<H3>{"7. Budget a finer stem"}</H3>

<Prose>{"Replace a 4×4 stride 4 stem with a 2×2 stride 2 stem on 224×224 inputs while keeping all later stage widths and depths unchanged. What happens to the first grid, the block MACs and the head's parameter count?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Track spatial area through the later stride 2 transitions. The classification head receives an averaged channel vector."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The first grid becomes 112×112 instead of 56×56. Every corresponding later grid has twice the side length, so block conv/linear MACs and transition MACs become four times larger. Stem MACs happen to remain equal here: four times as many outputs each use one quarter as many spatial weights. Stem parameters fall, but the global-average-pooling classifier's input width and head parameter count stay unchanged. Activation memory also grows; accuracy and latency cannot be deduced from this count alone."}</Prose>

</details>

<H3>{"8. Design an informative masked-learning extension"}</H3>

<Prose>{"You have 20 labeled specimens per class and many unlabeled images from repeated capture sessions. Propose a comparison to ask whether masked pretraining helps when labels are scarce. Include the split unit, baseline, preprocessing, model comparison and final evaluation."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Unlabeled access is still access to data. Keep the question of representation learning separate from the number of labels used by the readout."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Split capture sessions before training so related views do not cross partitions. Restrict both supervised and unlabeled pretraining inputs to training sessions. Fix the same labeled subset and feature-readout recipe for raw pixels, a random frozen encoder and the pretrained frozen encoder; optionally add a separately declared end-to-end supervised model. Fit preprocessing on training data, select any settings on development sessions, and report the chosen protocol once on held-out sessions. Record reconstruction and downstream task outcomes separately and repeat paired seeds. Do not call extra unlabeled access “the same data budget” unless that is explicitly the question."}</Prose>

</details>

<H2>{"9. Readiness, connections and other ways to learn"}</H2>

<Prose>{"You are ready to move on when you can explain a block using spatial and channel operations, mark the inputs used by each normalization statistic, trace a masked target without leaking it into the encoder, and state what the actual experiment demonstrates. Memorizing every model size or reproducing ImageNet training is not required."}</Prose>

<Prose>{"Next in this module is "}<a href={"/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals"}>{"Capsule Networks"}</a>{". It asks a different representation question: instead of only scalar feature activations, can groups of values represent a part's properties and help parts agree on a whole? ConvNeXt does not become obsolete at that transition; the proposed inductive bias changes."}</Prose>

<Prose>{"Useful routes through the references:"}</Prose>

<ul><li>{""}<a href={"https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py"}>{"Official ConvNeXt source"}</a>{": inspect the block, channel-first LayerNorm, stage transitions and initialization after working through §§2–3. Code reading is particularly useful for distinguishing logical axes from the diagram."}</li><li>{""}<a href={"https://arxiv.org/pdf/2201.03545"}>{"ConvNeXt V1 paper"}</a>{": read §2 as an experiment-design argument, then compare the small-regime roadmap in Appendix C with the final result table. The roadmap's roughly 82.0% average and final 82.1% checkpoint report are different records."}</li><li>{""}<a href={"https://arxiv.org/pdf/2301.00808"}>{"ConvNeXt V2 paper"}</a>{" and "}<a href={"https://cvpr.thecvf.com/media/cvpr-2023/Slides/22892_lw8881R.pdf"}>{"authors' CVPR slides"}</a>{": the paper supplies the mask/GRN details; the slides offer a visual second pass through masking, feature maps and co-design. Their full-scale experiments differ from our bounded dense-masked probe experiment."}</li><li>{""}<a href={"https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html"}>{"PyTorch channels-last tutorial"}</a>{": a hands-on storage-stride explanation. Its hardware results belong to the measured configurations; use the concepts to inspect your own workload."}</li><li>{""}<a href={"https://en.d2l.ai/chapter_convolutional-modern/cnn-design.html"}>{"Dive into Deep Learning: Designing Convolution Network Architectures"}</a>{": study the AnyNet→RegNet design-space argument as an alternative to memorizing model families. The chapter's broader historical rankings are time-specific; its distribution-of-designs perspective is the useful complement here."}</li><li>{""}<a href={"https://arxiv.org/pdf/2203.06717"}>{"RepLKNet"}</a>{", "}<a href={"https://arxiv.org/pdf/2206.04040"}>{"MobileOne"}</a>{", "}<a href={"https://arxiv.org/pdf/2106.04803"}>{"CoAtNet"}</a>{" and "}<a href={"https://arxiv.org/pdf/2204.01697"}>{"MaxViT"}</a>{": optional mechanism-focused extensions for large kernels, inference-time branch folding, stage ordering and local/global communication. Read the ablation conditions before generalizing a result."}</li></ul>

<Prose>All local experimental numbers come from the accompanying programs and retained results. Read the <a href={convnextAsset+"data-provenance.md"}>dataset provenance</a> and <a href={convnextAsset+"native-verification.json"}>current native verification record</a> for execution boundaries. The six original fits are conserved, with fresh native reconstruction and independent browser-model comparisons; no ImageNet training, pretrained photograph run or hardware timing is claimed.</Prose>
</div>
};
