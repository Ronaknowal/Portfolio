// Generated from the complete prepared manuscript by scripts/generate-hyena-lesson.mjs.
import {Prose,H2,H3,CodeBlock} from '../../components/content';
import {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';
import {LessonIntro} from '../../components/lesson-labs/LessonElements.jsx';
import {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';
import {DelayAddressFigure,EchoFigure,ToeplitzFigure,PaddingFigure,GateRailsFigure,CoefficientFigure,HierarchyFigure,CoordinateFigure,GradientFigure,DnaWindowFigure,DataRolesFigure,LearningCurvesFigure,CounterfactualFigure,OverlapFigure,ModesFigure,ApproximationFigure,MechanismMapFigure,StripedFigure,BenchmarkFigure,SequenceBlockFigure} from '../../components/lesson-labs/HyenaFigures.jsx';
import {ConvolutionLab,GateLab,BlockedFftLab,StreamingLab} from '../../components/lesson-labs/HyenaMechanismLabs.jsx';
import {DnaStudy,LearnedFilterFigure,HyenaProgram} from '../../components/lesson-labs/HyenaStudy.jsx';
export default {title:"Hyena & Long Convolution Models",readTime:"~75 min read + investigations and practice",hasIntegratedGuide:true,content:()=> <div className="neural-lesson neural-lesson-neutral hyena-lesson"><LessonIntro prerequisites="Weighted sums, matrix shapes and learning by reducing a loss. Fourier and signal-processing ideas are introduced here." sections={[["1-three-questions-about-a-long-sequence","1. Three questions about a long sequence"],["2-a-convolution-is-a-ledger-of-delayed-contributions","2. A convolution is a ledger of delayed contributions"],["3-why-an-fft-can-compute-the-same-sum-faster","3. Why an FFT can compute the same sum faster"],["4-gates-make-the-mixing-depend-on-the-input","4. Gates make the mixing depend on the input"],["5-generate-the-filter-from-position","5. Generate the filter from position"],["6-how-the-block-learns","6. How the block learns"],["7-a-real-sequence-task-identify-a-splice-boundary","7. A real sequence task: identify a splice boundary"],["8-deeper-route-long-filters-streaming-and-compact-state","8. Deeper route: long filters, streaming and compact state"],["9-deeper-route-what-the-family-adds","9. Deeper route: what the family adds"],["10-practice-reason-calculate-and-transfer","10. Practice: reason, calculate and transfer"],["11-references-and-another-way-to-learn","11. References and another way to learn"]]}>Follow delayed contributions, input-dependent gates and the state that carries a filter across chunks.</LessonIntro>
<Prose opening="exploration">{""}<strong>{"Explore as you read."}</strong>{" Change a signal, filter or gate and follow its delayed contributions. Compare causal and circular outputs, carry recurrent state across a new chunk boundary, and edit real DNA symbols to inspect a fixed classifier. Use the resulting paths, states and scores to distinguish future leakage, lost history and deliberate approximation."}</Prose>

<Prose>{"A short convolution asks what happened nearby. A long convolution lets a distant event still contribute to what happens now. Hyena adds a further choice: the input controls what gets transmitted through that long filter and how the receiver uses it."}</Prose>

<Prose>{"Imagine a sensor that produces a sharp pulse. A filter can make that pulse leave a fading echo. Now imagine a sequence of symbols in which some events are useful and others should be suppressed. A fixed echo pattern cannot make that decision by itself. Gates derived from the symbols supply the missing input dependence. Hyena combines these operations into a sequence mixer that can process a whole sequence with fast convolution algorithms."}</Prose>

<Prose>{"The preceding "}<a href={"/learn/path/full-curriculum/xlstm-extended-lstm?module=deep-learning-fundamentals"}>{"xLSTM lesson"}</a>{" built a memory state and asked how to read it. Here we start from the "}<strong>{"influence of an earlier position on a later position"}</strong>{". This change in viewpoint reveals both the appeal of long convolutions and a subtle problem: a fast whole-sequence computation does not automatically provide cheap one-token-at-a-time generation."}</Prose>

<Prose opening="route">{""}<strong>{"Your first pass:"}</strong>{" follow §§1–7, then try practice 1–6. You should be able to compute a causal convolution, identify circular wraparound, explain a gate-filter-gate block, and interpret the real sequence experiment. §8 develops streaming and recurrence extraction; §9 connects modern variants. Those branches and practice 7–10 provide the deeper route. You need weighted sums, matrix shapes and the idea of training by reducing a loss. We introduce the Fourier and signal-processing vocabulary locally."}</Prose>

<H2>{"1. Three questions about a long sequence"}</H2>

<Prose>{"Suppose a model sees "}<code>{"A 7 B 2 … B"}</code>{" and should answer "}<code>{"2"}</code>{". The last symbol must identify which earlier association matters. Contrast that with copying "}<code>{"4 7 2"}</code>{" after exactly three time steps: a fixed delay is sufficient for the latter. Both involve distant information, but only the first requires choosing an address from the content."}</Prose>

<Prose>{"Sequence mixers make different tradeoffs among three questions:"}</Prose>

<NeuralTable caption={"1. Three questions about a long sequence"} headers={[<>{"Question"}</>,<>{"What it means"}</>,<>{"A concrete diagnostic"}</>]} rows={[[<>{"Can information travel far?"}</>,<>{"An old input has a computational path to a new output."}</>,<>{"Place a nonzero input far back and inspect its contribution."}</>],[<>{"Can the input change which information matters?"}</>,<>{"The mixing weights depend on the sequence being processed."}</>,<>{"Change a key or a gate while keeping distance fixed."}</>],[<>{"Can the computation be evaluated efficiently?"}</>,<>{"The algorithm avoids unnecessary work or storage."}</>,<>{"Derive the operations actually performed, then measure a specified implementation."}</>]]} />

<Prose>{"Hyena’s design joins long filters, input-dependent gates and fast convolution. These are properties of its mechanism; whether training learns a useful solution is a separate empirical question. "}<a href={"https://arxiv.org/html/2302.10866v2"}>{"The original paper"}</a>{" used associative recall and other small controlled tasks to guide its design before studying language and vision."}</Prose>

<DelayAddressFigure/>

<H2>{"2. A convolution is a ledger of delayed contributions"}</H2>

<Prose>{"Let "}<InlineMath>{"u_t"}</InlineMath>{" be the input at position "}<InlineMath>{"t"}</InlineMath>{", starting at zero. Let "}<InlineMath>{"h_r"}</InlineMath>{" be the filter coefficient for a "}<strong>{"lag"}</strong>{" of "}<InlineMath>{"r"}</InlineMath>{" positions. The causal output is"}</Prose>

<div className="neural-equation"><MathBlock>{"y_t=\\sum_{j=0}^{t}h_{t-j}u_j\n    =h_0u_t+h_1u_{t-1}+\\cdots+h_tu_0."}</MathBlock></div>

<Prose>{"“Causal” means the output at "}<InlineMath>{"t"}</InlineMath>{" uses no input after "}<InlineMath>{"t"}</InlineMath>{". “Filter” means the collection of coefficients. The name "}<strong>{"impulse response"}</strong>{" describes the same object from a useful experiment: put a single unit pulse at "}<InlineMath>{"u_0=1"}</InlineMath>{", followed by zeros, and the output is "}<InlineMath>{"h_0,h_1,h_2,\\ldots"}</InlineMath>{". The filter is the echo of that pulse."}</Prose>

<Prose>{"Take "}<InlineMath>{"u=[1,2,3,4]"}</InlineMath>{" and "}<InlineMath>{"h=[1,0.5,0.25,0.125]"}</InlineMath>{". At position 2, the current input contributes 3, the previous input contributes "}<InlineMath>{"0.5\\times2=1"}</InlineMath>{", and the oldest input contributes "}<InlineMath>{"0.25\\times1=0.25"}</InlineMath>{". The total is 4.25."}</Prose>

<NeuralTable caption={"2. A convolution is a ledger of delayed contributions"} headers={[<>{"Position "}<InlineMath>{"t"}</InlineMath>{""}</>,<>{"Current-to-oldest contributions"}</>,<>{""}<InlineMath>{"y_t"}</InlineMath>{""}</>]} rows={[[<>{"0"}</>,<>{""}<InlineMath>{"1\\times1"}</InlineMath>{""}</>,<>{"1"}</>],[<>{"1"}</>,<>{""}<InlineMath>{"1\\times2+0.5\\times1"}</InlineMath>{""}</>,<>{"2.5"}</>],[<>{"2"}</>,<>{""}<InlineMath>{"1\\times3+0.5\\times2+0.25\\times1"}</InlineMath>{""}</>,<>{"4.25"}</>],[<>{"3"}</>,<>{""}<InlineMath>{"1\\times4+0.5\\times3+0.25\\times2+0.125\\times1"}</InlineMath>{""}</>,<>{"6.125"}</>]]} />

<EchoFigure/>

<Prose>{"The equivalent matrix is"}</Prose>

<div className="neural-equation"><MathBlock>{"\\begin{bmatrix}1&0&0&0\\\\.5&1&0&0\\\\.25&.5&1&0\\\\.125&.25&.5&1\\end{bmatrix}\n\\begin{bmatrix}1\\\\2\\\\3\\\\4\\end{bmatrix}\n=\\begin{bmatrix}1\\\\2.5\\\\4.25\\\\6.125\\end{bmatrix}."}</MathBlock></div>

<Prose>{"This is a "}<strong>{"lower-triangular Toeplitz matrix"}</strong>{": “lower triangular” encodes causality; “Toeplitz” says each diagonal uses the same lag coefficient. Every pair of positions at distance 2 receives "}<InlineMath>{"h_2"}</InlineMath>{". A length-"}<InlineMath>{"K"}</InlineMath>{" finite impulse response, or FIR, filter additionally sets "}<InlineMath>{"h_r=0"}</InlineMath>{" for "}<InlineMath>{"r\\ge K"}</InlineMath>{"."}</Prose>

<ToeplitzFigure/>

<Prose>{"For a length-"}<InlineMath>{"L"}</InlineMath>{" filter applied to "}<InlineMath>{"L"}</InlineMath>{" inputs, direct causal evaluation has "}<InlineMath>{"L(L+1)/2"}</InlineMath>{" coefficient-input products per channel. With many channels and long sequences, that work matters. Yet the matrix contains extensive repetition. The FFT exploits this structure."}</Prose>

<H2>{"3. Why an FFT can compute the same sum faster"}</H2>

<Prose>{"The discrete Fourier transform represents a finite signal using oscillating basis patterns. Think of changing coordinates: we can describe a sound by its samples over time or by the amplitudes and phases of its frequencies. We have not yet removed information."}</Prose>

<Prose>{"For two arrays padded to length "}<InlineMath>{"P"}</InlineMath>{", the Fourier convolution theorem gives"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{IFFT}\\left(\\operatorname{FFT}(u)\\odot\\operatorname{FFT}(h)\\right)."}</MathBlock></div>

<Prose>{"Here "}<InlineMath>{"\\odot"}</InlineMath>{" is elementwise multiplication. The forward transforms separate the oscillating components; multiplication applies the filter to those components; the inverse transform reconstructs samples. The FFT is an efficient algorithm for that coordinate change, with work proportional to "}<InlineMath>{"P\\log P"}</InlineMath>{"."}</Prose>

<Prose>{"There is a catch. A length-"}<InlineMath>{"P"}</InlineMath>{" transform treats indices modulo "}<InlineMath>{"P"}</InlineMath>{": values that run past the right edge wrap to the left. It naturally computes "}<strong>{"circular convolution"}</strong>{". To recover ordinary linear convolution of lengths "}<InlineMath>{"L"}</InlineMath>{" and "}<InlineMath>{"K"}</InlineMath>{", choose "}<InlineMath>{"P\\ge L+K-1"}</InlineMath>{", pad both arrays with zeros, and keep the desired outputs. For causal sequence mixing, keep the first "}<InlineMath>{"L"}</InlineMath>{"."}</Prose>

<PaddingFigure/>

<Prose>{"The wrong "}<InlineMath>{"P=4"}</InlineMath>{" computation for our example produces "}<code>{"[4, 3.875, 4.75, 6.125]"}</code>{". Its first output is"}</Prose>

<div className="neural-equation"><MathBlock>{"1\\times1+0.5\\times4+0.25\\times3+0.125\\times2=4."}</MathBlock></div>

<Prose>{"That first output already contains the future input 4. Changing only the final input from 4 to 8 changes the wrong first output from 4 to 6. In the correctly padded computation the first three outputs remain "}<code>{"[1,2.5,4.25]"}</code>{"."}</Prose>

<H3>{"A complete numerical program"}</H3>

<Prose>{"Install NumPy in your own environment with "}<code>{"python -m pip install numpy"}</code>{", save this as "}<code>{"causal_fft.py"}</code>{", and run "}<code>{"python causal_fft.py"}</code>{". Both inputs and expected outputs are included."}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\ndef causal_fft(values, kernel):\n    length = len(values)\n    full_length = length + len(kernel) - 1\n    fft_length = 1 << (full_length - 1).bit_length()\n    product = np.fft.rfft(values, fft_length) * np.fft.rfft(kernel, fft_length)\n    return np.fft.irfft(product, fft_length)[:length]\n\nvalues = np.array([1., 2., 3., 4.])\nkernel = np.array([1., .5, .25, .125])\ndirect = np.convolve(values, kernel)[:len(values)]\nfast = causal_fft(values, kernel)\nprint(np.round(direct, 6))\nprint(np.round(fast, 6))\nprint(np.allclose(direct, fast, atol=1e-12))"}</CodeBlock>

<Prose>{"The arrays both display "}<code>{"[1. 2.5 4.25 6.125]"}</code>{", followed by "}<code>{"True"}</code>{". The underlying floating-point arrays need not be bit-for-bit identical. In the saved calculation the first FFT output is "}<code>{"0.9999999999999996"}</code>{". That numerical roundoff is distinct from the large semantic wraparound error above. Always pass the intended inverse-transform length, especially for odd sizes; make normalization conventions agree. The "}<a href={"https://docs.pytorch.org/docs/main/generated/torch.fft.rfft.html"}>{"PyTorch FFT documentation"}</a>{" explains normalization and device/dtype restrictions."}</Prose>

<HyenaProgram filename="convolution_mechanisms.py"/>

<ConvolutionLab/>

<H2>{"4. Gates make the mixing depend on the input"}</H2>

<Prose>{"A useful basic Hyena block has three projected streams: "}<InlineMath>{"q"}</InlineMath>{", "}<InlineMath>{"k"}</InlineMath>{" and "}<InlineMath>{"v"}</InlineMath>{", each shaped "}<InlineMath>{"L\\times D"}</InlineMath>{". Here "}<InlineMath>{"D"}</InlineMath>{" is the channel width. These letters resemble attention’s query, key and value names, but the computation below contains no softmax and no pairwise query-key dot product."}</Prose>

<Prose>{"For one channel,"}</Prose>

<div className="neural-equation"><MathBlock>{"z_j=k_jv_j,\\qquad r_t=\\sum_{j\\le t}h_{t-j}z_j,\n\\qquad y_t=q_tr_t."}</MathBlock></div>

<Prose>{"The sending gate "}<InlineMath>{"k_j"}</InlineMath>{" changes what enters the filter. The receiving gate "}<InlineMath>{"q_t"}</InlineMath>{" changes how the result is used. Gates can amplify or reverse sign; they are not automatically probabilities in "}<InlineMath>{"[0,1]"}</InlineMath>{". A neural implementation derives them from the input using learned projections and small causal depthwise convolutions. “Depthwise” means each channel has its own short filter rather than mixing channels at that operation. Dense projections before and after it handle channel mixing."}</Prose>

<GateRailsFigure/>

<Prose>{"Use the earlier "}<InlineMath>{"v=[1,2,3,4]"}</InlineMath>{", filter "}<InlineMath>{"h"}</InlineMath>{", sending gates "}<InlineMath>{"k=[1,0,-1,2]"}</InlineMath>{", and receiving gates "}<InlineMath>{"q=[1,2,-1,0.5]"}</InlineMath>{". Then"}</Prose>

<div className="neural-equation"><MathBlock>{"kv=[1,0,-3,8],\\quad h*(kv)=[1,.5,-2.75,6.625],\n\\quad y=[1,1,2.75,3.3125]."}</MathBlock></div>

<Prose>{"At the last position the contributions before the receiving gate are "}<InlineMath>{"0.125+0-1.5+8=6.625"}</InlineMath>{". The final factor 0.5 gives 3.3125. The zero gate suppresses the second value along this path; the negative gate reverses the third."}</Prose>

<Prose>{"We can expose every pairwise coefficient without using a dense matrix to run the block:"}</Prose>

<div className="neural-equation"><MathBlock>{"H(q,k)=\\operatorname{diag}(q)\\,T_h\\,\\operatorname{diag}(k),\n\\qquad H_{tj}=q_t h_{t-j}k_j\\quad(j\\le t)."}</MathBlock></div>

<CoefficientFigure/>

<Prose>{"For fixed "}<InlineMath>{"q,k,h"}</InlineMath>{", this is linear in "}<InlineMath>{"v"}</InlineMath>{". A complete block is nonlinear in its original input because the same input also changes its gates. Consequently the displayed matrix is neither the full input-output Jacobian nor a guaranteed causal explanation of the network’s final prediction. It describes a specific internal path with gates held fixed."}</Prose>

<H3>{"Why “hierarchy” matters"}</H3>

<Prose>{"The general construction alternates filters and gates:"}</Prose>

<div className="neural-equation"><MathBlock>{"z^{(0)}=v,\\qquad z^{(n)}=x^{(n)}\\odot(h^{(n)}*z^{(n-1)}),\n\\quad n=1,\\ldots,N."}</MathBlock></div>

<Prose>{"The associated product is "}<InlineMath>{"D_{x^{(N)}}T_{h^{(N)}}\\cdots D_{x^{(1)}}T_{h^{(1)}}"}</InlineMath>{". Each additional stage introduces intermediate positions through which information can pass. With two stages,"}</Prose>

<div className="neural-equation"><MathBlock>{"y_t=q_t\\sum_{m\\le t}\\psi_{t-m}k_m\\sum_{j\\le m}\\varphi_{m-j}v_j,\n\\quad\nH_{tj}=q_t\\sum_{m=j}^{t}\\psi_{t-m}k_m\\varphi_{m-j}."}</MathBlock></div>

<Prose>{"The sum over "}<InlineMath>{"m"}</InlineMath>{" is the new feature: an input can reach the receiver through several intervening gates. Every path still goes forward in sequence order, proving causality if the gate-generating operations are also causal."}</Prose>

<HierarchyFigure/>

<Prose>{"Terminology differs across descriptions. The original paper’s formal "}<InlineMath>{"N"}</InlineMath>{"-stage hierarchy counts alternating filter/gate stages. HyenaDNA’s common "}<code>{"order=2"}</code>{" implementation has three projected streams, two gates and one long convolution after their short preprocessing. In this lesson the practical model is explicitly a "}<strong>{"one-long-filter Hyena-style block"}</strong>{", stacked twice. Inspect the actual equation or code rather than infer the computation from “order two.” "}<a href={"https://arxiv.org/html/2306.15794v2"}>{"HyenaDNA’s method"}</a>{" and "}<a href={"https://github.com/HazyResearch/hyena-dna/blob/main/standalone_hyenadna.py"}>{"the authors’ standalone code"}</a>{" show this convention."}</Prose>

<GateLab/>

<H2>{"5. Generate the filter from position"}</H2>

<Prose>{"An explicit length-"}<InlineMath>{"L"}</InlineMath>{", "}<InlineMath>{"D"}</InlineMath>{"-channel filter stores "}<InlineMath>{"LD"}</InlineMath>{" learned coefficients. An "}<strong>{"implicit filter"}</strong>{" instead learns a function whose input is lag and whose output is a vector of channel coefficients:"}</Prose>

<div className="neural-equation"><MathBlock>{"h_r=\\gamma_\\theta(\\operatorname{pos}(r))\\odot w(r)."}</MathBlock></div>

<Prose>{"The small neural network "}<InlineMath>{"\\gamma_\\theta"}</InlineMath>{" shares parameters across positions. The window "}<InlineMath>{"w(r)"}</InlineMath>{", often an exponential envelope, biases the filter’s range. A position representation may include the lag itself and sine/cosine features at several frequencies. A sine activation in the filter network makes oscillating and sharply varying filters easier to represent than an overly smooth initialization would suggest."}</Prose>

<LearnedFilterFigure/>

<Prose>{"Our experiment fixes a reference length "}<InlineMath>{"L_{\\rm ref}=60"}</InlineMath>{", sets "}<InlineMath>{"s_r=r/59"}</InlineMath>{", and uses"}</Prose>

<div className="neural-equation"><MathBlock>{"\\operatorname{pos}(r)=[s_r;(\\sin(2\\pi f s_r))_{f\\in F};(\\cos(2\\pi f s_r))_{f\\in F}],\\quad F=\\{1,2,4,8\\},\n\\quad\nh_r=W_2\\sin(W_1\\operatorname{pos}(r)+b_1)+b_2,"}</MathBlock></div>

<Prose>{"followed by multiplication by "}<InlineMath>{"\\exp[-s_r\\operatorname{softplus}(a)]"}</InlineMath>{", one decay rate per channel. The position vector has nine features. The network uses 32 hidden units and 16 outputs. Its learned count is "}<InlineMath>{"9\\times32+32+32\\times16+16+16=864"}</InlineMath>{", including the 16 decay parameters. Producing all 60 coefficient vectors still requires evaluating 60 positions and storing their outputs; fewer learned parameters do not make that work disappear."}</Prose>

<H3>{"A less obvious causality bug: moving the coordinate system"}</H3>

<Prose>{"If a four-token prefix uses coordinates "}<InlineMath>{"r/3"}</InlineMath>{", then the same prefix inside an eight-token sequence uses "}<InlineMath>{"r/7"}</InlineMath>{", its filter changes when later tokens arrive. Correct FFT padding cannot repair that problem. A causal model must use the same lag interpretation for the same prefix."}</Prose>

<Prose>{"For the illustrative function "}<InlineMath>{"h_r=e^{-s_r}\\cos(2\\pi s_r)"}</InlineMath>{", using reference length 8 gives initial coefficients "}<code>{"[1,0.54049,-0.16722,-0.58693]"}</code>{". Recomputing them with active length 4 gives "}<code>{"[1,-0.35827,-0.25671,0.36788]"}</code>{". The first four coefficients differ by as much as 0.95481. Our implementation stores a fixed position grid and slices it for shorter inputs."}</Prose>

<CoordinateFigure/>

<Prose>{"A filter function may be mathematically evaluable at longer lags, but those values were not necessarily trained. Extending a position buffer, changing frequency scales and continuing training are separate choices. Some official configurations also learn positional embeddings, adding length-dependent parameters. Read the chosen configuration before claiming that *all* parameters are independent of maximum length."}</Prose>

<Prose>{"The exponential window is an inductive bias, not a proof that every fitted coefficient decreases monotonically. The network can oscillate or grow inside the envelope; some implementations add a nonzero envelope floor. A finite filter without decay is perfectly well-defined. The practical question is which initialization and parameterization let the model learn useful filters stably."}</Prose>

<H2>{"6. How the block learns"}</H2>

<Prose>{"Start with one trainable filter and a transparent loss. For inputs "}<code>{"[1,2]"}</code>{", filter "}<code>{"[0.5,0.25]"}</code>{", and desired final output 2, the prediction is "}<InlineMath>{"2h_0+h_1=1.25"}</InlineMath>{". Use half squared error:"}</Prose>

<div className="neural-equation"><MathBlock>{"\\mathcal L=\\tfrac12(2h_0+h_1-2)^2=0.28125."}</MathBlock></div>

<Prose>{"The prediction error is "}<InlineMath>{"-0.75"}</InlineMath>{". Each coefficient’s gradient is this error multiplied by the input it weights: "}<InlineMath>{"\\nabla_h\\mathcal L=[-1.5,-0.75]"}</InlineMath>{". One gradient step with learning rate 0.1 gives "}<code>{"[0.65,0.325]"}</code>{", prediction 1.625 and loss 0.0703125. A coefficient grows because doing so moves this prediction toward its target."}</Prose>

<GradientFigure/>

<Prose>{"In a neural Hyena block, backpropagation follows the same chain through receiving gates, convolution, sending gates and the filter-generating network. FFT convolution is differentiable. A direct implementation and a correctly normalized FFT implementation must agree on the operation’s derivatives within numerical tolerance. The saved double-precision probe checks both outputs and gradients against an independently indexed direct sum."}</Prose>

<Prose>{"The full sequence block also has residual paths and a per-position feed-forward network. These keep the representation update separate from the long mixing operation. The small classifier below uses LayerNorm before each update, two blocks, a 16-dimensional token embedding and a three-logit head at the final position. Each block’s mixed update is"}</Prose>

<div className="neural-equation"><MathBlock>{"q\\odot\\{h*(k\\odot v)+b_{\\rm skip}\\odot(k\\odot v)\\}."}</MathBlock></div>

<Prose>{"The learned skip coefficient is another current-position path. It does not make the long filter noncausal. No probability interpretation is assigned to either gate; only the classifier’s final softmax produces class probabilities."}</Prose>

<Prose>{"For next-token language modeling, a different head produces vocabulary logits at each position and the loss compares each output with the following token. Shift targets consistently; apply any loss mask to predictions of the intended continuation. Teacher-forced accuracy, where preceding true tokens are supplied, and free-running generation answer different questions. The sequence classifier we now train is supervised classification, so it uses one label for the whole 60-position window."}</Prose>

<H2>{"7. A real sequence task: identify a splice boundary"}</H2>

<Prose>{"A gene can be transcribed into an RNA molecule containing regions called introns and exons. RNA splicing removes introns and joins retained exons; alternative splicing can produce different mature RNAs from one gene. The DNA itself is not cut up by this RNA-processing operation, and introns need not be biologically useless. The "}<a href={"https://www.genome.gov/about-genomics/educational-resources/fact-sheets/ribonucleic-acid-fact-sheet"}>{"NHGRI RNA explanation"}</a>{" gives the biological background."}</Prose>

<Prose>{"The historical "}<a href={"https://archive.ics.uci.edu/dataset/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences"}>{"UCI Splice-Junction dataset"}</a>{" asks whether the central boundary in a 60-character DNA window is "}<code>{"EI"}</code>{" (exon to intron), "}<code>{"IE"}</code>{" (intron to exon), or "}<code>{"N"}</code>{" (neither). We use those explicit boundary directions. Some original metadata reverses the donor/acceptor names; we do not use the conflicting names to define the labels. The central boundary sits between array indices 29 and 30. Display positions as −30…−1 and +1…+30, with no fictitious middle base at zero."}</Prose>

<Prose>{"The input alphabet is "}<code>{"A,C,G,T,D,N,R,S"}</code>{". The last four symbols describe uncertainty among bases: "}<code>{"D"}</code>{" means A/G/T, "}<code>{"N"}</code>{" any of the four, "}<code>{"R"}</code>{" A/G, "}<code>{"S"}</code>{" C/G. They are not four extra chemical bases. We keep them as separate input symbols rather than invent a measured probability for the alternatives. A sequence symbol "}<code>{"N"}</code>{" is also distinct from the *output class* "}<code>{"N"}</code>{"."}</Prose>

<DnaWindowFigure/>

<H3>{"Separate the data roles before fitting"}</H3>

<Prose>{"The source contains 3,190 rows and 3,005 distinct sequence strings. Two identical-input rows, source IDs 1022 and 1969, have conflicting labels; we exclude both from this exercise and disclose the decision. We keep the first occurrence of every other sequence, leaving 3,004 examples. Before splitting, we join source-record prefixes connected by an identical sequence, then keep every resulting group within one data role. This also keeps different windows from each observed source-record prefix together."}</Prose>

<NeuralTable caption={"Separate the data roles before fitting"} headers={[<>{"Role"}</>,<>{"Rows"}</>,<>{"Source/duplicate-connected groups"}</>,<>{"Use"}</>]} rows={[[<>{"Fit"}</>,<>{"2,128"}</>,<>{"1,008"}</>,<>{"Update parameters."}</>],[<>{"Validation"}</>,<>{"460"}</>,<>{"216"}</>,<>{"Choose an epoch for each predeclared model."}</>],[<>{"Assessment"}</>,<>{"416"}</>,<>{"216"}</>,<>{"Report the selected models once."}</>]]} />

<Prose>{"Exact source IDs, group assignments and class counts accompany the downloadable data. Distinct record names can still describe homologous biological sequences. This grouping addresses the relationships visible in the file; it does not establish independence across species, gene families or new patients. The assessment is a modest historical-data exercise, not a biological validation study or evidence of million-token performance."}</Prose>

<DataRolesFigure/>

<H3>{"Train a model you can inspect"}</H3>

<Prose>{"Download "}<a href={"/learn-code/hyena-long-convolution-models/splice_models.py"}>{"the complete training program"}</a>{", "}<a href={"/learn-code/hyena-long-convolution-models/splice.data"}>{"the data"}</a>{" and "}<a href={"/learn-code/hyena-long-convolution-models/splice.names"}>{"its original metadata"}</a>{" into one directory. The program needs Python, NumPy, PyTorch and scikit-learn. In a fresh environment run:"}</Prose>

<CodeBlock language={"text"}>{"python -m pip install numpy torch scikit-learn\npython splice_models.py"}</CodeBlock>

<Prose>{"The retained run used Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu and scikit-learn 1.9.1, with two CPU threads. It trains four predeclared fits: a positional one-hot linear classifier with seed 29, gated sequence models with seeds 29 and 71, and an ungated sequence model with seed 29. All use Adam at learning rate 0.003, batches of 128, 80 epochs and gradient norm clipping at 1. The epoch with smallest validation cross-entropy is retained, with the earliest exact tie. The assessment data does not choose the epoch."}</Prose>

<Prose>{"The positional linear classifier is a meaningful baseline: the target boundary is always at the same location, so a coefficient for “G at index 30” can be informative. The ungated model retains its nonlinear feed-forward blocks but sets the two mixing gates to one; it is "}<strong>{"not a purely linear network"}</strong>{". Its otherwise retained projections include unused gate branches, so its nominal parameter count should not be read as equal effective capacity."}</Prose>

<Prose>{"The training program contains data grouping, encoding, the filter network, FFT convolution, residual blocks, loss, optimizer, epoch selection, confusion matrices and saved weights. This is the core mixing function used there:"}</Prose>

<SequenceBlockFigure/><HyenaProgram filename="splice_models.py"/><HyenaProgram filename="author_calculations.py"/>

<CodeBlock language={"python"}>{"def causal_convolution(values, kernel):\n    # values: batch, length, channels; kernel: length, channels\n    length = values.shape[1]\n    size = 1 << (2 * length - 2).bit_length()\n    spectrum = torch.fft.rfft(values, n=size, dim=1)\n    filter_spectrum = torch.fft.rfft(kernel, n=size, dim=0)\n    return torch.fft.irfft(\n        spectrum * filter_spectrum[None], n=size, dim=1\n    )[:, :length]"}</CodeBlock>

<Prose>{"Padding is along the sequence axis, not the channel axis. The filter broadcasts across the batch. The full source imports "}<code>{"torch"}</code>{"; this excerpt belongs to that runnable program. It produces "}<code>{"splice-results.json"}</code>{" and "}<code>{"splice-fits.npz"}</code>{", containing role IDs, validation histories, selected weights and logits. You can reproduce the experiment without downloading a large pretrained model."}</Prose>

<H3>{"Read the actual outcome"}</H3>

<NeuralTable caption={"Read the actual outcome"} headers={[<>{"Model"}</>,<>{"Selected epoch"}</>,<>{"Validation errors / 460"}</>,<>{"Assessment errors / 416"}</>]} rows={[[<>{"Positional linear, seed 29"}</>,<>{"42"}</>,<>{"32"}</>,<>{"30"}</>],[<>{"Gated Hyena-style, seed 29"}</>,<>{"11"}</>,<>{"45"}</>,<>{"40"}</>],[<>{"Ungated sequence model, seed 29"}</>,<>{"25"}</>,<>{"82"}</>,<>{"86"}</>],[<>{"Gated Hyena-style, seed 71"}</>,<>{"11"}</>,<>{"50"}</>,<>{"57"}</>]]} />

<Prose>{"The positional baseline makes fewer errors than either small gated fit. That is an informative result: a fixed-location motif task can reward a direct representation. Gating helps relative to this ungated comparison, but changing seed also changes outcomes. These fits are not a universal ranking of linear classifiers, Hyena or other architectures."}</Prose>

<LearningCurvesFigure/>

<Prose>{"For the gated seed-29 model, the assessment confusion matrix is:"}</Prose>

<NeuralTable caption={"Read the actual outcome"} headers={[<>{"True \\ Predicted"}</>,<>{"EI"}</>,<>{"IE"}</>,<>{"N"}</>]} rows={[[<>{"EI"}</>,<>{"76"}</>,<>{"3"}</>,<>{"6"}</>],[<>{"IE"}</>,<>{"1"}</>,<>{"74"}</>,<>{"8"}</>],[<>{"N"}</>,<>{"13"}</>,<>{"9"}</>,<>{"226"}</>]]} />

<Prose>{"Rows reveal different errors: 9 of 85 EI examples, 9 of 83 IE examples and 22 of 248 N examples are misclassified. A single overall accuracy hides those denominators. A confidence value alone also does not establish calibration."}</Prose>

<H3>{"Change a sequence and observe a hypothesis"}</H3>

<Prose>{"Validation source row 3 is an EI example whose two bases immediately after the central divider are "}<code>{"GT"}</code>{". The gated seed-29 fit assigns class probabilities approximately "}<code>{"[0.99284,0.00648,0.00068]"}</code>{" in EI/IE/N order. Replacing those two input bases by "}<code>{"AA"}</code>{", without refitting, produces "}<code>{"[0.08177,0.28564,0.63259]"}</code>{": the prediction changes to N."}</Prose>

<Prose>{"This establishes that the fitted model is sensitive to that edit. It does not provide a newly measured biological label for the edited sequence. The original label belongs to the observed record. Sequence perturbation is a way to investigate a model’s behavior; a biological claim requires separate evidence."}</Prose>

<CounterfactualFigure/>

<DnaStudy/>

<Prose>{"A causal network can classify this *whole observed window* using its final position because that position has access to all 60 bases. It does not forecast a central boundary before the right-hand context arrives. If an application must decide before the right-hand context arrives, train and evaluate it using only the context available at that decision."}</Prose>

<H2>{"8. Deeper route: long filters, streaming and compact state"}</H2>

<Prose>{"FFT convolution is attractive when the input segment is already available. Autoregressive generation presents a different workload: produce one new token, feed it back, then produce the next. Re-running a full FFT for each token wastes earlier work. But “therefore Hyena must re-run a full FFT” is too strong. There are several computational representations."}</Prose>

<NeuralTable caption={"8. Deeper route: long filters, streaming and compact state"} headers={[<>{"Representation"}</>,<>{"What is retained"}</>,<>{"Work for an additional output, per channel"}</>,<>{"Appropriate question"}</>]} rows={[[<>{"Direct FIR, length "}<InlineMath>{"K"}</InlineMath>{""}</>,<>{"A buffer of the latest "}<InlineMath>{"K"}</InlineMath>{" inputs"}</>,<>{""}<InlineMath>{"O(K)"}</InlineMath>{""}</>,<>{"Is the filter short enough for a simple exact streaming implementation?"}</>],[<>{"Whole-sequence FFT"}</>,<>{"A segment and transform workspaces"}</>,<>{""}<InlineMath>{"O(P\\log P)"}</InlineMath>{" for the segment"}</>,<>{"Are enough input samples available to exploit parallel batch computation?"}</>],[<>{"Blocked convolution"}</>,<>{"Input blocks and overlap/history"}</>,<>{"Depends on block and filter sizes"}</>,<>{"Can we trade buffering latency against throughput?"}</>],[<>{""}<InlineMath>{"d"}</InlineMath>{"-mode recurrent filter"}</>,<>{""}<InlineMath>{"d"}</InlineMath>{" state values"}</>,<>{""}<InlineMath>{"O(d)"}</InlineMath>{""}</>,<>{"Does this filter have, or admit a good approximation by, a compact recurrence?"}</>]]} />

<Prose>{"These counts exclude projections, gates and feed-forward layers. A fixed-"}<InlineMath>{"K"}</InlineMath>{" buffer is constant in the *total stream duration*, yet can still be large in "}<InlineMath>{"K"}</InlineMath>{". An unrestricted implicit filter whose support grows with context does not automatically have a fixed small state."}</Prose>

<H3>{"Blocks must include the overlapping tail"}</H3>

<Prose>{"Split inputs "}<code>{"[1,2,3,4]"}</code>{" into "}<code>{"[1,2]"}</code>{" and "}<code>{"[3,4]"}</code>{". Convolve each block with the filter, shift the second result by two positions, then "}<strong>{"add the overlapping outputs"}</strong>{". This overlap-add construction gives the same linear convolution. If each chunk is filtered independently and only its first two outputs are retained, contributions crossing the chunk boundary vanish."}</Prose>

<OverlapFigure/>

<H3>{"Complete the blocked FFT route before choosing a package"}</H3>

<Prose>{"The preceding "}<code>{"overlap_add"}</code>{" oracle uses direct convolution in each block. It makes the overlap ledger clear but does not acquire FFT complexity just because the blocks are small. "}<a href={"/learn-code/hyena-long-convolution-models/blocked_convolution.py"}>{"blocked_convolution.py"}</a>{" supplies the corresponding efficient transform route: transform the kernel once, zero-pad each input block to prevent circular wraparound, multiply spectra, inverse-transform, and add the entire valid tail into its correct global positions. It never constructs a dense Toeplitz matrix."}</Prose>

<Prose>{"Keep it beside "}<a href={"/learn-code/hyena-long-convolution-models/convolution_mechanisms.py"}>{"convolution_mechanisms.py"}</a>{", then run "}<code>{"python blocked_convolution.py"}</code>{" with NumPy 2.3.5 and SciPy 1.18.1. The ordinary comparison uses "}<code>{"scipy.signal.oaconvolve(values, kernel, mode=\"full\")[:len(values)]"}</code>{". The "}<code>{"same"}</code>{" mode is centered, so replacing the causal slice by "}<code>{"mode=\"same\""}</code>{" changes alignment. "}<a href={"https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.oaconvolve.html"}>{"SciPy's mode and overlap-add contract"}</a>{"."}</Prose>

<CodeBlock language={"python"}>{"\"\"\"Bounded FFT blocks, a reusable filter spectrum, and the ordinary SciPy route.\n\nNumPy 2.3.5 and SciPy 1.18.1 targets. Real, finite, nonempty 1-D arrays.\nThe output is the first len(values) samples of linear causal convolution.\n\"\"\"\nimport numpy as np\nfrom scipy.signal import oaconvolve\nfrom convolution_mechanisms import direct, overlap_add\n\n\ndef overlap_add_fft(values, kernel, block_size):\n    values, kernel = np.asarray(values, dtype=float), np.asarray(kernel, dtype=float)\n    if values.ndim != 1 or kernel.ndim != 1 or min(len(values), len(kernel)) == 0:\n        raise ValueError(\"Need two nonempty vectors\")\n    if block_size < 1:\n        raise ValueError(\"block_size must be positive\")\n    size = 1 << (block_size + len(kernel)-2).bit_length()\n    filter_spectrum = np.fft.rfft(kernel, n=size)\n    # All requested output samples must be retained; no all-pairs Toeplitz matrix.\n    output = np.zeros(len(values))\n    for start in range(0, len(values), block_size):\n        block = values[start:start+block_size]\n        transformed = np.fft.rfft(block, n=size)\n        convolved = np.fft.irfft(transformed*filter_spectrum, n=size)\n        count = min(len(block)+len(kernel)-1, len(values)-start)\n        output[start:start+count] += convolved[:count]\n    return output\n\n\ndef main():\n    values = np.array([2., -1., 3., 0., 1., 2., -.5])\n    kernel = np.array([.5, 1., -.25, .125])\n    expected = direct(values, kernel)\n    library = oaconvolve(values, kernel, mode=\"full\")[:len(values)]\n    np.testing.assert_allclose(library, expected, atol=1e-12)\n    for block_size in (1, 2, 3, 8):\n        actual = overlap_add_fft(values, kernel, block_size)\n        np.testing.assert_allclose(actual, expected, atol=1e-12)\n        np.testing.assert_allclose(actual, overlap_add(values, kernel, block_size), atol=1e-12)\n        print(\"block\", block_size, \"output\", actual, \"maximum error\", abs(actual-expected).max())\n\n\nif __name__ == \"__main__\":\n    main()"}</CodeBlock>

<Prose>{"For input length N, filter length M, block size B and transform length F≥B+M−1, work is O(F log F + ceil(N/B) F log F). Retained workspace beyond the O(N) returned output is O(F); the transformed kernel is reused. This bound describes this program, not a measured speed victory. Very short kernels can favor direct convolution; unsuitable B can waste work. The authoring probe's seven-sample fixture gives "}<code>{"[1, 1.5, 0, 3.5, −.375, 2.375, 1.5]"}</code>{" at B=1,2,3,8, with maximum observed float64 difference 4.45×10⁻¹⁶ from the direct oracle. No training run or speed measurement was added."}</Prose>

<BlockedFftLab/><HyenaProgram filename="blocked_convolution.py"/>

<Prose>{"The trainable ordinary workflow remains "}<code>{"splice_models.py::causal_convolution"}</code>{", "}<code>{"ImplicitFilter"}</code>{" and "}<code>{"SequenceBlock"}</code>{": Torch FFT operations preserve the gradient path into the filter-generating network and both input-dependent gates. NumPy/SciPy are useful numeric references, not differentiable replacements inside that Torch training graph. The two-gate block is an explicitly scoped Hyena-style operation; StripedHyena's other filters/attention layers and pretrained checkpoints remain distinct family examples."}</Prose>

<Prose>{""}<strong>{"Change the contract."}</strong>{" Use a ten-sample signal, a five-tap signed filter and B=3; compare the whole output including the incomplete last block. Then change a future input only."}</Prose>

<details><summary>Hint and reasoned solution</summary>

<Prose>{"Choose F≥7, hence F=8 for this power-of-two implementation. The final input block contains one real sample; its convolution still has up to five valid terms, of which only positions inside the requested ten-sample output are returned. Add tails rather than overwrite them. A changed input at index j must leave output indices below j unchanged to rounding precision. A centered "}<code>{"same"}</code>{" slice or an undersized transform can violate that causality. The all-zero kernel is a useful null; B>N must still match the direct oracle. To stream output instead of retaining N samples, emit only a block's finalized prefix and carry the overlapping tail, taking care when M−1 exceeds B."}</Prose>

</details>

<H3>{"Some long filters are exactly recurrent"}</H3>

<Prose>{"For "}<InlineMath>{"h_r=a^r"}</InlineMath>{", define "}<InlineMath>{"s_t=a s_{t-1}+u_t"}</InlineMath>{", with "}<InlineMath>{"s_{-1}=0"}</InlineMath>{". Expanding the recurrence gives "}<InlineMath>{"s_t=u_t+a u_{t-1}+a^2u_{t-2}+\\cdots"}</InlineMath>{". That is exactly convolution with the exponential filter. The past has been compressed into one number because this particular filter has algebraic structure."}</Prose>

<Prose>{"A sum of exponentials uses several modes:"}</Prose>

<div className="neural-equation"><MathBlock>{"h_r=\\sum_{n=1}^{d}R_n\\lambda_n^r,\\qquad\ns_{n,t}=\\lambda_n s_{n,t-1}+u_t,\\qquad\ny_t=\\sum_n R_n s_{n,t}."}</MathBlock></div>

<Prose>{"Each "}<InlineMath>{"\\lambda_n"}</InlineMath>{" determines a mode’s retention and oscillation; "}<InlineMath>{"R_n"}</InlineMath>{" determines its contribution. A negative real pole alternates sign. Complex-conjugate pairs can produce real decaying oscillations. For an indefinitely sustained stable filter, poles inside the unit circle give decaying modes; a finite FIR remains well-defined without that infinite-horizon requirement."}</Prose>

<Prose>{"Take "}<InlineMath>{"R=[0.6,0.4]"}</InlineMath>{", "}<InlineMath>{"\\lambda=[0.5,-0.25]"}</InlineMath>{". The first six coefficients are "}<code>{"[1,0.2,0.175,0.06875,0.0390625,0.018359375]"}</code>{". For inputs "}<code>{"[1,-2,0.5,3,-1,2]"}</code>{", the output is "}<code>{"[1,-1.8,0.275,2.81875,-0.4109375,2.299609375]"}</code>{"."}</Prose>

<ModesFigure/>

<Prose>{"This complete NumPy program computes the recurrent output and independently checks convolution:"}</Prose>

<CodeBlock language={"python"}>{"import numpy as np\n\nvalues = np.array([1., -2., .5, 3., -1., 2.])\nresidues = np.array([.6, .4])\npoles = np.array([.5, -.25])\nstate = np.zeros(2)\noutputs = []\nfor value in values:\n    state = poles * state + value\n    outputs.append(residues @ state)\nkernel = (residues[:, None] * poles[:, None] ** np.arange(len(values))).sum(0)\nreference = np.convolve(values, kernel)[:len(values)]\nprint(np.round(outputs, 9))\nprint(np.allclose(outputs, reference, atol=1e-12))"}</CodeBlock>

<Prose>{"After the first three inputs, the state is "}<code>{"[-0.25,1.0625]"}</code>{". Carry that state into the remaining inputs to obtain the same continuation. Resetting it gives the different suffix "}<code>{"[3,-0.4,2.325]"}</code>{". State must travel across chunks just as convolution tails must travel across blocks."}</Prose>

<Prose>{"Inside a gated block, these modes accumulate the "}<strong>{"gated value"}</strong>{" "}<InlineMath>{"k_t v_t"}</InlineMath>{"; "}<InlineMath>{"q_t"}</InlineMath>{" multiplies the read afterward. Cache the short-convolution history as well. An attention layer elsewhere in a hybrid does not execute these updates on behalf of the convolution layers."}</Prose>

<H3>{"When the recurrence is an approximation"}</H3>

<Prose>{"A general filter network need not output a small sum of exponentials. "}<a href={"https://proceedings.neurips.cc/paper_files/paper/2023/file/371355cd42caaf83412c3fbef4688979-Paper-Conference.pdf"}>{"Laughing Hyena Distillery"}</a>{" studies converting trained long filters into compact state-space approximations. The teaching idea is: choose a target state size, fit a recurrent filter to the original coefficients or transfer function, then check both the approximation and the model using it."}</Prose>

<Prose>{"An elementary error bound explains what must be controlled. If "}<InlineMath>{"e=h-\\hat h"}</InlineMath>{" is the filter error, then"}</Prose>

<div className="neural-equation"><MathBlock>{"|y_t-\\hat y_t|\n =\\left|\\sum_{j\\le t}e_{t-j}u_j\\right|\n \\le\\|u\\|_\\infty\\sum_{r\\ge0}|e_r|\n =\\|u\\|_\\infty\\|e\\|_1."}</MathBlock></div>

<Prose>{"For one fixed-gate sandwich, multiply this bound by "}<InlineMath>{"\\|q\\|_\\infty\\|k\\|_\\infty"}</InlineMath>{". Later nonlinear layers and changing internal inputs require additional analysis; a small filter error alone is not a universal guarantee of unchanged predictions."}</Prose>

<Prose>{"Truncate our six-coefficient example after lag 1. Its omitted coefficient sum is 0.301171875 and the input’s largest magnitude is 3, so the bound is 0.903515625. The observed maximum output change is 0.499609375. The bound is deliberately conservative and the truncation is deliberately visible. Both filters can run quickly, but they do not compute the same result."}</Prose>

<ApproximationFigure/>

<Prose>{"For a deeper connection, extend this known analytic two-mode filter through lag 10 and arrange its coefficients into a "}<strong>{"Hankel matrix"}</strong>{", whose entries are constant on anti-diagonals: "}<InlineMath>{"A_{ij}=h_{i+j}"}</InlineMath>{". For a sum of "}<InlineMath>{"d"}</InlineMath>{" exponential modes, "}<InlineMath>{"A_{ij}=\\sum_n R_n\\lambda_n^i\\lambda_n^j"}</InlineMath>{", so it is a sum of at most "}<InlineMath>{"d"}</InlineMath>{" rank-one matrices. Our 6×6 example has two substantive singular values, about 1.08698 and 0.13949; the others are numerical roundoff below "}<InlineMath>{"10^{-16}"}</InlineMath>{". A rapidly decaying Hankel spectrum suggests that a smaller state may approximate a filter. It does not say that every neural filter has low rank or that one finite matrix certifies all future lags. The full system-realization theory refines the indexing, assumptions and minimal-state statement."}</Prose>

<StreamingLab/>

<H2>{"9. Deeper route: what the family adds"}</H2>

<H3>{"Different ways to specify a filter"}</H3>

<Prose>{"The "}<a href={"/learn/path/full-curriculum/state-space-models-s4-mamba-mamba-2?module=deep-learning-fundamentals"}>{"S4/Mamba lesson"}</a>{" showed how a time-invariant state-space system yields a convolution kernel. Hyena often starts with a direct neural function of lag. These parameterizations impose different structure. Once a model makes its state transition or input maps depend on the current token, the overall operator need not be one fixed convolution. Mamba’s selectivity and Hyena’s surrounding gates should be described by their actual equations rather than called interchangeable implementations."}</Prose>

<Prose>{"Continuous kernel convolution, or CKConv, developed the idea of evaluating a learned kernel function at relative positions. S4 supplied a structured state-space route to long kernels. H3 combined short shifts, longer state-space filters and multiplicative paths to support content-dependent mechanisms. SaShiMi explored multiscale state-space sequence modeling for audio. Hyena belongs to this set of ideas; the architectural differences concern which filters, gates, resolutions and states are used. "}<a href={"https://arxiv.org/abs/2102.02611"}>{"CKConv"}</a>{", "}<a href={"https://arxiv.org/abs/2212.14052"}>{"H3"}</a>{" and "}<a href={"https://arxiv.org/abs/2202.09729"}>{"SaShiMi"}</a>{" are useful primary pointers for those branches."}</Prose>

<Prose>{"For ordinary row-softmax attention, "}<InlineMath>{"A_{tj}\\propto\\exp(q_t^\\top k_j/\\sqrt{d_k})"}</InlineMath>{" over allowed positions. Its coefficients are nonnegative and normalized. Hyena’s signed distance-and-gate coefficients generally compute a different operator. "}<a href={"/learn/path/full-curriculum/sparse-linear-attention-variants?module=deep-learning-fundamentals"}>{"Sparse and linear attention"}</a>{" distinguishes changing an operator from finding a different algorithm for the same one."}</Prose>

<MechanismMapFigure/>

<H3>{"Why genomics is an interesting application"}</H3>

<Prose>{"Single DNA-base resolution and long context can both matter: a local change and a distant regulatory region may affect what a model should infer. HyenaDNA pretrained a causal decoder on the human reference genome, using individual nucleotide symbols and special tokens, then adapted it to downstream tasks. Its method includes length warm-up, task-specific supervised adaptation and learned soft prompts. Soft prompts are trainable input vectors; they are not newly observed DNA bases. The paper also studied a separately implemented bidirectional ablation. Its main pretrained architecture was not bidirectional. "}<a href={"https://arxiv.org/html/2306.15794v2"}>{"HyenaDNA"}</a>{""}</Prose>

<Prose>{"The practical workflow is to distinguish the pretraining task from your labeled downstream question, preserve the chosen tokenizer and positional conventions, train or validate the downstream head, and establish an appropriate biological split. A pretrained backbone does not arrive with a valid classifier for every possible new label. Our 60-base experiment demonstrates the trainable mechanism and model inspection; it does not reproduce the paper’s long-context experiments."}</Prose>

<H3>{"StripedHyena and StripedHyena 2"}</H3>

<Prose>{"StripedHyena’s 2023 release combined attention, gated convolutions and feed-forward layers. Its recurrent convolution representation and its attention KV cache coexist during generation. The "}<a href={"https://www.together.ai/blog/stripedhyena-7b"}>{"original release article"}</a>{" describes the architecture and the measured workloads of that release; its historical timing claims are not timings for the small program here."}</Prose>

<Prose>{"The 2025 StripedHyena 2 work takes a more specific mixture: short explicit filters for local mixing, medium explicit filters with decay regularization, and long implicit filters expressed using exponential modes, interleaved with attention. The long modal filters permit recurrent evaluation; finite short/medium filters retain bounded input history. Filter sharing across channel groups helps organize hardware-efficient computation. These are deliberate choices of memory range and algorithm, rather than an assertion that every layer should use the longest possible filter. The work supports the Evo 2 genomic model family. "}<a href={"https://arxiv.org/html/2503.01868v1"}>{"StripedHyena 2 methods"}</a>{""}</Prose>

<StripedFigure/>

<H3>{"Complexity is a guide to what to measure"}</H3>

<Prose>{"For "}<InlineMath>{"D"}</InlineMath>{" channels and a fixed small number of long-filter stages, convolution mixing has "}<InlineMath>{"O(DL\\log L)"}</InlineMath>{" work, while dense input/output projections and feed-forward updates add "}<InlineMath>{"O(LD^2)"}</InlineMath>{". Filter generation also costs work at every evaluated position. Attention’s pairwise mixing has "}<InlineMath>{"O(L^2D)"}</InlineMath>{" work, but optimized exact attention need not materialize an "}<InlineMath>{"L\\times L"}</InlineMath>{" score array in main device memory. A memory-efficient attention algorithm and a different sequence operator answer different questions."}</Prose>

<Prose>{"An operation-count sketch such as "}<InlineMath>{"L^2/(L\\log_2 L)=L/\\log_2 L"}</InlineMath>{" is not a speedup measurement. FFT workspaces, complex arithmetic, dtype, kernel fusion, hardware utilization, batch size and projections all affect wall time. "}<a href={"https://arxiv.org/abs/2311.05908"}>{"FlashFFTConv"}</a>{" investigates why FFT-based sequence convolutions need hardware-aware algorithms; later multi-hybrid work also uses blocked direct convolution for selected filter lengths."}</Prose>

<BenchmarkFigure/>

<Prose>{"When diagnosing a model, distinguish an operator bug from a modeling problem. Padding, lag direction, shifted targets, short-filter causality and moving position coordinates can change the intended computation. A correct model can still underfit, overfit, exploit a shortcut or fail on a new context length. Inspect the data roles, learning curves and changed-input behavior before attributing a result to the architecture family."}</Prose>

<Prose>{"The next topic in this module is "}<a href={"/learn/path/full-curriculum/ring-attention-sequence-parallelism?module=deep-learning-fundamentals"}>{"Ring Attention & Sequence Parallelism"}</a>{". It changes how attention work is distributed across devices. Keep that distinction: Hyena changes the sequence mixer; Ring Attention distributes an attention computation. Later, "}<a href={"/learn/path/full-curriculum/hybrid-ssm-transformer-architectures-jamba?module=deep-learning-fundamentals"}>{"Hybrid SSM–Transformer Architectures"}</a>{" combines state and attention paths in one model."}</Prose>

<section className="lesson-ending lesson-ending--practice" data-lesson-ending="practice"><H2>{"10. Practice: reason, calculate and transfer"}</H2>

<Prose>{"Try each task before opening its hint. The first six establish the core route. The final four develop architecture and streaming reasoning. A successful answer explains which computation or information boundary produced its result."}</Prose>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"1. A filter with a negative echo"}</H3>

<Prose>{"For input "}<code>{"[2,0,-1,3]"}</code>{" and filter "}<code>{"[1,-0.5,0.25]"}</code>{", compute the first four causal outputs. Which lag is absent at position 1, and why?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Write current, one-step-old and two-step-old contributions separately. Inputs before the sequence start are zero."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The outputs are "}<code>{"[2,-1,-0.5,3.5]"}</code>{". At position 1 there is no two-step-old input, so the lag-2 term contributes zero. At position 3 the contributions are "}<InlineMath>{"3+(-0.5)(-1)+0.25(0)=3.5"}</InlineMath>{". A negative filter coefficient can increase the output when the input it multiplies is also negative."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"2. Padding is part of the mathematics"}</H3>

<Prose>{"A length-6 signal is convolved with a length-4 filter. What is the smallest sufficient transform length? What convenient power-of-two length could you choose? Does keeping only the first six results make an unpadded length-6 transform causal?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Count the support of the full linear convolution before deciding what to crop."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The full length is "}<InlineMath>{"6+4-1=9"}</InlineMath>{". Any supported FFT length at least 9 suffices; 16 is a convenient power of two. Cropping a circular convolution does not remove tail contributions that have already wrapped into the first six positions. Padding must prevent that aliasing before the transform-domain product is inverted."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"3. Change the sender, then the receiver"}</H3>

<Prose>{"For "}<InlineMath>{"v=[2,1,-1]"}</InlineMath>{", "}<InlineMath>{"h=[1,0.5,0.25]"}</InlineMath>{", "}<InlineMath>{"k=[1,0,2]"}</InlineMath>{", and "}<InlineMath>{"q=[1,-1,0.5]"}</InlineMath>{", calculate the output. If "}<InlineMath>{"k_1"}</InlineMath>{" becomes 1, which outputs change? What happens if instead only "}<InlineMath>{"q_2"}</InlineMath>{" becomes zero?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"First compute "}<InlineMath>{"kv"}</InlineMath>{"; only then convolve and multiply by "}<InlineMath>{"q"}</InlineMath>{"."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{""}<InlineMath>{"kv=[2,0,-2]"}</InlineMath>{", the convolution is "}<code>{"[2,1,-1.5]"}</code>{", and the output is "}<code>{"[2,-1,-0.75]"}</code>{". Opening "}<InlineMath>{"k_1"}</InlineMath>{" gives "}<code>{"[2,-2,-0.5]"}</code>{": positions 1 and 2 change, while position 0 is unaffected. Setting only "}<InlineMath>{"q_2=0"}</InlineMath>{" erases only this block’s final output, giving "}<code>{"[2,-1,0]"}</code>{". In a full residual network other paths can still carry information."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"4. Is the matrix an explanation of everything?"}</H3>

<Prose>{"An author plots "}<InlineMath>{"H(q(u),k(u))"}</InlineMath>{" and says its entry "}<InlineMath>{"H_{tj}"}</InlineMath>{" is the derivative of the final network output with respect to input "}<InlineMath>{"u_j"}</InlineMath>{". Identify the missing terms and propose an accurate caption."}</Prose>

<details><summary>Hint</summary>

<Prose>{"The input affects more than the value stream. Consider the product rule and later layers."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Changing "}<InlineMath>{"u_j"}</InlineMath>{" can change "}<InlineMath>{"q"}</InlineMath>{", "}<InlineMath>{"k"}</InlineMath>{", "}<InlineMath>{"v"}</InlineMath>{", normalization and later updates. The full derivative includes these dependencies and all paths to the final output. A suitable caption is: “Conditional mixing coefficients of this channel with its current gates held fixed.” To study final sensitivity, compute or perturb the full model separately and state what that analysis measures."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"5. A classifier has learned something—but what?"}</H3>

<Prose>{"A positional linear baseline beats a gated sequence model on a centered splice task. Give two reasonable explanations and one additional evaluation that would answer a different, clearly stated question. Why should you not keep trying variants against the same assessment labels?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use the fixed boundary, optimization and data-role definitions. A larger model is not the only possible change."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The baseline can directly weight each known motif position; the sequence model must learn a useful transformation and may overfit or optimize less effectively with this data budget. A separate, appropriately constructed gene-family-held-out dataset could ask about transfer beyond related biological sequences. Alternatively a newly designed variable-boundary task could ask whether a model locates an unknown site, with labels and input availability defined accordingly. Repeatedly choosing variants based on assessment performance turns that set into validation data, so its original evaluation role is lost."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"6. An unchanged prefix should stay unchanged"}</H3>

<Prose>{"A model uses a causal FFT and recalculates position features as "}<InlineMath>{"r/(L-1)"}</InlineMath>{" for every input length. Appending tokens changes early outputs. Explain the bug and give two checks that distinguish it from ordinary float roundoff."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Compare the actual coefficient vectors used for a prefix alone and for that prefix inside a longer sequence."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The lag coordinate changes with active length, so the filter is different even at old lags. Fix a reference coordinate system and slice it consistently. Check that the old filter coefficients agree across prefix lengths, then compare full-network prefix outputs against a direct causal implementation at an appropriate numerical tolerance. Also alter only future token values at a fixed length. This separates a moving-coordinate bug from future-value leakage and small arithmetic differences."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"7. A long filter with one state"}</H3>

<Prose>{"For "}<InlineMath>{"h_r=0.75^r"}</InlineMath>{" and input "}<code>{"[2,-1,0,1]"}</code>{", compute the recurrent state/output. Continue after the second input using the saved state; compare with resetting it."}</Prose>

<details><summary>Hint</summary>

<Prose>{"Apply "}<InlineMath>{"s_t=0.75s_{t-1}+u_t"}</InlineMath>{", starting from zero."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The outputs are "}<code>{"[2,0.5,0.375,1.28125]"}</code>{". The state after the second input is 0.5. Carrying it gives the correct suffix "}<code>{"[0.375,1.28125]"}</code>{"; resetting gives "}<code>{"[0,1]"}</code>{". A state is not optional bookkeeping: it encodes earlier contributions to later outputs."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"8. A finite approximation bound"}</H3>

<Prose>{"Over the horizon of interest, an original filter is "}<code>{"[1,0.4,0.2,0.1]"}</code>{" and an approximation is "}<code>{"[1,0.4,0,0]"}</code>{". Inputs are bounded by magnitude 2. Bound the maximum convolution-output error. How does the bound change for fixed sending gates of magnitude at most 3 and receiving gates at most 0.5?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Use the absolute omitted coefficient mass. Keep gates separate from inputs."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"The omitted mass is 0.3, so the convolution bound is "}<InlineMath>{"2\\times0.3=0.6"}</InlineMath>{". The fixed-gate bound is "}<InlineMath>{"0.5\\times3\\times0.6=0.9"}</InlineMath>{". Cancellation can make the observed error smaller. This is a bound for the stated finite filter and fixed path, not for an unmeasured infinite extension or arbitrary downstream network."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"9. A copy-task shortcut"}</H3>

<Prose>{"An experiment always asks the model to copy each input token exactly five positions later. Construct a filter that solves it without content-dependent gates. Redesign the task to test content-dependent addressing instead."}</Prose>

<details><summary>Hint</summary>

<Prose>{"An impulse response can be a pure delay. A key-value query can require different delays in different examples."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Set "}<InlineMath>{"h_5=1"}</InlineMath>{" and every other coefficient to zero; then "}<InlineMath>{"y_t=u_{t-5}"}</InlineMath>{", channel by channel. Use zero initial history. For addressing, sample new key-value associations per example, vary their order and distances, and query one of the keys at the end. Hold out newly generated examples and compare controlled baselines. Fixed-delay success alone cannot establish the harder capability; conversely this construction does not prove that every ungated deep network must fail the redesigned task."}</Prose>

</details></div>

<div className="lesson-exercise" data-lesson-exercise=""><H3>{"10. Plan an honest efficiency comparison"}</H3>

<Prose>{"You have an FFT sequence mixer and an optimized attention implementation. Design separate experiments for full-sequence training and autoregressive decoding. What changes when the convolution is replaced by a modal approximation?"}</Prose>

<details><summary>Hint</summary>

<Prose>{"Separate the mathematical operator, the workload and the implementation. Include prediction quality in an approximation study."}</Prose>

</details>

<details><summary>Solution</summary>

<Prose>{"Fix model dimensions, input lengths, batch, dtype, device and implementation versions. For training, measure the specified forward/backward/update scope with warm-up and synchronization, recording peak storage and excluding setup consistently. For decoding, separate prompt prefill from incremental steps, specify generated length and cache/state handling, and record latency or throughput with its batch. A modal approximation changes the filter unless exact equivalence is established; report filter error and relevant output/task differences as well as runtime and memory. Do not divide asymptotic expressions and label the result a measured speedup."}</Prose>

</details></div></section>

<section className="lesson-ending lesson-ending--resources" data-lesson-ending="resources"><H2>{"11. References and another way to learn"}</H2>

<Prose>{"These links supplement the self-contained route above. Read a resource for a particular question rather than treating every linked model as a prerequisite."}</Prose>

<ul><li>{""}<a href={"https://arxiv.org/html/2302.10866v2"}>{"Hyena Hierarchy — original paper"}</a>{": the formal hierarchy, synthetic task design, filter parameterization and matrix interpretation. Read §3 after the gate example; Appendix B derives the conditional matrix, and Appendix D explains frequency-rich initialization."}</li><li>{""}<a href={"https://hazyresearch.stanford.edu/blog/2023-03-07-hyena"}>{"Hazy Research’s illustrated Hyena introduction"}</a>{": a more informal creator explanation of why gating and long filters were combined. Its reported comparisons describe the 2023 experiments."}</li><li>{""}<a href={"https://icml.cc/virtual/2023/poster/24143"}>{"ICML 2023 Hyena presentation page"}</a>{": the conference page identifies the paper, authors, poster and a video section. It offers a presentation route alongside the article. The page and identity were verified; the recording was not watched for this manuscript and playback availability may depend on the host."}</li><li>{""}<a href={"https://arxiv.org/html/2306.15794v2"}>{"HyenaDNA paper"}</a>{" and "}<a href={"https://github.com/HazyResearch/hyena-dna/blob/main/standalone_hyenadna.py"}>{"standalone author implementation"}</a>{": read method §3 for the practical gate-filter-gate operator, tokenization, length warm-up and adaptation. In code, inspect "}<code>{"fftconv"}</code>{", "}<code>{"PositionalEmbedding"}</code>{", "}<code>{"HyenaFilter"}</code>{" and "}<code>{"HyenaOperator"}</code>{" with the actual maximum-length contract."}</li><li>{""}<a href={"https://proceedings.neurips.cc/paper_files/paper/2023/file/371355cd42caaf83412c3fbef4688979-Paper-Conference.pdf"}>{"Laughing Hyena Distillery"}</a>{": the deeper path from a trained convolution to a compact recurrence, including approximation objectives, modal interpolation, Hankel spectra and deployment. Our two-mode example supplies the prerequisite intuition."}</li><li>{""}<a href={"https://www.together.ai/blog/stripedhyena-7b"}>{"StripedHyena release"}</a>{" and "}<a href={"https://arxiv.org/html/2503.01868v1"}>{"StripedHyena 2 methods"}</a>{": read the architectural design and storage contracts before the dated benchmark tables. The second paper explains short, medium and long filters plus blocked/context-parallel algorithms."}</li><li>{""}<a href={"https://arxiv.org/abs/2311.05908"}>{"FlashFFTConv"}</a>{": an advanced systems branch about tensor-core use and memory traffic in sequence convolutions. This is a primary pointer, not a reproduced benchmark here."}</li><li>{""}<a href={"https://docs.pytorch.org/docs/main/generated/torch.fft.rfft.html"}>{"PyTorch real FFT reference"}</a>{": normalization, dimensions, padding/trim behavior and supported device/dtype combinations. Consult the documentation matching your installed version."}</li><li>{""}<a href={"https://archive.ics.uci.edu/dataset/69/molecular%2Bbiology%2Bsplice%2Bjunction%2Bgene%2Bsequences"}>{"UCI splice data and license"}</a>{", "}<a href={"https://www.genome.gov/about-genomics/educational-resources/fact-sheets/ribonucleic-acid-fact-sheet"}>{"NHGRI RNA background"}</a>{", and "}<a href={"/learn-code/hyena-long-convolution-models/data-provenance.md"}>{"this packet’s provenance"}</a>{": distinguish biological meaning, historical data and our exact experimental split."}</li><li>{""}<a href={"/learn-code/hyena-long-convolution-models/splice_models.py"}>{"Complete experiment"}</a>{", "}<a href={"/learn-code/hyena-long-convolution-models/convolution_mechanisms.py"}>{"mechanism calculations"}</a>{", and "}<a href={"/learn-code/hyena-long-convolution-models/author_calculations.py"}>{"saved-fit inspection program"}</a>{": runnable companions for reproducing the numbers and investigating your own inputs. After running the training program, place these companions beside it and run "}<code>{"python convolution_mechanisms.py"}</code>{" or "}<code>{"python author_calculations.py"}</code>{"; the mechanism companion also needs SciPy ("}<code>{"python -m pip install scipy"}</code>{"). They contain the complete procedures, not pseudocode for unreported model training."}</li></ul></section>
</div>};
