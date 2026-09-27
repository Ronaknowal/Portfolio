# Concept-level author review: Convolution, Pooling and Receptive Fields

Read the entire current production lesson: all eleven numbered sections, additional complete pullback implementation section, eight worked core/deeper questions and changed-input implementation tasks. Production JSX is the content checkpoint; canonical manuscript/renderer agree. Browser and independent review remain pending root.

## Full progression and decisions

| Location / concepts | Support retained or changed |
| --- | --- |
| Opening → travelling local rule, correlation convention | Existing editable patch/product/output lab, fixed edge filter, dense comparison and overlapping-data caveat provide intuition. Retained. |
| §2 channels, biases, NCHW, grouping | Existing two-channel arithmetic and live mixing lab support axes/weight sharing. Groups/depthwise remain an explicit preview owned by the later depthwise lesson. |
| §3 shared weights, loss reduction, input overlap, update | Full worked gradients, new loss and live investigation already explain accumulation. Runnable same-input torch program preserved. |
| §4 stride, dilation, padding, floor, alignment | Live geometry covers all controls. Existing same/valid/full/boundary cases retained. |
| §5 max/average routing, negative padding, adaptive/global pooling | Existing pooling lab and counterexamples sufficient except adaptive bin overlap. Added live five-input bin-membership diagram with output count 1–7, named index boundaries, exact mean and repeated-bin case; no prediction gate. |
| §6 receptive region → jump → center → holes → branch unions | Existing ancestry and dilation diagrams strong. Added intermediate explanation deriving each recurrence from old-grid gaps, stride movement and padding shift. |
| §7 complete model → measured outcomes → shifted deployment | All twelve actual digit fits, maps, scores and qualified evidence retained. Removed demand to write a prediction before a changed experiment; retained explicit change/metric protocol. |
| §8 architectural reach versus gradient influence, shift phase | Averaging-profile live lab, threshold qualification, boundary and anti-aliasing lab already sufficient. |
| §9 transpose versus inverse, nullspace, overlap artifacts, output size | Existing scatter/coverage lab retained. Added lengths five and six both mapping to three, then output_padding selecting the possible size without reconstructing information. |
| §10 direct sums, im2col/fold, implementation algorithms and costs | Patch-matrix figure, MAC arithmetic, layout distinction and algorithm preview retained. Added concrete two-route frozen-BN folding figure after formula, making alpha's distribution over weights and bias visible. |
| Complete scratch/pullback section | Full vectorized convolution and pooling derivatives, contraction axes, spatial scatter, complexity and library equality retained. Adaptive prefix/range-add implementation and 5→3/3→5/global changed cases retained. |
| Practice, readiness, next lesson, references | Eight core/deeper questions and all solutions preserved; no expanded prerequisite burden. |

## Sources actually inspected

- PyTorch native adaptive boundary functions: https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/AdaptivePooling.h — start_index/end_index read and checked against all new control settings.
- PyTorch convolution/BN tutorial, folding code and evaluation-only condition: https://docs.pytorch.org/tutorials/intermediate/torch_compile_conv_bn_fuser.html — supports the local algebra; linked as another implementation route. Compiler benchmarking portions were not used as our evidence.

## Verification

JSX parses. Every adaptive bin at seven available output counts checked against the native integer formula, default means checked. BN algebra checked on three patches for negative, zero and positive scales; default output is 1. Forward-size ambiguity independently checked. Re-running the generator produces identical topic bytes. Existing untouched native fits were not rerun. Scoped amber/charcoal DOM grids preserve index columns, wrap labels and provide real range/reset controls. Root must inspect actual desktop and narrow rendering before closure.
