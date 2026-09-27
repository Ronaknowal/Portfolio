# MoE rendered review

Root observed production preview4197 on desktop1280×900 and phone320×900. Initial runtime and final bounded figure change were reviewed; final build20:41:17 binds current sources.

Routing starts [15/7,6/7], IDs3/1. Expert2 mass5 changes IDs2/3 and output[7/9,20/9]; pinned original survives. Shared logit+1 preserves this. k1 selected-normalization yields[-1,4], selected gate1 and all local derivatives0. Full-softmax yields[-.38462,1.53846] and nonzero unselected-denominator derivatives. Equal-output preset plus selected normalization yields[2,-1] with zero local derivatives.

Capacity2 drops3 assignments and all of token4's expert sum. Capacity3 gives sums[1.5,2,2.5,2.5,1] and only1 dropped assignment. Moving token4 first at capacity2 keeps the same3 drops but no fully lost token, sums[1.5,1.5,2,1.5,1.5]. Dropless sums[1.5,2,2.5,2.5,1.5] preserve original return IDs. Reset returns original order and controls.

Balance edited row0[.2,.7,.1] gives q=P=[.25,.5,.25], global1.125. Shared offset+2 changes only numerical scale, z-loss4. Per-row aggregation gives1.95 versus global1.125. Peaked log-normalized rows yield hard count[1,0,0], objective2.94 and near-zero z-loss4.479e-32. Fine-expert budget retains115200 stored matrices and23040 activeMACs, doubles router480→960 and payload18432→36864bytes; remote fraction0 sets only remote payload0.

Real4,762-parameter workspace loads only on opening. Source2030 class7 probability.96999671; zero lower half changes predicted class3 and class7 probability.00776563, assignments[16,11,4,1]. Already-zero upper-left patch is exact null. Temperature2 preserves route IDs/counts with class7 delta−.0001896. Disable used expert0 gives class7+.00009525; source1762 never uses expert0 and disabling it is exact null. Keyboard Space selects pixel(0,2), intensity4→16 changes class0 probability.9517158→.95303778. Patch14/head1 controls update the inspected internal path, keeping attention donors distinct from expert gates and all16-coordinate returns available.

Measured outcomes show all12 paired clean/stress points on0–1 axes without connected seeds; screenshot inspected. Dense17 selection removes router counts with explicit explanation. MoE.1/73 has total counts[2547,1965,2410,2678]; class0 has[211,302,95,352], sum960. Curves use actual recorded checkpoints. Full CPU training disclosure loads7,817 displayed characters; first independent solution gives2.375 versus1.9 under its two normalizers.

At320px document305/305; controls stack, diagrams retain nominal440px and keyboard ArrowRight changes local scrollLeft in263px viewport. All visible SVG text bounds checked. Blank expert count then one Reset restores4. The opening E0 label was clipped by3.4CSSpx on desktop; the topic-local12px inset and347px height now give17.27CSSpx clearance. Final screenshot and320px complete text-bound scan show no clipped labels.

Actual generated preview model asset was held temporarily to exercise a failed request: friendly retry appears, exact SHA256 bytes restored, retry restores class7.96999671 and original routes. No recovery file remains. Warning/error console list empty. No model fitting or GPU timing was claimed by this browser review.
