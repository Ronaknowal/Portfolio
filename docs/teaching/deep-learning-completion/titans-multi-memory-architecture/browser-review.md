# Titans — rendered review

Root integration review,27 September2026; actual production preview4197 at1280×960 and320×900. Initial function checks used build21:41:28. Final source-bound receipt follows the bounded display/accessibility fixes identified below.

## Five investigations

The associative-write lab starts with weight[1,0], query read1. Orthogonal second key leaves that read unchanged despite updating the second coordinate. Correlating the key gives preview gradient[−3,−3], update[1.5,1.5] and read2.5; clicking Write produces that exact state. Zero key/nonzero target gives loss12.5 but zero gradient/update. The retained-momentum case gives zero new gradient and update[0.5,0]; full decay still leaves next read0.5. The no-update setting is a genuine null. Blank rate preserves the last valid calculation and one Reset restores0.5.

Gated requests: carrying all state gives zero coordinate differences. Restarting the suffix changes token3 by[0.1532062,−0.1276064] and token4 by[−0.0250247,−0.0207976]. Reversing only the final token changes only that token's outputs; first three stay unchanged. Disabling writes from fresh zero memory yields zero outputs at every token, while the attention branch remains represented. The fixed request B stays independently initialized.

Outer differentiation: detaching the graph preserves forward loss2.434e−6 and reports None/disconnected, explicitly not numerical zero. Its recomputed finite difference−0.002493279307 agrees with analytic−0.002493279308. Changing rate to0.4 changes loss0.05700018178 despite the detached graph; finite difference0.383408303471 agrees with analytic0.383408303496.

Chunk comparison: default sequential endpoint1.25 versus anchor endpoint1.5. Changed practice starts1 with targets[3,−1], rate0.25: both first weights1.5, then sequential0.875 versus anchor1.0. Chunk size1 makes both0.875. Zero rate leaves both weights1 throughout.

Real rentals: seed3 onJuly2 has adaptive forecast5944.15199376 before arrival6227. Adding1000 to that arrival leaves the same-day forecast exactly unchanged but changes its subsequent loss0.0211005788→0.4340487721 and gradient norm0.392830324→1.7816692601. July3 adaptive forecast changes6470.74053991→6689.17398111; frozen-model forecast also changes because its now-observed input changes. The availability table names that causal distinction. Switching seed7 preserves edited7227 and inspected date.

Temporarily missing seed19 rental asset produces an explicit useful retry. Edited7227 survives failure and recovery; same-day original/current seed19 forecast remains5815.54500023 while post-arrival losses differ0.0446511669/0.5254388708. The dist-only asset was restored with exact SHA256d4eceb123255264a249498921b9486eb096185e9e0df92e8e8d232d826250f8e. No source/runtime asset is left hidden.

## Reading and display

Opened all four complete programs: neural_memory.py1627, rental_memory_study.py5738, memory_mechanisms.py6668, check_author_packet.py7790 displayed characters including headings. The library module is a reusable definition file; the three runnable programs show their entry points. Opened the changed chunk-practice solution and confirmed its complete derivation to0.875 versus1.0.

All23 bespoke figure SVGs had no painted text outside their viewport in tested states. Actual screenshots inspected the forecasting controls and enlarged causal-difference graph, not merely DOM dimensions. At320px, document scroll/client widths both305px; local diagram scroll is680px within244px and moves using ArrowRight. The arrival slider ArrowRight changes7227→7228 live. Console warnings/errors were empty.

Two findings were sent to the author: shared NeuralSelect did not associate visible labels with actual selects, requiring a topic-local accessible select; the signed forecast plot's negative tick extended5.72px left of its SVG and visually lost its minus sign. These are substantive accessibility/representation fixes. The final receipt records actual rechecks after correction; numerical experiments remain reusable because their sources did not change.

Final22:05:28 build recheck: exact accessible-name actions work for request continuation and saved seed. Maximum allowed arrival20000 produces signed difference ticks including−2178.95; actual desktop screenshot shows the entire minus sign. At320px its left painted clearance is16.146px inside400px SVG. All28 currently rendered SVGs have no out-of-bounds text; document remains305px without overflow. Both findings are closed.

Final independent-review delta on22:20:57 build: card1 key[2,2],target5,rate0.5,12same writes produces weight[−664300,−664300]. Preview explicitly names key·query=2 and exact next read1992905. New four-row legends show compact1.993e6/2.657e6 without clipping; all text in both vector/contribution plots fits their actual SVG. Desktop screenshot confirms complete readable legend and exact adjacent table; phone remains305/305px.
