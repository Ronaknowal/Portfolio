# Neural ODE browser review

Root reviewed the built lesson in the actual in-app browser at 1280×960 and 320×900. The initial full interaction pass used the 20:52 source; final footer and opening-copy corrections use immutable build `2026-09-26T21-08-02-718Z.json`. Numerical sources and all six investigation engines remained unchanged during that bounded correction.

## Live mechanisms

The fixed-step lab initially gives Euler error .09354521 versus RK4 5.061e-6, with 8 versus32 calls. Editing A00 to-.3 and initial z0 to1.2 changes endpoints to[-.452601,1.399663] and[-.391667,1.284375]. Equal-call selection grants Euler32 steps/calls and error .03151047 while keeping RK4's32 calls and6.187e-6 error. Zero field leaves both endpoints[1.2,.8] and both errors exactly0. Current RK stages show their separate tentative states and derivatives.

Adaptive default41 accepted+3 rejected attempts costs88 evaluations. Attempt0 starts at[1,.3], h.1, with Euler[.8,-1.2], Heun[.82,2.55] and ratio103.5895586; attempt1 retains time0 and the identical start, h.01, ratio8.5537225. Changing the fast rate to-30 and tolerance to.001 yields107 accepted+2 rejected and218 calls. The ratio strip includes all109 attempts, its threshold and original-value logarithmic labels. Zero-state mode gives zero trajectories/error/ratios and four total calls. Screenshots inspected actual intervals and the shared-ratio plot.

Gradient default shows the current trajectory ending .9800344 and loss .0001993126. Direct and finite-difference gradients agree at-.012800825 while continuous exact is-.0090209367 and approximate backsolve-.013696876. The changed decay case gives endpoint .31640625, direct .049108887, exact .061759395 and backsolve .036828763. Setting initial state0 gives zero for all four parameter gradients without NaN. The state/loss lane makes the distinct objective visible.

Lifted depth.7 gives A[-2,2.8], B[0,0], C[1,.7] with classes1/0/1; zero depth keeps all added coordinates0 and yields0/0/0. Record identities and horizontal positions remain fixed.

Observation query at.7 reads the post-update state-.144428174. Editing the future observation to2 leaves that result unchanged. Setting the middle value to0 gives .15557183; omitting it gives .222245466. Empty-history mode gives exactly0 and explicitly labels every observation absent. Duplicate times produce a local strict-order message, and Reset restores a valid timeline immediately. Presence, availability and actual applied events remain separate tables.

The density rotation control preserves area1, density1 and product1 with trace0. Restoring expansion/shear gives trace.1, area1.2214028, density.8187308 and log-density change-.2. Changing the velocity at the conditional-path crossing from0 to1 changes mean squared loss from4 to5; the display retains both target velocities.

## Actual saved classifier and recovery

Row70 defaults to raw[5.6,2.5,3.9,1.1]cm and NeuralODE37 probabilities[.014115504,.983768526,.002115970]. Petal length+ .6cm recomputes[.007935863,.984635726,.007428411]. Switching to Euler16, augmented model61 retains raw[5.6,2.5,4.5,1.1] and gives[.001553864,.985838393,.012607743]. All six hidden coordinates remain represented, with explicitly optional two-coordinate projection. Original/current comparisons use the same model and solver.

Withheld only generated dist `model-augmented_ode-13.json`, SHA2567a2fe72cd291f0d633ec5df317f595a3cadc1f948124f031f4f985021028c8fb. Selecting seed13 displayed a local model-loading error and retry button. Restored identical bytes and retried successfully: edited features and Euler16 survived, probabilities[.000358183,.993635257,.00600656]. No held asset remains. Reset after blank petal input restores3.9 on the first click.

## Rendering and accessibility

Desktop and phone screenshots inspected controls, flow diagrams, scalar gradients and real-model output. At320px, the document is305px client/305px scroll width. The measured-model diagram remains680px inside its219px keyboard-focusable region; ArrowRight moves its local scroll. Controls stack and retain amber focus. Complete solver bridge opens with2,584 visible characters and its main entry point; the complete training disclosure has7,197 characters. Practice6's solution correctly distinguishes .281534419 with a zero observation from .402192028 when omitted. Browser error/warning log is empty after recovery.

The initial SVG scan found one adaptive footer1.334px beyond its boundary. Author increased that SVG from260 to282px without altering its calculations or coordinates, and replaced the opening implementation-style paragraph with a learner-facing invitation. Final rendered verification is recorded in the source-bound browser receipt. All other inspected SVG text remained within its own nominal canvas. Local scrolling, not tiny text or document overflow, handles wide diagrams.

Final correction replay: actual282px SVG has20.667px painted footer clearance; screenshot reviewed. New opening invitation is visible. No numerical or other runtime changes occurred.
