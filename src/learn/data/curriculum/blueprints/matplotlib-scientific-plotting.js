// Authoring blueprint: Matplotlib & Scientific Plotting.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Turn validated arrays into interpretable figures by connecting the measurement, coordinate mapping and visual encoding, then diagnose a proposed model with residual plots.",
  "outcomes": [
    "Choose an encoding for a stated comparison and label its units",
    "Explain Figure/Axes/Axis/Artist ownership and predict scale or limit changes",
    "Trace histogram membership and density area and identify the exact quantity shown by an error bar",
    "Export and inspect a complete figure and solve a new diagnostic plotting task"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules",
    "NumPy: Arrays, Broadcasting & Vectorization"
  ],
  "sequence": [
    "State the question and distinguish trends, comparisons, relationships and distributions",
    "Build a complete curve and introduce the figure object hierarchy before extending it",
    "Move fixed data through linear, zoomed and log coordinate mappings",
    "Compare bar lengths and paired scatter observations without causal overclaims",
    "Trace observations into unequal-width bins and distinguish count height from density area",
    "Show raw observations alongside sample SD or SEM and expose duplicated-row pseudoreplication",
    "Map matrix entries through fixed colour limits; interpret masking and orientation",
    "Control shared axes, style, layout, accessibility and exported artifact properties",
    "Diagnose an invented straight-line proposal with observed-minus-predicted residuals",
    "Produce an independent two-panel distance/time report and predict a changed-data result"
  ],
  "visual": {
    "type": "Data-to-position mapping",
    "question": "Can the marker move without a measurement changing?",
    "interaction": "Select a measurement and switch limits/scales while its normalized coordinate calculation stays visible."
  },
  "visuals": [
    {
      "type": "Histogram membership and area",
      "question": "Does a taller rectangle always contain more observations?",
      "interaction": "Change bin edges and count/density; select a bin to see contributing values, width, mass and height."
    },
    {
      "type": "Raw values and specified interval",
      "question": "Why does changing SD to SEM shrink a bar without changing the runs?",
      "interaction": "Switch the interval definition and compare independent observations with literal copied rows."
    }
  ],
  "practice": {
    "task": "Build and explain a report using exported figures; create an independent distance/time comparison and residual panel, then change a measured value.",
    "success": "Nine runnable examples produce inspectable SVGs; plotted data, transforms, histogram normalization and interval endpoints agree with native Matplotlib/NumPy, and captions state meaning and limits."
  },
  "misconceptions": [
    "Axes is a panel, not the plural of Axis in the object API",
    "Log spacing represents ratios",
    "Unequal-width count areas are not probabilities",
    "Errorbar displays supplied intervals and does not infer a confidence level",
    "A residual pattern suggests investigation, not a unique causal diagnosis"
  ],
  "sources": [
    "https://matplotlib.org/stable/users/explain/quick_start.html",
    "https://matplotlib.org/stable/users/explain/artists/transforms_tutorial.html",
    "https://matplotlib.org/stable/gallery/statistics/histogram_normalization.html",
    "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html",
    "https://matplotlib.org/stable/users/explain/colors/colormaps.html"
  ],
  "depth": "core",
  "reviewFocus": "Keep actual exported figures as well as three bounded visual investigations; verify geometry against Matplotlib rather than against the JS model alone. Annotate the SciPy 2018 recording and companion notebooks with current-API caveats. The opening route continues to Git; actual navigation follows the selected route.",
  "designRecord": "docs/teaching/next-three-reimplementation.md"
};
