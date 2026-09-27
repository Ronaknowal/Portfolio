# Copy confirmation at the button

28 September 2026. A successful copy now changes the clicked button to
**✓ Copied** with amber emphasis for 2.2 seconds. Its reserved width prevents a
layout jump. The status is announced accessibly; a failed copy shows useful
manual-copy guidance in the toolbar. Repeated clicks restart the interval, stale
requests cannot overwrite newer feedback, and unmount clears its timer.

The production build passed. Existing exact-source handler checks passed with
additional repeated-click, reset and cleanup assertions. Desktop and 390px browser
checks observed the success state and reset with a constant 88px button width;
see [browser observations](copy-feedback-browser.json). This follows the prior
code-access review without changing source payloads, teaching material or labs.

The [receipt](copy-feedback-review.json) preserves the 81 current checkpoints and
96 historical rows. Verify it with:

```sh
node scripts/check-lesson-code-controls.mjs
node scripts/verify-lesson-code-access.mjs docs/teaching/lesson-code-access/copy-feedback-review.json
```

The original code-access receipt remains unchanged; its default historical
source-identity check predates this small feedback change.
