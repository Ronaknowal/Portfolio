# The Training Loop Portfolio — Design Reference Document

> A complete reference for the design philosophy, visual language, code architecture, and decision rationale behind the AI-themed portfolio website. Use this document when extending, refactoring, or redesigning the site.

---

## 1. Core Concept

### The Big Idea

The portfolio is framed as **a neural network being trained**. The visitor's scroll is the training loop — the deeper they scroll, the more the "model" (you) converges. This is not just a visual theme layered on top of a portfolio; the **metaphor IS the navigation and narrative structure**.

Two AI concepts are fused together:

| Concept | What it provides | How it manifests |
|---|---|---|
| **Training Loop** (Theme 3) | Narrative structure, section labeling, career-as-training metaphor | Epoch labels, loss curves, training logs, model architecture diagrams, "Run Inference" CTA |
| **Diffusion / Denoising** (Theme 6) | Visual language, scroll-linked aesthetics | Noise grain that fades on scroll, text that "denoises" from random characters, elements that clarify as they enter viewport |

### Why This Combination Works

- The training loop gives **meaning** to every section — it's not arbitrary, it maps to a real ML workflow
- The diffusion aesthetics give **visual drama** — the site feels alive, like something is being generated in real-time
- Together they create a feeling of **progressive revelation** — the visitor discovers more as the model "learns"
- It signals deep AI literacy without being pretentious — the metaphors are playful ("batch_size: life") while being technically grounded

---

## 2. Narrative Architecture

The site follows a strict narrative arc mapped to model training phases:

```
Epoch  0  → Hero          → "Initializing Weights"     → Maximum noise, name denoises
Epoch 12  → About         → "Early Convergence"        → Noise dropping, bio reveals
Epoch 24  → Experience    → "Pre-training"              → Foundational work experience
Epoch 34  → Projects      → "Fine-tuning Data"          → Side projects as specialization
Epoch 56  → Skills        → "Model Architecture"        → Tech stack as network layers
Epoch 78  → Journey       → "Convergence Analysis"      → Interactive loss curve timeline
Epoch 92  → Content       → "Model Outputs"             → What the trained model generates
Epoch 100 → Contact       → "Inference Ready"            → CTA as API call
```

### Section Design Decisions

**Hero — Epoch 0: Initializing Weights**
- Maximum visual noise (grain overlay at highest opacity)
- Name text uses `DenoisingText` component — characters scramble from unicode block characters into readable text
- Subtitle denoises slightly after the name (staggered via progress offset)
- "scroll to train" prompt with bouncing arrow anchors the scroll-as-training metaphor from the first screen
- *Why*: First impression must immediately communicate "this is different." The denoising effect is the hook.

**About — Epoch 12: Early Convergence**
- Noise is noticeably reduced — visitor can feel the "training" is working
- Bio text is in monospace, keeping the technical tone
- `cat config.yaml` terminal block shows personal info as a config file — playful but informative
- *Why*: About sections are usually boring. The config.yaml format makes it scannable and memorable while staying on-theme.

**Experience — Epoch 24: Pre-training**
- Framed as "pre-training" — the large-scale foundational training before fine-tuning
- Current job has a **pulsing green dot** with "Training" label — signals active/ongoing
- Past roles are completed training runs
- Each role has "Training Objectives" styled as a log with green checkmarks
- Bottom metrics: `dataset: enterprise-scale | compute: multi-GPU | duration: 2+ years`
- Capability tags are **green-tinted** (not amber) — visually distinguishing pre-trained capabilities from fine-tuning work
- `pretrain.summary()` terminal block ties it off
- *Why*: The pre-training metaphor elegantly solves the "day job vs side projects" tension. The job isn't secondary — it's foundational. This framing honors the work experience while making clear that side projects are where specialization happens.

**Projects — Epoch 34: Fine-tuning Data**
- Renamed from "Training Data" to "Fine-tuning Data" after adding the experience section
- Each project is labeled as `SAMPLE 001`, `SAMPLE 002` etc.
- Training metrics block on each card: `loss: 0.034 | acc: 97.2% | epochs: 847`
- These metrics are **metaphorical, not real**: high accuracy = project went well, low loss = few challenges, epochs = time invested
- Generation steps slider lets visitors scrub: concept → prototype → shipped
- Cards have a subtle grain overlay that disappears on hover (micro-denoising effect)
- *Why*: The SAMPLE labeling and fake metrics are the heart of the personality. They signal ML fluency while being fun. The slider is a nod to diffusion timesteps.

**Architecture — Epoch 56: Model Architecture**
- Tech stack rendered as a neural network architecture diagram
- Layers flow left-to-right: `Input(Curiosity) → Conv2d(Python, JS) → Dense(AI/ML) → Dense(Product) → Output(Bloospace)`
- Each layer has a distinct color (amber, green, blue, purple, pink)
- `model.summary()` terminal block with playful params: `Optimizer: AdamW(curiosity, caffeine)`
- Almost no noise left at this point — the model is "well-trained"
- *Why*: Skills sections are typically boring lists. Rendering them as a network diagram is both on-theme and actually more informative — it shows how skills connect and flow into outcomes.

**Journey — Epoch 78: Convergence Analysis**
- Full interactive SVG loss curve
- X-axis = time, Y-axis = loss (inversely, growth)
- Data points are hoverable — each reveals a career event with its year and "loss" value
- Loss spikes represent setbacks (pivots, failures); sharp drops represent breakthroughs
- Green glow effect on the curve line for visual emphasis
- *Why*: This is the emotional centerpiece. A traditional timeline is boring. A loss curve is immediately legible to anyone in ML, and the hover-to-reveal mechanic creates discovery moments.

**Content — Epoch 92: Model Outputs**
- Framed as "Inference Results" — what the trained model produces
- Simple list of content pieces (reels, series, posts) with hover effects
- Minimal design — the focus is on the work, not decoration
- *Why*: By this point in the scroll, the "model" is trained. The outputs should feel clean and confident, matching the narrative of convergence.

**Contact — Epoch 100: Inference Ready**
- CTA: "Run Inference" + "The model is converged. Submit your prompt."
- Contact info styled as an API POST request with JSON body
- Social links as simple text with amber hover
- *Why*: This is the payoff of the entire metaphor. The visitor has "trained" the model by scrolling. Now they can use it. The API format is both on-brand and practical.

**Footer**
- `model_version: 2025.02 | framework: life`
- `built with curiosity, caffeine, and gradient descent`
- Extremely muted (#222 on #050505) — barely visible, rewards close readers
- *Why*: The footer is an easter egg, not a UI element. It should whisper, not shout.

---

## 3. Visual Design System

### Color Palette

```
Background:     #050505  (near-black, not pure black — softer on eyes)
Primary Text:   #e8e8e8  (off-white, high contrast but not harsh)
Secondary Text: #888888  (for body copy, descriptions)
Muted Text:     #444444  (for metadata, labels, subtle info)
Ghost Text:     #222222  (for barely-visible footer, grid lines)
Border:         #1a1a1a  (cards, dividers — just barely visible)

Accent Amber:   #e2b55a  (primary accent — epoch labels, links, highlights)
Accent Green:   #4ade80  (secondary accent — loss values, active states, checkmarks)
Accent Blue:    #60a5fa  (architecture diagram — AI/ML layer)
Accent Purple:  #c084fc  (architecture diagram — Product layer)
Accent Pink:    #f472b6  (architecture diagram — Output layer)
```

**Why this palette:**
- Near-black background (#050505 not #000000) avoids the "hole in the screen" effect of pure black
- Amber (#e2b55a) was chosen over the more common cyan/blue because it evokes monitoring dashboards (Weights & Biases, Grafana) without being the cliché "hacker green" or "AI blue"
- Green (#4ade80) is reserved for "positive" signals — successful metrics, active states, checkmarks — mirroring how green is used in actual training dashboards
- The multi-color architecture diagram breaks the 2-color monotony but is contained to one section

### Color Semantics

| Color | Meaning | Used For |
|---|---|---|
| Amber `#e2b55a` | Identity, navigation, primary accent | Epoch labels, nav hover, links, config keys, fine-tuning tags |
| Green `#4ade80` | Positive, active, success | Loss values, accuracy, checkmarks, active status dot, pre-training capability tags |
| White `#e8e8e8` | Primary content | Headings, names, important text |
| Gray `#888–#444` | Supporting content | Body text, descriptions, metadata |
| Ghost `#222–#111` | Background structure | Grid lines, borders, footer text |

### Typography

```
Display / Headings:  Space Grotesk (weights: 300–700)
Code / Technical:    JetBrains Mono (weights: 300–700)
```

**Why Space Grotesk:**
- Geometric sans-serif with personality — not as sterile as Inter/Roboto
- Tight letter-spacing at large sizes creates a modern, confident feel
- The slightly quirky letterforms (look at the 'a' and 'g') give it character without being distracting

**Why JetBrains Mono:**
- Purpose-built for code readability
- Ligature support for a polished terminal look
- Used for ALL technical/metadata elements: epoch labels, config blocks, metrics, terminal outputs
- Creates a clear visual hierarchy: Space Grotesk = "what I'm saying", JetBrains Mono = "how the system works"

**Typography Scale:**
```
Hero name:        clamp(48px, 10vw, 120px) — responsive, max 120px
Section headings: 40px, weight 600, letter-spacing -0.03em
Card titles:      22px, weight 600, letter-spacing -0.02em
Body (mono):      13-14px, line-height 1.6-1.9
Metadata:         10-11px, letter-spacing 1-3px (uppercase)
Epoch labels:     10px, letter-spacing 3px, uppercase, color #333
```

**Why negative letter-spacing on headings:**
At large sizes, default spacing looks loose. Tightening to -0.02em / -0.03em makes headings feel dense and intentional — a hallmark of editorial design.

### Spacing & Layout

```
Max content width:  800-900px (centered)
Section padding:    120px vertical (generous breathing room)
Card padding:       28px 24px
Card gap:           20px
Card border-radius: 8px (subtle rounding, not bubbly)
```

**Why 120px section padding:**
Each section is a distinct "epoch" in the training process. The generous spacing creates a clear separation between phases, reinforcing the idea that each is a discrete step. It also ensures the scroll-linked noise transition has room to breathe.

**Why max-width 800-900px:**
Optimal reading width for monospace text is narrower than sans-serif. Since half the content is monospace, 800px keeps line lengths comfortable. Projects section is 900px to accommodate the card grid.

---

## 4. Interactive Systems

### 4.1 Scroll-Linked Noise Layer (`NoiseCanvas`)

**What it does:** A full-screen canvas element renders animated static grain. Its opacity is linked to scroll position — starts at 0.6, fades to 0 as you reach the bottom.

**Code architecture:**
```
- Fixed-position <canvas> at z-index 9999 (above content, below HUD)
- requestAnimationFrame loop generating random grayscale pixels
- Each pixel: R=G=B=random(0-255), Alpha=40 (very transparent)
- Opacity controlled by parent via prop: max(0.6 - scrollProgress * 1.2, 0)
- pointer-events: none so it doesn't block interaction
```

**Why canvas, not CSS filter or SVG:**
- CSS `backdrop-filter` noise doesn't animate and looks static/dead
- SVG `feTurbulence` is CPU-heavy at full viewport size
- Canvas with `createImageData` is the lightest way to get real-time animated grain
- The alpha of 40 per pixel keeps it subtle — you feel it more than see it

**Why the opacity formula `max(0.6 - scrollProgress * 1.2, 0)`:**
- Starts at 0.6 (noticeable but not overwhelming)
- Crosses zero around 50% scroll (by mid-page, the model is "mostly trained")
- Multiplier of 1.2 ensures it's fully gone before you reach the bottom
- The `transition: opacity 0.3s ease` on the canvas smooths out scroll jitter

**Performance note:** This renders every frame at full viewport resolution. On low-end devices, consider:
- Rendering at 0.5x resolution and CSS-scaling up
- Using a static grain PNG with CSS animation instead
- Adding a `prefers-reduced-motion` check to disable

### 4.2 Denoising Text (`DenoisingText`)

**What it does:** Text scrambles from unicode block characters into readable text, driven by a progress value (0 = fully scrambled, 1 = fully readable).

**Code architecture:**
```
- Characters: "█▓▒░╬╫╪┼╳※¤◊◈⬡⬢⎔⏣" (chosen for visual density)
- For each character in the target text:
  - If progress > (charIndex/textLength) + 0.3 → show real character
  - If progress > (charIndex/textLength) → 50/50 chance real or random
  - Else → show random noise character
- Spaces always pass through (prevents word-boundary scrambling)
```

**Why these specific unicode characters:**
- Block elements (█▓▒░) create a "resolving from density" effect
- Box-drawing characters (╬╫╪┼) echo terminal/matrix aesthetics
- Geometric shapes (◈⬡⬢) add visual variety
- Mixed widths create a "glitching" feel without looking broken

**Why the left-to-right reveal pattern:**
Characters resolve based on their position (`charIndex / textLength`), creating a left-to-right sweep. This mimics how diffusion models denoise — not all at once, but progressively across the space. The 50/50 zone (where characters flicker between real and noise) simulates the "uncertain" intermediate steps.

**Why progress is scroll-linked in hero, not time-based:**
Originally considered a timer-based denoise on page load. Changed to scroll-linked because:
- It reinforces the core metaphor (scrolling = training)
- It gives the visitor agency — they control the denoising speed
- It ensures the effect is seen (a timed animation might finish before the visitor notices it)

### 4.3 HUD — Mini Loss Curve (`MiniLossCurve`)

**What it does:** A persistent overlay in the top-right corner showing current epoch, loss value, and a tiny animated loss curve — all updating in real-time as you scroll.

**Code architecture:**
```
- Fixed position, z-index 10000 (above noise canvas)
- Loss formula: exp(-3.5 * t) + sin(t * 20) * 0.03 * (1 - t)
  - exp(-3.5t): exponential decay (realistic training curve shape)
  - sin(t*20)*0.03*(1-t): small oscillations that dampen over time (realistic noise)
- SVG polyline drawn point-by-point up to current scroll progress
- Epoch = floor(scrollProgress * 100)
- Loss = exp(-3.5 * scrollProgress) + 0.01
```

**Why the loss formula:**
- `exp(-3.5t)` gives the classic "sharp initial drop then plateau" shape of real training curves
- The sinusoidal noise term adds realism — real loss curves are never smooth
- The `(1-t)` damping on the noise means early training is volatile, late training is stable
- This matches real ML training behavior, which adds authenticity

**Why a persistent HUD, not just inline:**
- It creates a "monitoring dashboard" feel — like you're watching a training run in Weights & Biases
- It gives continuous feedback that scrolling = progress, reinforcing the metaphor at all times
- It's the kind of detail that makes someone go "oh that's cool" and remember the site

### 4.4 Scroll-Triggered Sections (`Section`)

**What it does:** Each section fades in and slides up when it enters the viewport.

**Code architecture:**
```
- IntersectionObserver with threshold: 0.1 (triggers when 10% visible)
- One-way trigger: once visible, stays visible (no exit animation)
- CSS transition: opacity 0.8s + translateY(40px → 0) 0.8s
```

**Why IntersectionObserver, not scroll events:**
- More performant (browser-native, doesn't fire on every pixel of scroll)
- Cleaner API (threshold-based instead of manual offset calculation)
- No debouncing needed

**Why one-way (no exit animation):**
- Sections should feel like permanent revelations, not flickering appearances
- Matches the training metaphor — once the model learns something, it doesn't "unlearn" it on scroll-up
- Reduces visual noise from constant re-animation

### 4.5 Interactive Loss Curve (`LossCurve`)

**What it does:** A full-size SVG chart where each data point represents a career event. Hovering reveals the event detail and its "loss" value.

**Code architecture:**
```
- SVG with manual coordinate mapping (getX, getY helper functions)
- Dashed grid lines at 0.2 intervals
- Green polyline for the curve + a blurred copy at 0.15 opacity for glow
- Circle elements for each event, scaling on hover (r: 4 → 6)
- Tooltip rendered as SVG rect + text, positioned relative to the point
```

**Why SVG, not a charting library (Recharts, Chart.js):**
- Full control over every visual element (glow effects, custom tooltips, grid styling)
- No dependency overhead for a single chart
- The chart needs to match the overall aesthetic precisely — library defaults would fight the design
- SVG is resolution-independent (crisp on all displays)

**Why the loss values for career events are hand-assigned:**
- Loss = 0.95 for "Started learning to code" (high uncertainty, just beginning)
- Loss = 0.72 for "Pivoted from initial idea" (spike — setback)
- Loss = 0.04 for "Currently training" (near-converged)
- These tell a story: the curve isn't monotonically decreasing, it has spikes (failures) followed by sharp drops (breakthroughs)
- This is both realistic (real training curves have spikes) and narratively compelling

### 4.6 Generation Steps Slider (Project Cards)

**What it does:** A range slider on each project card that scrubs between "concept", "prototype", and "shipped" stages.

**Current implementation:** Updates a label only. This is a scaffold for a richer feature.

**Future enhancement ideas:**
- Show different images/screenshots at each stage
- Change the card's grain overlay (noisy at concept, clean at shipped)
- Animate the training metrics (loss decreasing across stages)
- Show actual project evolution screenshots

### 4.7 Active Job Indicator (Experience Section)

**What it does:** A pulsing green dot next to "Training" text on the current job.

**Code architecture:**
```css
@keyframes pulse-dot {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}
```

**Why a pulse, not a static dot:**
- A static green dot says "active." A pulsing green dot says "actively running right now" — like a process indicator in a monitoring dashboard
- It's a subtle animation that draws the eye without being distracting
- Reinforces that the pre-training phase is ongoing (you're still employed there)

---

## 5. Visual Differentiation System

### Pre-training (Experience) vs Fine-tuning (Projects)

These two sections need to feel related but distinct:

| Property | Pre-training (Experience) | Fine-tuning (Projects) |
|---|---|---|
| Label color | Amber `#e2b55a` | Amber `#e2b55a` |
| Tag/badge color | Green `#4ade80` | Gray `#555` border |
| Tag background | `#4ade8008` (green tint) | none |
| Card label | Period (2023 — Present) | SAMPLE 001 |
| Metrics format | `dataset: enterprise-scale` | `loss: 0.034 \| acc: 97.2%` |
| Has slider | No | Yes (generation steps) |
| Active indicator | Pulsing green dot | None |

**Why green tags for experience, gray for projects:**
Green = "pre-trained capability" (something acquired and stable). Gray = "tools used" (more neutral, just descriptive). This subtly communicates that your job gave you **foundational capabilities** while your projects are where you **apply and specialize** them.

---

## 6. Component Architecture

```
Portfolio (root)
├── NoiseCanvas              — Full-screen animated grain, opacity linked to scroll
├── MiniLossCurve            — Persistent HUD with epoch/loss/curve
├── Nav                      — Fixed nav bar with scroll-triggered background
├── Hero Section             — DenoisingText for name + subtitle
├── Section (About)          — Config.yaml terminal block
├── Section (Experience)     — Job cards with active indicator + training log
├── Section (Projects)       — ProjectCard[] with metrics + generation slider
│   └── ProjectCard          — Individual project with grain overlay + hover state
├── Section (Architecture)   — ArchitectureDiagram + model.summary()
│   └── ArchitectureDiagram  — Network layer visualization
├── Section (Journey)        — Interactive LossCurve
│   └── LossCurve            — SVG chart with hover tooltips
├── Section (Outputs)        — Content list items with hover
├── Section (Contact)        — API-style CTA + social links
└── Footer                   — Muted metadata
```

### Shared Components

**`Section`** — Wraps every content section with IntersectionObserver-based reveal animation. Takes `id` for navigation anchoring and `style` for per-section layout.

**`DenoisingText`** — Reusable text scramble effect. Props: `text`, `progress` (0-1), `as` (HTML tag), `style`. Can be used anywhere a reveal effect is needed.

**`ProjectCard`** — Self-contained project display with hover states, grain overlay, metrics, slider, and tags. All data passed via `project` prop.

### State Management

All state is local (useState). No global state management needed because:
- Scroll progress is computed from `window.scrollY` on every scroll event
- Each section independently manages its own visibility via IntersectionObserver
- Each card independently manages its own hover and slider state
- No cross-component communication is required

### Scroll Progress Flow
```
window.scrollY
  → scrollProgress (0 to 1, global)
    → NoiseCanvas opacity (0.6 → 0)
    → MiniLossCurve epoch + loss + curve points
    → heroDenoised (0 to 1, hero-specific, faster ramp)
      → DenoisingText progress (name)
      → DenoisingText progress (subtitle, offset by -0.2)
```

---

## 7. Micro-Interactions & Motion Design

### Interaction Inventory

| Element | Trigger | Effect | Duration |
|---|---|---|---|
| Noise canvas | Scroll | Opacity fades from 0.6 → 0 | Continuous (0.3s CSS transition) |
| Hero text | Scroll | Characters denoise left-to-right | Continuous |
| Sections | Enter viewport | Fade in + slide up 40px | 0.8s ease |
| Nav background | Scroll past 5% | Transparent → blurred dark | 0.3s ease |
| Project cards | Hover | Border amber, bg amber tint, grain fades | 0.4s ease |
| Loss curve points | Hover | Radius 4→6, color green→amber, tooltip appears | 0.2s ease |
| HUD loss curve | Scroll | Curve draws progressively, metrics update | Continuous |
| Active job dot | Constant | Pulses opacity 1→0.4→1 | 2s infinite |
| Scroll arrow | Constant | Bounces vertically | 2s infinite |
| Nav links | Hover | Color #555 → #e2b55a | 0.2s |
| Output items | Hover | Border amber, bg amber tint | 0.3s |
| Contact links | Hover | Color #444 → #e2b55a | 0.2s |

### Motion Principles

1. **Scroll is the primary input** — Most animations are scroll-driven, not time-driven. This makes the visitor feel in control.
2. **Reveals are one-way** — Once something appears, it stays. No exit animations. The model doesn't "unlearn."
3. **Hover effects are subtle** — Border color shift + background tint. No scale transforms, no bouncing. Keeps the brutalist feel.
4. **Two constant animations only** — The pulsing dot and the bouncing arrow. Everything else is triggered. This prevents visual fatigue.
5. **Transitions are ease, not bounce** — The overall tone is technical/professional, not playful. Ease curves feel controlled; spring/bounce would undermine the aesthetic.

---

## 8. Tone & Content Voice

### Writing Rules

1. **Technical metaphors that non-technical people can still enjoy.** "Loss: 0.034" is fun even if you don't know what loss means. "batch_size: life" is universally funny.

2. **Lowercase for metadata, Title Case for headings.** Metadata (epoch labels, terminal blocks, metrics) is always lowercase/monospace. Headings are title case/sans-serif. This creates a clear visual hierarchy.

3. **Short, punchy subtitles.** "The samples I chose to specialize on." "Large-scale foundational training." "Hover over the data points to trace the journey." One line, sets context, moves on.

4. **Terminal blocks are personality vehicles.** The `cat config.yaml`, `model.summary()`, `pretrain.summary()`, and `POST /api/collaborate` blocks are where humor and personality live. They should feel like reading someone's actual terminal output — casual, informative, slightly witty.

5. **Never break the metaphor.** Every section title has an epoch number. Every label uses ML terminology. The metaphor is maintained from hero to footer. Consistency is what makes it feel intentional rather than gimmicky.

### Content Template (for adding new sections)

```
<epoch label>  EPOCH XX — [ML CONCEPT IN CAPS]
<heading>      [Human-readable section name]
<subtitle>     [One-line metaphor bridging ML concept to content]
<content>      [Actual content]
<terminal>     $ [playful command that summarizes the section]
               [key: value pairs in config/yaml style]
```

---

## 9. Future Enhancement Ideas

### High Impact

- **Real diffusion effect on profile photo:** Start as pure noise, progressively denoise into the actual photo on scroll. Could use a pre-generated sequence of images at different noise levels, swapped based on scroll progress.
- **Actual AI chat in Contact section:** Replace the static API block with a lightweight chatbot (using Claude API) that answers questions about you. "The model is ready — ask it anything."
- **Particle system background:** Replace or supplement the noise canvas with a WebGL particle system where particles slowly organize into a neural network graph as you scroll.
- **Sound design:** Subtle ambient tone that "resolves" from noise to a clean note as you scroll. Triggered by user interaction (not autoplay). Very experimental.

### Medium Impact

- **Generation steps slider with real content:** Show actual screenshots/images at each stage (concept sketch, wireframe, final product).
- **Animated architecture diagram:** Signals flow through the layers like a forward pass animation, pulsing from Input → Output.
- **Blog section with "inference temperature" toggle:** Higher temperature = more creative/experimental posts, lower = more technical/focused.
- **Dark/light mode as "training regime":** Dark = pre-training (large-scale, foundational), Light = fine-tuning (focused, refined). Toggle switches the entire aesthetic.

### Low Impact / Polish

- **Custom cursor:** Small crosshair or node cursor that leaves a faint trail, like tracing a gradient descent path.
- **Easter egg: Konami code** triggers a "model reset" — the whole page re-noises from the top.
- **Responsive HUD position:** Move the mini loss curve to bottom-center on mobile.
- **Print stylesheet:** Clean, no-noise version for PDF export.
- **View transitions API:** Smooth page-level transitions if converting to multi-page.

---

## 10. Technical Notes

### Performance Considerations

- **NoiseCanvas** renders at full resolution every frame. On 4K displays this is 3840×2160×4 bytes = ~33MB/s of pixel data. Consider rendering at 0.5x and CSS-scaling.
- **Scroll event handler** fires on every pixel. It only does simple math (division, max, min) so it's fine, but if more computation is added, consider `requestAnimationFrame` throttling.
- **Google Fonts** loaded via dynamically injected `<link>` tag. For production, consider self-hosting or using `font-display: swap`.

### Accessibility Notes

- All sections have `id` attributes for anchor navigation
- Text contrast ratios: #e8e8e8 on #050505 = 18.1:1 (AAA), #888 on #050505 = 6.4:1 (AA), #444 on #050505 = 3.1:1 (fails AA for small text — consider bumping metadata to #555)
- The noise canvas is `pointer-events: none` and purely decorative
- Consider adding `prefers-reduced-motion` media query to disable NoiseCanvas animation and DenoisingText scramble
- Screen readers will read DenoisingText's scrambled characters — consider `aria-label` with the final text

### Browser Compatibility

- `IntersectionObserver`: Supported in all modern browsers
- `canvas.getContext('2d')`: Universal support
- `backdropFilter`: Supported with `-webkit-` prefix in Safari
- `clamp()`: Supported in all modern browsers
- Unicode block characters: Rendering depends on system fonts; test across OS

### File Structure (for production build)

```
src/
├── components/
│   ├── NoiseCanvas.jsx
│   ├── MiniLossCurve.jsx
│   ├── DenoisingText.jsx
│   ├── Section.jsx
│   ├── ProjectCard.jsx
│   ├── LossCurve.jsx
│   └── ArchitectureDiagram.jsx
├── data/
│   ├── projects.js
│   ├── experience.js
│   ├── journey-events.js
│   └── outputs.js
├── styles/
│   └── tokens.js          (color, typography, spacing constants)
├── hooks/
│   └── useScrollProgress.js
└── App.jsx
```

---

## 11. Design Principles Summary

1. **The metaphor IS the product.** It's not decoration — every design decision serves the training loop narrative.
2. **Progressive revelation.** More information, less noise as you scroll. This mirrors training convergence AND creates engagement.
3. **Two type families, strict roles.** Space Grotesk for what you say, JetBrains Mono for how the system works.
4. **Two accent colors, strict semantics.** Amber for identity/navigation, green for positive/active states.
5. **Brutalist restraint with technical charm.** No gradients, no shadows, no rounded-everything. Just borders, monospace, and purposeful color.
6. **Playful precision.** The humor ("batch_size: life") works because the technical framework around it is rigorous. If the metaphor were sloppy, the jokes would feel lazy.
7. **Scrolling = agency.** The visitor controls the training. This creates investment in the experience.

---

*Document version: 1.0 — February 2025*
*Companion code: portfolio.jsx*