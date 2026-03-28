# ARV Architecture Diagram — Generative AI Prompt

## 프롬프트 (아래 전체를 복사하여 사용)

---

Create a publication-quality architecture diagram for a research paper titled **"Lightweight Image Forgery Detection via Asymmetric Risk-aware Veto Consensus"**, to be submitted to an academic conference (KIPS 2026). The diagram must look like it belongs in a top-tier computer vision or machine learning paper.

---

### REFERENCE STYLE

Emulate the clean, professional style seen in these influential papers' architecture figures:

- **"Attention Is All You Need" (Vaswani et al., 2017, Fig. 1)**: Stacked rectangular blocks with rounded corners, clear vertical data flow, labeled arrows, soft color fills (light orange, light blue, light green), thin dark borders.
- **"Focal Loss for Dense Object Detection" (Lin et al., 2017, Fig. 3 — RetinaNet)**: Horizontal pipeline with clear left-to-right flow, backbone feeding into sub-networks, color-coded paths.
- **"MobileNetV2" (Sandler et al., 2018, Fig. 3)**: Compact block-level view, minimal text inside blocks, dimension annotations on connecting arrows.
- **"Deep Residual Learning" (He et al., 2016, Fig. 2)**: Skip connections drawn as curved arrows bypassing blocks.
- **"U-Net" (Ronneberger et al., 2015, Fig. 1)**: Symmetric layout, color-coded operations, dimension labels along the flow.

The goal: **clean geometric shapes, soft pastel fills, thin dark borders, generous whitespace, no 3D effects, no gradients, no shadows, no decorative elements.** Every element must serve an informational purpose.

---

### SYSTEM OVERVIEW (what to draw)

The system is called **ARV (Asymmetric Risk-aware Veto)**. It is a two-stage decision system for image forgery detection using two lightweight models.

**Data flow (top to bottom):**

```
[Input Image]
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
┌─────────────┐          ┌─────────────┐
│    Base      │          │  Auxiliary   │
│  Classifier  │          │    Model    │
│  (3-class)   │          │  (binary)   │
└─────┬───────┘          └──────┬──────┘
      │                          │
      │  P_base(auth)            │  P_aux(auth)
      │  P_base(manip)           │  P_aux(manip)
      │  P_base(aigen)           │
      │                          │
      └──────────┬───────────────┘
                 │
                 ▼
       ┌─────────────────┐
       │   Stage 1:      │
       │ Aggressive      │
       │   Fusion        │
       └────────┬────────┘
                │
                │  y_stage1
                │
                ▼
        ┌───────────────┐
        │  Decision     │ ← "Does Stage 1 change
        │  Changed?     │    the base prediction?"
        └───┬───────┬───┘
         No │       │ Yes
            │       │
            ▼       ▼
      ┌─────────┐  ┌───────────────────┐
      │  Keep   │  │   Stage 2: ARV    │
      │  base   │  │  Risk-aware Veto  │
      │ decision│  └────────┬──────────┘
      └─────────┘           │
                      ┌─────┴─────┐
                      │           │
                      ▼           ▼
                ┌──────────┐ ┌──────────┐
                │  Keep    │ │  Revert  │
                │ (accept  │ │ (restore │
                │  change) │ │  base)   │
                └──────────┘ └──────────┘
                      │           │
                      └─────┬─────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Final Output │
                    │  y_final     │
                    └──────────────┘
```

---

### DETAILED ELEMENT SPECIFICATIONS

#### 1. INPUT IMAGE (top center)

- **Shape**: A small square icon representing a photograph (a simple landscape silhouette or a generic image icon — NOT a real photo).
- **Size**: Small, roughly 40×40px equivalent in the diagram.
- **Label**: Bold text below it: **"Input Image"**
- **Position**: Top center of the entire diagram.
- From this icon, **two arrows** diverge downward-left and downward-right (a Y-split).

#### 2. BASE CLASSIFIER (left branch)

- **Shape**: Rounded rectangle (corner radius ~8px).
- **Fill color**: `#DBEAFE` (soft sky blue, similar to Tailwind blue-100).
- **Border**: 1.5px solid `#3B82F6` (medium blue).
- **Internal layout** (top to bottom inside the box):
  - **Title line**: Bold, 11pt, `#1E3A5F` dark navy: **"Base Classifier"**
  - **Subtitle**: Regular, 9pt, `#64748B` slate gray: **"Dual-Stream MobileNetV2"**
  - **Separator**: thin horizontal line `#93C5FD`
  - **Class label**: Regular, 9pt: **"3-class: auth / manip / aigen"**
- **Size**: Width ~160px, Height ~90px.
- **Output arrow** going down from bottom center, labeled on the right side of the arrow:
  - Three lines of small text (8pt, monospace-style):
    - `P_base(auth)`
    - `P_base(manip)`
    - `P_base(aigen)`

#### 3. AUXILIARY MODEL (right branch)

- **Shape**: Rounded rectangle, same corner radius as base.
- **Fill color**: `#FEF3C7` (soft warm amber, Tailwind amber-100).
- **Border**: 1.5px solid `#F59E0B` (amber-500).
- **Internal layout**:
  - **Title**: Bold, 11pt, `#78350F` dark amber: **"Auxiliary Model"**
  - **Subtitle**: Regular, 9pt, `#92400E`: **"MobileNetV2 (binary)"**
  - **Separator**: thin horizontal line `#FCD34D`
  - **Training note**: Italic, 8pt, `#92400E`: **"Trained with w(x) = (1 − P_base(y|x))^γ"**
  - **Class label**: Regular, 9pt: **"2-class: auth / manip"**
- **Size**: Width ~160px, Height ~100px.
- **Annotation**: A small callout or side note (dashed rounded box, fill `#FFFBEB`) connected by a thin dashed line to the training note:
  - Text inside (8pt): **"Focuses on samples where\nbase classifier is weak"**
- **Output arrow** going down, labeled:
  - `P_aux(auth)`
  - `P_aux(manip)`

#### 4. STAGE 1: AGGRESSIVE FUSION (center, below both models)

- **Shape**: Rounded rectangle, slightly wider than the models.
- **Fill color**: `#E0E7FF` (soft indigo, Tailwind indigo-100).
- **Border**: 1.5px solid `#6366F1` (indigo-500).
- **Internal layout**:
  - **Stage tag**: Small badge/pill shape in top-left corner inside the box: `Stage 1` in white text on `#6366F1` indigo background, 8pt bold.
  - **Title**: Bold, 11pt, `#312E81`: **"Inverse-Confidence Weighted Fusion"**
  - **Equations** (centered, 9pt, math-style font):
    - `w_base = 1 / max(P_base)`
    - `w_aux = 1 / max(P_aux)`
    - `score(c) = w_base · P_base(c) + w_aux · P_aux(c)`
  - **Small note** (8pt italic, `#4338CA`): **"aigen: base only"**
- **Two input arrows** converge from Base Classifier and Auxiliary Model into this box.
- **One output arrow** going down, labeled: **y_stage1**

#### 5. DECISION DIAMOND (below Stage 1)

- **Shape**: Diamond (rhombus), classic flowchart decision shape.
- **Fill color**: `#FFF7ED` (soft orange-50).
- **Border**: 1.5px solid `#EA580C` (orange-600).
- **Text inside** (9pt, bold, centered): **"y_stage1 ≠ y_base ?"**
- **Size**: ~100×60px diamond.
- **Two exits**:
  - **Left exit (No)**: Arrow going down-left, labeled **"No"** in `#16A34A` green.
  - **Right exit (Yes)**: Arrow going down-right, labeled **"Yes"** in `#DC2626` red.

#### 6. "NO" PATH → KEEP BASE (left-bottom)

- **Shape**: Small rounded rectangle.
- **Fill color**: `#DCFCE7` (soft green-100).
- **Border**: 1.5px solid `#16A34A` (green-600).
- **Text**: Bold, 10pt: **"Keep base prediction"**
- Arrow from this box goes down to the Final Output.

#### 7. STAGE 2: ARV RISK-AWARE VETO (right-bottom, the core)

This is the most important block — make it visually prominent but not cluttered.

- **Shape**: Rounded rectangle, **slightly larger** than other blocks.
- **Fill color**: `#FEE2E2` (soft red-100, Tailwind red-100).
- **Border**: **2px** solid `#DC2626` (red-600) — slightly thicker than other blocks to indicate importance.
- **Internal layout**:
  - **Stage tag**: Small badge/pill in top-left: `Stage 2` in white text on `#DC2626` red background, 8pt bold.
  - **Title**: Bold, 12pt, `#7F1D1D` dark red: **"ARV: Risk-aware Veto"**
  - **Separator line**: `#FCA5A5`
  - **Three feature groups** listed vertically, each as a small sub-row:
    - `📊 Base features` → scores, margins, confidence (8pt)
    - `🌐 Context features` → subtype, dataset family (8pt)
    - `⚠️ Risk features` → direction, disagreement, OOD flag (8pt)
  - (Use small monochrome icons or bullet symbols instead of emoji if the rendering tool doesn't support emoji.)
  - **Asymmetric cost annotation** (bottom of box, 8pt italic):
    - **"manip→auth: cost 6.0 | auth→manip: cost 2.0"**
- **Two output arrows** from the bottom of this box:
  - **Left arrow**: labeled **"keep"** → goes down-left (green `#16A34A` color)
  - **Right arrow**: labeled **"revert"** → goes down-right (orange `#EA580C` color)

#### 8. KEEP / REVERT OUTCOMES

Two small terminal boxes below Stage 2:

**Keep box**:
- Fill: `#DCFCE7` (green-100), Border: `#16A34A`
- Text: **"Accept change"** (9pt)

**Revert box**:
- Fill: `#FFEDD5` (orange-100), Border: `#EA580C`
- Text: **"Restore base"** (9pt)

Both boxes have arrows converging down to the Final Output box.

#### 9. FINAL OUTPUT (bottom center)

- **Shape**: Rounded rectangle with slightly thicker border (2px).
- **Fill color**: `#F0FDF4` (very light green).
- **Border**: 2px solid `#15803D` (green-700).
- **Text**: Bold, 11pt: **"Final Prediction: y_final"**
- **Sub-text**: 9pt: **"authentic / manipulated / ai_generated"**

---

### LAYOUT AND SPACING

- **Overall layout**: Vertical (top-to-bottom), left-right symmetric around a center axis.
- **Total aspect ratio**: Roughly 3:4 (width:height) — suitable for a single-column academic paper figure (fits within ~85mm × 120mm print area).
- **Vertical spacing** between blocks: ~25-30px equivalent.
- **Horizontal spacing** between Base Classifier and Auxiliary Model: ~40-50px gap.
- **The center axis** runs through: Input Image → Stage 1 → Decision Diamond → Final Output.
- **Base Classifier** is offset to the left of center.
- **Auxiliary Model** is offset to the right of center.
- **Stage 2 ARV** is offset slightly to the right (since it's on the "Yes" path).
- **Keep base** and the "No" path are offset to the left.

---

### ARROW STYLE

- **All arrows**: Straight lines with pointed arrowheads. No curved or bezier arrows except for the Y-split from Input Image.
- **Line weight**: 1.5px for data flow arrows, 1px for annotation/callout dashed lines.
- **Arrow color**: `#374151` (gray-700) for main data flow. Use colored arrows only for the No/Yes branches from the diamond and the keep/revert outputs from ARV.
- **Labels on arrows**: 8-9pt, placed adjacent to the arrow (not on top of it), in `#374151` dark gray.

---

### TYPOGRAPHY

- **All text**: Sans-serif font (Inter, Helvetica, or Arial).
- **Title text** inside blocks: 11pt bold.
- **Subtitle/description**: 9pt regular.
- **Annotations**: 8pt regular or italic.
- **Math expressions**: Use a math-style font or italicize variable names. For example: *P*_base, *w*(*x*), *γ*.
- **No text should be smaller than 7pt** — everything must be legible when printed at 85mm width.

---

### COLOR PALETTE SUMMARY

| Element | Fill | Border | Text |
|---------|------|--------|------|
| Base Classifier | `#DBEAFE` sky blue | `#3B82F6` | `#1E3A5F` navy |
| Auxiliary Model | `#FEF3C7` amber | `#F59E0B` | `#78350F` dark amber |
| Stage 1 Fusion | `#E0E7FF` indigo | `#6366F1` | `#312E81` dark indigo |
| Decision Diamond | `#FFF7ED` orange-50 | `#EA580C` | `#9A3412` |
| Stage 2 ARV | `#FEE2E2` red-100 | `#DC2626` | `#7F1D1D` dark red |
| Keep (accept) | `#DCFCE7` green-100 | `#16A34A` | `#166534` |
| Revert (restore) | `#FFEDD5` orange-100 | `#EA580C` | `#9A3412` |
| Final Output | `#F0FDF4` green-50 | `#15803D` | `#14532D` |
| Arrows | — | `#374151` | `#374151` |
| Background | `#FFFFFF` white | — | — |

---

### WHAT TO AVOID

- ❌ No 3D effects, shadows, or drop shadows.
- ❌ No gradient fills.
- ❌ No decorative icons or clipart.
- ❌ No thick borders (max 2px, only for emphasis).
- ❌ No overlapping elements.
- ❌ No dense paragraphs of text inside blocks.
- ❌ No rounded-pill shapes for process blocks (use rounded rectangles).
- ❌ No hand-drawn or sketch style.
- ❌ No dark background.

---

### FINAL CHECKLIST

1. The diagram clearly shows **two parallel models** receiving the same input.
2. The **Stage 1 fusion** merges both outputs.
3. A **decision gate** checks if the fused result differs from the base prediction.
4. If no change → pass through directly.
5. If changed → **Stage 2 ARV** evaluates the risk of that change.
6. ARV outputs either **keep** (accept the correction) or **revert** (restore base prediction).
7. All paths converge to a single **Final Output**.
8. The diagram is print-ready at single-column width (~85mm / 3.35 inches).
9. All text is legible at print size.
10. The visual hierarchy makes Stage 2 ARV the most prominent element (thicker border, slightly larger).

---

### CONTEXT FOR THE AI

This figure will appear as **Figure 1** in the paper, captioned:
> **"Fig. 1.** Overall architecture of the proposed ARV system. The base classifier and auxiliary model run in parallel on the input image. Stage 1 aggressively fuses their outputs to generate correction candidates. Stage 2 (ARV) selectively vetoes risky verdict changes using directional, disagreement, and contextual features."

The target audience is academic reviewers and researchers in image forensics and edge AI. The diagram must communicate the two-stage logic instantly at a glance.
