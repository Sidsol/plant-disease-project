# Design System Strategy: Editorial Botany

## 1. Overview & Creative North Star
### The Creative North Star: "The Digital Herbarium"
This design system moves away from the sterile, plastic feel of generic health apps and toward the tactile, authoritative atmosphere of a high-end botanical journal. We are building a "Digital Herbarium"—an experience that feels curated, expert, and deeply rooted in the natural world.

To break the "template" look, we employ **Intentional Asymmetry**. Rather than perfectly centered grids, we use generous whitespace (negative space) to allow high-contrast typography to breathe. Layouts should feel like a well-composed editorial spread, utilizing overlapping elements (e.g., a plant image partially breaking the boundary of its container) to create movement and depth.

---

## 2. Colors
Our palette is a sophisticated journey through a sun-drenched forest. We utilize deep, authoritative greens and soft, atmospheric neutrals.

### The "No-Line" Rule
**Explicit Instruction:** Designers are prohibited from using 1px solid borders to section off content. Structural integrity is achieved through tonal shifts. A section is defined by moving from `surface` (#f8faf9) to `surface-container-low` (#f3f4f3). This creates a cleaner, more premium aesthetic that mimics natural horizons rather than digital boxes.

### Surface Hierarchy & Nesting
Treat the UI as a physical stack of fine paper. 
*   **Base Layer:** `surface` (#f8faf9)
*   **Secondary Content:** `surface-container-low` (#f3f4f3)
*   **Interactive Cards:** `surface-container-lowest` (#ffffff) sitting on top of `surface-container` (#edeeed) provides a "lifted" effect.

### Glassmorphism & Signature Textures
*   **Floating Navigation:** Use semi-transparent `surface` colors with a 20px backdrop-blur to create a "frosted glass" effect.
*   **CTAs:** Primary buttons should utilize a subtle linear gradient from `primary` (#0c2c1c) to `primary_container` (#234231) at a 135-degree angle to provide "visual soul."

---

## 3. Typography
We use a high-contrast pairing: **Epilogue** for display and **Manrope** for utility.

*   **Display & Headlines (Epilogue):** Large, bold, and authoritative. These are the "Journal Titles." Use `display-lg` (3.5rem) for hero moments to establish immediate expertise.
*   **Titles & Body (Manrope):** Chosen for its modern, clean legibility. `title-md` (1.125rem) should be used for card headings to ensure a "Trustworthy" feel.
*   **High-Contrast Scale:** To ensure readability and an editorial feel, maintain a strict ratio between headings and body text. Use `on_surface` (#191c1c) for all primary text to ensure it pops against the pale sage backgrounds.

---

## 4. Elevation & Depth
In this system, depth is felt, not seen. We favor **Tonal Layering** over heavy shadows.

*   **The Layering Principle:** Place a `surface-container-lowest` card on a `surface-container-low` background. The subtle shift in hex value provides enough contrast to define the object without adding visual clutter.
*   **Ambient Shadows:** Where floating is required (e.g., a "Classify" FAB), use an extra-diffused shadow: `box-shadow: 0 12px 32px rgba(12, 44, 28, 0.08)`. Note that the shadow is tinted with the `primary` green, not black.
*   **The "Ghost Border" Fallback:** If accessibility requires a border, use `outline_variant` at 15% opacity. Never use a 100% opaque border.

---

## 5. Components

### Buttons
*   **Primary:** Roundedness `md` (0.75rem). Background: Gradient of `primary` to `primary_container`. Text: `on_primary`.
*   **Secondary:** Ghost style. Background: Transparent. Border: `outline_variant` at 20%. Text: `primary`.

### Cards & Lists
*   **Zero-Divider Rule:** Forbid the use of line dividers between list items. Use Spacing `4` (1.4rem) or a subtle background shift between items.
*   **Plant Analysis Cards:** Use `surface_container_lowest` with a `lg` (1rem) corner radius.

### Input Fields
*   **Text Inputs:** Soft backgrounds (`surface_variant`) rather than outlined boxes. Use `title-sm` for labels to maintain the "expert" tone.

### Botanical Progress Bars
*   Instead of a standard flat bar, use a `primary` fill with a `secondary_container` background. The track should have `full` roundedness.

### Additional Specialty Components
*   **The "Micro-Journal" Card:** A small, high-density card for "Treatment Tips" using `tertiary_container` for the background to denote a "specialized" or "expert" insight.

---

## 6. Do's and Don'ts

### Do
*   **Do** use intentional asymmetry. Overlap a plant image across two different surface containers.
*   **Do** use `secondary` (#516447) for "Cultural" or "Organic" tips to reinforce the nature-inspired vibe.
*   **Do** leverage the full Spacing Scale. If a layout feels "crowded," jump from `spacing-6` to `spacing-10`.

### Don't
*   **Don't** use pure black (#000000) for text. Always use `on_surface` (#191c1c).
*   **Don't** use "bubbly" 100% round corners for cards; stick to `lg` (1rem) to maintain a professional, architectural feel.
*   **Don't** use 1px dividers. If you feel the need for a line, use white space instead.