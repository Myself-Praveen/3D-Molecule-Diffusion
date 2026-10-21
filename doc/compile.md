# How to compile / render this doc tree

This tree is intended to be processed by a single toolchain pass, not by hand.
The current target:

- **AsciiDoc** → **PDF** via **Asciidoctor PDF**, with a small custom theme for
  code callouts, table styling, and figure placement.
- **PlantUML** rendered into PNG/SVG and referenced by id from AsciiDoc figures.
- **Mermaid** diagrams kept where the reader is more likely to be online and
  viewing in a markdown-native viewer; a converter step is optional.

Workflow:

1. `plantuml` renders each `.puml` into `doc/figures/` by id.
2. `asciidoctor-pdf` renders `doc/index.adoc` → `doc/build/index.pdf`.
3. Any `mermaid` blocks are either left for a markdown viewer or converted with
   `mmdc` when a PDF-only artifact is required.

Why this split: the architecture diagrams (component map, data flow, error
taxonomy) are plantuml because they need precise edges and reusable chunks; the
small inline flowcharts are mermaid because readers hit those first and most
markdown tooling already renders them inline.
