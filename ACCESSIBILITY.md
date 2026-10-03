# Accessibility Statement

rAPIdtools is developed by the NHERI RAPID Facility at the University of
Washington. We want researchers, emergency managers and students to be able to
use the package regardless of disability, and we treat accessibility problems
as bugs. This page describes what rAPIdtools consists of, what we know about
the accessibility of each part, and how to tell us when something does not
work for you.

## What rAPIdtools consists of

- **A Python library and command-line tools.** Most users drive rAPIdtools
  from scripts, notebooks or a terminal. These work with whatever editor,
  terminal and assistive technology you already use; the package itself has no
  user interface of its own in this mode.
- **Documentation** at
  [rapid-facility.github.io/rAPIdtools](https://rapid-facility.github.io/rAPIdtools/),
  built with Sphinx and the Sphinx Book Theme.
- **A browser-based graphical interface** (`rapidtools-gui`) for detecting
  assets and running inference without writing code.

## Current status

### Library and command line

Output is plain text, GeoJSON, Shapefile and other standard formats that can
be read by screen readers and consumed by any downstream tool. If you find a
message or progress display that relies on colour or layout alone, please
report it.

### Documentation

The documentation site supports the following:

- Keyboard navigation of the sidebar, page contents and search box.
- Light and dark colour themes, selectable from the toolbar.
- Text that reflows and can be resized with the browser's zoom controls.
- Headings in a logical order and a skip-to-content link, provided by the
  theme.

Known limitations:

- Some figures in the examples are screenshots or rendered maps whose
  alternative text summarises rather than fully describes the image.
- Code samples are presented in a scrolling block; long lines may require
  horizontal scrolling.

### Graphical interface

The GUI has not yet been through a formal accessibility audit against the
Web Content Accessibility Guidelines (WCAG) 2.2. What we know today:

- Form controls carry visible text labels that are associated with their
  inputs.
- The page is written in English and declares its language, and its structure
  uses headings.
- The interactive map and the image viewers are drawn on an HTML canvas. Their
  content is not exposed to screen readers, and the detections they display
  are only available visually. The same information is available as a
  downloadable GeoJSON file.
- Several controls use fixed, small font sizes, and not every colour pairing
  has been checked for sufficient contrast.
- Not every control has an explicit accessible name or role for assistive
  technology, and keyboard focus is not always clearly visible.
- Animations do not yet honour the operating system's "reduce motion" setting.

We consider the GUI the part of the project most in need of improvement and
plan to address the items above in future releases.

## Compatibility

The documentation and GUI are tested in recent versions of Firefox and
Chromium-based browsers on Linux, macOS and Windows. We have not yet tested
systematically with screen readers such as NVDA, JAWS, VoiceOver or Orca, or
with switch and voice-control input. Reports from users of these tools are
especially welcome.

## Tell us about a problem

If any part of rAPIdtools is difficult or impossible for you to use, please
let us know. Include which part you were using (library, documentation or
GUI), the browser or terminal and any assistive technology involved, and what
happened.

- Open an issue at
  <https://github.com/RAPID-Facility/rAPIdtools/issues> with the label
  `accessibility`.
- If you would rather not use GitHub, email <uwrapid@uwrapid.org> with
  "rAPIdtools accessibility" in the subject line.

We aim to acknowledge reports within five working days. Accessibility fixes
are released on the same schedule as other bug fixes.

## Contributing accessibility improvements

Pull requests that improve accessibility are welcome and follow the normal
process in [CONTRIBUTING.md](CONTRIBUTING.md). Helpful contributions include
alternative text for figures, accessible names and keyboard handling for GUI
controls, contrast fixes, and a text or table alternative to the canvas-based
map view.

This statement was last reviewed on 3 October 2026 and will be updated as the
project changes.
