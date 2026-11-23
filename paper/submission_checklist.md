# Submission checklist

**Status (2026-11-22): no venue chosen.** The manuscript is written as a general-purpose paper in the
PLOS LaTeX template (working format). Venue-only packaging is deferred until a venue is chosen:
TIFF/EPS export, single-file flattening, fees and submission-system fields. The PLOS ONE and MELBA
rules below are kept for reference.

Decisions recorded 2026-11-22: authors are the four team members (Lalit Kharel, Carter Wald, Peyton
Nelson, Jacob Dimmitt), no instructor or advisor; order and corresponding author `[TODO]`. Short
title approved: "Attention, multi-scale blocks and fine-tuned SAM for thin retinal vessels". Code
licence: MIT `[TODO: confirm with co-authors]`. Ethics statement (revision 2): public dataset, no new human data, no further approval required `[TODO: authors
confirm]`; cites the FIVES approval (SAHZU Medical Ethics Committee, Declaration of Helsinki, NCT04718532) and consent;
does not call the data de-identified.
Holm (1979) is cited as a manually verified exception (no DOI; JSTOR 4615733).


Rules checked on 2026-11-22 against the official PLOS ONE pages listed at the end. Status:
`[x]` done, `[ ]` to do, `[TODO]` needs the authors.

## 1. Venue rules (PLOS ONE research article)

| Item | PLOS ONE rule | Plan / status |
|---|---|---|
| Length | "Manuscripts can be any length. There are no restrictions on word count, number of figures, or amount of supporting information." Concise presentation encouraged. | `[ ]` Keep the main text focused; secondary tables to Supporting information. |
| Title | Full title ≤ 250 characters, sentence case; short title ≤ 100 characters. | `[x]` Title changed in revision 2 (113 characters). `[TODO]` Confirm the short title. |
| Abstract | ≤ 300 words, one unstructured paragraph; no citations; abbreviations avoided where possible. | `[x]` 298 words, unstructured, no citations (Step 4). |
| Order | Title page (title, authors, affiliations) → Abstract → Introduction; middle sections can be renamed and ordered freely; then Acknowledgments → References → Supporting information captions. | `[x]` `main.tex` follows this order (Related work is a middle section). |
| Layout | Double-spaced, single column, page numbers and continuous line numbers, no footnotes, at most 3 heading levels. | `[x]` Template + `\doublespacing` + `lineno`. `[ ]` No footnotes in the text (Step 3). |
| Abbreviations | Define at first use; no non-standard abbreviations used fewer than 3 times. | `[x]` Step 4 consistency audit (2026-11-23): CNN, LoRA, IoU, CI defined at first use; DR, CLAHE and ViT-B appear once or twice in the text but also in tables and as method names. |
| References | Vancouver style via the template's `plos2025.bst`; preprints only with a citable DOI or arXiv URL; no unpublished work; avoid retracted work; give current details of corrected work. | `[x]` `refs.bib` (43 verified entries, see `refs_verification.md`); no retractions found; 2 published corrections recorded in the entries. |
| LaTeX | Submit LaTeX manuscripts as PDF; the source must be a single `.tex` file (no `\input`); no template package may be removed; no graphics or colour in the text. | `[ ]` Draft uses `\input`; Step 5 writes `main_submission.tex` (flattened, bibliography inlined, `\showfiguresfalse`). |
| Figures | TIFF or EPS only; width 789–2250 px at 300 dpi (6.68–19.05 cm); max height 22.23 cm; < 10 MB; text in Arial, Times or Symbol at 8–12 pt; multi-panel figures as one file; captions in the manuscript, after the first paragraph citing the figure; cite as "Fig". | `[ ]` Step 2 exports vector PDF (Overleaf previews, MELBA) **and** TIFF (LZW, ≥ 300 dpi) or EPS for PLOS. |
| Tables | Cell-based, editable; no nested tabulars, no graphics or coloured text (cell shading allowed); placed after the first paragraph citing them; `adjustwidth` for wide tables. | `[ ]` Step 2 generates `tables/*.tex` to these rules. |
| Supporting information | Uploaded separately; captions listed at the end of the manuscript (e.g. "S1 Table."), cited with `\nameref`. | `[ ]` Step 2/3. |
| Statistics | A statistical-analysis subsection in Methods: software names and versions with references, enough detail to reproduce, repository identifier for the analysis code. | `[ ]` Step 3. Versions in the analysis environment: PyTorch 2.14.1+cu126, scikit-image 0.26.0, SciPy 1.17.1, OpenCV 5.1.0 (all in `requirements.lock.txt`). |
| Cover letter | Separate file, at most 1 page: contribution, relation to published work, article type, prior interactions with PLOS. | `[ ]` Step 5. |

## 2. Statements required at submission

PLOS collects funding, competing interests, data availability and author contributions in
the submission system; their text is kept in `sections/declarations.tex`.

| Statement | Requirement | Status |
|---|---|---|
| Author list, affiliations, corresponding author | Names, affiliations, email of the corresponding author. | `[TODO]` Order, department, corresponding author, and whether the instructor/advisor is a co-author or acknowledged. |
| Author contributions | At least one CRediT role per author, entered in the submission system. | `[TODO]` Roles per author. |
| Financial disclosure (funding) | Funding sources, or a statement that there was none. | `[TODO]` |
| Competing interests | Declaration for every author. | `[TODO]` |
| Data availability | Statement published with the article; the "minimal data set" needed to replicate all findings must be available. For third-party data, give access details. | `[ ]` Draft: FIVES (Jin et al. 2022; figshare doi:10.6084/m9.figshare.19688169.v1, CC BY 4.0) via the Hugging Face mirror `tyluan/FIVES` (tagged CC BY 4.0); split files, per-image metrics, sweeps and all result tables in the public repository. |
| Code availability | Author-generated code that underpins the findings must be available without restriction at publication. PLOS recommends a repository that issues persistent identifiers (DOIs), an open-source licence and documentation. | `[ ]` Code: `https://github.com/userlk327/five-vessels-seg`; analysis decisions recorded before test evaluation at commit `0ee73608db87db9e0b6f2ff46d29f2faa75fd1dc`. `[TODO]` The repository has **no LICENSE file**: choose an open-source licence. `[TODO]` Archive a release with a DOI (e.g. Zenodo) before submission. |
| Ethics statement | Research with human participants needs IRB approval; for third-party personal data, declare risks to individuals' privacy or affirm there were none. | `[TODO]` Confirm with the UNL IRB (or the advisor) that secondary analysis of a public, de-identified dataset needs no approval. Draft text: public de-identified dataset, no new human data; burned-in text markings outside the eye were never read, transcribed or stored, and every figure pixelates the area outside the field of view. |
| AI-use disclosure | Must be in the article body (a Methods subsection or the Acknowledgments) and include: the tool names; how they were used; which aspects of the study, text, data or files were affected; how the authors checked the outputs. AI tools cannot be authors. | `[ ]` Step 3 (Acknowledgments or Methods). `[TODO]` Authors confirm the wording. |
| Reporting checklist | PLOS asks for adherence to relevant reporting guidelines. | `[x]` CLAIM 2024 (Tejani et al. 2024) filled in as S1 Checklist (`claim_checklist.md`); items 43–44 TODO. |
| Fees | Article processing charge on acceptance. | `[TODO]` Advisor approval. The PLOS fee page did not load for automated checking; third-party pages report $2,477 (October 2026) and a search snippet shows $2,670. Confirm on plos.org/fees. |

## 3. Conflicts between PAPER_PROMPT.md and PLOS ONE (and how they are resolved)

1. **Figure format:** PAPER_PROMPT asks for vector PDF figures; PLOS accepts only TIFF or EPS. Both are produced: PDF for drafting and the MELBA fallback, TIFF/EPS for PLOS.
2. **Single file:** PAPER_PROMPT's layout uses `sections/`, `numbers.tex` and `tables/*.tex`; PLOS requires one `.tex` file. Drafting stays modular; a flattening step produces the submission file.
3. **Declarations:** PAPER_PROMPT puts them in the manuscript; PLOS collects them in the submission system (except the AI disclosure, which must be in the body). The text is kept in `sections/declarations.tex` for copying into the system and for MELBA.
4. **Related work:** PLOS has no separate Related work section by default; middle sections may be renamed and ordered freely, so it stays as its own section between Introduction and Methods.

## 4. Fallback: MELBA

From `melba.scholasticahq.com/for-authors` (checked 2026-11-22):
- Submission in LaTeX with the MELBA template, through Scholastica; cover letter required.
- Typical length about 20 pages, without a strict limit; the published PDF is hosted on arXiv (arXiv posting at submission is encouraged).
- Scope: new ML methods or new applications to medical data; an evaluation study should state its contribution clearly in the cover letter.
- Fees: not stated on the authors page (the site describes the journal as free and open access); confirm before switching.

Adapting: the `sections/` files are written without PLOS-specific commands, so a MELBA main file can wrap them; figures as PDF; declarations go in the manuscript.

## 5. Sources (official pages, accessed 2026-11-22)

- PLOS ONE submission guidelines: https://journals.plos.org/plosone/s/submission-guidelines
- PLOS ONE LaTeX guidelines and template (v3.8, Apr 2026): https://journals.plos.org/plosone/s/latex
- PLOS ONE figures: https://journals.plos.org/plosone/s/figures
- PLOS ONE tables: https://journals.plos.org/plosone/s/tables
- Data availability: https://journals.plos.org/plosone/s/data-availability
- Materials, software and code sharing: https://journals.plos.org/plosone/s/materials-software-and-code-sharing
- Ethical publishing practice (AI tools and disclosures): https://journals.plos.org/plosone/s/ethical-publishing-practice
- Criteria for publication: https://journals.plos.org/plosone/s/criteria-for-publication
- MELBA for authors: https://melba.scholasticahq.com/for-authors
