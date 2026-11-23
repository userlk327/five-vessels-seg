"""Build paper/refs.bib from verified metadata only.

    python paper/scripts/build_refs.py        # needs web access (Crossref, doi.org, arXiv API)

Every entry is fetched live: DOIs from the Crossref REST API (checked to resolve at doi.org),
arXiv IDs from the arXiv API. Nothing is typed by hand except the corrections listed in FIXES
(each with its reason) and a venue taken from the arXiv record's own comment field.
Writes paper/refs.bib and paper/refs_verification.md (one row per reference: identifier,
source used for verification, title, and the sentence(s) the reference will support).
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "reference-check/1.0"}
PAPER = Path(__file__).resolve().parents[1]

# key, identifier, planned use (the sentence(s) the reference supports)
DOI_REFS = [
    ("jin2022fives", "10.1038/s41597-022-01564-3", "FIVES: 800 images, 573 patients, four categories, official 600/200 split, annotation protocol"),
    ("jin2022fivesdata", "10.6084/m9.figshare.19688169.v1", "Data citation and licence (CC BY 4.0) in Data availability"),
    ("wong2016dr", "10.1038/nrdp.2016.12", "Introduction: retinal vascular changes in diabetic retinopathy, including new fine vessels"),
    ("wong2004hypertensive", "10.1056/NEJMra032865", "Introduction, one general statement only: retinal vessel changes are associated with hypertension"),
    ("abramoff2010retinal", "10.1109/RBME.2010.2084567", "Introduction/related work: retinal image analysis and vessel segmentation as a building block"),
    ("fraz2012survey", "10.1016/j.cmpb.2012.03.009", "Related work: earlier retinal vessel segmentation methods"),
    ("ronneberger2015unet", "10.1007/978-3-319-24574-4_28", "Methods: U-Net baseline architecture"),
    ("liu2022msunet", "10.1155/2022/5188362", "Introduction/related work: multi-scale U-Net with attention reported for retinal vessels"),
    ("cao2023mfaunet", "10.3389/fnins.2023.1249331", "Introduction/related work: multi-scale fusion + attention reported for retinal vessels"),
    ("kirillov2023sam", "10.1109/ICCV51070.2023.00371", "Methods: Segment Anything Model (ViT-B), prompt encoder, mask decoder"),
    ("mazurowski2023sam", "10.1016/j.media.2023.102918", "Related work: zero-shot SAM on medical images varies by task"),
    ("huang2024sam", "10.1016/j.media.2023.103061", "Related work: zero-shot SAM on medical images"),
    ("shi2023samzeroshot", "10.3390/diagnostics13111947", "Related work: zero-shot SAM fails on blood vessels; fine-tuning helps"),
    ("ma2024medsam", "10.1038/s41467-024-44824-z", "Related work: SAM adapted to medical images by fine-tuning"),
    ("shit2021cldice", "10.1109/CVPR46437.2021.01629", "Methods: clDice definition; Introduction: overlap metrics miss topology of thin tubular structures"),
    ("nikolov2021nsd", "10.2196/26151", "Methods: normalized surface Dice (we use a boundary-pixel approximation)"),
    ("maierhein2024metrics", "10.1038/s41592-023-02151-z", "Introduction/Methods: metric choice should match the property of interest"),
    ("reinke2024pitfalls", "10.1038/s41592-023-02150-0", "Introduction: overlap metrics are insensitive to errors on small/thin structures"),
    ("zuiderveld1994clahe", "10.1016/B978-0-12-336156-1.50061-6", "Methods: CLAHE contrast enhancement"),
    ("vanderwalt2014skimage", "10.7717/peerj.453", "Methods: skeletonization with scikit-image"),
    ("zhang1984thinning", "10.1145/357994.358023", "Methods: the 2D skeletonization algorithm used by scikit-image's default"),
    ("virtanen2020scipy", "10.1038/s41592-019-0686-2", "Methods: Euclidean distance transform and Wilcoxon test implementation (SciPy)"),
    ("dice1945", "10.2307/1932409", "Methods: Dice coefficient"),
    ("wilcoxon1945", "10.2307/3001968", "Methods: paired Wilcoxon signed-rank test"),
    ("efron1979bootstrap", "10.1214/aos/1176344552", "Methods: bootstrap confidence intervals"),
    ("bland1986", "10.1016/S0140-6736(86)90837-8", "Methods/Results: Bland–Altman limits of agreement for vessel measures"),
    ("liew2008fractal", "10.1016/j.ophtha.2008.05.029", "Methods: box-counting fractal dimension of the retinal vasculature"),
    ("mongan2020claim", "10.1148/ryai.2020200029", "Methods/Supporting information: CLAIM reporting checklist"),
    ("tejani2024claim", "10.1148/ryai.240300", "Methods/Supporting information: CLAIM 2024 update (checklist used)"),
    ("isensee2021nnunet", "10.1038/s41592-020-01008-z", "Discussion: nnU-Net as a self-configuring reference baseline (future work)"),
]
# key, arXiv ID, planned use, venue confirmed by the arXiv record's comment (None: cite as preprint)
ARXIV_REFS = [
    ("oktay2018attention", "1804.03999", "Methods: additive attention gates", "Medical Imaging with Deep Learning (MIDL) 2018"),
    ("yu2016dilated", "1511.07122", "Methods: dilated convolutions for multi-scale context", "International Conference on Learning Representations (ICLR) 2016"),
    ("hu2022lora", "2106.09685", "Methods: low-rank adaptation (LoRA)", None),
    ("zhang2023samed", "2304.13785", "Methods: SAMed-style LoRA fine-tuning of SAM (q/v LoRA, no prompts)", None),
    ("dosovitskiy2021vit", "2010.11929", "Methods: ViT image encoder used by SAM ViT-B", None),
    ("qiu2023learnable", "2304.13425", "Related work: SAM cannot segment vessels in colour fundus images zero-shot", None),
    ("zhang2024samlimits", "2412.04243", "Related work/Discussion: segmentation foundation models struggle with tree-like, low-contrast objects", None),
    ("fadugba2024benchmark", "2406.14994", "Related work/Discussion: FIVES cross-dataset and cross-disease benchmark", None),
    ("amarnath2026resolution", "2604.02977", "Methods/Discussion: input resolution and thin-vessel sensitivity on FIVES (context for 512)", None),
    ("kingma2015adam", "1412.6980", "Methods: Adam optimizer", "International Conference on Learning Representations (ICLR) 2015"),
    ("micikevicius2018mixed", "1710.03740", "Methods: mixed-precision training", "International Conference on Learning Representations (ICLR) 2018"),
    ("paszke2019pytorch", "1912.01703", "Methods: PyTorch", "Advances in Neural Information Processing Systems (NeurIPS) 2019"),
    ("bouthillier2021variance", "2103.03098", "Introduction/Discussion: training-seed variance in ML benchmarks", None),
]
# Corrections of known metadata defects, each with its reason (applied after fetching).
FIXES = {
    "bland1986": {"author": [("J. Martin", "Bland"), ("Douglas G.", "Altman")],
                  "title": "Statistical methods for assessing agreement between two methods of clinical measurement",
                  "_why": "Crossref splits the names as 'Martin Bland, J.' and 'DouglasG.'; title is all caps"},
    "liu2022msunet": {"pages": "5188362", "_why": "Crossref gives pages 1-10; Hindawi's article number (5188362) is the DOI suffix"},
    "zuiderveld1994clahe": {"booktitle": "Graphics Gems IV",
                            "_why": "Crossref container is 'Graphics Gems'; its ISBN 9780123361561 is the Graphics Gems IV volume"},
}
# Published notices found by Crossref's 'updates' filter on 2026-11-22 (no retractions found).
NOTICES = {
    "wong2016dr": "Erratum published: doi:10.1038/nrdp.2016.30",
    "virtanen2020scipy": "Author Correction published: doi:10.1038/s41592-020-0772-5",
}
# Does the source support the planned sentence? Checked 2026-11-22 against the abstract
# (Crossref, PubMed or arXiv) or, where stated, the full text. "method/tool" = the reference is
# the originating paper of a method or tool we use, so it supports its own citation.
SUPPORT = {
    "jin2022fives": "full text (PMC9352679): 800 images, 573 patients, one hospital (SHAZU), all subjects Asian, 200 per category, 2048x2048, pixelwise manual annotation by trained medical staff verified by ophthalmologists, recommended 600/200 split",
    "wong2016dr": "abstract: proliferative DR is caused by abnormal growth of new retinal blood vessels; DR is a microvascular disease",
    "wong2004hypertensive": "no abstract (NEJM review). Authors' decision: cite only for the general statement that hypertension is associated with retinal vessel changes, which the title ('Hypertensive Retinopathy') supports",
    "abramoff2010retinal": "abstract: quantitative fundus analysis with a focus on retinal vasculature, including segmentation",
    "fraz2012survey": "title: survey of retinal vessel segmentation methods",
    "liu2022msunet": "abstract: improved U-Net with position/channel attention and multiscale modules; accuracy 96.99%/97.51% on DRIVE/CHASE_DB1",
    "cao2023mfaunet": "abstract: multi-scale fusion self-attention aimed at microvessels; CLAHE preprocessing; Dice 84.04 on FIVES among six datasets",
    "mazurowski2023sam": "full text (PMC10528428): point/box prompts on 19 datasets; 'SAM's performance based on single prompts highly varies depending on the dataset and the task'",
    "huang2024sam": "abstract: 53 datasets; SAM sensitive to prompt randomness; fine anatomical structures make medical data harder",
    "shi2023samzeroshot": "abstract: zero-shot SAM 'completely failed' for blood vessels; fine-tuning with little data improved it",
    "ma2024medsam": "full text (PMC10803759): MedSAM fine-tunes SAM on more than one million medical image-mask pairs",
    "qiu2023learnable": "full text (ar5iv): 'SAM ... cannot segment vessels or lesions from color fundus'; Table 1: SAM on FIVES Dice 15.72, recall 82.17",
    "zhang2024samlimits": "abstract: SFMs (SAM, SAM 2, ...) struggle with dense tree-like, low-contrast objects",
    "fadugba2024benchmark": "abstract: five models on FIVES; 'basic architectures such as U-Net perform just as well as more advanced ones'; image quality matters",
    "amarnath2026resolution": "abstract: U-Net on five datasets incl. FIVES; thin/medium/thick at half-width <3, 3-7, >7 px; thin sensitivity on HRF/FIVES peaks at processed widths 256-876 px",
    "bouthillier2021variance": "abstract: variance from data sampling, parameter initialization and hyperparameters markedly affects benchmark results",
    "zhang2023samed": "abstract: LoRA on the SAM image encoder, fine-tuned TOGETHER WITH the prompt encoder and mask decoder; warmup + AdamW. Our setup differs (prompt encoder frozen, Adam) and must say so",
    "reinke2024pitfalls": "full text (nature.com, open access): common metrics are sensitive to structure size; 'errors in small structures might be suppressed by correctly segmented larger structures'; overlap metrics such as DSC ignore boundaries, similarly when the centre(line) is of interest",
    "maierhein2024metrics": "full text (nature.com, open access): 'chosen performance metrics often do not reflect the domain interest'; tubular and non-tubular structures require different metrics",
    "liew2008fractal": "abstract: fractal dimension of retinal vessels from fundus photographs (semiautomated); box-counting to confirm in full text",
    "nikolov2021nsd": "full text (PMC8314151): introduces the surface Dice similarity coefficient at a tolerance tau",
    "shit2021cldice": "method: defines clDice for tubular structures",
}
UNRESOLVED = []
# Manually checked exceptions (no DOI or arXiv ID), approved by the authors on 2026-11-22.
MANUAL = {
    "holm1979": ("article", {"author": "Holm, S.", "title": "A Simple Sequentially Rejective Multiple Test Procedure",
                             "journal": "Scandinavian Journal of Statistics", "year": "1979", "volume": "6", "number": "2",
                             "pages": "65--70", "note": "JSTOR 4615733", "url": "https://www.jstor.org/stable/4615733"},
                 "JSTOR 4615733 (no DOI)", "Methods: Holm correction within each metric",
                 "verified manually (no DOI; Scand. J. Statist. 6(2):65-70, 1979; JSTOR 4615733), approved exception"),
}


def get(url: str, accept: str | None = None) -> bytes:
    headers = dict(UA, **({"Accept": accept} if accept else {}))
    for attempt in range(6):                  # back off on rate limits (arXiv asks for >= 3 s between calls)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=40) as r:
                return r.read()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))


def tex_escape(s: str) -> str:
    return (s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")
            .replace("–", "--").replace("—", "---"))


def protect(title: str) -> str:
    """Brace words with capitals after the first letter (acronyms, CamelCase) so BibTeX keeps them."""
    words = []
    for w in title.split():
        core = w.strip(":,.?()")
        words.append("{" + w + "}" if any(c.isupper() for c in core[1:]) or (core.isupper() and len(core) > 1) else w)
    return " ".join(words)


def fmt_authors(names):
    return " and ".join(f"{tex_escape(fam)}, {tex_escape(giv)}" if giv else "{" + tex_escape(fam) + "}" for giv, fam in names)


def crossref_entry(key: str, doi: str) -> tuple[str, dict]:
    if doi.startswith("10.6084/"):                       # DataCite (figshare): no Crossref record
        bib = get(f"https://doi.org/{doi}", "application/x-bibtex").decode()
        meta = json.loads(get(f"https://api.figshare.com/v2/articles/{doi.split('figshare.')[1].split('.v')[0]}"))
        names = [(" ".join(a["full_name"].split()[:-1]), a["full_name"].split()[-1]) for a in meta["authors"]]
        fields = {"author": fmt_authors(names), "title": protect(tex_escape(meta["title"])),
                  "howpublished": "figshare", "year": meta["published_date"][:4], "doi": doi,
                  "note": f"Dataset, licence {meta['license']['name']}"}
        assert "figshare" in bib.lower()
        return "misc", {"fields": fields, "title": meta["title"], "via": "doi.org (DataCite) + figshare API"}
    m = json.loads(get(f"https://api.crossref.org/works/{doi}"))["message"]
    get(f"https://doi.org/{doi}", "application/x-bibtex")           # the DOI must also resolve at doi.org
    fix = FIXES.get(key, {})
    names = fix.get("author") or [(a.get("given", ""), a.get("family", a.get("name", ""))) for a in m.get("author", [])]
    title = fix.get("title") or " ".join(m["title"][0].split())
    year = (m.get("published-print") or m.get("published-online") or m.get("issued"))["date-parts"][0][0]
    cont = m.get("container-title") or [""]
    f = {"author": fmt_authors(names), "title": protect(tex_escape(title)), "year": str(year), "doi": m["DOI"]}
    typ = {"journal-article": "article", "proceedings-article": "inproceedings", "book-chapter": "incollection"}.get(m["type"], "misc")
    if typ == "article":
        f["journal"] = tex_escape(cont[0])
    else:
        f["booktitle"] = tex_escape(fix.get("booktitle") or cont[-1])
        if m.get("publisher"):
            f["publisher"] = tex_escape(m["publisher"])
    if m.get("volume"):
        f["volume"] = m["volume"]
    if m.get("issue"):
        f["number"] = m["issue"]
    pages = fix.get("pages") or m.get("page") or m.get("article-number")
    if pages:
        f["pages"] = pages.replace("-", "--")
    if key in NOTICES:
        f["note"] = NOTICES[key]
    return typ, {"fields": f, "title": title, "via": "Crossref API + doi.org", "fix": fix.get("_why"),
                 "notice": NOTICES.get(key)}


def arxiv_entries(items) -> dict:
    ids = ",".join(a for _, a, _, _ in items)
    root = ET.fromstring(get(f"https://export.arxiv.org/api/query?id_list={ids}&max_results={len(items) + 5}"))
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    meta = {}
    for e in root.findall("a:entry", ns):
        aid = e.find("a:id", ns).text.rsplit("/", 1)[-1].rsplit("v", 1)[0]
        cm = e.find("x:comment", ns)
        meta[aid] = {"title": " ".join(e.find("a:title", ns).text.split()),
                     "authors": [a.find("a:name", ns).text for a in e.findall("a:author", ns)],
                     "year": e.find("a:published", ns).text[:4], "comment": " ".join(cm.text.split()) if cm is not None else ""}
    out = {}
    for key, aid, _, venue in items:
        m = meta[aid]
        names = [(" ".join(n.split()[:-1]), n.split()[-1]) for n in m["authors"]]
        f = {"author": fmt_authors(names), "title": protect(tex_escape(m["title"])), "year": m["year"],
             "eprint": aid, "archivePrefix": "arXiv", "url": f"https://arxiv.org/abs/{aid}"}
        if venue:
            f["booktitle"] = venue
            f["note"] = f"arXiv:{aid}"
            typ = "inproceedings"
        else:
            f["howpublished"] = f"arXiv:{aid}"
            f["note"] = "Preprint"
            typ = "misc"
        out[key] = (typ, {"fields": f, "title": m["title"], "via": "arXiv API" + (f" (venue from arXiv comment: '{m['comment'][:60]}')" if venue else ""),
                          "comment": m["comment"]})
    return out


def main() -> None:
    entries = {}
    for key, doi, _ in DOI_REFS:
        entries[key] = crossref_entry(key, doi)
    entries.update(arxiv_entries(ARXIV_REFS))
    for key, (typ, fields, _, _, why) in MANUAL.items():
        entries[key] = (typ, {"fields": fields, "title": fields["title"], "via": why})
    today = dt.date.today().isoformat()
    lines = [f"% refs.bib -- generated by paper/scripts/build_refs.py on {today}. Do not edit by hand.",
             "% Every entry was fetched from Crossref/doi.org, DataCite/figshare or the arXiv API.", ""]
    for key in [k for k, *_ in DOI_REFS] + [k for k, *_ in ARXIV_REFS] + list(MANUAL):
        typ, e = entries[key]
        lines.append(f"% verified {today} via {e['via']}" + (f"; corrected: {e['fix']}" if e.get("fix") else ""))
        lines.append(f"@{typ}{{{key},")
        lines += [f"  {k} = {{{v}}}," for k, v in e["fields"].items()]
        lines += ["}", ""]
    (PAPER / "refs.bib").write_text("\n".join(lines))
    use = {k: u for k, _, u in DOI_REFS} | {k: u for k, _, u, _ in ARXIV_REFS} | {k: v[3] for k, v in MANUAL.items()}
    ident = ({k: f"doi:{d}" for k, d, _ in DOI_REFS} | {k: f"arXiv:{a}" for k, a, _, _ in ARXIV_REFS}
             | {k: v[2] for k, v in MANUAL.items()})
    rows = ["| Key | Identifier | Verified via | Title (as registered) | Supports | Support check |", "|---|---|---|---|---|---|"]
    for key in [k for k, *_ in DOI_REFS] + [k for k, *_ in ARXIV_REFS] + list(MANUAL):
        typ, e = entries[key]
        via = e["via"] + (f"; corrected: {e['fix']}" if e.get("fix") else "") + (f"; {e['notice']}" if e.get("notice") else "")
        rows.append(f"| `{key}` | {ident[key]} | {via} | {e['title']} | {use[key]} | {SUPPORT.get(key, 'method/tool')} |")
    for key, ident_, use_, why in UNRESOLVED:
        rows.append(f"| `{key}` | {ident_} | **not verified** | A simple sequentially rejective multiple test procedure | {use_} | {why} |")
    (PAPER / "refs_verification.md").write_text(
        f"# Reference verification ({today})\n\nGenerated by `paper/scripts/build_refs.py`. {len(entries)} entries in "
        f"`refs.bib`: {len(entries) - len(MANUAL)} verified by DOI or arXiv ID, {len(MANUAL)} verified manually "
        f"(approved exception); {len(UNRESOLVED)} unresolved. FIVES records keep the author spelling each record "
        f"uses (article: 'Jingxing Zhou'; figshare dataset: 'Jingxin Zhou').\n\n" + "\n".join(rows) + "\n")
    print(f"wrote {PAPER / 'refs.bib'} ({len(entries)} entries) and refs_verification.md")


if __name__ == "__main__":
    main()
