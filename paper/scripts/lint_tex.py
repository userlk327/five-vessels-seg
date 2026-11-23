"""Static checks of the LaTeX sources (no TeX installation needed).

    python paper/scripts/lint_tex.py

Resolves \\input recursively from main.tex and supplement.tex and checks:
  * every \\Macro{} used is defined (numbers.tex, captions.tex, the main file) or is a known LaTeX command;
  * every \\ref / \\nameref target has a \\label; every \\cite key is in refs.bib; every \\input file exists;
  * braces balance and $ signs pair up in each file; no character outside ASCII except those pdfLaTeX's utf8
    input maps with the template's packages;
and reports word counts (abstract; main text from Introduction to Conclusion, without tables, captions and math).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]
KNOWN = set("""documentclass usepackage newcommand renewcommand begin end section subsection subsubsection paragraph
caption label ref nameref cite input includegraphics textbf textit emph texttt url bf it textsuperscript textsubscript
item hline thickhline multicolumn centering raggedright newline bigskip medskip vspace clearpage newgeometry linenumbers
nolinenumbers bibliography bibliographystyle times sigma psi phi theta mathrm lvert rvert left right frac cap cup oplus
partial tau le ge leq geq dots quad textminus pm textdegree in times alpha ldots infty sum textcurrency Yinyang ddag dag
textpilcrow figurename makeatletter makeatother quad setlength parindent textwidth textheight raggedbottom
DisableLigatures newcolumntype newlength noalign global arrayrulewidth cline vskip savedwidth vrule pagestyle fancyhf
rfoot lfoot pageref today headrulewidth footrule hrule fancyheadoffset fancyfootoffset biblabel doublespacing
captionsetup DeclareCaptionLabelFormat Large large newif ifshowfigures showfigurestrue showfiguresfalse fi figpreview
linewidth else thetable thefigure arabic noindent par smallskip and hfill qquad mid text boldsymbol mathcal Big big
labelformat footnote thepage circ max min sitab""".split())
OK_UNICODE = set("–—·°")


def resolve(path: Path, seen=None) -> list[Path]:
    seen = seen or []
    if path in seen:
        return seen
    seen.append(path)
    for inc in re.findall(r"\\input\{([^}]+)\}", strip_comments(path.read_text())):
        f = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
        if not f.exists():
            print(f"MISSING input: {inc} (from {path.name})")
            continue
        resolve(f, seen)
    return seen


def strip_comments(s: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in s.splitlines())


def defined(files) -> set[str]:
    out = set()
    for f in files:
        out |= set(re.findall(r"\\newcommand\{\\(\w+)\}", f.read_text()))
        out |= set(re.findall(r"\\newcommand\\(\w+)", f.read_text()))
        out |= set(re.findall(r"\\newif\\if(\w+)", f.read_text()))
    return out


def words(tex: str) -> int:
    t = strip_comments(tex)
    t = re.sub(r"\\begin\{(table|figure|equation)\}.*?\\end\{\1\}", " ", t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", " x ", t)
    t = re.sub(r"\\(cite|ref|nameref|label|input)\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\{\}", " n ", t)          # number macros count as one word
    t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
    t = re.sub(r"[{}~]", " ", t)
    return len([w for w in t.split() if re.search(r"[A-Za-z0-9]", w)])


def main() -> int:
    problems = 0
    bib = set(re.findall(r"@\w+\{([^,]+),", (PAPER / "refs.bib").read_text()))
    for top in ("main.tex", "supplement.tex"):
        files = resolve(PAPER / top)
        defs = defined(files)
        labels, uses_ref, cites = set(), set(), set()
        for f in files:
            s = strip_comments(f.read_text())
            labels |= set(re.findall(r"\\label\{([^}]+)\}", s))
            uses_ref |= set(re.findall(r"\\(?:ref|nameref)\{([^}]+)\}", s))
            for c in re.findall(r"\\cite\{([^}]+)\}", s):
                cites |= {k.strip() for k in c.split(",")}
            for m in sorted(set(re.findall(r"\\([A-Za-z]+)", s)) - defs - KNOWN):
                print(f"UNDEFINED \\{m} in {f.relative_to(PAPER)}")
                problems += 1
            depth = 0
            for ch in re.sub(r"\\[{}$%&_#]", "", s):
                depth += (ch == "{") - (ch == "}")
            if depth:
                print(f"BRACES unbalanced by {depth} in {f.relative_to(PAPER)}")
                problems += 1
            if len(re.findall(r"(?<!\\)\$", s)) % 2:
                print(f"ODD number of $ in {f.relative_to(PAPER)}")
                problems += 1
            bad = sorted({c for c in s if ord(c) > 127 and c not in OK_UNICODE})
            if bad:
                print(f"NON-ASCII {''.join(bad)!r} in {f.relative_to(PAPER)}")
                problems += 1
        for r in sorted(uses_ref - labels - {"#1"}):
            print(f"[{top}] UNRESOLVED reference {r}")
            problems += 1 if top == "main.tex" else 0
        for k in sorted(cites - bib):
            print(f"[{top}] UNKNOWN citation {k}")
            problems += 1
        unused = sorted(bib - cites) if top == "main.tex" else []
        print(f"[{top}] {len(files)} files, {len(cites)} cited keys, {len(labels)} labels"
              + (f"; refs.bib entries never cited: {unused}" if unused else ""))
    sec = PAPER / "sections"
    abstract = words((sec / "abstract.tex").read_text())
    body = sum(words((sec / f"{n}.tex").read_text()) for n in ("introduction", "related", "methods", "results", "discussion", "conclusion"))
    print(f"words: abstract {abstract}; main text (Introduction to Conclusion, without tables, figure captions and display "
          f"math) {body}")
    print("OK" if problems == 0 else f"{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
