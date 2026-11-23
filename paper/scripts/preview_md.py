"""Write paper/draft_preview.md: a readable Markdown copy of the manuscript (no TeX needed).

    python paper/scripts/preview_md.py

Approximate rendering for reading only: number macros replaced by their values, citations shown as [key],
figures as image links to the PNG previews, tables as pointers to assets_review.md. The LaTeX files remain
the source of truth.
"""
from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1]


def macros() -> dict[str, str]:
    out = {}
    for f in ("numbers.tex", "captions.tex"):
        for name, body in re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$", (PAPER / f).read_text(), flags=re.M):
            out[name] = body
    return out


def render(tex: str, M: dict[str, str]) -> str:
    t = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())
    for _ in range(3):                                        # captions contain number macros
        t = re.sub(r"\\(\w+)\{\}", lambda m: M.get(m.group(1), m.group(0)), t)
        t = re.sub(r"\\([A-Za-z]+)(?![A-Za-z])", lambda m: M.get(m.group(1), m.group(0)), t)
        t = re.sub(r"\\(Cap\w+)\b", lambda m: M.get(m.group(1), m.group(0)), t)
    t = re.sub(r"\\begin\{figure\}.*?\\figpreview(?:\[[^]]*\])?\{figures/([^}]+)\}.*?\\caption\{(.*?)\}\s*\\label\{[^}]+\}\s*\\end\{figure\}",
               lambda m: f"\n![{m.group(1)}](figures/preview/{Path(m.group(1)).stem}.png)\n\n*{m.group(2)}*\n", t, flags=re.S)
    t = re.sub(r"\\input\{tables/(\w+)\}", r"\n> **[Table `\1`: see assets_review.md]**\n", t)
    t = re.sub(r"\\subsection\*\{([^}]*)\}", r"\n### \1\n", t)
    t = re.sub(r"\\subsubsection\*\{([^}]*)\}", r"\n#### \1\n", t)
    t = re.sub(r"\\paragraph\*\{([^}]*)\}", r"\n**\1** ", t)
    t = re.sub(r"\\cite\{([^}]*)\}", r"[\1]", t)
    t = re.sub(r"Fig~\\ref\{fig:(\w+)\}", r"Fig (\1)", t)
    t = re.sub(r"Table~\\ref\{tab:(\w+)\}", r"Table (\1)", t)
    t = re.sub(r"\\nameref\{si:(\w+)\}", r"[SI: \1]", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = re.sub(r"\\(emph|textbf|texttt|url)\{([^}]*)\}", r"*\2*", t)
    t = re.sub(r"\{\\bf ([^}]*)\}", r"**\1**", t)
    t = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", t).replace("\\item", "-")
    t = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", lambda m: "\n$$" + " ".join(m.group(1).split()) + "$$\n", t, flags=re.S)
    rep = {"\\textminus{}": "−", "\\textminus": "−", "\\times": "×", "$\\times$": "×", "\\%": "%", "~": " ", "--": "–",
           "\\pm": "±", "\\le ": "≤ ", "\\ge ": "≥ ", "\\,": " ", "\\ ": " ", "\\_": "_", "\\&": "&"}
    for a, b in rep.items():
        t = t.replace(a, b)
    t = re.sub(r"\$([^$\n]{1,40})\$", r"\1", t)
    return re.sub(r"\n{3,}", "\n\n", t)


def main():
    M = macros()
    main_tex = (PAPER / "main.tex").read_text()
    title = re.search(r"\\textbf\{(Attention gates.*?)\}", main_tex).group(1)
    out = [f"# {title}", "", "*Reading copy generated from the LaTeX sources by `paper/scripts/preview_md.py`; "
           "the LaTeX files are the source of truth. Authors: [TODO].*", ""]
    for sec, head in (("abstract", "Abstract"), ("introduction", "Introduction"), ("related", "Related work"),
                      ("methods", "Materials and methods"), ("results", "Results"), ("discussion", "Discussion"),
                      ("conclusion", "Conclusion"), ("declarations", "Declarations"), ("acknowledgments", "Acknowledgments"),
                      ("supporting", "Supporting information")):
        out += [f"## {head}", render((PAPER / "sections" / f"{sec}.tex").read_text(), M), ""]
    (PAPER / "draft_preview.md").write_text("\n".join(out))
    left = sorted(set(re.findall(r"\\[A-Za-z]+", (PAPER / "draft_preview.md").read_text())))
    print(f"wrote paper/draft_preview.md; LaTeX commands left in the preview: {left[:20]}")


if __name__ == "__main__":
    main()
