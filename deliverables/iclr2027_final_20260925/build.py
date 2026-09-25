"""Build the independent ICLR submission from Markdown; no experiments run."""
from pathlib import Path
import json
import re
import subprocess
import shutil
import hashlib
from datetime import datetime, timezone
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent
METADATA = json.loads((ROOT / "metadata.json").read_text())
TITLE = METADATA["title"]
KEYWORDS = "; ".join(METADATA["keywords"])

CITATIONS = {
    "lake2018generalization": ("Lake and Baroni", "2018"),
    "keysers2020measuring": ("Keysers et al.", "2020"),
    "press2023measuring": ("Press et al.", "2023"),
    "yan2021positive": ("Yan et al.", "2021"),
    "jacobsen2019excessive": ("Jacobsen et al.", "2019"),
    "kaushik2020learning": ("Kaushik et al.", "2020"),
    "he2016deep": ("He et al.", "2016"),
    "deng2009imagenet": ("Deng et al.", "2009"),
    "johnson2017clevr": ("Johnson et al.", "2017"),
}


def convert(source, target):
    subprocess.run(["pandoc", str(ROOT/source), "--from=markdown", "--to=latex", "--natbib", "--wrap=none", "-o", str(ROOT/target)], check=True)
    p = ROOT/target
    s = p.read_text().replace(".png}", ".pdf}")
    # Pandoc emits figures as vector PDF in the submission and PNG in Markdown.
    p.write_text(s)


def main():
    archive = ROOT / "archive"
    archive.mkdir(exist_ok=True)
    for name in ("submission.pdf", "submission.md"):
        source = ROOT / name
        target = archive / ("combined_before_split" + source.suffix)
        if source.exists() and not target.exists():
            shutil.copyfile(source, target)
    for a,b in [("abstract.md","abstract.tex"),("main_body.md","main_body.tex"),("statements.md","statements.tex"),("supplement_draft.md","supplement.tex")]:
        convert(a,b)
    tex = r"""\documentclass{article}
\usepackage{iclr2027_conference,times}
\usepackage{amsmath,amssymb,graphicx,booktabs,longtable,array,calc}
\usepackage{microtype}
\usepackage{hyperref}
\usepackage{url}
\usepackage{float}
\setkeys{Gin}{keepaspectratio}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\providecommand{\passthrough}[1]{#1}
\newcommand{\real}{\mathbb{R}}
\hypersetup{pdftitle={TITLEHERE},pdfauthor={Anonymous},colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}
\title{TITLEHERE}
\author{Anonymous authors\\Paper under double-blind review}
\begin{document}
\maketitle
\begin{abstract}
\input{abstract}
\end{abstract}
\noindent\textbf{Keywords:} KEYWORDSHERE
\input{main_body}
\label{lastmainpage}
\clearpage
\input{statements}
\bibliography{references}
\bibliographystyle{iclr2027_conference}
\end{document}
""".replace("TITLEHERE", TITLE).replace("KEYWORDSHERE", KEYWORDS.replace(";", ","))
    (ROOT/"main.tex").write_text(tex)
    supplement_tex = tex.split(r"\begin{document}")[0]
    supplement_tex = supplement_tex.replace(
        r"\title{" + TITLE + "}",
        r"\title{" + TITLE + r"\\Supplementary Material}",
    )
    supplement_tex += r"""\begin{document}
\maketitle
\noindent Figure and table numbers without a supplementary prefix refer to the main paper.
\appendix
\input{supplement}
\end{document}
"""
    (ROOT/"supplementary.tex").write_text(supplement_tex)
    engine = Path("/home/huyudi/.TinyTeX/bin/x86_64-linux/pdflatex")
    bibtex = engine.with_name("bibtex")
    commands = [[str(engine),"-interaction=nonstopmode","-halt-on-error","main.tex"], [str(bibtex),"main"], [str(engine),"-interaction=nonstopmode","-halt-on-error","main.tex"], [str(engine),"-interaction=nonstopmode","-halt-on-error","main.tex"], [str(engine),"-interaction=nonstopmode","-halt-on-error","supplementary.tex"], [str(engine),"-interaction=nonstopmode","-halt-on-error","supplementary.tex"]]
    for i,cmd in enumerate(commands):
        p=subprocess.run(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (ROOT/f"split_build_{i+1}.log").write_text(p.stdout)
        if p.returncode:
            print(p.stdout[-6000:]);raise SystemExit(p.returncode)
    # Resolve the manuscript's fixed bibliography without requiring citeproc.
    supplement=re.sub(r"<!--.*?-->","",(ROOT/"supplement_draft.md").read_text(),flags=re.S)
    main_md = f"# {TITLE}\n\n**Anonymous submission to ICLR 2027**\n\n## Abstract\n\n"+(ROOT/"abstract.md").read_text()+f"\n**Keywords:** {KEYWORDS}\n\n"+(ROOT/"main_body.md").read_text()+"\n\n"+(ROOT/"statements.md").read_text()
    main_md = re.sub(r"\[(@[^\]]+)\]", lambda m: "(" + "; ".join(", ".join(CITATIONS[k.strip().lstrip("@")]) for k in m.group(1).split(";")) + ")", main_md)
    main_md = re.sub(r"@([A-Za-z0-9_]+)", lambda m: CITATIONS[m.group(1)][0] + " (" + CITATIONS[m.group(1)][1] + ")", main_md)
    chunks = re.split(r"\\bibitem(?:\[.*?\])?\{[^}]+\}", (ROOT/"main.bbl").read_text(), flags=re.S)[1:]
    references = []
    for item in chunks:
        item = item.replace(r"\end{thebibliography}", "").replace(r"\newblock", " ").replace("~", " ").replace(r"\&", "&")
        item = re.sub(r"\\(?:em|textit|textbf|url|href|natexlab)\b", "", item)
        item = item.replace('\\"o', "ö").replace("\\'e", "é").replace("{", "").replace("}", "")
        references.append(re.sub(r"\s+", " ", item).strip())
    main_md += "\n# References\n\n" + "\n\n".join(references) + "\n"
    main_md = main_md.replace("{width=100%}", "").replace("{.unnumbered}", "")
    (ROOT/"main.md").write_text(main_md)
    (ROOT/"supplementary.md").write_text(f"# {TITLE}\n\n**Supplementary Material — Anonymous submission to ICLR 2027**\n\nFigure and table numbers without a supplementary prefix refer to the main paper.\n\n" + supplement)
    aux=(ROOT/"main.aux").read_text()
    hit=re.search(r"\\newlabel\{lastmainpage\}\{\{[^}]*\}\{(\d+)\}",aux)
    receipt = {"completed_utc": datetime.now(timezone.utc).isoformat(), "main_text_pages": int(hit.group(1)) if hit else None, "documents": {}, "result_recomputation": False, "git_commit_or_push": False, "ai_disclosure": "Two sentences; actual research uses retained"}
    for stem in ("main", "supplementary"):
        log=(ROOT/f"{stem}.log").read_text()
        with pdfium.PdfDocument(str(ROOT/f"{stem}.pdf")) as pdf:
            count = len(pdf)
        receipt["documents"][stem] = {"pdf": f"{stem}.pdf", "markdown": f"{stem}.md", "pages": count, "overfull": log.count("Overfull "), "undefined": len(re.findall(r"(?:Citation|Reference).*undefined", log)), "pdf_sha256": hashlib.sha256((ROOT/f"{stem}.pdf").read_bytes()).hexdigest()}
    (ROOT/"SPLIT_DELIVERY_RECEIPT.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
