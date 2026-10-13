# arXiv upload package

The final manuscript package contains 19 files, with `main.tex` at its root,
`authors.tex`, `appendix.tex`, `.bib` and `.bbl` bibliography files, required style
files and ten PDF figures. It builds using pdfLaTeX/BibTeX without shell escape,
network access or external image conversion. The manuscript has nine main-text
pages and eighteen pages including references and appendices.

The prepared artifacts are `m2o_arxiv.pdf` and `m2o_arxiv_source.zip` in the
release-artifacts folder. Upload the **source ZIP** as the manuscript source;
inspect arXiv's generated PDF before completing submission. Code and checkpoint files are not LaTeX source.

Title: When More Calls Help: Dataset-Free Low-Compute Self-Distillation for MeanFlow

Authors, in order:
1. Zhicheng Jiang — Massachusetts Institute of Technology; 2077 AI.
2. Ningshan Ma — Massachusetts Institute of Technology.

Contact: jzc_2007@mit.edu, in a first-page footnote.

The original ICLR template is retained with named authors and a blank running
header. There is no under-review or accepted-publication claim. No arXiv ID,
journal reference, DOI or public repository URL is invented. A submission
license and subject classification are selected by the authors during upload;
the code's MIT license does not select a manuscript license automatically.

Checks completed locally: clean source compilation, resolved references and
citations, embedded fonts, metadata, all eighteen rendered pages and all figure
pages. Server-side arXiv compilation and publication have not occurred.

Official packaging guidance: https://info.arxiv.org/help/submit_tex.html
