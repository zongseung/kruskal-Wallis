"""Convert reproduce_paper_tables.py text output into LaTeX tables.

The text output and result/paper_tables.tex used to drift apart (the .tex
was hand-curated and lagged behind the script). This converter regenerates
result/paper_tables.tex directly from a captured run so the two stay in sync.

The emitted tables match the layout used in the paper (revision/kw-test.tex):

  * the working ``N`` and ``SRC`` columns are dropped;
  * ``CHI`` is moved to sit just before ``ED``;
  * SD2/SDC2 carry the gamma-based saddlepoint (Wood, Booth & Butler 1993);
  * each table is wrapped in threeparttable + adjustbox;
  * group sizes are typeset as ``$n_{1}, \\ n_{2}, \\ n_{3}$``.

Usage
-----
    # 1. capture a run
    uv run python examples/reproduce_paper_tables.py > result/paper_tables_raw.txt
    # 2. convert
    uv run python examples/tables_to_latex.py

    # or specify paths explicitly
    uv run python examples/tables_to_latex.py <raw.txt> <out.tex>
"""
import os
import re
import sys

# Header tokens that get an en-dash in LaTeX; everything else is verbatim.
HEADER_MAP = {"E-Q": "E--Q", "E-P": "E--P", "ED-HK": "ED--HK"}

DROP_COLUMNS = {"N", "SRC"}


def parse_blocks(text):
    """Parse the text output into a list of table blocks.

    Each block = dict(section, alpha, label, columns, rows).

    Section titles sit between two rows of '=' characters. The opening and
    closing delimiters look identical, so ``title_done`` marks that the next
    '=' row is a closer (not a new opener).
    """
    lines = text.splitlines()
    blocks = []
    section = ""
    alpha = "0.10"
    label = None
    armed = False        # saw an opening '===' -- next text line is the title
    title_done = False   # just captured a title -- the next '===' is its closer

    i = 0
    while i < len(lines):
        s = lines[i].strip()

        if s and set(s) == {"="}:
            if title_done:
                title_done = False        # closing delimiter
            else:
                armed = True              # opening delimiter
            i += 1
            continue

        if armed:
            if s:                         # the title line itself
                section = s
                label = None
                m = re.search(r"alpha\s*=\s*([0-9.]+)", s)
                alpha = f"{float(m.group(1)):.2f}" if m else "0.10"
                armed = False
                title_done = True
            i += 1
            continue

        if s:
            title_done = False

        # "--- alpha = X ---" marker
        m = re.match(r"-+\s*alpha\s*=\s*([0-9.]+)\s*-+", s)
        if m:
            alpha = f"{float(m.group(1)):.2f}"
            label = None
            i += 1
            continue

        # "[Sub-table label]" line (exactly bracketed)
        if s.startswith("[") and s.endswith("]"):
            label = s[1:-1].strip()
            i += 1
            continue

        # Column header line (contains the PAG(4) column)
        if "PAG(4)" in s and "SD1" in s:
            columns = s.split()
            rows = []
            started = False
            j = i + 1
            while j < len(lines):
                t = lines[j].strip()
                if t and set(t) <= {"-"}:        # dashed separator
                    if started:
                        break
                    j += 1
                    continue
                if not t:
                    if started:
                        break
                    j += 1
                    continue
                if t[0] in "=[":
                    break
                if ":" in t:                      # legend line
                    break
                fields = t.split()
                if len(fields) != len(columns):
                    raise ValueError(
                        f"column mismatch in section '{section}': "
                        f"{len(fields)} fields vs {len(columns)} headers\n  {t}"
                    )
                rows.append(fields)
                started = True
                j += 1
            blocks.append(dict(section=section, alpha=alpha, label=label,
                               columns=columns, rows=rows))
            i = j
            continue

        i += 1

    return blocks


def project_block(columns, rows):
    """Apply the paper column layout: drop N/SRC, move CHI before ED."""
    keep = [i for i, c in enumerate(columns) if c not in DROP_COLUMNS]
    cols = [columns[i] for i in keep]
    rows = [[r[i] for i in keep] for r in rows]

    if "CHI" in cols and "ED" in cols:
        ci = cols.index("CHI")
        chi_name = cols.pop(ci)
        chi_vals = [r.pop(ci) for r in rows]
        ei = cols.index("ED")
        cols.insert(ei, chi_name)
        for r, v in zip(rows, chi_vals):
            r.insert(ei, v)
    return cols, rows


def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def config_header(g):
    """LaTeX header for the group-size column, e.g. $n_{1}, \\ n_{2}, \\ n_{3}$."""
    if not g:
        return "Config"
    return "$" + r", \ ".join(f"n_{{{i}}}" for i in range(1, g + 1)) + "$"


def make_caption(block):
    section = block["section"]
    base = re.sub(r"\s*[,(]\s*alpha\s*=.*$", "", section).strip().rstrip(":")
    base = base.replace("&", r"\&").replace("%", r"\%")
    if block["label"]:
        base += f" --- {block['label']}"
    return f"{base} ($\\alpha = {float(block['alpha']):.2f}$)"


def build_notes(columns):
    notes = []
    if "E-Q" in columns:
        notes.append(r"\item E--Q: Exact (or simulated) upper quantile $c_{\alpha}$.")
    if "E-P" in columns:
        notes.append(r"\item E--P: Corresponding tail probability $\Pr(H \geq c_{\alpha})$.")
    if "SIM-CV" in columns:
        notes.append(r"\item SIM-CV: Simulated upper quantile $c_{\alpha}$ (10{,}000 iterations).")
    if "SIM" in columns:
        notes.append(r"\item SIM: Corresponding simulated tail probability.")
    if "ED-HK" in columns:
        notes.append(r"\item ED--HK: Hall \& Kolassa Edgeworth expansion "
                     r"(asymptotic cumulants).")
    return notes


def _parse_cfg_tuple(s):
    """Parse "3, 4, 5" -> (3, 4, 5) for sorting; degrades gracefully on bad input."""
    try:
        return tuple(int(x.strip()) for x in s.split(",") if x.strip())
    except ValueError:
        return ()


def emit_table(block, idx):
    columns, rows = project_block(block["columns"], block["rows"])
    cfg_idx = 1 if columns and columns[0] == "Category" else 0
    n_text = cfg_idx + 1
    ncol = len(columns)
    colspec = "l" * n_text + "r" * (ncol - n_text)

    # Sort rows by configuration tuple (n1, n2, n3, ...) ascending.
    # If a Category column is present (comprehensive study), preserve the
    # category grouping by sorting on (category, cfg_tuple).
    if cfg_idx == 1:
        rows = sorted(rows, key=lambda r: (r[0], _parse_cfg_tuple(r[1])))
    else:
        rows = sorted(rows, key=lambda r: _parse_cfg_tuple(r[0]))

    group_sizes = {row[cfg_idx].count(",") + 1 for row in rows}
    g = group_sizes.pop() if len(group_sizes) == 1 else None

    header = " & ".join(
        config_header(g) if j == cfg_idx else HEADER_MAP.get(c, c)
        for j, c in enumerate(columns)
    ) + r" \\"

    body = []
    for row in rows:
        cells = list(row)
        cells[cfg_idx] = cells[cfg_idx].replace(",", r", \ ")
        body.append(" & ".join(cells) + r" \\")

    caption = make_caption(block)
    label = f"tab:{slugify(re.sub(r'.alpha.*', '', caption))[:44]}-{idx}"

    out = [
        r"\begin{table}[!htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.05}",
        "",
        r"\begin{threeparttable}",
        r"\begin{adjustbox}{max width=\textwidth}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header,
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        "",
        r"\begin{tablenotes}",
        r"\footnotesize",
        *build_notes(columns),
        r"\end{tablenotes}",
        "",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ]
    return "\n".join(out)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    raw_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(root, "result", "paper_tables_raw.txt")
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "result", "paper_tables.tex")

    with open(raw_path) as f:
        blocks = parse_blocks(f.read())

    parts = [
        "%% ============================================================",
        "%% Paper Tables: Kruskal-Wallis Higher Order Asymptotic Approximations",
        "%% Auto-generated by examples/tables_to_latex.py",
        f"%% Source: {os.path.relpath(raw_path, root)}",
        "%% Do not edit by hand -- regenerate from the script output instead.",
        "%%",
        "%% Layout matches revision/kw-test.tex: the N and SRC working columns",
        "%% are dropped and CHI is placed just before ED. SD1 = Easton-Ronchetti",
        "%% saddlepoint; SD2 = gamma-based saddlepoint (Wood, Booth & Butler 1993).",
        "%% ============================================================",
        "",
    ]
    for idx, block in enumerate(blocks, 1):
        parts.append(emit_table(block, idx))

    with open(out_path, "w") as f:
        f.write("\n".join(parts))

    total_rows = sum(len(b["rows"]) for b in blocks)
    print(f"Parsed {len(blocks)} blocks -> {len(blocks)} LaTeX tables, "
          f"{total_rows} data rows.")
    print(f"Wrote {os.path.relpath(out_path, root)}")


if __name__ == "__main__":
    main()
