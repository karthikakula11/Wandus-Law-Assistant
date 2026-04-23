from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

from xhtml2pdf import pisa

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Wandus-Full-Project-Code-StepByStep.pdf"

INCLUDE_EXTS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".json", ".md", ".yml", ".yaml", ".sql", ".css", ".html", ".txt", ".toml", ".ini", ".sh"
}

EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".cursor",
    ".vscode",
    "eval/results",
}

EXCLUDE_FILES = {
    ".env",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "Wandus-Full-Project-Code-StepByStep.pdf",
}


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in EXCLUDE_FILES:
        return True
    for d in EXCLUDE_DIRS:
        if rel == d or rel.startswith(d + "/"):
            return True
    return False


def gather_files() -> list[Path]:
    out: list[Path] = []
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if should_exclude(p):
            continue
        if p.suffix.lower() not in INCLUDE_EXTS:
            continue
        out.append(p)
    out.sort(key=lambda p: p.relative_to(ROOT).as_posix())
    return out


def read_text(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_bytes().decode("utf-8", errors="replace")


def line_numbered(content: str) -> str:
    lines = content.splitlines()
    width = max(3, len(str(max(len(lines), 1))))
    return "\n".join(f"{i:{width}d} | {line}" for i, line in enumerate(lines, 1))


def build_html(files: list[Path]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    toc_items = []
    body_blocks = []

    for i, path in enumerate(files, 1):
        rel = path.relative_to(ROOT).as_posix()
        step_title = f"Step {i}: {rel}"
        toc_items.append(f"<li>{escape(step_title)}</li>")

        code = line_numbered(read_text(path))
        body_blocks.append(
            "\n".join(
                [
                    '<div class="file-block">',
                    f"<h2>{escape(step_title)}</h2>",
                    '<p class="meta">Read this file in order; this is one code block in the project flow.</p>',
                    f"<pre>{escape(code)}</pre>",
                    "</div>",
                ]
            )
        )

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Strict//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-strict.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
<title>Wandus Full Project Code (Step-by-Step)</title>
<style type="text/css">
  @page {{ size: A4; margin: 1.1cm; }}
  body {{ font-family: Helvetica, Arial, sans-serif; font-size: 9pt; color: #111; }}
  h1 {{ font-size: 18pt; margin: 0 0 6px 0; }}
  h2 {{ font-size: 11pt; margin: 0 0 4px 0; color: #0b3d91; }}
  .subtitle {{ font-size: 9pt; color: #444; margin: 0 0 8px 0; }}
  .meta {{ font-size: 8pt; color: #555; margin: 0 0 6px 0; }}
  .toc {{ margin: 8px 0 12px 16px; font-size: 8pt; }}
  .file-block {{ page-break-before: always; }}
  pre {{ font-family: Courier New, monospace; font-size: 7pt; line-height: 1.25; background: #f7f7f7; border: 1px solid #ddd; padding: 6px; white-space: pre-wrap; word-wrap: break-word; }}
</style>
</head>
<body>
  <h1>Wandus Project Code - Step by Step</h1>
  <p class="subtitle">Generated: {escape(now)} | Root: {escape(str(ROOT))}</p>
  <p class="meta">This PDF includes project source/config files in a readable sequence. Secrets and heavy generated files are excluded.</p>
  <h2>File Order (Steps)</h2>
  <ol class="toc">
    {''.join(toc_items)}
  </ol>
  {''.join(body_blocks)}
</body>
</html>
"""


def main() -> None:
    files = gather_files()
    if not files:
        raise SystemExit("No files matched selection.")

    html = build_html(files)
    buf = BytesIO()
    status = pisa.CreatePDF(BytesIO(html.encode("utf-8")), dest=buf, encoding="utf-8")
    if status.err:
        raise SystemExit(f"PDF generation failed with {status.err} errors")

    OUT.write_bytes(buf.getvalue())
    print(f"Created: {OUT}")
    print(f"Included files: {len(files)}")


if __name__ == "__main__":
    main()
