import markdown
from pathlib import Path

src = Path('report.md').read_text(encoding='utf-8')
body = markdown.markdown(src, extensions=['tables', 'fenced_code', 'sane_lists'])
html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
@page {{ size: A4; margin: 15mm 15mm 15mm 15mm; }}
body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif; font-size: 10.2pt;
       line-height: 1.36; color: #111; }}
h1 {{ font-size: 17pt; margin: 0 0 4pt 0; line-height: 1.2; }}
h2 {{ font-size: 12.6pt; margin: 10pt 0 3pt 0; border-bottom: 0.6pt solid #bbb; padding-bottom: 1pt; }}
h3 {{ font-size: 10.9pt; margin: 8pt 0 2pt 0; }}
p {{ margin: 3pt 0; }}
ul, ol {{ margin: 2pt 0 2pt 16pt; padding: 0; }}
li {{ margin: 1pt 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 4pt 0; font-size: 8.8pt; }}
th, td {{ border: 0.5pt solid #999; padding: 2pt 3pt; text-align: left; vertical-align: top; }}
th {{ background: #eee; }}
code {{ font-family: Consolas, monospace; font-size: 8.6pt; background: #f3f3f3; }}
pre {{ font-family: Consolas, monospace; font-size: 8.2pt; background: #f6f6f6;
       padding: 4pt; border: 0.5pt solid #ddd; white-space: pre-wrap; }}
blockquote {{ margin: 3pt 0 3pt 8pt; padding-left: 6pt; border-left: 2pt solid #ccc; }}
hr {{ border: none; border-top: 0.6pt solid #ccc; margin: 8pt 0; }}
</style></head><body>
{body}
</body></html>"""
Path('report.html').write_text(html, encoding='utf-8')
print('wrote report.html')
