"""Rebuild the research homepage and legacy report from released summaries.

Run: python -m pip install -r requirements.txt && python build_report.py
No GPU, account credentials, network requests, or original project paths required.
"""
import csv
import hashlib
import html
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / 'docs'
DATA = DOCS / 'data'
ASSETS = DOCS / 'assets'
ASSETS.mkdir(parents=True, exist_ok=True)
paper = json.loads((ROOT / 'manuscript.json').read_text())
primary = json.loads((DATA / 'primary.json').read_text())
extension = json.loads((DATA / 'extension.json').read_text())
accounting = json.loads((DATA / 'training-accounting.json').read_text())
A = primary['aggregate']
ARMS = ['clean', 'random_matched', 'paired', 'paired_answer', 'random_answer']
NAMES = {'clean': 'Clean GRPO', 'random_matched': 'Random injection', 'paired': 'Boundary paired',
         'paired_answer': 'Paired + answer CE', 'random_answer': 'Random + answer CE'}
METRICS = ['clean_em', 'attack_em', 'attack_target_hit', 'attack_format_failure', 'attack_searches']
SEEDS = ['20261012', '20261013', '20261014']

def dump_csv(name, rows):
    with (DATA / name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

# Recalculate every aggregate and attacked metric from the complete condition summaries.
for arm in ARMS:
    for metric, values in A[arm].items():
        xs = list(values['per_seed'].values())
        assert abs(statistics.mean(xs) - values['mean']) < 1e-12
        assert abs(statistics.stdev(xs) - values['sample_sd']) < 1e-12
    for seed in SEEDS:
        m = primary['per_run'][seed + '/' + arm]['metrics']
        assert all(v['n'] == 1000 for v in m.values())
        for metric, source in [('attack_em', 'em'), ('attack_target_hit', 'target_hit_rate_all'),
                              ('attack_format_failure', 'format_failure_rate'), ('attack_searches', 'mean_searches')]:
            expected = (m['test_override'][source] + m['test_authority'][source]) / 2
            assert abs(expected - A[arm][metric]['per_seed'][seed]) < 1e-12

seed_rows = [{'configuration': arm, 'seed': seed, **{m: A[arm][m]['per_seed'][seed] for m in METRICS}}
             for arm in ARMS for seed in SEEDS]
dump_csv('seeds.csv', seed_rows)
dump_csv('aggregate.csv', [{'configuration': a, 'metric': m, **{k: A[a][m][k] for k in ['mean', 'sample_sd']}}
                           for a in ARMS for m in A[a]])
dump_csv('conditions.csv', [{'configuration': key.split('/')[1], 'seed': key.split('/')[0], 'condition': cond, **vals}
                           for key, row in primary['per_run'].items() for cond, vals in row['metrics'].items()])
dump_csv('costs.csv', [{'configuration': r['arm'], 'seed': r['seed'],
    'loop_seconds': r['budget']['loop_seconds'], 'consumed_attacks': r['budget']['actual_exposed'],
    'optimizer_steps': r['budget']['effective_steps'],
    'accepted_ce_examples': r['training']['auxiliary_sources'].get('examples', 0),
    'answer_loss_tokens': r['training']['auxiliary_sources'].get('answer_tokens', 0),
    'training_allocated_gpu_hours': r['timing']['training_allocated_gpu_hours'],
    **r['training']['cost']} for r in accounting.values()])
for arm, examples, tokens in [('random_answer', 219, 825), ('paired_answer', 200, 805)]:
    selected = [v for v in accounting.values() if v['arm'] == arm]
    assert sum(x['training']['auxiliary_sources']['examples'] for x in selected) == examples
    assert sum(x['training']['auxiliary_sources']['answer_tokens'] for x in selected) == tokens

main_header = ['Configuration', 'Clean EM ↑', 'Attack EM ↑', 'Target hit ↓', 'Format fail ↓', 'Searches']
main_rows = [[NAMES[a]] + [f'{A[a][m]["mean"] * 100:.2f} ± {A[a][m]["sample_sd"] * 100:.2f}' for m in METRICS[:3]]
             + [f'{A[a]["attack_format_failure"]["mean"] * 100:.2f}', f'{A[a]["attack_searches"]["mean"]:.3f}'] for a in ARMS]
seed_header = ['Configuration', 'Seed', 'Clean EM ↑', 'Attack EM ↑', 'Target hit ↓', 'Format fail ↓']
seed_table = [[NAMES[r['configuration']], r['seed']] + [f'{r[m] * 100:.2f}' for m in METRICS[:4]] for r in seed_rows]
context_header = ['Auxiliary context', 'Clean EM ↑', 'Attack EM ↑', 'Target hit ↓']
context_rows = [[name] + [f'{extension["context_aggregate"][arm][m]["mean"] * 100:.2f}' for m in METRICS[:3]]
                for arm, name in [('context_clean', 'Clean context'), ('context_attack', 'Attacked context')]]
main_caption = 'Table 1. Historical export-level confirmation results. EM and target hit show mean ± sample SD over three seeds; format failure and searches show means. All rates are percentages. Attacked metrics average both test templates. Search count is a cost measure, not an accuracy score.'
seed_caption = 'Table 2. Every primary run; rates in percent. No seeds are omitted or combined into a synthetic best run.'
context_caption = 'Table 3. Separate extension confirmation; three-seed means in percent. Exposure and supervision dose are not fully matched.'

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                     'axes.spines.top': False, 'axes.spines.right': False})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.65), sharey=True)
palette = ['#9aabb6', '#697a90', '#897798', '#488a91', '#165c4b']
for ax, m, title in zip(axes, METRICS[:3], ['Clean EM · higher is better', 'Attacked EM · higher is better', 'Target hit · lower is better']):
    for i, arm in enumerate(ARMS):
        vals = [A[arm][m]['per_seed'][s] * 100 for s in SEEDS]
        ax.scatter(vals, [i - .11, i, i + .11], s=25, color=palette[i], alpha=.65, zorder=3)
        ax.scatter([A[arm][m]['mean'] * 100], [i], marker='|', s=200, linewidths=2, color='#17262b', zorder=4)
    ax.set_title(title, fontsize=10, loc='left', pad=15)
    ax.set_xlim(0, 42)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_xlabel('Percent')
    ax.grid(axis='x', color='#dfe3df', linewidth=.6)
axes[0].set_yticks(range(5), list(NAMES.values()))
axes[0].invert_yaxis()
fig.tight_layout(pad=1.2)
for ext in ['svg', 'png', 'pdf']:
    fig.savefig(ASSETS / ('confirmation.' + ext), dpi=180, bbox_inches='tight', facecolor='white')
plt.close(fig)
figure_caption = 'Figure 1. Each dot is one training seed; the dark tick is its configuration mean. Seed dispersion is shown directly, without treating templates or repeated questions as independent samples.'

def esc(s):
    return html.escape(str(s), quote=True)

def table_html(header, rows, caption):
    return '<div class="table-wrap"><table><caption>' + esc(caption) + '</caption><thead><tr>' + ''.join('<th scope="col">' + esc(v) + '</th>' for v in header) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(('<th scope="row">' if i == 0 else '<td>') + esc(v) + ('</th>' if i == 0 else '</td>') for i, v in enumerate(row)) + '</tr>' for row in rows) + '</tbody></table></div>'

def html_section(s):
    h = '<section id="' + s['id'] + '"><h2>' + esc(s['title']) + '</h2>'
    h += ''.join('<p>' + esc(p) + '</p>' for p in s['paragraphs'])
    if s.get('main_table'): h += table_html(main_header, main_rows, main_caption)
    if s.get('seed_table'): h += table_html(seed_header, seed_table, seed_caption)
    if s.get('context_table'): h += table_html(context_header, context_rows, context_caption)
    if s.get('figure'): h += '<figure><img src="assets/confirmation.svg" alt="Three panels show every seed for clean exact match, attacked exact match, and target hit. Full values appear in Tables 1 and 2."><figcaption>' + esc(figure_caption) + '</figcaption></figure>'
    h += ''.join('<p>' + esc(p) + '</p>' for p in s.get('after', []))
    return h + '</section>'

downloads = [('manuscript.pdf', 'Read the PDF'), ('data/seeds.csv', 'Seed results'), ('data/conditions.csv', 'All conditions'),
             ('data/costs.csv', 'Training costs'), ('data/protocol.json', 'Protocol'), ('data/provenance.json', 'Provenance')]
if (DOCS / 'emnlp-manuscript.pdf').exists():
    downloads[0] = ('emnlp-manuscript.pdf', 'Revised manuscript PDF')
page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Retrieval Instruction Robustness in Search-R1 | Yiwen Lu</title>
<meta name="description" content="A technical report on retrieval instruction robustness, auxiliary answer supervision, and a three-seed evaluation of a Qwen2.5-3B search agent.">
<meta name="author" content="Yiwen Lu"><link rel="icon" href="assets/favicon.svg" type="image/svg+xml"><link rel="stylesheet" href="assets/style.css"></head><body>
<a class="skip" href="#abstract">Skip to manuscript</a><div class="reading-bar"><a href="#top">SEARCH-R1 / ROBUSTNESS</a><a href="manuscript.pdf">PDF ↗</a></div>
<main id="top"><header><p class="eyebrow">RESEARCH &amp; ENGINEERING REPORT</p><h1>''' + esc(paper['title']) + '</h1><p class="subtitle">' + esc(paper['subtitle']) + '</p><p class="byline">' + esc(paper['author']) + ' <span>·</span> ' + esc(paper['date']) + '</p><p class="status">' + esc(paper['status']) + '</p><nav class="downloads" aria-label="Downloads">' + ''.join('<a href="' + url + '">' + label + '</a>' for url, label in downloads) + '</nav></header>'
page += '<nav class="contents" aria-label="Contents">' + ''.join('<a href="#' + s['id'] + '">' + esc(label) + '</a>' for s, label in zip(paper['sections'], ['Question', 'Method', 'Evaluation', 'Results', 'Engineering', 'Limitations'])) + '<a href="#seeds">Supplement</a></nav>'
if (DOCS / 'emnlp-manuscript.pdf').exists():
    page += '<aside class="abstract"><h2>SARRA: Attack Resistance Is Only Half the Answer for Search Agents</h2><p>A saved film-comparison error opens the revised manuscript: avoiding the attacker\'s target can still leave the user with a wrong answer. The narrative follows accuracy gains, paired outcomes, transfer, and recovery diagnostics, and now adds author 3B/7B development results and a paired new-versus-repeated-evidence intervention. The manuscript has eight pages of main text and an expanded reproducibility and diagnostics appendix with exact prompts, training and scoring details, full per-run accounting, all transfer conditions, and exploratory supervision/recovery diagnostics. It retains 45 scholarly references plus one software reference, three trace-grounded cases, and a paired analysis of 6,000 attack-condition pairs. <a href="emnlp-manuscript.pdf">Read the revised PDF</a> or download the <a href="emnlp-source.zip">LaTeX source</a>. It is a working preprint, not an accepted paper. The HTML below preserves the original technical report; its <a href="manuscript.pdf">earlier PDF</a> remains available.</p></aside>'
page += '<section id="abstract" class="abstract"><h2>Abstract</h2><p>' + esc(paper['abstract']) + '</p></section>'
page += ''.join(html_section(s) for s in paper['sections'])
page += '<div class="supplement-label">SUPPLEMENTARY MATERIAL</div>' + ''.join(html_section(s) for s in paper['appendices'])
page += '<section id="references"><h2>References</h2><ol class="references">' + ''.join('<li id="ref-' + r['id'] + '"><a href="' + esc(r['url']) + '">' + esc(r['text']) + '</a></li>' for r in paper['references']) + '</ol></section>'
page += '<footer><p>Reporting files: <a href="data/primary.json">primary summary</a> · <a href="data/extension.json">extension summary</a> · <a href="data/aggregate.csv">aggregate CSV</a> · <a href="data/training-accounting.json">cost ledger</a> · <a href="data/model-hashes.json">model hashes</a></p><p>Yiwen Lu · Search-R1 robustness project · Technical report, 2026</p></footer></main></body></html>'
(DOCS / 'report.html').write_text(page.replace('assets/style.css', 'assets/report.css'))
(DOCS / '.nojekyll').touch()

# The legacy report HTML and PDF use the same paragraphs and computed tables.
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='PaperTitle', fontName='Times-Bold', fontSize=24, leading=27, spaceAfter=10))
styles.add(ParagraphStyle(name='PaperSubtitle', fontName='Times-Roman', fontSize=15, leading=19, spaceAfter=12))
styles.add(ParagraphStyle(name='PaperBody', fontName='Times-Roman', fontSize=10.5, leading=14, spaceAfter=7))
styles.add(ParagraphStyle(name='PaperHeading', fontName='Helvetica-Bold', fontSize=12, leading=16, spaceBefore=15, spaceAfter=8, keepWithNext=True))
styles.add(ParagraphStyle(name='PaperCaption', fontName='Helvetica', fontSize=8, leading=11, spaceBefore=6, spaceAfter=10))
styles.add(ParagraphStyle(name='PaperCell', fontName='Helvetica', fontSize=7.4, leading=10))
styles.add(ParagraphStyle(name='PaperMeta', fontName='Helvetica', fontSize=9, leading=13, spaceAfter=8))

def pdftext(s):
    return esc(s.replace('↑', '(+)').replace('↓', '(-)').replace('θ', 'theta'))

def para(s, style='PaperBody'):
    return Paragraph(pdftext(s), styles[style])

story = [para(paper['title'], 'PaperTitle'), para(paper['subtitle'], 'PaperSubtitle'),
         para(paper['author'] + ' | ' + paper['date'] + ' | Technical report v1.0', 'PaperMeta'),
         para('Abstract', 'PaperHeading'), para(paper['abstract'])]

def pdf_table(header, rows, caption):
    n = len(header)
    widths = ([126, 72, 72, 72, 60, 50] if n == 6 and header[1] != 'Seed' else
              [137, 67, 62, 62, 62, 62] if n == 6 else [180, 90, 90, 92])
    cells = [[para(c, 'PaperCell') for c in row] for row in [header] + rows]
    tab = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e9eeea')),
                            ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#526858')),
                            ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d9dfdb')),
                            ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 6),
                            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    story.extend([tab, para(caption, 'PaperCaption')])

for i, s in enumerate(paper['sections'] + paper['appendices']):
    if i == len(paper['sections']): story.append(PageBreak())
    story.append(para(s['title'], 'PaperHeading'))
    story.extend(para(p) for p in s['paragraphs'])
    if s.get('main_table'): pdf_table(main_header, main_rows, main_caption)
    if s.get('seed_table'): pdf_table(seed_header, seed_table, seed_caption)
    if s.get('context_table'): pdf_table(context_header, context_rows, context_caption)
    if s.get('figure'):
        from PIL import Image as PILImage
        with PILImage.open(ASSETS / 'confirmation.png') as im: ratio = im.height / im.width
        story.extend([Image(str(ASSETS / 'confirmation.png'), width=452, height=452 * ratio), para(figure_caption, 'PaperCaption')])
    story.extend(para(p) for p in s.get('after', []))
story.append(para('References', 'PaperHeading'))
for r in paper['references']:
    story.append(Paragraph('[' + r['id'] + '] ' + pdftext(r['text']) + ' <link href="' + esc(r['url']) + '">' + esc(r['url']) + '</link>', styles['PaperBody']))

def page_number(canvas, doc):
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#60716b'))
    canvas.drawString(72, 36, 'Search-R1 retrieval instruction robustness · Technical report')
    canvas.drawRightString(540, 36, str(doc.page))

SimpleDocTemplate(str(DOCS / 'manuscript.pdf'), pagesize=(612, 792), rightMargin=72, leftMargin=72,
    topMargin=54, bottomMargin=54, title=paper['title'], author=paper['author']).build(story, onFirstPage=page_number, onLaterPages=page_number)
from project_site import build_site
build_site()
manifest = {str(p.relative_to(DOCS)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(DOCS.rglob('*'))
            if p.is_file() and p.name != 'release-sha256.json'}
(DOCS / 'release-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('Built website, PDF, figures, and CSVs. All primary aggregates and seed values verified.')
