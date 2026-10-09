"""Build the project homepage from released evidence; Python standard library only."""
from pathlib import Path
import json, re, html, hashlib
ROOT=Path(__file__).resolve().parent
DOCS=ROOT/'docs'

def development_section():
    cross=json.loads((DOCS/'data/cross-model-development.json').read_text())
    evidence=json.loads((DOCS/'data/evidence-branch-7b.json').read_text())
    labels={'clean':'Clean','test_override':'Override','test_authority':'Authority',
            'neutral_override':'Neutral / override','quoted_override':'Quoted / override',
            'neutral_authority':'Neutral / authority','quoted_authority':'Quoted / authority'}
    rows=''
    for condition,models in cross['conditions'].items():
        for model,r in models.items():
            cells=''.join(f'<td>{100*r["rates"][m]:.2f}</td>' for m in ['em','target_hit','other_wrong','format_failure'])
            rows+=f'<tr data-condition="{condition}" data-model="{model}"><th scope="row">{labels[condition]}</th><td>{model.split("_")[-1].upper()}</td>{cells}<td>{r["rates"]["searches"]:.3f}</td><td>{100*r["rates"]["actual_exposure"]:.0f}</td></tr>'
    cards='';paired=''
    for c,r in evidence['conditions'].items():
        new=100*r['branches']['new']['eligible_mean']['em'];repeat=100*r['branches']['repeat']['eligible_mean']['em'];delta=100*r['new_minus_repeat_em'];lo,hi=[100*x for x in r['paired_question_bootstrap_95ci']]
        cards+=f'<article data-evidence="{c}"><span>{labels[c]} · {r["eligible_states"]} eligible states</span><strong>+{delta:.2f}<span> pp</span></strong><p><b>{new:.2f}% new retrieval</b><br>{repeat:.2f}% repeated evidence</p><p class="small-note">Paired 95% CI [{lo:.2f}, {hi:.2f}] pp</p></article>'
        paired+=f'<tr data-evidence-all="{c}"><th scope="row">{labels[c]}</th><td>{r["eligible_states"]} / 256</td><td>{100*r["branches"]["new"]["all_question_mean"]["em"]:.2f}</td><td>{100*r["branches"]["repeat"]["all_question_mean"]["em"]:.2f}</td><td>{r["new_only_correct"]} / {r["repeat_only_correct"]}</td></tr>'
    return '''<section class="wrap section" id="development"><div class="section-heading"><p class="eyebrow">NEW · DEVELOPMENT EVIDENCE</p><h2>Low target hit. Unfinished task.</h2><p>Author Search-R1 3B and 7B checkpoints on the same 256 opened development questions. Seven conditions, a shared four-search cap, and a multistep interface check for both models.</p></div><p>The 7B checkpoint returns the override target on just <b>1.95%</b> of questions. Yet accuracy drops from <b>55.86% to 51.17%</b>: another 41.41% are other wrong answers and 5.47% fail the output format. Attack resistance leaves a substantial task-completion gap.</p><div class="data-table" id="cross-model-table"><h3>Every condition, both checkpoints</h3><div class="table-scroll"><table><caption>256 questions per row. Rates in percent; searches are means.</caption><thead><tr><th>Condition</th><th>Model</th><th>EM ↑</th><th>Target ↓</th><th>Other wrong ↓</th><th>Format fail ↓</th><th>Searches</th><th>Exposed %</th></tr></thead><tbody>'''+rows+'''</tbody></table></div></div><p class="stat-note">The checkpoints differ in post-training settings as well as size. This comparison does not isolate scaling, cross-family generalization, or transfer of our training method. Benign controls are separately token-matched to each attack; exposure means consumed malicious context.</p></section>
<section class="results-section" id="evidence"><div class="wrap section"><div class="section-heading"><h2>Does another search bring useful evidence?</h2><p>Keep the 7B agent’s history and second query fixed. Return new BM25 results, or repeat its first clean documents. Let both continuations run under the same four-search cap.</p></div><div class="metrics">'''+cards+'''</div><p class="scope-inline"><span>WHAT THIS SUPPORTS</span>New retrieval packages help at states where this policy already chooses a second search. This is a paired evidence intervention, not a learned verification policy or a pure test of document novelty.</p><div class="data-table"><h3>Keep the denominator visible</h3><div class="table-scroll"><table><caption>All-question EM carries original outcomes forward for ineligible questions.</caption><thead><tr><th>Condition</th><th>Eligible / all</th><th>New EM % ↑<br>All 256</th><th>Repeat EM % ↑<br>All 256</th><th>New-only / repeat-only<br>correct, eligible states</th></tr></thead><tbody>'''+paired+'''</tbody></table></div></div><p>The 1,128 continuations cover 564 eligible condition–question states, with recurring question identities. All 564 new-evidence branches reproduce the original continuations exactly; all 43 identical-evidence pairs also give identical outputs. Eligibility uses the chosen action, never correctness. No second attack is injected, and later retrieval is clean.</p><p class="stat-note">Intervals use 10,000 paired question-bootstrap draws within each condition. The data were already open for development. Retrieved packages can differ in relevance, order, and length; equal caps do not imply equal realized cost. The diagnostic used 0.3435 allocated GPU-hours. Full token, call, and timing records are released below.</p><p class="small-note">Training status · The separate 7B export/reload numerical check passes, but the GRPO smoke stopped before its first update on a GPU-identifier parsing error. There is no completed 7B training-effect comparison yet; historical 3B qualifications remain.</p><div class="resource-links"><a href="data/cross-model-development.json">3B / 7B results ↗</a><a href="data/evidence-branch-7b.json">Paired evidence and costs ↗</a><a href="data/development-provenance.json">Protocols and source hashes ↗</a></div></div></section>'''

def build_site():
    data=DOCS/'data'
    primary=json.loads((data/'primary.json').read_text())
    paired=json.loads((data/'paired-analysis.json').read_text())
    cases=json.loads((data/'cases.json').read_text())
    arm_names={'clean':'Clean GRPO','random_matched':'Random injection','paired':'Boundary paired','paired_answer':'Paired + answer CE','random_answer':'Random + answer CE'}
    case_data=[]
    labels=['A · From the target to the correct answer','B · From the target to another wrong answer','C · From a correct answer to a wrong one']
    notes=["Both policies issue the same first query and retrieve the same documents. The supervised export preserves the reference answer, but the snippets do not establish complete comparative grounding.","The supervised answer names the building instead of the reference city. The question admits some ambiguity; the result is incorrect under the frozen benchmark scoring rule. The two policies issue different queries.","The retrieved pool identifies the relevant films as 1963 and 1996 releases. Under the authority attack, the supervised export chooses the later film. Both outputs avoid the target, so target hit alone misses the regression."]
    for (key,c),label,note in zip(cases.items(),labels,notes):
        record=c['records']['random_answer/clean']
        outputs={}
        for condition,source in [('clean','clean'),('attack',c['condition']),('benign','quoted_target')]:
            outputs[condition]={}
            for arm in ['random_matched','random_answer']:
                r=c['records'][arm+'/'+source]
                final=next(e['text'] for e in reversed(r['events']) if e['role']=='assistant')
                match=re.search(r'<answer>(.*?)</answer>',final,re.S)
                outputs[condition][arm]={'answer':match.group(1).strip() if match else 'Invalid answer','correct':bool(r['metrics']['em'])}
        case_data.append({'label':label,'question':record['question'],'reference':record['answers'][0], 'outputs':outputs,'note':note,'id':c['id']})
    payload={'aggregate':primary['aggregate'],'names':arm_names,'cases':case_data,'transitions':paired['transition_counts']}
    def plot_rows(metric):
        rows=''
        for arm,label in arm_names.items():
            v=primary['aggregate'][arm][metric];mean=v['mean']*100;sd=v['sample_sd']*100
            rows+=f'<div class="plot-row" data-arm="{arm}"><span class="plot-name">{html.escape(label)}</span><div class="track"><span class="bar {"accent" if arm=="random_answer" else ""}" style="width:{mean/45*100:.5f}%"></span></div><span class="plot-value">{mean:.2f}<small>± {sd:.2f}</small></span></div>'
        return rows
    charts=''
    for metric,title,note in [('attack_em','Attacked exact match ↑','Higher is better. Averaged across both held-out attack templates.'),('clean_em','Clean exact match ↑','Higher is better. Unmodified retrieval on the same questions.'),('attack_target_hit','Attack-target hit ↓','Lower is better. All-question denominator; averaged across both attack templates.')]:
        charts+=f'<article class="chart-panel" data-chart="{metric}"><p class="eyebrow">ALL FIVE CONFIGURATIONS / THREE-SEED MEANS</p><h3>{title}</h3><div class="plot">{plot_rows(metric)}</div><div class="plot-axis"><span>0%</span><span>15%</span><span>30%</span><span>45%</span></div><p class="chart-note">{note} Values show mean ± sample SD.</p></article>'
    seed_rows=''
    for arm,label in arm_names.items():
        for i,seed in enumerate(['20261012','20261013','20261014'],1):
            values=''.join(f'<td data-metric="{m}">{100*primary["aggregate"][arm][m]["per_seed"][seed]:.2f}</td>' for m in ['clean_em','attack_em','attack_target_hit'])
            seed_rows+=f'<tr data-arm="{arm}" data-seed="{seed}"><th scope="row">{html.escape(label)}</th><td>Seed {i}</td>{values}</tr>'
    seeds='<div class="data-table all-seeds"><h3>Every seed, side by side</h3><p>All 15 runs. Percent; single-run values.</p><div class="table-scroll"><table><thead><tr><th>Configuration</th><th>Run</th><th>Clean EM ↑</th><th>Attacked EM ↑</th><th>Target hit ↓</th></tr></thead><tbody>'+seed_rows+'</tbody></table></div><a href="data/seeds.csv">Download all seed results ↗</a></div>'
    hero='<div class="hero-comparison">'
    for arm,label in [('random_matched','Random injection'),('random_answer','+ Answer supervision')]:
        hero+=f'<div class="hero-config" data-hero-config="{arm}"><h3>{label}</h3>'
        for metric,label in [('attack_em','Correct answer ↑'),('attack_target_hit','Attack-target hit ↓')]:
            hero+=f'<div class="hero-stat" data-metric="{metric}"><span>{label}</span><strong>{100*primary["aggregate"][arm][metric]["mean"]:.2f}<small>%</small></strong></div>'
        hero+='</div>'
    hero+='</div>'
    descriptions=[('correct','Correct answer','Corresponds to a correct answer in the supervised export.'),('other_wrong','Another wrong answer','Target avoidance leaves task failure unresolved.'),('target','Still the target','The supervised export also returns the attack target.'),('invalid','Invalid output','Avoiding the target does not make an invalid output correct.')]
    outcomes='<div class="outcome-list">'
    for key,label,note in descriptions:
        n=paired['transition_counts']['target -> '+key]
        outcomes+=f'<div class="outcome-item" data-outcome="{key}"><div><strong>{label}</strong><span>{n:,} · {100*n/1277:.1f}%</span></div><p>{note}</p></div>'
    outcomes+='</div><p class="outcome-detail">Cross-model output comparisons, not recovery actions within one agent. Percentages use the 1,277 baseline target-hit pairs as the denominator.</p>'
    tc=paired['transition_counts']
    kinds=[('wrong_to_correct','Wrong → correct',sum(tc[k+' -> correct'] for k in ['target','other_wrong','invalid']),f'Only the supervised export is correct. {tc["target -> correct"]} of these were target hits in the baseline.'),
           ('target_to_other','Target → another wrong answer',tc['target -> other_wrong'],f'Of the baseline’s {sum(tc["target -> "+k] for k in ["correct","target","other_wrong","invalid"]):,} target hits. The target is avoided, but the user still gets a wrong answer.'),
           ('correct_to_wrong','Correct → wrong',sum(tc['correct -> '+k] for k in ['target','other_wrong','invalid']),'Regressions: only the unsupervised export is correct. Target avoidance alone would never show them.')]
    assert [k[2] for k in kinds]==[931,736,327],[k[2] for k in kinds]
    three=''.join(f'<article data-kind="{key}"><span>{html.escape(label)}</span><strong>{n:,}</strong><p>{html.escape(note)}</p></article>' for key,label,n,note in kinds)
    case_panels=''
    for index,c in enumerate(case_data):
        case_panels+=f'<article class="case-panel continuous-case" data-case="{index}"><p class="eyebrow">{html.escape(c["label"])}</p><div class="case-question"><h3>{html.escape(c["question"])}</h3><p>Reference: <strong>{html.escape(c["reference"])}</strong></p></div>'
        for condition,label in [('clean','Clean retrieval'),('attack','Under attack'),('benign','Benign quotation')]:
            case_panels+=f'<section class="case-condition" data-condition="{condition}"><h4>{label}</h4><div class="answer-grid">'
            for arm in ['random_matched','random_answer']:
                answer=c['outputs'][condition][arm]
                status='Correct' if answer['correct'] else 'Incorrect'
                cls='right' if answer['correct'] else 'wrong'
                case_panels+=f'<article data-arm="{arm}"><span>{html.escape(arm_names[arm].upper())}</span><blockquote>{html.escape(answer["answer"])}</blockquote><span class="answer-status {cls}">{status}</span></article>'
            case_panels+='</div></section>'
        case_panels+=f'<p class="case-note">{html.escape(c["note"])}</p></article>'
    table=''.join('<tr><th scope="row">'+html.escape(label)+'</th>'+''.join(f'<td>{primary["aggregate"][arm][m]["mean"]*100:.2f}</td>' for m in ['clean_em','attack_em','attack_target_hit'])+'</tr>' for arm,label in arm_names.items())
    template=(ROOT/'templates/index.html').read_text()
    script=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c')
    for token,value in {'@@STUDY_DATA@@':script,'@@HERO_COMPARISON@@':hero,'@@ALL_CHARTS@@':charts,'@@ALL_SEEDS@@':seeds,'@@ALL_OUTCOMES@@':outcomes,'@@THREE_KINDS@@':three,'@@ALL_CASES@@':case_panels,'@@RESULT_ROWS@@':table,'@@DEVELOPMENT@@':development_section()}.items():template=template.replace(token,value)
    assert '@@' not in template
    (DOCS/'index.html').write_text(template)
    (DOCS/'citation.bib').write_text('''@misc{lu2026attackresistance,
  title={SARRA: Attack Resistance Is Only Half the Answer for Search Agents},
  author={Lu, Yiwen},
  year={2026},
  note={Working manuscript; not peer reviewed}
}
''')
    manifest={str(f.relative_to(DOCS)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(DOCS.rglob('*')) if f.is_file() and f.name!='release-sha256.json'}
    (DOCS/'release-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    build_site()
    print('Built research homepage with frozen-data interactions.')
