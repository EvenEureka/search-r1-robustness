"""Check the continuous research page and its frozen evidence in Chromium."""
from pathlib import Path
import argparse,json
from urllib.parse import urlparse,unquote
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8765/');parser.add_argument('--output',default=str(ROOT.parent/'website_review'));args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);checks=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,args=['--no-sandbox','--disable-dev-shm-usage'])
 context=browser.new_context(viewport={'width':1440,'height':1000},permissions=['clipboard-read','clipboard-write'])
 page=context.new_page();errors=[];requests=[];page.on('pageerror',lambda e:errors.append(str(e)));page.on('request',lambda r:requests.append(r.url))
 page.goto(args.url,wait_until='networkidle');study=json.loads(page.locator('#study-data').text_content())
 for arm in ['random_matched','random_answer']:
  for metric in ['attack_em','attack_target_hit']:
   node=page.locator(f'[data-hero-config="{arm}"] [data-metric="{metric}"] strong')
   assert node.is_visible() and node.inner_text()==f'{100*study["aggregate"][arm][metric]["mean"]:.2f}%'
 checks.append('Both hero configurations are visible together and match saved means.')
 for metric in ['attack_em','clean_em','attack_target_hit']:
  panel=page.locator(f'[data-chart="{metric}"]');assert panel.is_visible()
  for arm in study['names']:
   node=panel.locator(f'[data-arm="{arm}"] .plot-value')
   assert node.evaluate('(e)=>e.firstChild.textContent')==f'{100*study["aggregate"][arm][metric]["mean"]:.2f}'
   for seed in ['20261012','20261013','20261014']:
    node=page.locator(f'.all-seeds [data-arm="{arm}"][data-seed="{seed}"] [data-metric="{metric}"]')
    assert node.inner_text()==f'{100*study["aggregate"][arm][metric]["per_seed"][seed]:.2f}'
 checks.append('All 15 chart means and 45 individual-seed values match frozen evidence.')
 for index,c in enumerate(study['cases']):
  for condition in ['clean','attack','benign']:
   for arm in ['random_matched','random_answer']:
    node=page.locator(f'[data-case="{index}"] [data-condition="{condition}"] [data-arm="{arm}"]')
    assert node.is_visible()
    assert node.locator('blockquote').inner_text()==c['outputs'][condition][arm]['answer']
    assert node.locator('.answer-status').inner_text()==('Correct' if c['outputs'][condition][arm]['correct'] else 'Incorrect')
 checks.append('All 18 case answers and scoring labels are visible without clicking.')
 for key in ['correct','other_wrong','target','invalid']:
  n=study['transitions']['target -> '+key];node=page.locator(f'[data-outcome="{key}"]')
  assert node.is_visible() and f'{n:,} · {100*n/1277:.1f}%' in node.inner_text()
 checks.append('All four paired outcomes and their denominators are visible together.')
 tc=study['transitions']
 kinds={'wrong_to_correct':sum(tc[k+' -> correct'] for k in ['target','other_wrong','invalid']),'target_to_other':tc['target -> other_wrong'],'correct_to_wrong':sum(tc['correct -> '+k] for k in ['target','other_wrong','invalid'])}
 for key,n in kinds.items():
  node=page.locator(f'[data-kind="{key}"]');assert node.is_visible() and f'{n:,}' in node.inner_text(),key
 checks.append('The three kinds of paired change (wrong to correct, target to another wrong answer, correct to wrong) match the released transition counts.')
 cross=json.loads((ROOT/'docs/data/cross-model-development.json').read_text())
 branch=json.loads((ROOT/'docs/data/evidence-branch-7b.json').read_text())
 for condition,models in cross['conditions'].items():
  for model,r in models.items():
   cells=page.locator(f'#cross-model-table [data-condition="{condition}"][data-model="{model}"] td').all_text_contents()
   expected=[model.split('_')[-1].upper()]+[f'{100*r["rates"][m]:.2f}' for m in ['em','target_hit','other_wrong','format_failure']]+[f'{r["rates"]["searches"]:.3f}',f'{100*r["rates"]["actual_exposure"]:.0f}']
   assert cells==expected,(condition,model,cells,expected)
 for condition,r in branch['conditions'].items():
  card=page.locator(f'[data-evidence="{condition}"]');assert card.is_visible()
  for value in [f'{100*r["new_minus_repeat_em"]:.2f}',f'{100*r["branches"]["new"]["eligible_mean"]["em"]:.2f}',f'{100*r["branches"]["repeat"]["eligible_mean"]["em"]:.2f}']:
   assert value in card.inner_text()
  cells=page.locator(f'[data-evidence-all="{condition}"] td').all_text_contents()
  assert cells==[f'{r["eligible_states"]} / 256',f'{100*r["branches"]["new"]["all_question_mean"]["em"]:.2f}',f'{100*r["branches"]["repeat"]["all_question_mean"]["em"]:.2f}',f'{r["new_only_correct"]} / {r["repeat_only_correct"]}']
 checks.append('All 14 cross-checkpoint rows and three paired evidence contrasts match released records, including conditional and all-question denominators.')

 assert page.locator('details, [role="tab"], [role="tabpanel"], [hidden]').count()==0
 checks.append('No collapsed sections, tabs, or hidden evidence panels.')
 for href in page.locator('a[href]').evaluate_all('(xs)=>xs.map(x=>x.getAttribute("href"))'):
  if href.startswith('#'):assert page.locator(href).count(),href
  elif not urlparse(href).scheme:assert (ROOT/'docs'/unquote(href.split('#')[0])).is_file(),href
 checks.append('All local links and section anchors resolve.')
 page.locator('#copy-citation').click();assert 'Lu, Yiwen' in page.evaluate('navigator.clipboard.readText()')
 checks.append('Citation clipboard action works.')
 for width,height,label in [(1440,1250,'desktop'),(768,1024,'tablet'),(390,1200,'mobile'),(320,740,'small-mobile')]:
  page.set_viewport_size({'width':width,'height':height});page.evaluate('scrollTo({top:0,behavior:"instant"})');page.wait_for_timeout(100)
  assert page.evaluate('document.documentElement.scrollWidth<=innerWidth'),label
  page.screenshot(path=str(out/(label+'-release.png')))
  if label in ['desktop','mobile']:page.screenshot(path=str(out/(label+'-full.png')),full_page=True)
  page.locator('[data-chart="clean_em"]').scroll_into_view_if_needed();assert page.locator('[data-chart="clean_em"]').is_visible()
  page.screenshot(path=str(out/(label+'-results.png')))
 checks.append('Continuous scroll and layout pass at 1440, 768, 390 and 320 px.')
 page.set_viewport_size({'width':1440,'height':1100});page.evaluate('scrollTo({top:0,behavior:"instant"})')
 page.locator('#theme-toggle').click();assert page.locator('html').get_attribute('data-theme')=='dark'
 page.reload(wait_until='networkidle');assert page.locator('html').get_attribute('data-theme')=='dark'
 page.screenshot(path=str(out/'desktop-dark.png'))
 page.locator('#theme-toggle').click();assert page.locator('html').get_attribute('data-theme')=='light'
 checks.append('Dark/light theme switching persists across reloads; light mode is restored.')
 page.emulate_media(reduced_motion='reduce');assert page.evaluate('getComputedStyle(document.documentElement).scrollBehavior')=='auto'
 page.goto(args.url,wait_until='networkidle');page.keyboard.press('Tab');assert page.evaluate('document.activeElement.className')=='skip-link'
 assert not errors,errors
 assert all(urlparse(u).hostname in ['127.0.0.1','localhost',urlparse(args.url).hostname] for u in requests)
 checks.append('Keyboard entry, reduced motion, no browser errors and no external asset requests pass.')
 context.close()
 nojs=browser.new_context(java_script_enabled=False,viewport={'width':390,'height':844});pg=nojs.new_page();pg.goto(args.url)
 assert pg.locator('[data-chart]').count()==3 and pg.locator('.case-condition blockquote').count()==18
 for node in pg.locator('[data-chart], .case-condition, .scope-details, .all-seeds').all():assert node.is_visible()
 checks.append('Every chart, case condition, seed table and scope section is available without JavaScript.')
 nojs.close();browser.close()
report={'status':'PASS','checks':checks,'browser':'Chromium via Playwright','new_experiments':False,'publication_verified':False}
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
