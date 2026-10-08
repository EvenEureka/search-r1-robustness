'use strict';
(() => {
  const themeButton=document.getElementById('theme-toggle');
  const setTheme=(dark)=>{
    document.documentElement.dataset.theme=dark?'dark':'light';
    themeButton.textContent=dark?'Light mode':'Dark mode';
    themeButton.setAttribute('aria-label',dark?'Switch to light theme':'Switch to dark theme');
  };
  try{setTheme(localStorage.getItem('searchr1-theme')==='dark')}catch{setTheme(false)}
  themeButton.addEventListener('click',()=>{
    const dark=document.documentElement.dataset.theme!=='dark';setTheme(dark);
    try{localStorage.setItem('searchr1-theme',dark?'dark':'light')}catch{}
  });
  document.getElementById('copy-citation').addEventListener('click',async e=>{
    const button=e.currentTarget;const bib='@misc{lu2026attackresistance,\n  title={Attack Resistance Is Only Half the Answer: Evaluating Search-Agent Reliability},\n  author={Lu, Yiwen},\n  year={2026},\n  note={Working manuscript; not peer reviewed}\n}';
    try{await navigator.clipboard.writeText(bib);button.textContent='Copied ✓';document.getElementById('copy-status').textContent='BibTeX copied to clipboard.';setTimeout(()=>button.textContent='Copy BibTeX',2200)}
    catch{document.getElementById('copy-status').textContent='Clipboard unavailable. Use Download .bib.';button.textContent='Use Download .bib'}
  });
})();
