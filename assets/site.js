'use strict';
const menu=document.querySelector('.menu-toggle');
const nav=document.querySelector('#site-navigation');
menu.addEventListener('click',()=>{const expanded=menu.getAttribute('aria-expanded')!=='true';menu.setAttribute('aria-expanded',expanded);nav.classList.toggle('open',expanded);});
nav.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>{nav.classList.remove('open');menu.setAttribute('aria-expanded','false');}));
document.addEventListener('keydown',e=>{if(e.key==='Escape'){nav.classList.remove('open');menu.setAttribute('aria-expanded','false');}});
const byId=id=>document.getElementById(id);
const summaries={
 balanced:{period:'397 shared dates, 2025–Oct 2026',reference:'Balanced log-ridge vs. HAR-DRD',headline:'Lower forecast losses<br>and observed portfolio risk.',description:'Loss reductions are similar in direction and approximate size to the paper. Annualized long-only risk falls from 9.722% to 9.511%, before costs. Its paired interval includes zero.',link:'#results',linkText:'Read the current results',caveat:'Retrospective selection; the risk advantage remains uncertain.',rows:[['Euclidean loss','−2.68%',89],['Frobenius loss','−2.94%',98],['QLIKE loss','−1.19%',40]]},
 replication:{period:'623 one-session forecasts, 2022–2024',reference:'GHAR(GL,L) vs. HAR-DRD',headline:'Short-horizon gains.<br>A different long-run picture.',description:'Euclidean loss falls at 1 and 5 sessions. At 22 sessions, Frobenius loss rises. The full model ranking and portfolio advantage did not replicate.',link:'#replication',linkText:'Read the replication',caveat:'Forecast accuracy and portfolio risk are separate outcomes.',rows:[['1-session, Euclidean','−1.65%',55],['5-session, Euclidean','−2.60%',87],['22-session, Frobenius','+0.45%',15,true]]},
 spectral:{period:'443 shared forecast dates, 2025–Oct 2026',reference:'Geometric Spectral Gate vs. HAR-DRD',headline:'Lower matrix loss.<br>Risk evidence stays uncertain.',description:'Frobenius loss falls by 1.79%. Annualized long-only risk moves from 9.435% to 9.406%, before costs. Its paired risk interval crosses zero.',link:'#earlier-spectral-results',linkText:'Read the earlier spectral results',caveat:'The predefined joint accuracy-and-risk criterion is not passed.',rows:[['Euclidean loss','−1.61%',54],['Frobenius loss','−1.79%',60],['QLIKE loss','−0.39%',13]]}
};
const tabs=[...document.querySelectorAll('[data-benchmark]')];
function selectBenchmark(button){
 tabs.forEach(t=>{const on=t===button;t.setAttribute('aria-selected',on);t.tabIndex=on?0:-1;});
 const s=summaries[button.dataset.benchmark];
 byId('benchmark-content').setAttribute('aria-labelledby',button.id);
 for(const key of ['period','reference','description','caveat'])byId(`benchmark-${key}`).textContent=s[key];
 byId('benchmark-headline').innerHTML=s.headline;
 byId('benchmark-link').href=s.link;byId('benchmark-link').textContent=s.linkText;
 byId('loss-chart').innerHTML='<div class="chart-key">Change in forecast loss, lower is better</div>'+s.rows.map(([name,value,width,adverse])=>`<div class="chart-row"><span>${name}</span><div class="bar-track"><div class="bar${adverse?' adverse':''}" style="width:${width}%"></div></div><strong>${value}</strong></div>`).join('');
}
tabs.forEach((t,i)=>{t.addEventListener('click',()=>selectBenchmark(t));t.addEventListener('keydown',e=>{if(['ArrowRight','ArrowLeft','Home','End'].includes(e.key)){e.preventDefault();const j=e.key==='Home'?0:e.key==='End'?tabs.length-1:(i+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length;selectBenchmark(tabs[j]);tabs[j].focus();}});});
const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting)nav.querySelectorAll('a').forEach(a=>a.classList.toggle('active',a.hash==='#'+e.target.id));});},{rootMargin:'-10% 0px -65% 0px'});
document.querySelectorAll('main>section[id]').forEach(s=>observer.observe(s));
function renderMath(){if(typeof renderMathInElement==='function')renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false}],throwOnError:false});}
if(document.readyState==='complete')renderMath();else window.addEventListener('load',renderMath);
