(() => {
'use strict';
const steps=[...document.querySelectorAll('.pipeline-step')];
const expand=document.getElementById('expand-pipeline');
expand.addEventListener('click',()=>{const open=!steps.every(s=>s.open);steps.forEach(s=>s.open=open);syncExpand();});
function syncExpand(){const all=steps.every(s=>s.open);expand.textContent=all?'Collapse all steps':'Expand all steps';expand.setAttribute('aria-expanded',String(all));}
steps.forEach(s=>s.addEventListener('toggle',syncExpand));
document.querySelectorAll('.table-wrap').forEach(t=>{const hint=document.createElement('p');hint.className='table-hint';hint.textContent='Scroll the table horizontally to compare all metrics. Model names stay visible.';t.before(hint);});
const progress=document.querySelector('.reading-progress span');
function updateReading(){const max=document.documentElement.scrollHeight-innerHeight;progress.style.width=`${max>0?scrollY/max*100:0}%`;}
window.addEventListener('scroll',updateReading,{passive:true});window.addEventListener('resize',updateReading);updateReading();
// Reuse the explainer's six-stock toy construction in the opening mechanism.
const names=['A','B','C','D','E','F'],values=[.9,.25,.4,.2,.15,.3];
const xy=[[80,95],[255,56],[220,180],[70,282],[425,122],[280,308]];
const edges=[[0,1],[0,2],[0,3],[1,2],[2,4],[3,5],[4,5],[1,4]];
const svg=document.getElementById('hero-graph');svg.setAttribute('role','group');
const ns='http://www.w3.org/2000/svg';
function el(tag,attrs,parent=svg){const e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));parent.append(e);return e;}
let selected=0;
const lines=edges.map(([a,b])=>el('line',{x1:xy[a][0],y1:xy[a][1],x2:xy[b][0],y2:xy[b][1],class:'hero-edge'}));
const packets=edges.map(()=>el('circle',{r:3,class:'hero-flow'}));
const nodes=names.map((name,i)=>{const g=el('g',{class:'hero-stock',transform:`translate(${xy[i][0]},${xy[i][1]})`,tabindex:0,role:'button','aria-label':`Inspect illustrative stock ${name}`,'aria-pressed':i===0});el('circle',{r:35,class:'hit-target'},g);const circle=el('circle',{r:25,class:'node-disc',fill:'#faf7f2',stroke:'#b9a997'},g);el('text',{y:7,class:'stock-name'},g).textContent=name;el('text',{y:48,class:'stock-value'},g).textContent=values[i].toFixed(2);function pick(){selected=i;update();}g.addEventListener('click',pick);g.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();pick();}});return {g,circle};});
const matrix=el('g',{});el('rect',{x:358,y:224,width:170,height:148,rx:12,class:'hero-matrix'},matrix);el('text',{x:374,y:248,class:'hero-matrix-label'},matrix).textContent='Covariance';
const features=names.map((_,i)=>names.map((_,j)=>i===j?1:edges.some(e=>e.includes(i)&&e.includes(j))?.12:0));
const gram=features.map(a=>features.map(b=>a.reduce((sum,x,k)=>sum+x*b[k],0)));
for(let i=0;i<6;i++)for(let j=0;j<6;j++){const v=gram[i][j]/Math.sqrt(gram[i][i]*gram[j][j])*Math.sqrt(values[i]*values[j]);el('rect',{x:374+j*22,y:260+i*16,width:19,height:13,rx:2,fill:`hsl(${i===j?155:30} 22% ${92-v*53}%)`},matrix);}
const readout=el('text',{x:20,y:370,class:'hero-aggregation'});
function update(){nodes.forEach(({g,circle},i)=>{circle.setAttribute('fill',i===selected?'#c5dece':'#faf7f2');circle.setAttribute('stroke',i===selected?'#347259':'#b9a997');g.setAttribute('aria-pressed',String(i===selected));});lines.forEach((l,i)=>{const focused=edges[i].includes(selected);l.style.stroke=focused?'#53846b':'#c6b9ac';l.style.opacity=focused?1:.6;});const peers=edges.filter(e=>e.includes(selected)).map(e=>e[0]===selected?e[1]:e[0]);readout.textContent=`Context(${names[selected]}) = ${peers.reduce((sum,i)=>sum+values[i],0)/peers.length===0?'0':(peers.reduce((sum,i)=>sum+values[i],0)/peers.length).toFixed(3)}`;}
const media=matchMedia('(prefers-reduced-motion: reduce)'),pause=document.getElementById('hero-pause');
let paused=media.matches,visible=true,frame=0,phase=0,last=0;
function syncPause(){pause.setAttribute('aria-pressed',String(paused));pause.setAttribute('aria-label',paused?'Resume graph animation':'Pause graph animation');pause.innerHTML=paused?'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="m9 5 10 7-10 7Z"/></svg>':'<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M9 5v14M15 5v14"/></svg>';packets.forEach(p=>p.style.display=paused?'none':'');}
function tick(t){frame=0;const delta=last?Math.min((t-last)/1000,.05):0;last=t;phase+=delta*.25;packets.forEach((p,i)=>{const [a,b]=edges[i],f=(phase+i*.14)%1;const toward=b===selected?f:1-f;p.setAttribute('cx',xy[a][0]+(xy[b][0]-xy[a][0])*toward);p.setAttribute('cy',xy[a][1]+(xy[b][1]-xy[a][1])*toward);p.style.opacity=edges[i].includes(selected)?1:0;});if(!paused&&visible&&!document.hidden)frame=requestAnimationFrame(tick);}
function start(){if(!frame&&!paused&&visible&&!document.hidden){last=0;frame=requestAnimationFrame(tick);}}
pause.addEventListener('click',()=>{paused=!paused;syncPause();start();});media.addEventListener('change',()=>{paused=media.matches;syncPause();start();});
new IntersectionObserver(([e])=>{visible=e.isIntersecting;start();}).observe(svg);document.addEventListener('visibilitychange',start);
syncPause();update();start();
})();
