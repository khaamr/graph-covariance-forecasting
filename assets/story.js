(() => {
'use strict';
const $=id=>document.getElementById(id),ns='http://www.w3.org/2000/svg';
const svg=$('graph'),media=matchMedia('(prefers-reduced-motion: reduce)');
const names=['A','B','C','D','E','F'],initial=[.9,.25,.4,.2,.15,.3];
const positions=[[145,88],[405,75],[305,205],[125,310],[490,260],[345,338]];
const sparse=[[0,1],[0,2],[0,3],[1,2],[2,4],[3,5],[4,5],[1,4]];
const pairs=[];for(let a=0;a<6;a++)for(let b=a+1;b<6;b++)pairs.push([a,b]);
const pairEdges=[];pairs.forEach((a,i)=>pairs.forEach((b,j)=>{if(j>i&&a.some(x=>b.includes(x)))pairEdges.push([i,j]);}));
const ring=pairs.map((_,i)=>[320+200*Math.cos(-Math.PI/2+i*2*Math.PI/15),215+167*Math.sin(-Math.PI/2+i*2*Math.PI/15)]);
const titles=['Individual stock signals','Connections define context','A weighted aggregation pass','Stock pairs become graph nodes','Repeated pair aggregation','Assemble the covariance matrix'];
let values=[...initial],focus=0,step=0,pairFocus=0,passes=0,topology='sparse',matrixMode='covariance',cell=[0,1];
let paused=media.matches,playing=false,visible=false,frame=0,last=0,phase=0,autoplay=0,morph=1,flight=-1;
const chapters=[...document.querySelectorAll('.chapter')],stepButtons=[...document.querySelectorAll('[data-step]')];
let announcementTimer=0;
function announce(message,delay=0){clearTimeout(announcementTimer);announcementTimer=setTimeout(()=>{const status=$('story-announcement');if(status)status.textContent=message;},delay);}
function announceCurrent(delay=0){announce(`${$('readout-label').textContent}. ${$('readout-value').textContent}`,delay);}
function experiment(index){return index<3?'stock context':index<5?'pair operator':'covariance reconstruction';}
function selectCell(i,j,moveFocus=false){
 cell=[Math.max(0,Math.min(5,i)),Math.max(0,Math.min(5,j))];
 stop();updateMatrix();updateReadout();
 if(moveFocus)matrixCells[cell[0]*6+cell[1]].g.focus();
 announceCurrent();
}
function matrixKey(e,i,j){
 let row=i,column=j;
 if(e.key==='ArrowLeft')column=Math.max(0,j-1);
 else if(e.key==='ArrowRight')column=Math.min(5,j+1);
 else if(e.key==='ArrowUp')row=Math.max(0,i-1);
 else if(e.key==='ArrowDown')row=Math.min(5,i+1);
 else if(e.key==='Home'){column=0;if(e.ctrlKey||e.metaKey)row=0;}
 else if(e.key==='End'){column=5;if(e.ctrlKey||e.metaKey)row=5;}
 else {activate(e,()=>selectCell(i,j));return;}
 e.preventDefault();selectCell(row,column,true);
}
function el(tag,attrs={},parent=svg){const n=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));parent.append(n);return n;}
function text(value,attrs,parent){const n=el('text',attrs,parent);n.textContent=value;return n;}
const stockLayer=el('g',{'data-scene':'stocks'}),stockEdgeLayer=el('g',{},stockLayer),packetLayer=el('g',{},stockLayer);
const allStockLines=pairs.map(([a,b])=>el('line',{x1:positions[a][0],y1:positions[a][1],x2:positions[b][0],y2:positions[b][1],class:'story-edge'},stockEdgeLayer));
const packets=pairs.map(()=>el('circle',{r:4,class:'story-packet'},packetLayer));
const stockNodes=positions.map(([x,y],i)=>{
 const g=el('g',{class:'story-stock',role:'button',tabindex:0,transform:`translate(${x},${y})`,'aria-label':`Select stock ${names[i]}`},stockLayer);
 el('circle',{r:39,class:'story-hit'},g);const disc=el('circle',{r:29,class:'story-disc'},g);
 text(names[i],{y:7,class:'story-node-label'},g);const value=text(values[i].toFixed(2),{y:54,class:'story-node-value'},g);
 const select=()=>{stop();focus=i;update();announceCurrent();};g.addEventListener('click',select);g.addEventListener('keydown',e=>activate(e,select));return {g,disc,value};
});
const pairLayer=el('g',{'data-scene':'pairs'}),pairEdgeLayer=el('g',{},pairLayer);
const pairLines=pairEdges.map(()=>el('line',{class:'story-pair-edge'},pairEdgeLayer));
const pairNodes=pairs.map(([a,b],i)=>{
 const g=el('g',{class:'story-pair',role:'button',tabindex:-1,'aria-label':`Select pair ${names[a]}–${names[b]}`},pairLayer);
 el('circle',{r:25,class:'story-hit'},g);const disc=el('circle',{r:21,class:'story-pair-disc'},g);
 text(`${names[a]}–${names[b]}`,{y:5,class:'story-pair-label'},g);
 const select=()=>{stop();pairFocus=i;update();announceCurrent();};g.addEventListener('click',select);g.addEventListener('keydown',e=>activate(e,select));return {g,disc};
});
const pairLegend=text('',{x:320,y:418,class:'pair-scene-legend'},pairLayer);
const contractLayer=el('g',{'data-scene':'contraction'});
text('Pair feature before and after aggregation',{x:35,y:38,class:'plot-title'},contractLayer);
el('rect',{x:35,y:58,width:13,height:7,fill:'#c6b9ad'},contractLayer);text('Input',{x:55,y:67,class:'plot-label'},contractLayer);
el('rect',{x:120,y:58,width:13,height:7,fill:'#2c7b5c'},contractLayer);text('After k passes',{x:140,y:67,class:'plot-label'},contractLayer);
el('line',{x1:40,y1:330,x2:610,y2:330,class:'plot-axis'},contractLayer);
const meanLine=el('line',{x1:40,x2:610,class:'mean-line'},contractLayer);
const meanLabel=text('Preserved mean = 1/15',{x:605,class:'mean-label'},contractLayer);
const bars=pairs.map(([a,b],i)=>{
 const x=48+i*37;
 const raw=el('rect',{x,y:330,width:10,height:0,rx:2,fill:'#c6b9ad'},contractLayer);
 const smoothed=el('rect',{x:x+12,y:330,width:10,height:0,rx:2,fill:'#2c7b5c',class:'smooth-bar'},contractLayer);
 text(names[a]+names[b],{x:x+11,y:353,class:'bar-label'},contractLayer);return {raw,smoothed};
});
const contractFoot=text('',{x:320,y:397,class:'contract-foot'},contractLayer);
const matrixLayer=el('g',{'data-scene':'matrix'});
const matrixTitle=text('',{x:320,y:30,class:'matrix-scene-title'},matrixLayer);
const matrixCells=[];
for(let i=0;i<6;i++){
 text(names[i],{x:133,y:91+i*49,class:'story-matrix-label'},matrixLayer);
 text(names[i],{x:173+i*49,y:49,class:'story-matrix-label'},matrixLayer);
 for(let j=0;j<6;j++){
  const g=el('g',{class:'story-matrix-cell',role:'button',tabindex:-1,'aria-label':`Inspect matrix cell ${names[i]}, ${names[j]}`},matrixLayer);
  const rect=el('rect',{x:150+j*49,y:66+i*49,width:45,height:45,rx:6},g);
  const value=text('',{x:173+j*49,y:91+i*49,class:'story-cell-value'},g);
  const select=()=>selectCell(i,j);g.addEventListener('click',select);g.addEventListener('keydown',e=>matrixKey(e,i,j));matrixCells.push({g,rect,value,i,j});
 }
}
text('Select a cell, arrow keys move, Home/End cross the row',{x:320,y:396,class:'contract-foot'},matrixLayer);
function activate(e,fn){if(e.key==='Enter'||e.key===' '){e.preventDefault();fn();}}
function stockEdges(){if(topology==='complete')return pairs;if(topology==='empty')return [];if(topology==='sector')return pairs.filter(([a,b])=>Math.floor(a/2)===Math.floor(b/2));return sparse;}
function neighbors(i){return stockEdges().filter(e=>e.includes(i)).map(e=>e[0]===i?e[1]:e[0]);}
function pairNeighbors(i){return pairs.map((p,j)=>j!==i&&p.some(x=>pairs[i].includes(x))?j:-1).filter(j=>j>=0);}
function average(i){const n=neighbors(i);return n.length?n.reduce((sum,j)=>sum+values[j],0)/n.length:0;}
function operator(x){return pairs.map((_,i)=>pairNeighbors(i).reduce((sum,j)=>sum+x[j],0)/8);}
function contraction(k=passes){let x=pairs.map((_,i)=>i===pairFocus?1:0);for(let p=0;p<k;p++)x=operator(x);return x;}
function retained(k=passes){const x=contraction(k),mean=1/15;return Math.sqrt(x.reduce((sum,v)=>sum+(v-mean)**2,0)/(14/15));}
function covariance(){const edges=stockEdges();const f=names.map((_,i)=>names.map((_,j)=>i===j?1:edges.some(e=>e.includes(i)&&e.includes(j))?.12:0));const gram=f.map(a=>f.map(b=>a.reduce((sum,x,k)=>sum+x*b[k],0)));const correlation=gram.map((r,i)=>r.map((v,j)=>v/Math.sqrt(gram[i][i]*gram[j][j])));return {correlation,matrix:correlation.map((r,i)=>r.map((v,j)=>v*Math.sqrt(values[i]*values[j])))};}
function updateMatrix(){
 const {correlation,matrix}=covariance();
 matrixTitle.textContent=matrixMode==='variance'?'D (standard deviations)':matrixMode==='correlation'?'R (correlations)':'Σ = D R D, covariance';
 matrixCells.forEach(({g,rect,value,i,j})=>{
  const v=matrixMode==='variance'?(i===j?Math.sqrt(values[i]):0):matrixMode==='correlation'?correlation[i][j]:matrix[i][j];
  const selected=cell[0]===i&&cell[1]===j;
  rect.setAttribute('fill',`hsl(${i===j?157:266} ${i===j?24:16}% ${96-v*(i===j?44:50)}%)`);
  rect.setAttribute('stroke',selected?'#184f3a':'transparent');rect.setAttribute('stroke-width',2);
  value.textContent=v.toFixed(2);
  g.setAttribute('aria-label',`${names[i]}, ${names[j]}: ${matrixMode==='variance'?'standard deviation':matrixMode==='correlation'?'correlation':'covariance'} ${v.toFixed(3)}`);
  g.setAttribute('aria-pressed',String(selected));g.setAttribute('tabindex',step===5&&selected?'0':'-1');
 });
 const [i,j]=cell;
 if($('story-matrix-row'))$('story-matrix-row').value=i;
 if($('story-matrix-column'))$('story-matrix-column').value=j;
 $('cell-equation').textContent=`Σ${names[i]}${names[j]} = √${values[i].toFixed(2)} × ${correlation[i][j].toFixed(3)} × √${values[j].toFixed(2)} = ${matrix[i][j].toFixed(3)}`;
}
function updateContraction(){const x=contraction(),ratio=retained();bars.forEach(({raw,smoothed},i)=>{const h0=(i===pairFocus?1:0)*230,h=x[i]*230;raw.setAttribute('height',h0);raw.setAttribute('y',330-h0);smoothed.setAttribute('height',h);smoothed.setAttribute('y',330-h);});meanLine.setAttribute('y1',330-230/15);meanLine.setAttribute('y2',330-230/15);meanLabel.setAttribute('y',330-230/15-9);contractFoot.textContent=`k = ${passes}, centered norm ratio ${ratio.toFixed(5)}, spectral bound ${(0.25**passes).toFixed(5)}`;$('retained-value').textContent=(ratio*100).toFixed(passes>2?3:1)+'%';$('bound-value').textContent=(0.25**passes*100).toFixed(passes>2?3:1)+'%';$('passes-output').value=passes;$('story-passes').value=passes;}
function updateContributions(){const peers=neighbors(focus),mean=average(focus);$('contribution-list').innerHTML=peers.length?peers.map(i=>`<div class="contribution-row" data-source="${i}"><span>${names[i]} <small>${values[i].toFixed(2)}</small></span><div><span style="width:${values[i]*100}%"></span></div><span>× 1/${peers.length}</span><strong>${(values[i]/peers.length).toFixed(3)}</strong></div>`).join(''):'<p class="empty-context">No neighbors: graph context is 0. The stock’s own history remains available to the forecasting model.</p>';$('aggregation-formula').textContent=peers.length?`context(${names[focus]}) = (${peers.map(i=>values[i].toFixed(2)).join(' + ')}) / ${peers.length} = ${mean.toFixed(3)}`:`context(${names[focus]}) = 0 (empty neighborhood)`;$('run-pass').disabled=!peers.length;$('topology-summary').textContent=`${stockEdges().length} undirected edges, ${peers.length} neighbors for ${names[focus]}. ${topology==='empty'?'Graph aggregation contributes zero.':'Weights are normalized separately for each receiving stock.'}`;}
function updatePairs(){const [a,b]=pairs[pairFocus],peers=pairNeighbors(pairFocus);$('story-pair').value=pairFocus;$('pair-neighbor-list').innerHTML=[a,b].map(endpoint=>`<div><span class="endpoint-label">Shared ${names[endpoint]}</span><span>${peers.filter(i=>pairs[i].includes(endpoint)).map(i=>pairs[i].map(j=>names[j]).join('–')).join(', ')}</span></div>`).join('');pairLegend.textContent=`${names[a]}–${names[b]}, 4 neighbors through ${names[a]}, 4 through ${names[b]}`;pairNodes.forEach(({g,disc},i)=>{const on=i===pairFocus,linked=peers.includes(i);disc.setAttribute('fill',on?'#d8d1e8':linked?'#f1e9e1':'#faf7f3');disc.setAttribute('stroke',on?'#6d538e':linked?'#a99a8c':'#d3c9c0');g.setAttribute('aria-pressed',String(on));});}
function updateReadout(){
 const peers=neighbors(focus),mean=average(focus),[a,b]=pairs[pairFocus];
 const labels=[`Signal at ${names[focus]}`,`Neighborhood of ${names[focus]}`,`Graph context for ${names[focus]}`,`Pair ${names[a]}–${names[b]}`,`Separate operator experiment, ${passes} passes`,'Separate reconstruction example, selected cell'];
 const [i,j]=cell;const {correlation,matrix}=covariance();
 const descriptions=[
  `${values[focus].toFixed(2)}, scaled past-volatility feature`,
  peers.length?`${peers.map(i=>names[i]).join(', ')}, each receives weight 1/${peers.length}`:'No edges, graph contribution is zero',
  `Σ neighbor contributions = ${mean.toFixed(3)}, own feature = ${values[focus].toFixed(2)}`,
  `8 neighbors, share ${names[a]} or ${names[b]}, disjoint pairs are excluded`,
  `New unit signal at ${names[a]}–${names[b]}; mean ${(1/15).toFixed(4)} preserved, ${(retained()*100).toFixed(3)}% centered variation retained`,
  `${names[i]}, ${names[j]}, ${matrixMode==='variance'?'D = '+(i===j?Math.sqrt(values[i]):0).toFixed(3):matrixMode==='correlation'?'R = '+correlation[i][j].toFixed(3):'Σ = '+matrix[i][j].toFixed(3)}, uses stock signals and topology, not contraction output`
 ];
 $('readout-label').textContent=labels[step];$('readout-value').textContent=descriptions[step];
 $('scene-note').textContent=[
  'Stock-context demonstration: edit a stock signal and inspect its neighbors.',
  'The selected topology determines the stock neighborhood; no market graph is estimated here.',
  'A packet represents a weighted historical feature, not a causal shock.',
  'Pair-operator demonstration: all 15 pairs share endpoints, regardless of sparse stock edges.',
  'Independent input: a new unit pair signal. Edited stock signals are not used in this experiment.',
  'Reconstruction uses a Gram correlation from stock topology and edited stock variances. It does not use pair-pass output.'
 ][step];
 $('story-context').textContent=step===3?'Pair operator, 15 pairs, degree 8':step===4?'Pair operator, unit-signal input, toy bound 0.25ᵏ':step===5?'Reconstruction, edited stock variances + topology Gram correlation':`Stock context, ${stockEdges().length} illustrative edges, ${names[focus]} selected`;
}
function update(){
 flight=-1;$('run-pass').textContent='Run one pass';$('contribution-list').classList.remove('pass-complete');
 $('story-focus').value=focus;$('signal').value=values[focus];$('signal-output').value=values[focus].toFixed(2);$('slider-title').textContent=`Signal at ${names[focus]}`;
 const edges=stockEdges(),peers=neighbors(focus);stockNodes.forEach(({g,disc,value},i)=>{disc.setAttribute('r',25+values[i]*7);disc.setAttribute('fill',i===focus?'#c4dece':'#faf7f2');disc.setAttribute('stroke',i===focus?'#2c7656':'#b9a99a');value.textContent=values[i].toFixed(2);g.setAttribute('aria-pressed',String(i===focus));});
 pairs.forEach(([a,b],i)=>{const exists=edges.some(e=>e[0]===a&&e[1]===b);allStockLines[i].style.opacity=step>0&&exists?(a===focus||b===focus?1:.3):0;allStockLines[i].style.stroke=a===focus||b===focus?'#448366':'#b7a89b';packets[i].style.display=step===2&&exists&&pairs[i].includes(focus)?'':'none';});
 updateContributions();updatePairs();updateContraction();updateMatrix();updateReadout();schedule();
}
function switchStep(index){const previous=step;if(experiment(previous)!==experiment(index)&&playing)stop();step=index;document.querySelector('.story-lab').hidden=step===3||step===4;$('motion').hidden=step!==2;chapters.forEach((ch,i)=>ch.hidden=i!==step);stepButtons.forEach((b,i)=>{if(i===step)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});$('scene-title').textContent=titles[step];$('scene-counter').textContent=`${step+1} / 6`;$('step-position').textContent=`Step ${step+1} of 6`;$('previous-step').disabled=step===0;$('next-step').disabled=step===5;$('progress').style.width=`${step/5*100}%`;
 const layers=[stockLayer,pairLayer,contractLayer,matrixLayer],active=step<=2?0:step===3?1:step===4?2:3;
 layers.forEach((g,i)=>{g.classList.toggle('scene-active',i===active);g.setAttribute('aria-hidden',i!==active);g.style.pointerEvents=i===active?'auto':'none';g.querySelectorAll('[tabindex]').forEach(n=>n.setAttribute('tabindex',i===active?'0':'-1'));});
 if(step===3&&previous<3)morph=media.matches?1:0;
 else morph=1;
 $('story-lab-note')?.remove();update();draw();if(previous!==step)announce(`${titles[step]}. ${$('story-context').textContent}. ${$('scene-note').textContent}`);
}
function stop(){playing=false;autoplay=0;$('play').textContent='Play walkthrough';$('play').setAttribute('aria-pressed','false');}
function goTo(i){stop();switchStep(i);}
function draw(){
 const selected=pairs[pairFocus];const coords=pairs.map(([a,b],i)=>{const mid=[(positions[a][0]+positions[b][0])/2,(positions[a][1]+positions[b][1])/2];const ease=1-(1-morph)**3;return [mid[0]+(ring[i][0]-mid[0])*ease,mid[1]+(ring[i][1]-mid[1])*ease];});
 pairNodes.forEach(({g},i)=>g.setAttribute('transform',`translate(${coords[i][0]},${coords[i][1]})`));
 pairEdges.forEach(([a,b],i)=>{const l=pairLines[i];l.setAttribute('x1',coords[a][0]);l.setAttribute('y1',coords[a][1]);l.setAttribute('x2',coords[b][0]);l.setAttribute('y2',coords[b][1]);const other=a===pairFocus?b:b===pairFocus?a:-1;const shared=other>=0?selected.find(x=>pairs[other].includes(x)):-1;l.style.stroke=shared===selected[0]?'#3b8467':shared===selected[1]?'#9b6a3b':'#c7bbcf';l.style.opacity=other>=0?.8:.12;l.style.strokeWidth=other>=0?1.7:1;});
 pairs.forEach(([a,b],i)=>{const connected=a===focus||b===focus;const t=flight>=0?flight:(phase+i*.13)%1;const from=a===focus?b:a,to=focus;const f=Math.min(1,t);packets[i].setAttribute('cx',positions[from][0]+(positions[to][0]-positions[from][0])*f);packets[i].setAttribute('cy',positions[from][1]+(positions[to][1]-positions[from][1])*f);packets[i].style.opacity=connected&&(flight>=0||!paused)?1:0;});
}
function schedule(){if(!frame)frame=requestAnimationFrame(tick);}
function tick(time){frame=0;const delta=last?Math.min((time-last)/1000,.05):0;last=time;if(!paused)phase+=delta*.45;if(morph<1)morph=Math.min(1,morph+delta/0.7);if(flight>=0){flight+=delta/0.9;if(flight>=1){flight=-1;$('run-pass').textContent='Run one pass';$('contribution-list').classList.add('pass-complete');$('readout-value').textContent=`Received graph context = ${average(focus).toFixed(3)}. Own signal stays ${values[focus].toFixed(2)}.`;announceCurrent();}}if(playing&&visible){autoplay+=delta;if(autoplay>5){autoplay=0;if(step<5)switchStep(step+1);else stop();}}draw();if(visible&&!document.hidden&&(playing||morph<1||flight>=0||(!paused&&step===2)))schedule();}
names.forEach((n,i)=>$('story-focus').add(new Option(n,i)));pairs.forEach(([a,b],i)=>$('story-pair').add(new Option(`${names[a]}–${names[b]}`,i)));
$('story-focus').addEventListener('change',e=>{stop();focus=Number(e.target.value);update();announceCurrent();});$('signal').addEventListener('input',e=>{stop();values[focus]=Number(e.target.value);flight=-1;update();announceCurrent(250);});
$('story-topology').addEventListener('change',e=>{stop();topology=e.target.value;update();announceCurrent();});$('story-pair').addEventListener('change',e=>{stop();pairFocus=Number(e.target.value);update();announceCurrent();});$('story-passes').addEventListener('input',e=>{stop();passes=Number(e.target.value);update();announceCurrent(200);});
if($('story-matrix-row'))$('story-matrix-row').addEventListener('change',e=>selectCell(Number(e.target.value),cell[1]));
if($('story-matrix-column'))$('story-matrix-column').addEventListener('change',e=>selectCell(cell[0],Number(e.target.value)));
document.querySelectorAll('[data-matrix]').forEach(b=>b.addEventListener('click',()=>{stop();matrixMode=b.dataset.matrix;document.querySelectorAll('[data-matrix]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));update();announceCurrent();}));
$('run-pass').addEventListener('click',()=>{stop();flight=media.matches||!visible?-1:0;$('contribution-list').classList.remove('pass-complete');$('run-pass').textContent=media.matches?'Run one pass':'Aggregating…';if(media.matches||!visible){$('run-pass').textContent='Run one pass';$('contribution-list').classList.add('pass-complete');$('readout-value').textContent=`Received graph context = ${average(focus).toFixed(3)}.`;announceCurrent();}schedule();});
stepButtons.forEach(b=>b.addEventListener('click',()=>goTo(Number(b.dataset.step))));$('previous-step').addEventListener('click',()=>goTo(Math.max(0,step-1)));$('next-step').addEventListener('click',()=>goTo(Math.min(5,step+1)));
$('story-reset').addEventListener('click',()=>{stop();values=[...initial];focus=0;pairFocus=0;passes=0;topology='sparse';matrixMode='covariance';cell=[0,1];$('story-topology').value=topology;document.querySelectorAll('[data-matrix]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.matrix===matrixMode)));update();announce('Example reset. Stock signals, pair selection, graph topology, pass count and matrix inspection have returned to their initial values.');});
function syncMotion(){$('motion').textContent=paused?'Resume flow':'Pause flow';$('motion').setAttribute('aria-pressed',String(paused));}
$('motion').addEventListener('click',()=>{paused=!paused;syncMotion();schedule();});$('play').addEventListener('click',()=>{if(playing){stop();return;}if(media.matches){goTo((step+1)%6);return;}if(step===5)switchStep(0);playing=true;autoplay=0;$('play').textContent='Pause walkthrough';$('play').setAttribute('aria-pressed','true');schedule();});
media.addEventListener('change',()=>{paused=media.matches;morph=1;flight=-1;$('run-pass').textContent='Run one pass';stop();syncMotion();draw();});document.addEventListener('visibilitychange',()=>{last=0;if(document.hidden)stop();else schedule();});
new IntersectionObserver(([e])=>{visible=e.isIntersecting;if(!visible)stop();else{last=0;schedule();}}).observe(svg);
window.GraphCovStory={getState:()=>({step,focus,pairFocus,passes,topology,matrixMode,selectedMatrixCell:[...cell],experiment:experiment(step),values:[...values],stockEdges:stockEdges().length,pairCount:pairs.length,pairDegree:pairNeighbors(pairFocus).length,context:average(focus),retained:retained(),bound:.25**passes,covariance:covariance().matrix,correlation:covariance().correlation}),operator};
syncMotion();switchStep(0);
})();
