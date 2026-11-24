
// 1. DATA: 27 stocks with sectors and positions
const STOCKS = [
  // Technology (Cyan: #22d3ee)
  { ticker: 'AAPL', sector: 'Technology', color: '#22d3ee', x: 0.35, y: 0.28 },
  { ticker: 'MSFT', sector: 'Technology', color: '#22d3ee', x: 0.42, y: 0.22 },
  { ticker: 'INTC', sector: 'Technology', color: '#22d3ee', x: 0.30, y: 0.20 },
  { ticker: 'CSCO', sector: 'Technology', color: '#22d3ee', x: 0.26, y: 0.32 },
  { ticker: 'CRM',  sector: 'Technology', color: '#22d3ee', x: 0.48, y: 0.28 },
  { ticker: 'IBM',  sector: 'Technology', color: '#22d3ee', x: 0.32, y: 0.38 },

  // Finance (Blue: #3b82f6)
  { ticker: 'JPM',  sector: 'Finance', color: '#3b82f6', x: 0.65, y: 0.30 },
  { ticker: 'GS',   sector: 'Finance', color: '#3b82f6', x: 0.72, y: 0.24 },
  { ticker: 'AXP',  sector: 'Finance', color: '#3b82f6', x: 0.60, y: 0.38 },
  { ticker: 'TRV',  sector: 'Finance', color: '#3b82f6', x: 0.75, y: 0.36 },

  // Healthcare (Emerald: #10b981)
  { ticker: 'JNJ',  sector: 'Healthcare', color: '#10b981', x: 0.22, y: 0.58 },
  { ticker: 'UNH',  sector: 'Healthcare', color: '#10b981', x: 0.28, y: 0.65 },
  { ticker: 'MRK',  sector: 'Healthcare', color: '#10b981', x: 0.18, y: 0.68 },
  { ticker: 'AMGN', sector: 'Healthcare', color: '#10b981', x: 0.32, y: 0.74 },

  // Industrials (Rose: #f43f5e)
  { ticker: 'BA',   sector: 'Industrials', color: '#f43f5e', x: 0.52, y: 0.52 },
  { ticker: 'CAT',  sector: 'Industrials', color: '#f43f5e', x: 0.58, y: 0.60 },
  { ticker: 'HON',  sector: 'Industrials', color: '#f43f5e', x: 0.48, y: 0.64 },
  { ticker: 'MMM',  sector: 'Industrials', color: '#f43f5e', x: 0.62, y: 0.52 },

  // Discretionary (Violet: #8b5cf6)
  { ticker: 'HD',   sector: 'Discretionary', color: '#8b5cf6', x: 0.70, y: 0.72 },
  { ticker: 'MCD',  sector: 'Discretionary', color: '#8b5cf6', x: 0.78, y: 0.65 },
  { ticker: 'NKE',  sector: 'Discretionary', color: '#8b5cf6', x: 0.72, y: 0.82 },

  // Staples (Sky: #38bdf8)
  { ticker: 'PG',   sector: 'Staples', color: '#38bdf8', x: 0.38, y: 0.82 },
  { ticker: 'KO',   sector: 'Staples', color: '#38bdf8', x: 0.45, y: 0.88 },
  { ticker: 'WMT',  sector: 'Staples', color: '#38bdf8', x: 0.32, y: 0.88 },

  // Communication (Purple: #a855f7)
  { ticker: 'DIS',  sector: 'Communication', color: '#a855f7', x: 0.40, y: 0.45 },
  { ticker: 'VZ',   sector: 'Communication', color: '#a855f7', x: 0.34, y: 0.52 },

  // Energy (Amber: #f59e0b)
  { ticker: 'CVX',  sector: 'Energy', color: '#f59e0b', x: 0.82, y: 0.48 }
];

// Edges connecting stocks (intra-sector + inter-sector GLASSO edges)
const EDGES = [
  // Tech intra
  [0,1],[0,2],[0,3],[0,4],[0,5],[1,2],[1,4],[2,3],[3,5],
  // Finance intra
  [6,7],[6,8],[6,9],[7,8],[8,9],
  // Health intra
  [10,11],[10,12],[10,13],[11,12],[11,13],
  // Industrials intra
  [14,15],[14,16],[14,17],[15,16],[16,17],
  // Discretionary intra
  [18,19],[18,20],[19,20],
  // Staples intra
  [21,22],[21,23],[22,23],
  // Communication intra
  [24,25],
  // Inter-sector GLASSO spillovers
  [0,6],[1,7],[0,24],[6,14],[14,26],[15,26],[10,21],[18,21],[4,6],[0,14],[6,18],[10,24]
];

(() => {
'use strict';
const N=STOCKS.length,PAIRS=[];
for(let i=0;i<N;i++)for(let j=i+1;j<N;j++)PAIRS.push([i,j]);
const M=PAIRS.length,DEGREE=2*(N-2),RHO=(N-4)/DEGREE;
const byId=id=>document.getElementById(id);
const canvas=byId('graph-canvas'),ctx=canvas.getContext('2d'),viewport=byId('graph-viewport');
const reduced=matchMedia('(prefers-reduced-motion: reduce)');
const state={mode:'stock',stock:0,column:0,pair:0,passes:0,injected:false,injectionStock:null,playing:!reduced.matches};
let width=1,height=1,dpr=1,phase=0,last=0,frame=0,inView=true,dirty=true;
let transition=null;
let visual={stock:1,pair:0,matrix:0,lift:0,contract:0};
let spectrum=[],matrix=[];
const stockSelect=byId('stock-select'),pairSelect=byId('network-pair-select'),columnSelect=byId('network-column-select');
STOCKS.forEach((s,i)=>{stockSelect.add(new Option(s.ticker,i));columnSelect.add(new Option(s.ticker,i));});
PAIRS.forEach(([a,b],i)=>pairSelect.add(new Option(`${STOCKS[a].ticker} / ${STOCKS[b].ticker}`,i)));
canvas.tabIndex=0;canvas.setAttribute('role','img');
canvas.setAttribute('aria-label','Interactive illustrative stock network. Use the stock and pair selectors above, or arrow keys, to inspect values.');
const sectors=[...new Set(STOCKS.map(s=>s.sector))];
document.querySelectorAll('.graph-legend .legend-dot').forEach((dot,i)=>{dot.style.background=STOCKS.find(s=>s.sector===sectors[i])?.color||'#87988d';});
const legend=document.querySelector('.graph-legend');
const sectorLegendHTML=legend?legend.innerHTML:'';
function syncLegend(){
 if(!legend)return;
 if(state.mode==='stock'){legend.innerHTML=sectorLegendHTML;return;}
 const entries=state.mode==='line'?[['#f3dfb7','Selected pair'],['#9ccbb3','Shares one endpoint'],['#5c7b86','Other pair nodes']]:state.mode==='contraction'?[['#84d3a8','Positive centered contrast'],['#af9ed2','Negative centered contrast'],['#f3dfb7','Selected pair ring']]:[['#659a7b','Diagonal: individual variance'],['#2f4947','Off-diagonal: covariance'],['#d6b986','Selected row and column']];
 legend.innerHTML=entries.map(([color,label])=>`<span class="legend-item"><span class="legend-dot" style="background:${color}"></span>${label}</span>`).join('')+(state.mode==='contraction'?'<span class="network-legend-note">Color intensity rescales within each pass. Compare the centered-norm readout to measure contraction.</span>':state.mode==='matrix'?'<span class="network-legend-note">Lighter = larger covariance, nonlinear color scale</span>':'');
}

function neighbors(i){return EDGES.filter(e=>e.includes(i)).map(e=>e[0]===i?e[1]:e[0]);}
function pairNeighbors(i){const a=PAIRS[i];return PAIRS.reduce((out,b,j)=>{if(i!==j&&(a[0]===b[0]||a[0]===b[1]||a[1]===b[0]||a[1]===b[1]))out.push(j);return out;},[]);}
function multiply(x){const endpoint=new Float64Array(N);PAIRS.forEach(([a,b],i)=>{endpoint[a]+=x[i];endpoint[b]+=x[i];});return PAIRS.map(([a,b],i)=>(endpoint[a]+endpoint[b]-2*x[i])/DEGREE);}
function normCentered(x){const mean=x.reduce((a,b)=>a+b,0)/x.length;return Math.sqrt(x.reduce((a,b)=>a+(b-mean)**2,0));}
function updateSpectrum(){let x=PAIRS.map((_,i)=>i===state.pair?1:0);spectrum=[x];for(let k=1;k<=6;k++){x=multiply(x);spectrum.push(x);} }
function stockValue(i){return .28+.085*(i%7)+(state.injected && i===state.injectionStock ? .6 : 0);}
function updateMatrix(){
 const maxDegree=Math.max(...STOCKS.map((_,i)=>neighbors(i).length));
 const f=STOCKS.map((_,i)=>STOCKS.map((_,j)=>i===j?1:EDGES.some(e=>e.includes(i)&&e.includes(j))?.12/maxDegree:0));
 const g=f.map(a=>f.map(b=>a.reduce((sum,x,k)=>sum+x*b[k],0)));
 matrix=g.map((row,i)=>row.map((v,j)=>v/Math.sqrt(g[i][i]*g[j][j])*Math.sqrt(stockValue(i)*stockValue(j))));
}
function pairLabel(i){const [a,b]=PAIRS[i];return `${STOCKS[a].ticker}–${STOCKS[b].ticker}`;}
function format(x){return x.toFixed(4);}
function sync(){
 const pairMode=state.mode==='line'||state.mode==='contraction';
 byId('network-stock-control').hidden=pairMode;byId('network-pair-control').hidden=!pairMode;
 byId('network-column-control').hidden=state.mode!=='matrix';byId('network-stock-label').textContent=state.mode==='matrix'?'Matrix row':'Inspect a stock';stockSelect.setAttribute('aria-label',state.mode==='matrix'?'Select matrix row stock':'Select stock');
 byId('network-pass-control').hidden=state.mode!=='contraction';
 byId('network-inject').hidden=state.mode==='line'||state.mode==='contraction';
 byId('network-inject').textContent=state.injected?'Reset example signal':'Add example signal';
 byId('network-pass-output').value=state.passes;byId('network-passes').value=state.passes;
 stockSelect.value=state.stock;pairSelect.value=state.pair;columnSelect.value=state.column;syncLegend();
 byId('btn-toggle-flow').hidden=state.mode==='matrix'||state.mode==='contraction';
 byId('btn-toggle-flow').textContent=state.playing?'Pause dynamics':'Play dynamics';
 byId('btn-toggle-flow').setAttribute('aria-pressed',String(state.playing));
 [['stock','stock'],['line','line'],['contraction','contract'],['matrix','matrix']].forEach(([mode,suffix])=>{const b=byId(`btn-mode-${suffix}`);b.classList.toggle('active',state.mode===mode);b.setAttribute('aria-pressed',String(state.mode===mode));});
 let facts=[],guide='';
 if(state.mode==='stock'){
  const peers=neighbors(state.stock),mean=peers.reduce((sum,i)=>sum+stockValue(i),0)/peers.length;
  facts=[`<span><strong>${STOCKS[state.stock].ticker}</strong>, ${STOCKS[state.stock].sector}</span>`,`<span><strong>${peers.length}</strong> connected stocks</span>`,`<span>Own signal <strong>${stockValue(state.stock).toFixed(3)}</strong></span>`,`<span>Neighbor mean <strong>${mean.toFixed(3)}</strong></span>`];
  guide='Select a stock to inspect the equal-weight mean of its connected neighbors. Moving dots carry historical features; they do not show causal shocks. The sparse edges and signals are illustrative.';
 }else if(state.mode==='line'){
  facts=[`<span>Pair <strong>${pairLabel(state.pair)}</strong></span>`,`<span><strong>351</strong> pair nodes</span>`,`<span><strong>${pairNeighbors(state.pair).length}</strong> neighbors share one endpoint</span>`];
  guide='Every unordered pair is a node, including pairs absent from the illustrated sparse stock graph. The highlighted 50 connections share one endpoint with the selected pair; other links are suppressed for readability.';
 }else if(state.mode==='contraction'){
  const ratio=normCentered(spectrum[state.passes])/normCentered(spectrum[0]),bound=RHO**state.passes;
  facts=[`<span>Pass <strong>${state.passes}</strong>, ${pairLabel(state.pair)}</span>`,`<span class="contract-measure">Centered norm <strong>${format(ratio)}</strong> of initial</span>`,`<span class="contract-measure">Worst-case bound <strong>${format(bound)}</strong></span>`];
  guide='A unit signal starts at the selected pair. Each pass applies the exact row-normalized complete-pair line-graph operator via endpoint sums. The displayed centered norm is computed, while 0.46ᵏ is its spectral upper bound. All signals are illustrative.';
 }else{
  facts=[`<span><strong>27 × 27</strong> symmetric matrix</span>`,`<span><strong>${STOCKS[state.stock].ticker} × ${STOCKS[state.column].ticker}</strong> = ${matrix[state.stock][state.column].toFixed(4)}</span>`,`<span><strong>Positive definite</strong> by Gram construction</span>`];
  guide='This toy covariance is built from a nonsingular graph-feature matrix, normalized to a correlation matrix and rescaled by positive stock variances. It is not a fitted HAR forecast. Choose a row and column, tap a cell, or use arrow keys on the matrix to inspect each covariance.';
 }
 byId('network-status').innerHTML=facts.join('');byId('network-guide').textContent=guide;
 canvas.setAttribute('aria-label',`${state.mode} view. ${byId('network-status').textContent} ${guide}`);
 dirty=true;schedule();
}
function setMode(mode){
 if(!['stock','line','contraction','matrix'].includes(mode))return;
 state.mode=mode;
 const target=mode==='stock'?{stock:1,pair:0,matrix:0,lift:0,contract:0}:mode==='matrix'?{stock:0,pair:0,matrix:1,lift:1,contract:0}:{stock:0,pair:1,matrix:0,lift:1,contract:mode==='contraction'?1:0};
 if(reduced.matches){visual=target;transition=null;}else transition={from:{...visual},to:target,start:performance.now(),duration:650};
 sync();
}
window.setGraphMode=setMode;
window.triggerVolatilityShock=()=>{state.injected=!state.injected;state.injectionStock=state.injected?state.stock:null;updateMatrix();sync();};
window.toggleParticleFlow=()=>{state.playing=!state.playing;sync();};
stockSelect.addEventListener('change',()=>{state.stock=Number(stockSelect.value);hoveredCell=null;updateMatrix();sync();});
columnSelect.addEventListener('change',()=>{state.column=Number(columnSelect.value);hoveredCell=null;sync();});
pairSelect.addEventListener('change',()=>{state.pair=Number(pairSelect.value);updateSpectrum();sync();});
byId('network-passes').addEventListener('input',e=>{state.passes=Number(e.target.value);sync();});
function resize(){const rect=viewport.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);canvas.style.width=width+'px';canvas.style.height=height+'px';ctx.setTransform(dpr,0,0,dpr,0,0);dirty=true;schedule();}
new ResizeObserver(resize).observe(viewport);
function stockPosition(i){const s=STOCKS[i];return [35+(s.x-.1)/.8*(width-70),55+(s.y-.08)/.78*(height-125)];}
function pairPosition(i){const [a,b]=PAIRS[i],pa=stockPosition(a),pb=stockPosition(b);const contract=visual.contract;const radius=Math.sqrt((i+.5)/M)*Math.min(width*.43,height*(.39-.11*contract));const angle=i*2.399963229728653;const target=[width*.5+Math.cos(angle)*radius,height*(.47-.11*contract)+Math.sin(angle)*radius];return [(pa[0]+pb[0])*.5*(1-visual.lift)+target[0]*visual.lift,(pa[1]+pb[1])*.5*(1-visual.lift)+target[1]*visual.lift];}
function circle(x,y,r,fill,stroke){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=fill;ctx.fill();if(stroke){ctx.strokeStyle=stroke;ctx.lineWidth=1;ctx.stroke();}}
function text(value,x,y,size=12,color='#b5c5ca',align='left'){ctx.font=`${size}px Geist,system-ui,sans-serif`;ctx.fillStyle=color;ctx.textAlign=align;ctx.textBaseline='middle';ctx.fillText(value,x,y);}
function drawStocks(){
 ctx.save();ctx.globalAlpha=visual.stock;
 EDGES.forEach(([a,b],i)=>{const p=stockPosition(a),q=stockPosition(b),highlight=a===state.stock||b===state.stock;ctx.beginPath();ctx.moveTo(...p);ctx.lineTo(...q);ctx.strokeStyle=highlight?'#739f8c':'#4f6972';ctx.lineWidth=highlight?1.6:1;ctx.stroke();if(highlight&&state.playing&&!reduced.matches){const t=(phase*.35+i*.17)%1,f=b===state.stock?t:1-t;circle(p[0]+(q[0]-p[0])*f,p[1]+(q[1]-p[1])*f,2.5,'#c1e7d3');}});
 STOCKS.forEach((s,i)=>{const [x,y]=stockPosition(i),selected=i===state.stock;circle(x,y,(selected?12:7)+(state.injected&&i===state.injectionStock?4:0),s.color,selected?'#fff':null);if(selected)circle(x,y,18,'#ffffff00','#a7d6bc');text(s.ticker,x,y-(selected?27:18),selected?13:width<500?9:11,selected?'#f0f7f2':'#c2d1d5','center');});
 text('Sparse stock graph, illustrative edges',20,height-25,12,'#91a9ad');ctx.restore();
}
function drawPairs(){
 ctx.save();ctx.globalAlpha=visual.pair;
 const peers=pairNeighbors(state.pair),peerSet=new Set(peers),selected=pairPosition(state.pair),contract=state.mode==='contraction';
 // Only the inspected neighborhood is drawn; the full graph has 8,775 edges.
 peers.forEach((j,i)=>{const p=pairPosition(j);ctx.beginPath();ctx.moveTo(...selected);ctx.lineTo(...p);ctx.strokeStyle=contract?'#6a8b7c60':'#75b39950';ctx.lineWidth=.8;ctx.stroke();if(state.playing&&!reduced.matches&&!contract){const t=(phase*.3+i*.047)%1;circle(p[0]+(selected[0]-p[0])*t,p[1]+(selected[1]-p[1])*t,1.6,'#d1eada');}});
 const x=spectrum[state.passes],mean=1/M,maxContrast=Math.max(...x.map(v=>Math.abs(v-mean)),1e-12);
 PAIRS.forEach((_,i)=>{const p=pairPosition(i);let color=i===state.pair?'#f3dfb7':peerSet.has(i)?'#9ccbb3':'#5c7b86';if(contract){const z=(x[i]-mean)/maxContrast;color=z>=0?`rgba(132,211,168,${.2+.8*Math.abs(z)})`:`rgba(175,158,210,${.2+.8*Math.abs(z)})`;}circle(...p,i===state.pair?6:peerSet.has(i)?3.2:2.2,color);});
 circle(...selected,10,'#ffffff00','#f3dfb7');
 text(pairLabel(state.pair),selected[0],Math.max(24,selected[1]-20),12,'#fff2d8','center');
 if(contract)drawContractionChart();else{text('351 pair nodes, selected neighborhood only',20,height-27,width<500?10:12,'#a2b9bd');text('50 neighbors',width-20,height-27,12,'#c7dfd1','right');}
 ctx.restore();
}
function drawContractionChart(){
 const x0=width<500?34:54,y0=height-35,w=width-(width<500?64:108),h=70;
 ctx.fillStyle='#111b20ed';ctx.fillRect(x0-14,y0-h-23,w+28,h+42);
 ctx.strokeStyle='#33484b';ctx.beginPath();ctx.moveTo(x0,y0-h);ctx.lineTo(x0,y0);ctx.lineTo(x0+w,y0);ctx.stroke();
 const ratios=spectrum.map(x=>normCentered(x)/normCentered(spectrum[0]));
 for(const [values,color,dash] of [[ratios,'#a2d9b8',[]],[Array.from({length:7},(_,i)=>RHO**i),'#ad9fc4',[4,4]]]){ctx.strokeStyle=color;ctx.lineWidth=1.6;ctx.setLineDash(dash);ctx.beginPath();values.forEach((v,i)=>{const x=x0+i*w/6,y=y0-v*h;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);}
 ratios.forEach((v,i)=>circle(x0+i*w/6,y0-v*h,i===state.passes?4:2,'#a2d9b8'));
 text('0',x0,y0+13,10,'#9eb4b8','center');text('6 passes',x0+w,y0+13,10,'#9eb4b8','right');text('Computed norm',x0+10,y0-h-10,10,'#a2d9b8');text('0.46ᵏ bound',x0+w,y0-h-10,10,'#ad9fc4','right');
}
function matrixGeometry(){const size=Math.min(width-80,height-115),left=(width-size)/2,top=50;return {size,left,top,cell:size/N};}
let hoveredCell=null;
function drawMatrix(){
 ctx.save();ctx.globalAlpha=visual.matrix;const {size,left,top,cell}=matrixGeometry();
 matrix.forEach((row,i)=>row.forEach((v,j)=>{ctx.fillStyle=`hsl(${i===j?155:175} ${i===j?29:15}% ${v>0?20+Math.sqrt(Math.min(1,v))*45:13}%)`;ctx.fillRect(left+j*cell,top+i*cell,Math.max(1,cell-1),Math.max(1,cell-1));}));
 ctx.strokeStyle='#d6b986';ctx.lineWidth=1.2;ctx.strokeRect(left,top+state.stock*cell,size,cell);ctx.strokeRect(left+state.column*cell,top,cell,size);ctx.strokeStyle='#f2ead2';ctx.lineWidth=1.6;ctx.strokeRect(left+state.column*cell,top+state.stock*cell,cell,cell);
 for(let i=0;i<N;i++){const label=STOCKS[i].ticker;if(width>650||i%3===0){text(label,left-8,top+(i+.5)*cell,width<500?8:9,'#b9ccd0','right');ctx.save();ctx.translate(left+(i+.5)*cell,top-9);ctx.rotate(-Math.PI/2);text(label,0,0,width<500?8:9,'#b9ccd0');ctx.restore();}}
 const [i,j]=hoveredCell||[state.stock,state.column];text(`${STOCKS[i].ticker} × ${STOCKS[j].ticker} = ${matrix[i][j].toFixed(4)}`,width/2,height-32,13,'#d5e8df','center');text('Toy covariance, diagonal = variance, off-diagonal = co-movement',width/2,height-13,width<500?8:11,'#92aaad','center');ctx.restore();
}
function draw(){ctx.clearRect(0,0,width,height);ctx.fillStyle='#111b20';ctx.fillRect(0,0,width,height);if(visual.stock>.001)drawStocks();if(visual.pair>.001)drawPairs();if(visual.matrix>.001)drawMatrix();}
function schedule(){if(!frame&&inView&&!document.hidden)frame=requestAnimationFrame(tick);}
function tick(t){frame=0;const dt=last?Math.min((t-last)/1000,.05):0;last=t;if(state.playing&&!reduced.matches)phase+=dt;
 if(transition){const p=Math.min(1,(t-transition.start)/transition.duration),e=1-(1-p)**4;Object.keys(visual).forEach(k=>visual[k]=transition.from[k]+(transition.to[k]-transition.from[k])*e);if(p===1)transition=null;dirty=true;}
 const flowing=state.playing&&!reduced.matches&&(state.mode==='stock'||state.mode==='line');
 if(dirty||flowing){draw();dirty=false;}
 if((transition||flowing)&&inView&&!document.hidden)schedule();
}
function matrixCellAt(x,y){const {left,top,cell}=matrixGeometry();const j=Math.floor((x-left)/cell),i=Math.floor((y-top)/cell);return i>=0&&i<N&&j>=0&&j<N?[i,j]:null;}
canvas.addEventListener('pointermove',e=>{if(state.mode!=='matrix')return;const rect=canvas.getBoundingClientRect();hoveredCell=matrixCellAt(e.clientX-rect.left,e.clientY-rect.top);dirty=true;schedule();});
canvas.addEventListener('pointerleave',()=>{hoveredCell=null;dirty=true;schedule();});
canvas.addEventListener('click',e=>{const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;if(state.mode==='matrix'){const cell=matrixCellAt(x,y);if(cell){state.stock=cell[0];state.column=cell[1];hoveredCell=null;sync();}return;}const count=state.mode==='stock'?N:M;let best=-1,distance=state.mode==='stock'?28:14;for(let i=0;i<count;i++){const p=state.mode==='stock'?stockPosition(i):pairPosition(i),d=Math.hypot(x-p[0],y-p[1]);if(d<distance){best=i;distance=d;}}if(best>=0){if(state.mode==='stock'){state.stock=best;updateMatrix();}else{state.pair=best;updateSpectrum();}sync();}});
canvas.addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;e.preventDefault();const delta=e.key==='ArrowLeft'||e.key==='ArrowUp'?-1:1;if(state.mode==='matrix'){if(e.key==='ArrowLeft'||e.key==='ArrowRight')state.column=Math.max(0,Math.min(N-1,state.column+delta));else state.stock=Math.max(0,Math.min(N-1,state.stock+delta));hoveredCell=null;}else if(state.mode==='stock'){state.stock=(state.stock+delta+N)%N;updateMatrix();}else{state.pair=(state.pair+delta+M)%M;updateSpectrum();}sync();});
new IntersectionObserver(([e])=>{inView=e.isIntersecting;if(inView){last=0;dirty=true;schedule();}}).observe(viewport);
document.addEventListener('visibilitychange',()=>{if(!document.hidden){last=0;dirty=true;schedule();}});
reduced.addEventListener('change',()=>{if(reduced.matches){state.playing=false;if(transition){visual=transition.to;transition=null;}}sync();});
window.GraphCovNetwork={getState:()=>({...state,n:N,m:M,degree:DEGREE,rho:RHO,neighborCount:pairNeighbors(state.pair).length,normRatio:normCentered(spectrum[state.passes])/normCentered(spectrum[0]),bound:RHO**state.passes,matrix:matrix.map(row=>[...row]),selectedMatrixCell:[state.stock,state.column],selectedMatrixValue:matrix[state.stock][state.column],matrixGeometry:matrixGeometry(),pairSignal:[...spectrum[state.passes]],visual:{...visual},transitioning:!!transition}),multiplyLineGraph:x=>multiply(x),setMode};
updateSpectrum();updateMatrix();sync();resize();
})();
