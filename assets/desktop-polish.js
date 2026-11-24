(() => {
'use strict';
const workbench=document.getElementById('study-workbench');
function openHash(){let target;try{target=document.getElementById(decodeURIComponent(location.hash.slice(1)));}catch{return;}if(target&&(target.id==='graph-dynamics'||workbench.contains(target))){workbench.open=true;requestAnimationFrame(()=>target.scrollIntoView({block:'start',behavior:'instant'}));}}
window.addEventListener('hashchange',openHash);if(location.hash)openHash();
document.querySelectorAll('a[href="#graph-dynamics"]').forEach(a=>a.addEventListener('click',()=>{workbench.open=true;}));
const stages=document.querySelectorAll('[data-step]');
function reflectExperiment(){const current=Number(document.querySelector('[data-step][aria-current]')?.dataset.step||0),experiment=current<3?0:current<5?1:2;document.querySelectorAll('.experiment-map>div').forEach((e,i)=>e.classList.toggle('active-experiment',i===experiment));}
stages.forEach(b=>b.addEventListener('click',reflectExperiment));
new MutationObserver(reflectExperiment).observe(document.querySelector('.story-steps'),{subtree:true,attributes:true,attributeFilter:['aria-current']});reflectExperiment();
})();
