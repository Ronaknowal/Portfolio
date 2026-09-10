export function initialNotebook(){return {sourceOffset:2,rate:null,mean:null,output:null,counter:0,counts:[null,null,null],note:'A fresh kernel has no names; no output is saved.'};}

export function notebookAction(state,action,value){let s={...state,counts:[...state.counts]}; if(action==='edit'){s.sourceOffset=value;s.note='Document edited. Kernel and old output have not executed.';return s;}
 if(action==='restart'){return {...s,rate:null,mean:null,counter:0,note:'Kernel memory cleared; the document still carries its historical output and execution counts.'};}
 if(action==='all'){for(const a of ['inputs','calculate','display'])s=notebookAction(s,a);return {...s,note:'All three cells executed top to bottom using the current source.'};}
 s.counter++;const index=['inputs','calculate','display'].indexOf(action);s.counts[index]=s.counter;
 if(action==='inputs'){s.rate=s.sourceOffset;s.note=`Kernel offset now ${s.rate}; mean and output do not recalculate.`;}
 if(action==='calculate'){if(s.rate===null)s.note='NameError: offset is undefined. Run the input cell first.';else{s.mean=20-s.rate;s.note=`Compute (10 + 20 + 30) / 3 − ${s.rate} = ${s.mean}.`;}}
 if(action==='display'){if(s.mean===null){s.output='NameError';s.note='Display failed: mean is undefined in this kernel.';}else{s.output=s.mean;s.note=`Save output ${s.mean}; it reflects the stored mean, not a live formula.`;}}
 return s;
}

export function randomConsumers(separate=false,extra=false){const splitTape=[4,8,1,6],modelTape=separate?[7,2,9,3]:splitTape;const split=[splitTape[0],...(extra?[splitTape[1]]:[])];const modelIndex=separate?0:split.length;return {split,model:modelTape[modelIndex],splitTape,modelTape,modelIndex,separate,extra};}

export function provenanceModel(change='none',complete=true){const saved={file:'readings.json',data:'[10,20,30]',offset:2,code:'v1'};const current={...saved};if(change==='sameMean')current.data='[9,20,31]';if(change==='offset')current.offset=5;if(change==='data')current.data='[10,20,60]';if(change==='code')current.code='v2';const result=JSON.parse(current.data).reduce((a,b)=>a+b,0)/3-current.offset;const key=v=>complete?JSON.stringify([v.file,v.data,v.offset,v.code,'fixed environment']):v.file;return {saved,current,result,cached:18,hit:key(saved)===key(current),sameBytes:current.data===saved.data};}
