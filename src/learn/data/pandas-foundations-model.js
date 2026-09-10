export function alignmentModel(mode='labels', reversed=false, incomplete=false) {
  const orders=(reversed?[{label:'a',value:20},{label:'b',value:10}]:[{label:'b',value:10},{label:'a',value:20}]);
  const fees=incomplete?[{label:'b',value:2},{label:'c',value:7}]:[{label:'a',value:1},{label:'b',value:2}];
  return {orders,fees,rows:orders.map((row,i)=>{
    const source=mode==='labels'?fees.findIndex(f=>f.label===row.label):i;
    return {...row,source,fee:source<0?null:fees[source].value,total:source<0?null:row.value+fees[source].value};
  })};
}
export const cleaningInputs=['10','','bad','0'];
export function cleaningModel(policy='known') {
  return cleaningInputs.map((raw,index)=>{
    const value=raw===''||raw==='bad'?null:Number(raw);
    const mask=policy==='known'?value!==null:value===null?null:value>0;
    return {index,raw,value,mask,keep:mask===true,reason:raw===''?'Missing in source':raw==='bad'?'Invalid numeric text':'Observed value'};
  });
}
export const groupInputs=[{id:101,region:'North',amount:10},{id:102,region:'North',amount:null},{id:103,region:'South',amount:30},{id:104,region:null,amount:40}];
export function groupingModel(keepMissing=true, fillZero=false) {
  const keys=['North','South',...(keepMissing?[null]:[])];
  const groups=keys.map(key=>{
    const members=groupInputs.filter(r=>r.region===key);
    const values=members.map(r=>r.amount===null&&fillZero?0:r.amount).filter(v=>v!==null);
    return {key,members:members.map(r=>r.id),size:members.length,count:values.length,sum:values.reduce((a,b)=>a+b,0),mean:values.length?values.reduce((a,b)=>a+b,0)/values.length:null};
  });
  return {groups,transformed:groupInputs.map(r=>({id:r.id,mean:groups.find(g=>g.key===r.region)?.mean??null}))};
}
