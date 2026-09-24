export function meanTrace(values=[18,24], early=true) {
  let total=0; const states=[{total,consumed:0,result:null,note:'No reading has contributed yet.'}];
  values.forEach((value,index)=>{ if(early&&index>0)return; total+=value; states.push({total,consumed:index+1,result:null,note:`Add reading ${value}; total is now ${total}.`}); });
  states.push({total,consumed:early?1:values.length,result:total/values.length,note:early?'Return leaves the function immediately. Remaining readings never contribute.':'Every reading contributed before the single return.'});return states;
}

export const temperatureCases=[{id:'zero',label:'Reference: 0°C → 32°F',x:0,expected:32},{id:'boil',label:'Reference: 100°C → 212°F',x:100,expected:212},{id:'negative',label:'Reference: −40°C → −40°F',x:-40,expected:-40},{id:'difference',label:'Property: +10°C → +18°F',x:5,expected:18}];

export const temperatureMutants=[{id:'correct',label:'1.8 × C + 32',fn:x=>1.8*x+32},{id:'offset',label:'1.8 × C (missing offset)',fn:x=>1.8*x},{id:'slope',label:'2 × C + 32 (wrong slope)',fn:x=>2*x+32},{id:'constant',label:'Always return 32',fn:()=>32}];

export function testMatrix(selected=['difference']) {return temperatureMutants.map(m=>({...m,checks:temperatureCases.filter(c=>selected.includes(c.id)).map(c=>{const actual=c.id==='difference'?m.fn(c.x+10)-m.fn(c.x):m.fn(c.x);return {id:c.id,actual,pass:Math.abs(actual-c.expected)<1e-10};})}));}

export function dependencyModel(modern=false) {const versions=[1,2,3,4];const a=versions.filter(v=>v>=2&&v<4);const b=versions.filter(v=>modern?v>=3&&v<5:v>=1&&v<2);return {versions,a,b,common:a.filter(v=>b.includes(v))};}
