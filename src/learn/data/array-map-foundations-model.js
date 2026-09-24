export function arrayMovementTrace(operation='insert',full=false) {
  let cells=['A','B','C',...(full?[]:[null])],length=3,writes=0,oldCells=null;
  const trace=[];
  const save=(note,from=null,to=null)=>trace.push({cells:[...cells],oldCells:oldCells?[...oldCells]:null,length,capacity:cells.length,writes,note,from,to});
  save('Three logical elements occupy consecutive slots. An unused slot is capacity, not a fourth element.');
  const index=operation==='front'?0:operation==='append'?3:1;
  if(operation==='delete') {
    for(let i=index;i<length-1;i++){cells[i]=cells[i+1];writes++;save(`Move slot ${i+1} into slot ${i}. The trailing duplicate is not a new logical item.`,i+1,i);}
    cells[length-1]=null;length--;save('Shorten the logical sequence and clear the released slot. The backing capacity stays available.');
  } else {
    if(length===cells.length){const old=[...cells];oldCells=old;cells=Array(cells.length*2).fill(null);save('Allocate a larger backing array. The original three items still exist in the old storage.',null,null);
      for(let i=0;i<old.length;i++){cells[i]=old[i];writes++;save(`Copy old slot ${i} into new slot ${i}. Growth copies existing references/values; it does not insert X yet.`,i,i);}
      oldCells=null;save('All existing elements are copied. Use the new backing array and release the old storage.');
    }
    for(let i=length;i>index;i--){cells[i]=cells[i-1];writes++;save(`Shift slot ${i-1} to ${i}, moving right-to-left so unread items are not overwritten.`,i-1,i);}
    cells[index]='X';length++;writes++;save(`Write X at index ${index}; the logical length is now ${length}.`,null,index);
  }
  return trace;
}

export const textCases={ascii:'cat',composed:'café',decomposed:'cafe\u0301',emoji:'🙂'};
export function textModel(kind='decomposed',normalize=false) {
  const original=textCases[kind],text=normalize?original.normalize('NFC'):original;
  let byteOffset=0;
  const units=Array.from(text).map((char,index)=>{const bytes=Array.from(new TextEncoder().encode(char)),unit={char,index,code:'U+'+char.codePointAt(0).toString(16).toUpperCase().padStart(4,'0'),bytes,byteOffset};byteOffset+=bytes.length;return unit;});
  return {original,text,changed:original!==text,units,bytes:units.flatMap(u=>u.bytes),length:units.length};
}

export const hashEntries=[[10,2],[14,5],[18,1]];
export function hashTrace(key=18,capacity=4,operation='get') {
  const buckets=Array.from({length:capacity},()=>[]);hashEntries.forEach(([k,v])=>buckets[k%capacity].push({key:k,value:v}));
  const bucket=key%capacity,trace=[];let comparisons=0;
  const save=(note,selected=null,result=null)=>trace.push({buckets:structuredClone(buckets),bucket,key,comparisons,selected,result,note});
  save(`Compute ${key} remainder ${capacity} = bucket ${bucket}. A bucket narrows the candidates; it does not establish key equality.`);
  for(let i=0;i<buckets[bucket].length;i++) {
    const entry=buckets[bucket][i];comparisons++;
    if(entry.key===key){if(operation==='set')entry.value=99;save(operation==='set'?`Key ${key} is equal: replace its value with 99. Do not create a duplicate key.`:`Key ${key} is equal: return ${entry.value}.`,i,entry.value);return trace;}
    save(`Compare ${key} with stored key ${entry.key}: different keys can share a bucket. Continue within this bucket.`,i);
  }
  if(operation==='set'){buckets[bucket].push({key,value:99});save(`No equal key exists. Append a new ${key} → 99 entry to this bucket.`,buckets[bucket].length-1,99);}
  else save(`This bucket is exhausted. Key ${key} is absent; do not return a colliding key's value.`,null,'absent');
  return trace;
}
