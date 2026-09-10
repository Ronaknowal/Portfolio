export function reverseTrace(size=3,broken=false) {
  const ids=['A','B','C'].slice(0,size),nodes=ids.map((id,i)=>({id,value:[4,7,9][i],next:ids[i+1]??null}));
  let previous=null,current=ids[0]??null,saved=null,head=current;
  const trace=[];
  const save=(note,phase)=>trace.push({nodes:structuredClone(nodes),previous,current,saved,head,note,phase});
  save('Previous starts empty; current starts at the head. No node or link has changed.','Initialize');
  while(current!==null){const node=nodes.find(n=>n.id===current);
    if(broken){node.next=previous;save('The link was overwritten before saving its successor. The original route to the remaining nodes is lost.','Faulty order');saved=node.next;previous=current;current=saved;head=previous;save('The loop ends after A because the overwritten link is empty. B and C, if present, are no longer reachable from this head.','Lost suffix');return trace;}
    saved=node.next;save(`Remember ${current}'s old successor ${saved??'∅'} before changing its next link.`,'Save successor');
    node.next=previous;save(`Make ${current}.next point to ${previous??'∅'}. The reversed prefix gains one node; saved still preserves the remaining suffix.`,'Reverse one link');
    previous=current;current=saved;saved=null;save('Advance previous to the new prefix head and current to the saved suffix. Every original node is still in one of those parts.','Advance');
  }
  head=previous;save('The unprocessed suffix is empty. Publish previous as the new head; only links changed, not node values.','Finish');
  return trace;
}

export function bracketTrace(text='([])') {
  const stack=[],trace=[{index:-1,char:null,stack:[],result:null,note:'Before scanning, the processed prefix is empty and there are no unmatched opening brackets.'}];
  const matching={')':'(',']':'[','}':'{'};
  for(let index=0;index<text.length;index++){
    const char=text[index];let result=null,note;
    if('([{'.includes(char)){stack.push({char,index});note=`Opening ${char}: push it. Its closer must arrive before any earlier opener can close.`;}
    else if(char in matching){if(!stack.length){result=false;note=`Closing ${char} has no unmatched opener. Reject at position ${index}.`;}
      else if(stack.at(-1).char!==matching[char]){result=false;note=`Closing ${char} needs ${matching[char]}, but the most recent unmatched opener is ${stack.at(-1).char}. Crossing pairs are invalid.`;}
      else {stack.pop();note=`Closing ${char} matches the top. Pop exactly that opener; earlier unmatched openers remain.`;}}
    else note='This recognizer ignores non-bracket characters; it does not understand quotes or comments.';
    trace.push({index,char,stack:structuredClone(stack),result,note});if(result===false)return trace;
  }
  trace.push({index:text.length,char:null,stack:structuredClone(stack),result:stack.length===0,note:stack.length?'End of input, but opening brackets remain unmatched. Reject.':'End of input and no unmatched openers remain. Accept.'});return trace;
}

export const queueEvents=[['put','A'],['put','B'],['put','C'],['take'],['put','D'],['put','E'],['put','F'],['take'],['take'],['take'],['take'],['take']];
export function ringTrace(capacity=4) {
  let head=0,size=0;const cells=Array(capacity).fill(null),output=[],trace=[];
  const save=(note,event=null,changed=null)=>trace.push({cells:[...cells],head,size,tail:(head+size)%capacity,logical:Array.from({length:size},(_,i)=>cells[(head+i)%capacity]),output:[...output],note,event,changed});
  save('Empty queue. Size distinguishes empty from full even when the insertion slot equals head.');
  for(const [op,value]of queueEvents){let note,changed=null;
    if(op==='put'){if(size===capacity)note=`Reject ${value}: all ${capacity} slots are occupied. Pending work is retained.`;
      else{const slot=(head+size)%capacity;cells[slot]=value;size++;changed=slot;note=`Enqueue ${value} into slot ${slot}. Existing items do not move; the oldest remains at head.`;}}
    else if(!size)note='Reject dequeue: the queue is empty. Do not read an unused slot as an item.';
    else{const item=cells[head];output.push(item);changed=head;cells[head]=null;head=(head+1)%capacity;size--;note=`Dequeue ${item}, then advance head with wraparound. The remaining FIFO order is preserved.`;}
    save(note,op==='put'?'enqueue '+value:'dequeue',changed);
  }
  return trace;
}
