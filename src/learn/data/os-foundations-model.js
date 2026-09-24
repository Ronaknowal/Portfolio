export function scheduleTrace(quantum=1,io=true) {
  const jobs={A:{pc:0,value:0,state:'ready',wake:null,code:io?['+1','I/O','+1']:['+1','+1','+1']},B:{pc:0,value:0,state:'ready',wake:null,code:['+10','+10','+10']}};
  let ready=['A','B'],running=null,slice=0,time=0;
  const timeline=[],trace=[];
  const save=(note,cpu=null)=>trace.push({time,cpu,note,ready:[...ready],jobs:structuredClone(jobs),timeline:[...timeline]});
  save('Both processes are ready. Their saved counters and next-instruction positions are separate.');
  while(Object.values(jobs).some(j=>j.state!=='exited')&&time<30) {
    for(const id of ['A','B'])if(jobs[id].state==='waiting'&&jobs[id].wake<=time){jobs[id].state='ready';ready.push(id);}
    if(!running&&ready.length){running=ready.shift();slice=quantum;jobs[running].state='running';}
    const cpu=running;
    let note;
    if(!cpu){note='No process is ready; this CPU interval is idle while the I/O wait continues.';timeline.push('idle');}
    else {
      const job=jobs[cpu],instruction=job.code[job.pc];job.pc++;slice--;timeline.push(cpu);
      if(instruction==='I/O'){job.state='waiting';job.wake=time+3;running=null;note=`${cpu} requests I/O. It cannot run for the next two intervals; its next instruction is saved at ${job.pc}.`;}
      else {job.value+=Number(instruction);note=`${cpu} executes ${instruction}: counter = ${job.value}. Only this process's counter changes.`;
        if(job.pc===job.code.length){job.state='exited';running=null;note+=' It has finished.';}
        else if(!slice){job.state='ready';ready.push(cpu);running=null;note+=' The time slice ends; save its position and put it at the back of the ready queue.';}
      }
    }
    time++;save(note,cpu);
  }
  return trace;
}

export function translationModel(process='A',address=22,access='read') {
  const pageSize=16,page=Math.floor(address/pageSize),offset=address%pageSize;
  const table=[{page:0,frame:0,rights:'read',resident:true},{page:1,frame:process==='A'?3:5,rights:'read/write',resident:true},{page:2,frame:null,rights:'read/write',resident:false}];
  const entry=table.find(p=>p.page===page);
  let kind='resident',frame=entry?.frame??null,value=page===1?(process==='A'?17:42):8;
  if(!entry){kind='unmapped';value=null;}
  else if(access==='write'&&entry.rights==='read'){kind='protection';value=null;}
  else if(!entry.resident){kind='demand';frame=6;value=0;}
  if(access==='write'&&['resident','demand'].includes(kind))value=99;
  const physical=frame===null||value===null?null:frame*pageSize+offset;
  return {process,address,access,pageSize,page,offset,table,entry:entry??null,kind,frame,physical,value};
}

export function sharingTrace(shared=false) {
  const initial={aFrame:2,bFrame:2,frames:{2:7},aValue:7,bValue:7};
  return [
    {...initial,note:'Both mappings currently reach the same data, value 7. Sharing physical storage does not by itself decide what a later write means.',phase:'Before the write'},
    {...initial,note:shared?'B has a writable shared mapping. Its write is permitted to affect this shared storage.':'B attempts a write to a private copy-on-write page. A protection fault enters the kernel; B is entitled to a private writable copy.',phase:shared?'Check shared-write permission':'Handle the write fault'},
    {...initial,...(shared?{}:{bFrame:5,frames:{2:7,5:7}}),note:shared?'Keep both mappings on frame 2. No private copy is needed for this shared update.':'Copy the old contents to frame 5 and remap B. A still reaches frame 2. The values are equal, but future writes are now separate.',phase:shared?'Keep the shared frame':'Copy and remap'},
    {...initial,...(shared?{frames:{2:9},aValue:9,bValue:9}:{bFrame:5,frames:{2:7,5:9},bValue:9}),note:shared?'B writes 9. After coordinating access, A can read 9 through its shared mapping too.':'Retry B’s write: frame 5 becomes 9. A’s private view remains 7.',phase:'Complete the write'},
  ];
}
