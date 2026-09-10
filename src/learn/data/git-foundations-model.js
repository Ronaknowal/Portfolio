export function stagingTrace(restage=false) {
  const versions=[1,1,1];
  const steps=[];
  const add=(command,transfer,note)=>steps.push({command,transfer,note,versions:[...versions],status:(versions[1]!==versions[0]?'M':' ')+(versions[2]!==versions[1]?'M':' ')});
  add('Start at a committed version 1','No transfer','HEAD, index and working file agree.');
  versions[2]=2;add('Edit report.txt to version 2','Edit → working file','Editing does not update the proposed snapshot.');
  versions[1]=2;add('git add -- report.txt','Working file → index','Add captures the current contents, version 2.');
  versions[2]=3;add('Edit report.txt to version 3','Edit → working file','Now the staged change and later unstaged change both exist.');
  if(restage){versions[1]=3;add('git add -- report.txt','Working file → index','Staging again replaces the proposed contents with version 3.');}
  else add('git diff --staged','Read index versus HEAD','Inspection changes no content. The proposed version is still 2.');
  versions[0]=versions[1];add('git commit -m "Update report"','Index → new HEAD commit',restage?'The new commit records version 3; all three locations agree.':'The new commit records version 2; version 3 remains an unstaged edit.');
  return steps;
}
const graphNode=(id,parents,files)=>({id,parents,files});
export function branchTrace(diverged=false) {
  const a=graphNode('A',[],['report']),b=graphNode('B',['A'],['report','note']),c=graphNode('C',['A'],['report','units']),m=graphNode('M',['C','B'],['report','note','units']);
  const base={nodes:[a],main:'A',feature:null,head:'main'};
  return [
    {...base,command:'Start on main',note:'A commit is a snapshot; main points to A.'},
    {...base,feature:'A',head:'feature',command:'git switch -c feature',note:'Create another pointer to A. No new commit or copied project is needed.'},
    {...base,nodes:[a,b],feature:'B',head:'feature',command:'Add note; commit on feature',note:'Only feature advances to B. B records A as its parent.'},
    {...base,nodes:[a,b],feature:'B',command:'git switch main',note:'HEAD names main again. Its snapshot A has no note.'},
    {...base,nodes:diverged?[a,b,c]:[a,b],main:diverged?'C':'A',feature:'B',command:diverged?'Add units; commit on main':'Inspect unchanged main',note:diverged?'C and B both descend from A; neither is an ancestor of the other.':'Main still points to A, an ancestor of B.'},
    {...base,nodes:diverged?[a,b,c]:[a,b],main:diverged?'C':'B',feature:'B',command:'git merge --ff-only feature',note:diverged?'Refused: main cannot move to B without abandoning its C history. Inspect before selecting an integration policy.':'Fast-forward: main moves to B. No merge commit is created.'},
    {...base,nodes:diverged?[a,b,c,m]:[a,b],main:diverged?'M':'B',feature:'B',command:diverged?'git merge --no-edit feature':'Inspect integrated history',note:diverged?'M records both parent histories and their combined files. Feature still points to B.':'Both branch names point to B; there are still only two commits.'},
  ];
}
export function remoteTrace(localChange=false) {
  const start={local:'A',tracking:'A',shared:'A',working:'version 1'};
  return [
    {...start,command:'Start after cloning',note:'All three references know commit A.'},
    {...start,shared:'B',command:'Colleague commits and pushes B',note:'The shared branch changes. Your local tracking reference has not contacted it yet.'},
    {...start,shared:'B',local:localChange?'C':'A',working:localChange?'version 1 + local note':'version 1',command:localChange?'Make a local commit C':'Inspect before fetching',note:localChange?'Your main advances to C, independently of the colleague’s B.':'Your files still show A. A remote-tracking name is not a live connection.'},
    {...start,shared:'B',tracking:'B',local:localChange?'C':'A',working:localChange?'version 1 + local note':'version 1',command:'git fetch origin',note:'Fetch updates origin/main and downloads needed objects. It does not change your checked-out main or working file.'},
    {...start,shared:'B',tracking:'B',local:localChange?'C':'B',working:localChange?'version 1 + local note':'version 2',command:'git merge --ff-only origin/main',note:localChange?'Refused because C and B diverge. Your local note is preserved.':'Main can fast-forward to B; now the working file changes to version 2.'},
  ];
}
export const conflictVersions={base:'Report',ours:'Report by model',theirs:'Report by region'};
export function conflictTrace(resolution='combined') {
  const final=resolution==='combined'?'Report by model and region':conflictVersions.ours;
  return [
    {phase:'Inspect common base',working:conflictVersions.base,unmerged:false,committed:false,note:'Both branches started with the same one-line file.'},
    {phase:'Attempt merge',working:'<<<<<<< HEAD\nReport by model\n=======\nReport by region\n>>>>>>> feature',unmerged:true,committed:false,note:'Git has three versions to compare but cannot decide the intended meaning.'},
    {phase:'Edit the intended resolution',working:final,unmerged:true,committed:false,note:'Editing removes markers. The index still has unresolved stages until you add this path.'},
    {phase:'git add -- title.txt',working:final,unmerged:false,committed:false,note:'Add replaces the conflict stages with the chosen resolution. Review and test it before committing.'},
    {phase:'git commit',working:final,unmerged:false,committed:true,note:resolution==='combined'?'The merge records both parents and the combined title.':'Git accepts this resolution too, but it discards the region wording. Success does not prove the intended meaning was preserved.'},
  ];
}
