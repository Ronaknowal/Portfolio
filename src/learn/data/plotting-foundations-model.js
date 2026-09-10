export const plotViews={full:{min:0,max:30,scale:'linear'},zoom:{min:9,max:21,scale:'linear'},log:{min:1,max:100,scale:'log'}};
export function coordinateModel(view='full') {
  const settings=plotViews[view];
  const transform=x=>settings.scale==='log'?Math.log10(x):x;
  return { ...settings,points:[10,20].map(value=>({value,fraction:(transform(value)-transform(settings.min))/(transform(settings.max)-transform(settings.min))})) };
}
export const histogramValues=[1,2,2,3,7,9];
export function histogramModel(layout='three', density=false) {
  const edges=layout==='three'?[0,3,6,10]:[0,2,10];
  return edges.slice(0,-1).map((lo,i)=>{
    const hi=edges[i+1],indices=histogramValues.map((v,j)=>v>=lo&&(v<hi||(i===edges.length-2&&v===hi))?j:-1).filter(j=>j>=0);
    const count=indices.length,width=hi-lo;
    return {lo,hi,indices,count,width,height:density?count/(histogramValues.length*width):count,mass:count/histogramValues.length};
  });
}
export function intervalModel(kind='sd', repeated=false) {
  const values=repeated?[10,12,14,10,12,14]:[10,12,14];
  const mean=values.reduce((a,b)=>a+b,0)/values.length;
  const sd=Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/(values.length-1));
  const halfWidth=kind==='sd'?sd:sd/Math.sqrt(values.length);
  return {values,n:values.length,mean,sd,halfWidth,low:mean-halfWidth,high:mean+halfWidth};
}
