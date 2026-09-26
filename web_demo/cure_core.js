/* Thuật toán CURE cho sandbox độc lập: tính trên dữ liệu, không trên pixel. */
(function(root){
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
function representatives(points,c,alpha){
 const mean=points[0].map((_,d)=>points.reduce((s,p)=>s+p[d],0)/points.length);
 let chosen=[];
 if(points.length<=c)chosen=points.map((_,i)=>i);
 else{
  let first=0;for(let i=1;i<points.length;i++)if(distance(points[i],mean)>distance(points[first],mean))first=i;
  chosen=[first];
  while(chosen.length<c){
   let best=-1,max=-1;
   for(let i=0;i<points.length;i++){
    if(chosen.includes(i))continue;
    const d=Math.min(...chosen.map(j=>distance(points[i],points[j])));
    if(d>max){max=d;best=i;}
   }
   chosen.push(best);
  }
 }
 return {mean,reps:chosen.map(i=>points[i].map((v,d)=>v+alpha*(mean[d]-v)))};
}
function initialize(points){
 if(!points.length||points.some(p=>p.length!==2||p.some(v=>!Number.isFinite(v))))throw Error('Cần ít nhất một điểm có 2 tọa độ hữu hạn.');
 return points.map((p,i)=>({id:i,indices:[i],points:[p.slice()],mean:p.slice(),reps:[p.slice()]}));
}
function step(groups,k,c,alpha){
 if(!Number.isInteger(k)||k<1||!Number.isInteger(c)||c<1||!Number.isFinite(alpha)||alpha<0||alpha>1)throw Error('Tham số không hợp lệ.');
 if(groups.length<=k)return null;
 let best=Infinity,pair;
 for(let i=0;i<groups.length;i++)for(let j=i+1;j<groups.length;j++){
  let d=Infinity;
  for(const a of groups[i].reps)for(const b of groups[j].reps)d=Math.min(d,distance(a,b));
  if(d<best){best=d;pair=[i,j];}
 }
 const [i,j]=pair,a=groups[i],b=groups[j];
 const points=a.points.concat(b.points),indices=a.indices.concat(b.indices);
 groups[i]={id:a.id,points,indices,...representatives(points,c,alpha)};groups.splice(j,1);
 return {left:a.id,right:b.id,distance:best,size:points.length};
}
const api={distance,representatives,initialize,step};
if(typeof module!=='undefined')module.exports=api;else root.CureCore=api;
})(globalThis);
