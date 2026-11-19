/* Same camera-led poses as replay.csv; no browser-side gap filling. */
"use strict";
(async function () {
  const box = document.getElementById("replay");
  if (!box) return;
  const label = box.querySelector(".replay-status");
  try {
    const response = await fetch(box.dataset.url);
    if (!response.ok) throw new Error("Replay download failed");
    const data = await response.json(), frames = data.frames;
    if (!frames.length) throw new Error("No replay poses available");
    const canvas = box.querySelector("canvas"), ctx = canvas.getContext("2d");
    const slider = box.querySelector("input"), play = box.querySelector("button");
    const rate = box.querySelector("select");
    const view = box.querySelector(".replay-view"), coordinates = box.querySelector(".replay-coordinates");
    const speedplot = box.querySelector(".speedplot");
    slider.max = frames.length - 1;
    const add = (a,b) => a.map((v,i)=>v+b[i]);
    const mul = (a,s) => a.map(v=>v*s);
    const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
    const unit = a => mul(a,1/Math.hypot(...a));
    const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const up = unit(data.up), right = unit(add([1,0,0],mul(up,-up[0])));
    const forward = cross(up,right);
    const origin = frames[0].knob;
    const world = p => { const q=add(p,mul(origin,-1));return [dot(q,right),dot(q,forward),dot(q,up)]; };
    const poses = frames.map(f=>({knob:world(f.knob),tip:world(add(f.knob,mul(f.axis,data.length_m)))}));
    const all = poses.flatMap(p=>[p.knob,p.tip]);
    const lo=[0,1,2].map(i=>Math.min(...all.map(p=>p[i])));
    const hi=[0,1,2].map(i=>Math.max(...all.map(p=>p[i])));
    const center=lo.map((v,i)=>(v+hi[i])/2);
    const radius=Math.max(.4,...hi.map((v,i)=>(v-lo[i])/2));
    let yaw=-.9,pitch=.4,zoom=1,index=0,playing=false,last=0,position=frames[0].t;
    const pink="#e0187a", orange="#d97706";
    function project(p) {
      const [x,y,z]=p.map((v,i)=>v-center[i]);
      const a=Math.cos(yaw)*x-Math.sin(yaw)*y, b=Math.sin(yaw)*x+Math.cos(yaw)*y;
      const scale=Math.min(canvas.width,canvas.height)*.29/radius*zoom;
      return [canvas.width/2+a*scale,canvas.height/2-(Math.cos(pitch)*z-Math.sin(pitch)*b)*scale];
    }
    function segment(a,b,color,dashed=false,width=2) {
      a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;
      ctx.setLineDash(dashed?[7,5]:[]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
    }
    function textAt(p,text,color="#555",ratio=1) {
      const q=project(p);ctx.fillStyle=color;ctx.font=`${12*ratio}px system-ui`;ctx.fillText(text,q[0]+5*ratio,q[1]-5*ratio);
    }
    function drawSpeed(ratio) {
      if(!speedplot)return;
      speedplot.width=Math.max(300,Math.round(speedplot.clientWidth*ratio));speedplot.height=170*ratio;
      const s=speedplot.getContext("2d"),w=speedplot.width,h=speedplot.height;
      const peak=Math.max(10,...frames.map(f=>f.speed===null?0:f.speed))*1.1;
      const x=t=>45*ratio+(w-65*ratio)*(t-frames[0].t)/(frames[frames.length-1].t-frames[0].t||1);
      const y=v=>h-32*ratio-(h-62*ratio)*v/peak;
      s.fillStyle="#fff";s.fillRect(0,0,w,h);s.font=`${12*ratio}px system-ui`;
      s.fillStyle="#333";s.fillText("Camera-supported barrel speed (mph)",45*ratio,16*ratio);
      for(let i=0;i<=4;i++) {
        const value=peak*i/4;s.strokeStyle="#ddd";s.beginPath();s.moveTo(x(frames[0].t),y(value));s.lineTo(w-20*ratio,y(value));s.stroke();
        s.fillStyle="#555";s.fillText(value.toFixed(0),8*ratio,y(value)+4*ratio);
        const time=frames[0].t+(frames[frames.length-1].t-frames[0].t)*i/4;
        s.fillText(time.toFixed(2)+" s",x(time)-10*ratio,h-10*ratio);
      }
      s.strokeStyle=pink;s.lineWidth=2*ratio;s.beginPath();let connected=false;
      for(let i=0;i<frames.length;i++) {
        const f=frames[i];if(f.speed===null){connected=false;continue;}
        if(connected&&i&&f.t-frames[i-1].t<.06)s.lineTo(x(f.t),y(f.speed));else s.moveTo(x(f.t),y(f.speed));connected=true;
      }
      s.stroke();s.strokeStyle="#333";s.beginPath();s.moveTo(x(frames[index].t),25*ratio);s.lineTo(x(frames[index].t),h-32*ratio);s.stroke();
    }
    function draw() {
      const ratio=window.devicePixelRatio||1;
      canvas.width=Math.max(300,Math.round(canvas.clientWidth*ratio));canvas.height=420*ratio;
      ctx.fillStyle="#fff";ctx.fillRect(0,0,canvas.width,canvas.height);
      for(let i=-2;i<=2;i++) {
        const y=center[1]+i*radius/2,x=center[0]+i*radius/2,z=lo[2];
        segment([center[0]-radius,y,z],[center[0]+radius,y,z],"#e5e7eb",false,1);
        segment([x,center[1]-radius,z],[x,center[1]+radius,z],"#e5e7eb",false,1);
        textAt([x,center[1]-radius,z],x.toFixed(2),"#666",ratio);
        textAt([center[0]-radius,y,z],y.toFixed(2),"#666",ratio);
      }
      const corner=[center[0]-radius,center[1]-radius,lo[2]];
      const zspan=Math.max(.05,hi[2]-lo[2]);
      const axes=[[2*radius,0,0],[0,2*radius,0],[0,0,zspan]];
      axes.forEach((axis,i)=>{const end=add(corner,axis);segment(corner,end,["#bd2424","#16803d","#1d4ed8"][i],false,2*ratio);textAt(end,["X (m)","Y (m)","Z up (m)"][i],"#222",ratio);});
      for(let i=0;i<=4;i++){const z=lo[2]+i*zspan/4;textAt([corner[0],corner[1],z],z.toFixed(2),"#555",ratio);}
      for(let i=1;i<frames.length;i++) {
        const estimated=frames[i].source!=="camera_supported"||frames[i-1].source!=="camera_supported";
        const fraction=data.sweet_spot_s_m/data.length_m;
        const a=add(poses[i-1].knob,mul(add(poses[i-1].tip,mul(poses[i-1].knob,-1)),fraction));
        const b=add(poses[i].knob,mul(add(poses[i].tip,mul(poses[i].knob,-1)),fraction));
        segment(a,b,estimated?orange:pink,estimated,2*ratio);
      }
      const f=frames[index],p=poses[index],estimated=f.source!=="camera_supported";
      segment(p.knob,p.tip,estimated?orange:"#222",estimated,5*ratio);
      for(const s of Object.values(data.marker_s_m)) {
        const point=project(add(p.knob,mul(add(p.tip,mul(p.knob,-1)),s/data.length_m)));
        ctx.setLineDash([]);ctx.fillStyle=estimated?orange:pink;
        ctx.beginPath();ctx.arc(...point,5*ratio,0,Math.PI*2);ctx.fill();
      }
      slider.value=index;
      if(coordinates) {
        const barrel=add(p.knob,mul(add(p.tip,mul(p.knob,-1)),data.sweet_spot_s_m/data.length_m));
        const fmt=q=>q.map((v,i)=>["X","Y","Z"][i]+" "+v.toFixed(3)).join("   ")+" m";
        coordinates.textContent="Knob:    "+fmt(p.knob)+"\nBarrel:  "+fmt(barrel);
      }
      label.textContent=`${f.t.toFixed(3)} s | ${estimated?"IMU/model estimate":"Camera-supported pose"}`+
        ` | ${f.speed===null?"speed withheld":f.speed.toFixed(1)+" mph (supported estimate)"}`+
        ` | nearest camera support ${f.distance_to_camera_ms.toFixed(0)} ms`;
      drawSpeed(ratio);
    }
    play.onclick=()=>{if(index===frames.length-1){index=0;position=frames[0].t;}playing=!playing;play.textContent=playing?"Pause":"Play";last=0;draw();};
    slider.oninput=()=>{playing=false;play.textContent="Play";index=Number(slider.value);position=frames[index].t;draw();};
    if(view)view.onchange=()=>{[yaw,pitch]=({iso:[-.9,.4],xy:[0,Math.PI/2],xz:[0,0],yz:[Math.PI/2,0]})[view.value];draw();};
    function tick(now) {
      if(playing&&last) {
        position+=(now-last)/1000*Number(rate.value);
        while(index<frames.length-1&&frames[index+1].t<=position)index++;
        if(index===frames.length-1){playing=false;play.textContent="Replay";}
        draw();
      }
      last=now;requestAnimationFrame(tick);
    }
    let pointer=null;
    canvas.onpointerdown=e=>{pointer=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);};
    canvas.onpointermove=e=>{if(pointer){yaw+=(e.clientX-pointer[0])*.01;pitch=Math.max(-1.4,Math.min(1.4,pitch+(e.clientY-pointer[1])*.01));pointer=[e.clientX,e.clientY];draw();}};
    canvas.onpointerup=canvas.onpointercancel=()=>{pointer=null;};
    canvas.onwheel=e=>{e.preventDefault();zoom=Math.max(.25,Math.min(4,zoom*Math.exp(-e.deltaY*.001)));draw();};
    window.addEventListener("resize",draw);
    draw();requestAnimationFrame(tick);
  } catch(error) { label.textContent=error.message; }
})();
