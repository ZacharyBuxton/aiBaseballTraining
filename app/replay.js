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
    slider.max = frames.length - 1;
    const add = (a,b) => a.map((v,i)=>v+b[i]);
    const mul = (a,s) => a.map(v=>v*s);
    const dot = (a,b) => a.reduce((s,v,i)=>s+v*b[i],0);
    const unit = a => mul(a,1/Math.hypot(...a));
    const cross = (a,b) => [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
    const up = unit(data.up), right = unit(add([1,0,0],mul(up,-up[0])));
    const forward = cross(up,right);
    const world = p => [dot(p,right),dot(p,forward),dot(p,up)];
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
      const scale=Math.min(canvas.width,canvas.height)*.37/radius*zoom;
      return [canvas.width/2+a*scale,canvas.height/2-(Math.cos(pitch)*z-Math.sin(pitch)*b)*scale];
    }
    function segment(a,b,color,dashed=false,width=2) {
      a=project(a);b=project(b);ctx.strokeStyle=color;ctx.lineWidth=width;
      ctx.setLineDash(dashed?[7,5]:[]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();
    }
    function draw() {
      const ratio=window.devicePixelRatio||1;
      canvas.width=Math.max(300,Math.round(canvas.clientWidth*ratio));canvas.height=420*ratio;
      ctx.fillStyle="#fff";ctx.fillRect(0,0,canvas.width,canvas.height);
      for(let i=-2;i<=2;i++) {
        const y=center[1]+i*radius/2,x=center[0]+i*radius/2,z=lo[2];
        segment([center[0]-radius,y,z],[center[0]+radius,y,z],"#e5e7eb",false,1);
        segment([x,center[1]-radius,z],[x,center[1]+radius,z],"#e5e7eb",false,1);
      }
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
      label.textContent=`${f.t.toFixed(3)} s | ${estimated?"IMU/model estimate":"Camera-supported pose"}`+
        ` | ${f.speed===null?"speed withheld":f.speed.toFixed(1)+" mph (supported estimate)"}`+
        ` | nearest camera support ${f.distance_to_camera_ms.toFixed(0)} ms`;
    }
    play.onclick=()=>{if(index===frames.length-1){index=0;position=frames[0].t;}playing=!playing;play.textContent=playing?"Pause":"Play";last=0;draw();};
    slider.oninput=()=>{playing=false;play.textContent="Play";index=Number(slider.value);position=frames[index].t;draw();};
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
