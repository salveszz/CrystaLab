// FILE PURPOSE: Edit native-image polygons and request explicit local draft/review operations.
"use strict";
const $ = id => document.getElementById(id);
const names = {seed:"Main crystal",secondary_crystal:"Other crystal",inclusion_candidate:"Possible inclusion"};
const statusNames = {proposed:"Proposed",draft:"Draft",reviewed:"Reviewed",stale_review:"Review changed"};
const copy = value => JSON.parse(JSON.stringify(value));
const state = {session:null,current:null,doc:null,saved:null,image:null,index:0,selected:0,vertex:null,tool:"edit",drawing:[],history:[],hidden:false,space:false,drag:null,busy:false,loaded:false,view:{x:0,y:0,scale:1,w:0,h:0}};
const canvas = $("canvas"), ctx = canvas.getContext("2d");
const locked = () => !state.current || state.session.read_only || state.current.review !== null;
const dirty = () => state.doc && JSON.stringify(state.doc) !== JSON.stringify(state.saved);

async function api(path,payload) {
  const options = payload === undefined ? {} : {method:"POST",headers:{"Content-Type":"application/json","X-CrystaLab-Session":state.session.session_token},body:JSON.stringify(payload)};
  const response = await fetch(path,options);
  const value = await response.json();
  if(!response.ok) throw new Error(value.error || "The request could not be completed.");
  return value;
}
function message(text,error=false){ $("message").textContent=text;$("message").classList.toggle("error",error);$("message").hidden=!text; }
async function action(fn){if(state.busy)return;state.busy=true;refreshControls();try{await fn();}catch(error){message(error.message,true);}finally{state.busy=false;refreshControls();}}
function clearAcknowledgment(){$("ack").checked=false;$("empty-reviewed").checked=false;}
function remember(){clearAcknowledgment();state.history.push(copy(state.doc));if(state.history.length>50)state.history.shift();}
function setTool(tool){state.tool=tool;if(tool!=="draw")state.drawing=[];state.vertex=null;$("viewport").classList.toggle("pan",tool==="pan");refreshControls();draw();}
function updateFromInputs(){clearAcknowledgment();state.doc.review_note=$("notes").value;state.doc.flags={...state.doc.flags,uncertain:$("image-uncertain").checked};refreshControls();}
function populateFields(){ $("notes").value=state.doc.review_note || "";$("image-uncertain").checked=Boolean(state.doc.flags?.uncertain);refreshShapes(); }
function refreshShapes(){
  const list=$("shape-list");list.replaceChildren();
  if(!state.doc)return;
  if(state.selected>=state.doc.shapes.length)state.selected=Math.max(0,state.doc.shapes.length-1);
  state.doc.shapes.forEach((shape,index)=>{
    const button=document.createElement("button");button.className="shape-item"+(index===state.selected?" selected":"");button.type="button";
    const label=document.createElement("span");label.textContent=`${index+1}. ${names[shape.label]}`;
    const detail=document.createElement("small");detail.textContent=shape.flags?.uncertain?"uncertain":`${shape.points.length} points`;
    button.append(label,detail);button.addEventListener("click",()=>{state.selected=index;state.vertex=null;setTool("edit");refreshShapes();});list.append(button);
  });
  if(!state.doc.shapes.length){const empty=document.createElement("div");empty.className="empty-shapes";empty.textContent="No outlines yet. Add an outline, or check the image before marking it as empty.";list.append(empty);}
  const selected=state.doc.shapes[state.selected];
  if(state.tool!=="draw")$("label").value=selected?.label || "seed";
  $("shape-uncertain").checked=Boolean(selected?.flags?.uncertain);
  refreshControls();draw();
}
function refreshControls(){
  if(!state.session)return;
  const unavailable=locked() || state.busy;
  const changed=dirty(),drawing=state.tool==="draw";
  const selected=state.doc?.shapes[state.selected];
  ["edit","draw"].forEach(id=>{$(id).disabled=unavailable;$(id).setAttribute("aria-pressed",String(state.tool===id));});
  $("pan").setAttribute("aria-pressed",String(state.tool==="pan"));
  $("finish").hidden=!drawing;$("cancel-draw").hidden=!drawing;$("finish").disabled=state.drawing.length<3 || unavailable;
  $("draw-help").hidden=!drawing;$("save").disabled=unavailable || !changed || drawing;
  $("discard").disabled=unavailable || !changed;$("undo").disabled=unavailable || !state.history.length;
  $("remove-shape").disabled=unavailable || !selected || drawing;
  $("label").disabled=unavailable || (!selected && !drawing);$("shape-uncertain").disabled=unavailable || !selected || drawing;
  ["notes","image-uncertain"].forEach(id=>$(id).disabled=unavailable);
  $("previous").disabled=state.busy || state.index===0;$("next").disabled=state.busy || state.index>=state.session.samples.length-1;
  const uncertain=Boolean(state.doc?.flags?.uncertain) || Boolean(state.doc?.shapes.some(s=>s.flags?.uncertain));
  const empty=state.doc?.shapes.length===0;$("empty-row").hidden=!empty;
  const canReview=!unavailable && state.loaded && !drawing && !uncertain && $("reviewer").value.trim() && $("ack").checked && (!empty || $("empty-reviewed").checked);
  $("review").disabled=!canReview;$("reviewer").disabled=unavailable;$("ack").disabled=unavailable || uncertain;$("empty-reviewed").disabled=unavailable || uncertain;
  $("revise").hidden=!state.current?.review || state.session.read_only;$("revise").disabled=state.busy;
  $("export").disabled=state.busy || state.session.read_only || state.session.reviewed===0 || state.session.stale_reviews>0;
  $("review-help").textContent=state.session.read_only?"This preview does not change labels or reviews.":state.current?.review?(state.current.status==="stale_review"?"These labels changed after review. Open a revision to review the corrected outlines.":"Review saved. Open a revision if this image needs further corrections."):uncertain?"Keep this as a draft until the uncertain boundaries or objects are resolved.":"Review applies only to this image. Draft saves do not record a review.";
  if(state.current){const status=changed?"draft":state.current.status;$("status").textContent=statusNames[status];$("status").className="badge "+status;}
  $("image-info").textContent=state.image?`${state.image.naturalWidth} × ${state.image.naturalHeight} px · ${Math.round(state.view.scale*100)}%`:"—";
}
function refreshSession(){
  $("material").textContent=state.session.material.replaceAll("_"," ").toUpperCase();$("version").textContent=state.session.bundle_name;
  $("read-only").hidden=!state.session.read_only;$("bundle-path").textContent=state.session.bundle_path;
  $("progress-label").textContent=`${state.session.reviewed} / ${state.session.samples.length} reviewed`;$("progress").max=state.session.samples.length;$("progress").value=state.session.reviewed;
  const tabs=$("samples");tabs.replaceChildren();
  state.session.samples.forEach(sample=>{const button=document.createElement("button");button.className="sample-tab";button.type="button";button.setAttribute("aria-current",String(sample.index===state.index));button.title=statusNames[sample.status];
    const dot=document.createElement("span");dot.className="dot "+sample.status;dot.setAttribute("aria-hidden","true");const text=document.createElement("span");text.textContent=`${String(sample.index+1).padStart(2,"0")}  Frame ${sample.frame_index}`;button.append(dot,text);button.addEventListener("click",()=>action(()=>navigate(sample.index)));tabs.append(button);});
  refreshControls();
}
async function reloadSummary(){state.session=await api("/api/session");refreshSession();}
function renderSourceContext(){
  const source=state.current.source;$("source-context").hidden=!source.phase_context;
  if(!source.phase_context)return;
  $("source-context-title").textContent=`${source.caption_id || "Movie"} · Published source context`;
  $("source-phase").textContent=source.phase_context;
  const clocks=[];
  if(Number.isFinite(source.display_elapsed_s))clocks.push(`Display ${source.display_elapsed_s.toFixed(3)} s`);
  if(Number.isFinite(source.published_speedup_relative_elapsed_s))clocks.push(`Published-speed-up estimate ${source.published_speedup_relative_elapsed_s.toFixed(3)} s since clip start`);
  clocks.push("Acquisition timestamps unavailable; growth-run start unknown.");
  $("source-clock").textContent=clocks.join(" · ");
  $("source-provenance").textContent=[source.license,source.original_video_file,source.caption_page?`Caption ${source.caption_page}`:null].filter(Boolean).join(" · ");
  const article=$("source-paper");article.hidden=true;article.removeAttribute("href");
  if(typeof source.source_url==="string" && /^https?:\/\//.test(source.source_url)){article.href=source.source_url;article.hidden=false;}
}
async function load(index,payload=null){
  state.loaded=false;$("image-loading").hidden=false;state.index=index;state.current=payload || await api(`/api/sample/${index}`);state.doc=copy(state.current.annotation);state.saved=copy(state.doc);state.history=[];state.selected=0;state.vertex=null;state.drawing=[];state.tool="edit";$("viewport").classList.remove("pan");
  $("ack").checked=false;$("empty-reviewed").checked=false;$("sample-name").textContent=state.current.source.sample_id;$("source-label").textContent=`Run ${state.current.source.run_id} · Source frame ${state.current.source.frame_index}`;
  populateFields();refreshSession();renderSourceContext();
  const image=new Image();await new Promise((resolve,reject)=>{image.onload=resolve;image.onerror=()=>reject(new Error("The original image could not be loaded."));image.src=state.current.image_url+"?version="+state.current.revision;});
  if(image.naturalWidth!==state.doc.imageWidth || image.naturalHeight!==state.doc.imageHeight)throw new Error("The image dimensions differ from the annotation.");
  state.image=image;state.loaded=true;$("image-loading").hidden=true;focusOutline();refreshControls();
}
async function saveDraft(){
  if(locked())throw new Error("This image cannot be edited in the current view.");
  if(state.tool==="draw")throw new Error("Finish or cancel the new outline before saving.");
  state.current=await api(`/api/sample/${state.index}/draft`,{bundle_id:state.session.bundle_id,revision:state.current.revision,shapes:state.doc.shapes,image_uncertain:Boolean(state.doc.flags?.uncertain),note:state.doc.review_note || ""});
  state.doc=copy(state.current.annotation);state.saved=copy(state.doc);state.history=[];await reloadSummary();populateFields();message("Draft saved. No review was recorded.");
}
async function navigate(index){
  if(index===state.index)return;
  if(state.drawing.length)throw new Error("Finish or cancel the new outline before changing images.");
  if(dirty())await saveDraft();message("");await load(index);
}
function fitBox(box){const view=state.view;const scale=Math.min(view.w/box.w,view.h/box.h)*.9;view.scale=Math.max(.03,Math.min(16,scale));view.x=view.w/2-(box.x+box.w/2)*view.scale;view.y=view.h/2-(box.y+box.h/2)*view.scale;draw();refreshControls();}
function fitImage(){if(state.loaded)fitBox({x:0,y:0,w:state.image.naturalWidth,h:state.image.naturalHeight});}
function focusOutline(){
  if(!state.loaded)return;const points=state.doc.shapes[state.selected]?.points;
  if(!points?.length){fitImage();return;}
  const xs=points.map(p=>p[0]),ys=points.map(p=>p[1]);const w=Math.max(50,Math.max(...xs)-Math.min(...xs)),h=Math.max(50,Math.max(...ys)-Math.min(...ys));
  fitBox({x:Math.min(...xs)-w*.65,y:Math.min(...ys)-h*.65,w:w*2.3,h:h*2.3});
}
function zoom(factor,cx=state.view.w/2,cy=state.view.h/2){if(!state.loaded)return;const v=state.view,old=v.scale,next=Math.max(.03,Math.min(16,old*factor));v.x=cx-(cx-v.x)*next/old;v.y=cy-(cy-v.y)*next/old;v.scale=next;draw();refreshControls();}
function draw(){
  const ratio=window.devicePixelRatio || 1,v=state.view;ctx.setTransform(ratio,0,0,ratio,0,0);ctx.fillStyle="#202b36";ctx.fillRect(0,0,v.w,v.h);if(!state.loaded)return;
  ctx.translate(v.x,v.y);ctx.scale(v.scale,v.scale);ctx.drawImage(state.image,0,0);if(state.hidden)return;
  state.doc.shapes.forEach((shape,index)=>{const selected=index===state.selected;const color=shape.flags?.uncertain?"#ef8b68":selected?"#ffc34d":"#4fc9be";ctx.beginPath();shape.points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=(selected?2.2:1.7)/v.scale;ctx.setLineDash(shape.flags?.uncertain?[6/v.scale,4/v.scale]:[]);ctx.stroke();ctx.setLineDash([]);
    if(selected && !locked() && state.tool==="edit")shape.points.forEach((p,i)=>{ctx.beginPath();ctx.arc(p[0],p[1],(i===state.vertex?6:4.5)/v.scale,0,Math.PI*2);ctx.fillStyle=i===state.vertex?color:"#213547";ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=1.5/v.scale;ctx.stroke();});});
  if(state.drawing.length){ctx.beginPath();state.drawing.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.strokeStyle="#7bddd2";ctx.lineWidth=2/v.scale;ctx.stroke();state.drawing.forEach(p=>{ctx.beginPath();ctx.arc(p[0],p[1],4.5/v.scale,0,Math.PI*2);ctx.fillStyle="#7bddd2";ctx.fill();});}
}
function position(event){const rect=canvas.getBoundingClientRect();return {x:event.clientX-rect.left,y:event.clientY-rect.top};}
function world(p){return [(p.x-state.view.x)/state.view.scale,(p.y-state.view.y)/state.view.scale];}
function bounded(p){return [Math.round(Math.max(0,Math.min(state.doc.imageWidth-1,p[0]))*1000)/1000,Math.round(Math.max(0,Math.min(state.doc.imageHeight-1,p[1]))*1000)/1000];}
function segmentPoint(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],length=dx*dx+dy*dy;const t=length?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/length)):0;return [a[0]+t*dx,a[1]+t*dy];}
function inside(p,points){let hit=false;for(let i=0,j=points.length-1;i<points.length;j=i++){const a=points[i],b=points[j];if((a[1]>p[1])!==(b[1]>p[1]) && p[0]<(b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0])hit=!hit;}return hit;}
function finish(){if(state.drawing.length<3 || state.busy || locked())return;remember();state.doc.shapes.push({label:$("label").value,points:copy(state.drawing),group_id:null,shape_type:"polygon",flags:{proposed:false,uncertain:false},description:""});state.selected=state.doc.shapes.length-1;setTool("edit");refreshShapes();}
canvas.addEventListener("pointerdown",event=>{
  if(!state.loaded || state.busy)return;canvas.focus();const p=position(event);const point=world(p);event.preventDefault();
  if(event.button===1 || state.space || state.tool==="pan"){state.drag={kind:"pan",start:p,x:state.view.x,y:state.view.y};}
  else if(!locked() && state.tool==="draw"){state.drawing.push(bounded(point));refreshControls();draw();return;}
  else if(!locked() && state.tool==="edit"){
    const shape=state.doc.shapes[state.selected];let vertex=-1;
    if(shape){let distance=9/state.view.scale;shape.points.forEach((v,i)=>{const d=Math.hypot(v[0]-point[0],v[1]-point[1]);if(d<distance){vertex=i;distance=d;}});
      if(vertex>=0){remember();state.vertex=vertex;state.drag={kind:"vertex",shape:state.selected,vertex};}
      else{let edge=-1,nearest=null,min=7/state.view.scale;shape.points.forEach((a,i)=>{const q=segmentPoint(point,a,shape.points[(i+1)%shape.points.length]);const d=Math.hypot(q[0]-point[0],q[1]-point[1]);if(d<min){min=d;edge=i;nearest=q;}});
        if(edge>=0){remember();shape.points.splice(edge+1,0,bounded(nearest));state.vertex=edge+1;state.drag={kind:"vertex",shape:state.selected,vertex:edge+1};refreshShapes();}}
    }
    if(!state.drag){for(let i=state.doc.shapes.length-1;i>=0;i--)if(inside(point,state.doc.shapes[i].points)){state.selected=i;state.vertex=null;refreshShapes();break;}}
  }
  if(state.drag){try{canvas.setPointerCapture(event.pointerId);}catch{}draw();}
});
canvas.addEventListener("pointermove",event=>{if(!state.drag)return;const p=position(event);if(state.drag.kind==="pan"){state.view.x=state.drag.x+p.x-state.drag.start.x;state.view.y=state.drag.y+p.y-state.drag.start.y;}else{state.doc.shapes[state.drag.shape].points[state.drag.vertex]=bounded(world(p));refreshControls();}draw();});
function release(){if(state.drag?.kind==="vertex")refreshShapes();state.drag=null;}
canvas.addEventListener("pointerup",release);canvas.addEventListener("pointercancel",release);canvas.addEventListener("contextmenu",event=>event.preventDefault());
canvas.addEventListener("wheel",event=>{event.preventDefault();const p=position(event);zoom(Math.exp(-event.deltaY*.0012),p.x,p.y);},{passive:false});
new ResizeObserver(()=>{const rect=$("viewport").getBoundingClientRect(),v=state.view;const cx=(v.w/2-v.x)/v.scale,cy=(v.h/2-v.y)/v.scale;v.w=rect.width;v.h=rect.height;canvas.width=Math.round(v.w*(window.devicePixelRatio||1));canvas.height=Math.round(v.h*(window.devicePixelRatio||1));v.x=v.w/2-cx*v.scale;v.y=v.h/2-cy*v.scale;draw();}).observe($("viewport"));
$("edit").onclick=()=>setTool("edit");$("pan").onclick=()=>setTool("pan");$("draw").onclick=()=>{if(locked())return;setTool("draw");$("label").value=state.doc.shapes.length?"secondary_crystal":"seed";};$("finish").onclick=finish;$("cancel-draw").onclick=()=>setTool("edit");
$("hide").onclick=()=>{state.hidden=!state.hidden;$("hide").textContent=state.hidden?"Show outlines":"Hide outlines";$("hide").setAttribute("aria-pressed",String(state.hidden));draw();};
$("focus-outline").onclick=focusOutline;$("fit-image").onclick=fitImage;$("native-size").onclick=()=>zoom(1/state.view.scale);$("zoom-in").onclick=()=>zoom(1.25);$("zoom-out").onclick=()=>zoom(.8);
$("undo").onclick=()=>{if(locked() || !state.history.length)return;clearAcknowledgment();state.doc=state.history.pop();state.vertex=null;populateFields();draw();};
$("discard").onclick=()=>{clearAcknowledgment();state.doc=copy(state.saved);state.history=[];setTool("edit");populateFields();message("Unsaved edits discarded.");};
$("remove-shape").onclick=()=>{if(locked() || !state.doc.shapes[state.selected])return;remember();state.doc.shapes.splice(state.selected,1);state.vertex=null;refreshShapes();};
$("label").onchange=()=>{if(locked() || state.tool==="draw" || !state.doc.shapes[state.selected])return;remember();state.doc.shapes[state.selected].label=$("label").value;refreshShapes();};
$("shape-uncertain").onchange=()=>{if(locked() || !state.doc.shapes[state.selected])return;remember();const shape=state.doc.shapes[state.selected];shape.flags={...shape.flags,uncertain:$("shape-uncertain").checked};refreshShapes();};
$("notes").oninput=updateFromInputs;$("image-uncertain").onchange=()=>{remember();updateFromInputs();};
["reviewer","ack","empty-reviewed"].forEach(id=>$(id).addEventListener(id==="reviewer"?"input":"change",refreshControls));
$("save").onclick=()=>action(saveDraft);$("previous").onclick=()=>action(()=>navigate(state.index-1));$("next").onclick=()=>action(()=>navigate(state.index+1));
$("review").onclick=()=>action(async()=>{if(dirty())await saveDraft();state.current=await api(`/api/sample/${state.index}/review`,{bundle_id:state.session.bundle_id,revision:state.current.revision,reviewer:$("reviewer").value,ack_image_reviewed:$("ack").checked,empty_reviewed:$("empty-reviewed").checked});state.doc=copy(state.current.annotation);state.saved=copy(state.doc);$("ack").checked=false;$("empty-reviewed").checked=false;await reloadSummary();populateFields();message("Review recorded for this image. It is eligible for the reviewed-label export.");});
$("revise").onclick=()=>action(async()=>{const result=await api(`/api/sample/${state.index}/revise`,{bundle_id:state.session.bundle_id,revision:state.current.revision});state.session={...result.summary,session_token:state.session.session_token};await load(state.index,result.sample);message("A new revision is open. The previous review remains saved in the earlier version.");});
$("export").onclick=()=>action(async()=>{const result=await api("/api/export",{bundle_id:state.session.bundle_id});message(`Exported ${result.reviewed_images} reviewed image(s) to ${result.output}. ${result.unreviewed_images} unreviewed image(s) were excluded.`);});
window.addEventListener("keydown",event=>{if(state.busy || ["INPUT","TEXTAREA","SELECT"].includes(document.activeElement.tagName))return;if(event.code==="Space"){state.space=true;event.preventDefault();}if(event.key==="Escape")setTool("edit");if(event.key==="Enter" && state.tool==="draw"){event.preventDefault();finish();}if(event.key==="Delete" && !locked() && state.vertex!==null){const shape=state.doc.shapes[state.selected];if(shape.points.length>3){remember();shape.points.splice(state.vertex,1);state.vertex=null;refreshShapes();}else message("An outline needs three points. Use Remove selected outline to remove the object.",true);event.preventDefault();}});
window.addEventListener("keyup",event=>{if(event.code==="Space")state.space=false;});window.addEventListener("blur",()=>{state.space=false;state.drag=null;});
window.addEventListener("beforeunload",event=>{if(dirty() || state.drawing.length){event.preventDefault();event.returnValue="";}});
action(async()=>{state.session=await api("/api/session");refreshSession();await load(0);message("");});
