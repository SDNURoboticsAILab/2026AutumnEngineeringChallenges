#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零依赖本地标注工具（只用 Python 标准库，不需要 LabelImg / PyQt5）。

为什么用它：本机 Python 3.14 装不了 pip 版 LabelImg（依赖 PyQt5），
而你的项目正卡在“要给 obstacle 补框”这一步，所以这里用一个浏览器界面直接标。

启动（在 yolo_project 目录下）：
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe label_tool.py
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe label_tool.py --batch label_batches/batch_01.txt
    C:\\Users\\dell\\yolo-env\\Scripts\\python.exe label_tool.py --port 8899
然后浏览器打开 http://127.0.0.1:8888

操作：
    W 或鼠标直接拖拽     画一个框（当前类别见顶栏）
    1 / 2 / 3            切换类别：1=obstacle  2=cola  3=football
    点击框              选中；Delete / Backspace 删除选中
    D / →                下一张（自动保存）
    A / ←                上一张（自动保存）
    Ctrl+S               保存当前张
保存位置：dataset/labels/<split>/<图片名>.txt（YOLO 格式），
        同名文件首次覆盖前会自动备份为 .txt.bak。
"""
import argparse
import json
import shutil
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJ = Path(__file__).resolve().parent
DATASET = PROJ / 'dataset'
NAMES = {0: 'obstacle', 1: 'cola', 2: 'football'}
EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
DEFAULT_BATCH = PROJ / 'label_batches' / 'batch_01.txt'


def parse_batch(path):
    """batch 文件里是图片绝对路径，每行一个。"""
    items = []
    if path and Path(path).exists():
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            split = p.parent.name if p.parent.name in ('train', 'val') else 'train'
            items.append((split, p.name))
        return items
    for split in ('train', 'val'):
        d = DATASET / 'images' / split
        if d.exists():
            items += [(split, p.name) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return items


def read_boxes(split, name):
    lbl = DATASET / 'labels' / split / (Path(name).stem + '.txt')
    out = []
    if lbl.exists():
        for line in lbl.read_text(encoding='utf-8', errors='ignore').splitlines():
            parts = line.split()
            if len(parts) == 5:
                try:
                    out.append([int(float(parts[0]))] + [round(float(v), 6) for v in parts[1:]])
                except ValueError:
                    pass
    return out


def write_boxes(split, name, boxes):
    lbl = DATASET / 'labels' / split / (Path(name).stem + '.txt')
    lbl.parent.mkdir(parents=True, exist_ok=True)
    if lbl.exists():
        bak = lbl.with_name(lbl.stem + '.txt.bak')
        if not bak.exists():
            shutil.copy2(lbl, bak)
    lines = []
    for b in boxes:
        cls = int(b[0])
        x, y, w, h = [min(1.0, max(0.0, float(v))) for v in b[1:5]]
        w, h = max(0.0, min(w, 1.0)), max(0.0, min(h, 1.0))
        if w < 0.002 or h < 0.002:
            continue
        lines.append('%d %.6f %.6f %.6f %.6f' % (cls, x, y, w, h))
    lbl.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return len(lines)


PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>YOLO 标注工具</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,"Microsoft YaHei",sans-serif;background:#1c1c1e;color:#eaeaea;display:flex;height:100vh;overflow:hidden}
#left{flex:1;display:flex;flex-direction:column;min-width:0}
#bar{padding:8px 12px;background:#2c2c2e;display:flex;gap:10px;align-items:center;flex-wrap:wrap;font-size:14px}
#bar button{background:#3a3a3c;border:1px solid #555;color:#eaeaea;padding:5px 12px;border-radius:6px;cursor:pointer}
#bar button.active{background:#0a84ff;border-color:#0a84ff;font-weight:600}
#bar .sp{flex:1}
#wrap{flex:1;position:relative;overflow:auto;display:flex;align-items:center;justify-content:center;background:#111}
canvas{cursor:crosshair;display:block}
#hint{padding:6px 12px;background:#2c2c2e;font-size:12px;color:#9a9a9a}
#right{width:290px;background:#242426;overflow:auto;font-size:13px;border-left:1px solid #333}
#right h4{margin:0;padding:10px;font-size:13px;color:#8e8e93;border-bottom:1px solid #333}
.item{padding:6px 10px;border-bottom:1px solid #2f2f31;cursor:pointer;display:flex;gap:8px;align-items:center;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.item:hover{background:#2f2f31}
.item.cur{background:#0a84ff44}
.dot{width:9px;height:9px;border-radius:50%;background:#666;flex:0 0 9px}
.dot.done{background:#32d74b}
#stat{padding:8px 12px;font-size:13px;color:#32d74b;background:#2c2c2e}
</style></head><body>
<div id="left">
  <div id="bar">
    <span>类别：</span>
    <button data-c="0" class="active">1 obstacle</button>
    <button data-c="1">2 cola</button>
    <button data-c="2">3 football</button>
    <span class="sp"></span>
    <button id="prev">← 上一张 (A)</button>
    <button id="next">下一张 (D) →</button>
    <button id="save">保存 (Ctrl+S)</button>
    <span id="pos"></span>
  </div>
  <div id="wrap"><canvas id="cv"></canvas></div>
  <div id="hint">拖拽画框 · 1/2/3 切类别 · 点框选中 · Delete 删除 · A/D 翻页（自动保存）· Ctrl+S 保存</div>
  <div id="stat"></div>
</div>
<div id="right"><h4>本批清单（绿点=已有 obstacle 框）</h4><div id="list"></div></div>
<script>
let items=[],idx=0,boxes=[],cur=0,img=null,scale=1,offX=0,offY=0,sel=-1,drag=null;
const cv=document.getElementById('cv'),ctx=cv.getContext('2d');
const COLORS={0:'#ff453a',1:'#ffd60a',2:'#32d74b'};
const NAMES={0:'obstacle',1:'cola',2:'football'};
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}
async function load(){
  const r=await fetch('/api/list');items=await r.json();
  const i=items.findIndex(it=>!it.has0);idx=i<0?0:i;
}
function show(){
  const it=items[idx];if(!it)return;
  boxes=JSON.parse(JSON.stringify(it.boxes));sel=-1;
  img=new Image();
  img.onload=()=>{fit();draw();};
  img.src=it.url+'?t='+Date.now();
  document.getElementById('pos').textContent=(idx+1)+' / '+items.length;
  document.querySelectorAll('#list .item').forEach((e,i)=>e.classList.toggle('cur',i===idx));
  document.getElementById('stat').textContent='';
}
function fit(){
  const wrap=document.getElementById('wrap');
  const maxW=wrap.clientWidth-20,maxH=wrap.clientHeight-20;
  scale=Math.min(maxW/img.width,maxH/img.height,1);
  cv.width=img.width*scale;cv.height=img.height*scale;
}
function draw(){
  if(!img)return;
  ctx.clearRect(0,0,cv.width,cv.height);
  ctx.drawImage(img,0,0,cv.width,cv.height);
  ctx.lineWidth=2;ctx.font='13px system-ui';
  boxes.forEach((b,i)=>{
    const [c,x,y,w,h]=b;
    const X=(x-w/2)*cv.width,Y=(y-h/2)*cv.height,W=w*cv.width,H=h*cv.height;
    ctx.strokeStyle=COLORS[c]||'#fff';ctx.strokeRect(X,Y,W,H);
    const t=NAMES[c]||c;
    ctx.fillStyle=COLORS[c]||'#fff';ctx.fillText(t,X+2,Y+13);
    if(i===sel){ctx.lineWidth=3;ctx.strokeStyle='#fff';ctx.strokeRect(X-1,Y-1,W+2,H+2);ctx.lineWidth=2;}
  });
  if(drag){ctx.setLineDash([5,4]);ctx.strokeStyle='#0a84ff';ctx.strokeRect(drag.x,drag.y,drag.w,drag.h);ctx.setLineDash([]);}
}
function pos(e){const r=cv.getBoundingClientRect();return{x:clamp((e.clientX-r.left)/cv.width,0,1),y:clamp((e.clientY-r.top)/cv.height,0,1)};}
cv.addEventListener('mousedown',e=>{const p=pos(e);drag={x0:p.x,y0:p.y,x:p.x*0+ (p.x),y:p.y,w:0,h:0,px:e.clientX-r0(e),py:e.clientY-r0(e),cx:0,cy:0};
  const r=cv.getBoundingClientRect();drag.ax=e.clientX-r.left;drag.ay=e.clientY-r.top;drag.x=drag.ax;drag.y=drag.ay;draw();});
function r0(e){return {left:0,top:0};}
cv.addEventListener('mousemove',e=>{
  if(!drag)return;const r=cv.getBoundingClientRect();
  const cx=e.clientX-r.left,cy=e.clientY-r.top;
  drag.x=Math.min(drag.ax,cx);drag.y=Math.min(drag.ay,cy);drag.w=Math.abs(cx-drag.ax);drag.h=Math.abs(cy-drag.ay);draw();
});
cv.addEventListener('mouseup',e=>{
  if(!drag)return;const d=drag;drag=null;
  if(d.w>4&&d.h>4){
    const x=(d.x+d.w/2)/cv.width,y=(d.y+d.h/2)/cv.height,w=d.w/cv.width,h=d.h/cv.height;
    boxes.push([cur,+x.toFixed(6),+y.toFixed(6),+w.toFixed(6),+h.toFixed(6)]);
  }
  draw();autosave();
});
cv.addEventListener('click',e=>{
  const p=pos(e);sel=-1;
  for(let i=boxes.length-1;i>=0;i--){const [c,x,y,w,h]=boxes[i];
    if(Math.abs(p.x-x)<=w/2&&Math.abs(p.y-y)<=h/2){sel=i;break;}}
  draw();
});
async function save(){
  const it=items[idx];if(!it)return;
  const r=await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({split:it.split,name:it.name,boxes})});
  const j=await r.json();
  it.boxes=JSON.parse(JSON.stringify(boxes));
  it.has0=boxes.some(b=>b[0]===0);
  const row=document.querySelectorAll('#list .item')[idx];
  if(row)row.querySelector('.dot').className='dot'+(it.has0?' done':'');
  document.getElementById('stat').textContent='已保存 '+j.lines+' 个框 -> '+j.file;
  refreshStat();
}
function refreshStat(){
  const done=items.filter(i=>i.has0).length;
  document.getElementById('stat').textContent=(document.getElementById('stat').textContent||'')
    +'   |   进度 '+done+' / '+items.length;
}
let t=null;function autosave(){clearTimeout(t);t=setTimeout(save,400);}
document.addEventListener('keydown',e=>{
  if(e.ctrlKey&&e.key.toLowerCase()==='s'){e.preventDefault();save();return;}
  if(e.key==='d'||e.key==='ArrowRight'){e.preventDefault();save().then(()=>{idx=Math.min(idx+1,items.length-1);show();});return;}
  if(e.key==='a'||e.key==='ArrowLeft'){e.preventDefault();save().then(()=>{idx=Math.max(idx-1,0);show();});return;}
  if(e.key==='Delete'||e.key==='Backspace'){if(sel>=0){boxes.splice(sel,1);sel=-1;draw();autosave();}return;}
  if('123'.includes(e.key)){cur=+e.key-1;document.querySelectorAll('#bar button[data-c]').forEach(b=>b.classList.toggle('active',+b.dataset.c===cur));}
});
document.querySelectorAll('#bar button[data-c]').forEach(b=>b.onclick=()=>{cur=+b.dataset.c;
  document.querySelectorAll('#bar button[data-c]').forEach(x=>x.classList.toggle('active',x===b));});
document.getElementById('next').onclick=()=>save().then(()=>{idx=Math.min(idx+1,items.length-1);show();});
document.getElementById('prev').onclick=()=>save().then(()=>{idx=Math.max(idx-1,0);show();});
document.getElementById('save').onclick=save;
window.addEventListener('resize',()=>{fit();draw();});
function buildList(){
  const box=document.getElementById('list');box.innerHTML='';
  items.forEach((it,i)=>{const d=document.createElement('div');d.className='item';
    d.innerHTML='<span class="dot'+(it.has0?' done':'')+'"></span>'+it.name;
    d.onclick=()=>{save().then(()=>{idx=i;show();});};box.appendChild(d);});
}
(async()=>{await load();buildList();show();})();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    items = []

    def log_message(self, *args):
        pass

    def _send(self, body, ctype, code=200):
        if isinstance(body, str):
            body = body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == '/':
            return self._send(PAGE, 'text/html; charset=utf-8')
        if path == '/api/list':
            data = []
            for split, name in self.items:
                boxes = read_boxes(split, name)
                data.append({'split': split, 'name': name,
                             'url': '/img/%s/%s' % (split, name),
                             'boxes': boxes,
                             'has0': any(b[0] == 0 for b in boxes)})
            return self._send(json.dumps(data), 'application/json')
        if path.startswith('/img/'):
            _, _, split, name = path.split('/', 3)
            f = DATASET / 'images' / unquote(split) / unquote(name)
            if f.exists():
                data = f.read_bytes()
                return self._send(data, 'image/jpeg')
            return self._send('not found', 'text/plain', 404)
        return self._send('not found', 'text/plain', 404)

    def do_POST(self):
        if urlparse(self.path).path != '/api/save':
            return self._send('not found', 'text/plain', 404)
        length = int(self.headers.get('Content-Length', 0))
        payload = json.loads(self.rfile.read(length).decode('utf-8'))
        n = write_boxes(payload['split'], payload['name'], payload['boxes'])
        lbl = DATASET / 'labels' / payload['split'] / (Path(payload['name']).stem + '.txt')
        return self._send(json.dumps({'ok': True, 'lines': n, 'file': str(lbl)}),
                          'application/json')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default=str(DEFAULT_BATCH), help='batch 清单文件（图片绝对路径）')
    ap.add_argument('--port', type=int, default=8888)
    ap.add_argument('--host', default='127.0.0.1')
    args = ap.parse_args()

    Handler.items = parse_batch(args.batch)
    if not Handler.items:
        raise SystemExit('没有可标注的图片，请检查 --batch 或 dataset/images')
    done = sum(1 for s, n in Handler.items if any(b[0] == 0 for b in read_boxes(s, n)))
    print('标注批次: %s' % args.batch)
    print('图片数量: %d，其中已有 obstacle 框: %d' % (len(Handler.items), done))
    print('请在浏览器打开: http://%s:%d' % (args.host, args.port))
    print('按 Ctrl+C 结束。')
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print('已停止。')


if __name__ == '__main__':
    main()
