from flask import Flask, Response, render_template_string, jsonify, request
from ultralytics import YOLO
import cv2
import threading
import queue
import time
import requests
from requests.auth import HTTPDigestAuth

# ── Ayarlar ───────────────────────────────────────────────────────────────────
RTSP_URL     = "rtsp://admin:Bisavunma8503@192.168.1.108:554/live"
MODEL_PATH   = "runs/flir/yolo11s_flir/weights/best.pt"
IMGSZ        = 640
DISPLAY_SIZE = (1280, 720)

PERSON_CONF    = 0.50   # yolo11m F1 zirvesi ~0.55, 0.50 biraz daha geniş recall
CAR_CONF       = 0.35

MIN_PERSON_H   = 0.04   # frame yüksekliğinin %4'ünden kısa → gürültü
MIN_CAR_H      = 0.03
PERSON_AR_MIN  = 0.35   # aspect ratio min: h/w — yatık kutular insan değil
PERSON_AR_MAX  = 6.0    # aspect ratio max: h/w — çok ince de olmamalı
TEMPORAL_WIN   = 3      # son N frame'de en az 2'sinde görünmeli → confirmed

CAM_IP   = "192.168.1.108"
CAM_USER = "admin"
CAM_PASS = "Bisavunma8503"
CAM_AUTH = HTTPDigestAuth(CAM_USER, CAM_PASS)
# ─────────────────────────────────────────────────────────────────────────────

# FLIR modeli → araç tespiti
model_car    = YOLO(MODEL_PATH)

# COCO yolo11m → insan tespiti (20M param, yolo11s'ten çok daha iyi P/R dengesi)
model_person = YOLO("yolo11m.pt")

print(f"Araç modeli : {MODEL_PATH}")
print(f"İnsan modeli: yolo11m.pt (COCO, 20M param)")

app          = Flask(__name__)
output_frame = None
lock         = threading.Lock()
frame_queue  = queue.Queue(maxsize=1)
stats        = {"fps": 0.0, "person": 0, "car": 0, "total": 0}

# Temporal consistency: son TEMPORAL_WIN frame'deki ham person tespitleri
from collections import deque
person_history = deque(maxlen=TEMPORAL_WIN)  # her eleman: [(x1,y1,x2,y2), ...]

# ── Renk & çizim ──────────────────────────────────────────────────────────────
COLORS = {
    0: (57, 255, 20),    # person — neon yeşil
    2: (0, 165, 255),    # car    — turuncu
}

def draw_box(img, x1, y1, x2, y2, label, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>KASIRGA // Görüntü Analizi V 2.0</title>
  <link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&display=swap" rel="stylesheet">
  <style>
    :root { --cyan:#00f5ff; --pink:#ff006e; --dark:#020408; --panel:#050d14; --grid:rgba(0,245,255,0.04); }
    * { box-sizing:border-box; margin:0; padding:0; }
    body {
      background:var(--dark); color:var(--cyan);
      font-family:'Share Tech Mono',monospace; min-height:100vh;
      display:flex; flex-direction:column; align-items:center; padding:14px 16px;
      background-image: linear-gradient(var(--grid) 1px,transparent 1px),
                        linear-gradient(90deg,var(--grid) 1px,transparent 1px);
      background-size:40px 40px;
    }
    header {
      width:100%; max-width:1280px; display:flex; align-items:center;
      justify-content:space-between; padding:10px 0 12px;
      border-bottom:1px solid rgba(0,245,255,0.2); margin-bottom:12px; position:relative;
    }
    header::after {
      content:''; position:absolute; bottom:-1px; left:0;
      width:35%; height:1px; background:var(--pink); box-shadow:0 0 8px var(--pink);
    }
    .logo-top { font-family:'Orbitron',sans-serif; font-weight:900; font-size:1.5rem;
      letter-spacing:5px; color:var(--cyan); text-shadow:0 0 14px var(--cyan); }
    .logo-sub { font-size:0.62rem; letter-spacing:2px; color:rgba(0,245,255,0.4); text-transform:uppercase; }
    .badges { display:flex; gap:8px; align-items:center; }
    .badge {
      background:var(--panel); border:1px solid rgba(0,245,255,0.18);
      padding:6px 16px; font-size:0.78rem; letter-spacing:1px; color:rgba(0,245,255,0.55);
      clip-path:polygon(8px 0%,100% 0%,calc(100% - 8px) 100%,0% 100%);
    }
    .badge .lbl { display:block; font-size:0.5rem; letter-spacing:3px; color:rgba(0,245,255,0.28); margin-bottom:1px; }
    .badge.alert { border-color:var(--pink); color:var(--pink);
      box-shadow:0 0 16px rgba(255,0,110,0.35); animation:pulse 0.9s ease-in-out infinite; }
    .badge.alert .lbl { color:rgba(255,0,110,0.45); }
    @keyframes pulse { 0%,100%{box-shadow:0 0 16px rgba(255,0,110,0.35);} 50%{box-shadow:0 0 30px rgba(255,0,110,0.65);} }
    .feed-wrap {
      width:100%; max-width:1280px; position:relative;
      border:1px solid rgba(0,245,255,0.25); box-shadow:0 0 28px rgba(0,245,255,0.07);
    }
    .feed-wrap::before { content:''; position:absolute; top:-1px; left:-1px; width:16px; height:16px;
      border-top:2px solid var(--pink); border-left:2px solid var(--pink); z-index:2; }
    .feed-wrap::after  { content:''; position:absolute; bottom:-1px; right:-1px; width:16px; height:16px;
      border-bottom:2px solid var(--pink); border-right:2px solid var(--pink); z-index:2; }
    .corner { position:absolute; width:16px; height:16px; z-index:2; }
    .corner.tr { top:-1px; right:-1px; border-top:2px solid var(--pink); border-right:2px solid var(--pink); }
    .corner.bl { bottom:-1px; left:-1px; border-bottom:2px solid var(--pink); border-left:2px solid var(--pink); }
    .feed-wrap img { width:100%; display:block; }
    .cam-label { position:absolute; top:8px; left:10px; font-size:0.55rem;
      letter-spacing:3px; color:rgba(0,245,255,0.35); z-index:3; }
    .ts { position:absolute; top:8px; right:10px; font-size:0.55rem;
      letter-spacing:2px; color:rgba(0,245,255,0.3); z-index:3; }
    .ctrl-panel {
      width:100%; max-width:1280px; margin-top:10px;
      display:grid; grid-template-columns:auto auto 1fr auto; gap:10px; align-items:start;
    }
    .ctrl-section {
      background:var(--panel); border:1px solid rgba(0,245,255,0.12);
      padding:14px 12px 12px; position:relative;
    }
    .ctrl-section::before {
      content:attr(data-label); position:absolute; top:0; left:10px;
      font-size:0.44rem; letter-spacing:3px; color:rgba(0,245,255,0.35);
      background:var(--panel); padding:0 4px; transform:translateY(-50%);
    }
    .ptz-grid { display:grid; grid-template-columns:repeat(3,46px); grid-template-rows:repeat(3,46px); gap:4px; }
    .ptz-btn {
      background:rgba(0,245,255,0.03); border:1px solid rgba(0,245,255,0.18);
      color:var(--cyan); font-size:1.15rem; cursor:pointer;
      display:flex; align-items:center; justify-content:center;
      user-select:none; transition:background 0.08s,border-color 0.08s;
      -webkit-tap-highlight-color:transparent;
    }
    .ptz-btn:active,.ptz-btn.pressed {
      background:rgba(0,245,255,0.14); border-color:var(--cyan);
      box-shadow:0 0 12px rgba(0,245,255,0.35);
    }
    .ptz-center { background:rgba(255,0,110,0.06); border-color:rgba(255,0,110,0.28);
      color:var(--pink); font-size:0.55rem; letter-spacing:1px; }
    .ptz-status { margin-top:6px; font-size:0.48rem; letter-spacing:2px;
      color:rgba(0,245,255,0.3); min-height:12px; text-align:center; }
    .zoom-col { display:flex; flex-direction:column; gap:4px; }
    .zoom-btn {
      background:rgba(0,245,255,0.03); border:1px solid rgba(0,245,255,0.18);
      color:rgba(0,245,255,0.7); font-family:'Share Tech Mono',monospace;
      font-size:0.62rem; letter-spacing:1px; padding:10px 14px; cursor:pointer;
      user-select:none; white-space:nowrap; -webkit-tap-highlight-color:transparent;
    }
    .zoom-btn:active,.zoom-btn.pressed { background:rgba(0,245,255,0.12);
      border-color:var(--cyan); box-shadow:0 0 10px rgba(0,245,255,0.3); color:var(--cyan); }
    .speed-block { display:flex; flex-direction:column; gap:10px; }
    .ctrl-label { font-size:0.46rem; letter-spacing:2.5px; color:rgba(0,245,255,0.35); margin-bottom:3px; display:block; }
    input[type=range] { -webkit-appearance:none; width:100%; height:2px; background:rgba(0,245,255,0.18); outline:none; }
    input[type=range]::-webkit-slider-thumb { -webkit-appearance:none; width:13px; height:13px;
      background:var(--cyan); cursor:pointer; box-shadow:0 0 8px var(--cyan); }
    .speed-val { font-size:0.75rem; color:var(--cyan); margin-left:6px; }
    .preset-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; }
    .preset-btn {
      background:rgba(0,245,255,0.03); border:1px solid rgba(0,245,255,0.15);
      color:rgba(0,245,255,0.5); font-family:'Share Tech Mono',monospace;
      font-size:0.58rem; letter-spacing:1px; padding:6px 4px; cursor:pointer; text-align:center;
    }
    .preset-btn:hover { background:rgba(0,245,255,0.08); border-color:rgba(0,245,255,0.38); color:var(--cyan); }
    .action-col { display:flex; flex-direction:column; gap:5px; }
    .act-btn {
      background:rgba(0,245,255,0.03); border:1px solid rgba(0,245,255,0.18);
      color:rgba(0,245,255,0.65); font-family:'Share Tech Mono',monospace;
      font-size:0.6rem; letter-spacing:1.5px; padding:9px 14px; cursor:pointer;
      clip-path:polygon(6px 0%,100% 0%,calc(100% - 6px) 100%,0% 100%);
      text-align:center; white-space:nowrap;
    }
    .act-btn:hover { background:rgba(0,245,255,0.09); border-color:rgba(0,245,255,0.45);
      box-shadow:0 0 12px rgba(0,245,255,0.18); color:var(--cyan); }
    .act-btn.snap { border-color:rgba(255,0,110,0.35); color:rgba(255,0,110,0.7); }
    .act-btn.snap:hover { background:rgba(255,0,110,0.09); border-color:var(--pink);
      box-shadow:0 0 12px rgba(255,0,110,0.25); color:var(--pink); }
    footer { margin-top:10px; width:100%; max-width:1280px; display:flex;
      justify-content:space-between; align-items:center;
      font-size:0.58rem; letter-spacing:1.5px; color:rgba(0,245,255,0.22); }
    .dot { display:inline-block; width:6px; height:6px; border-radius:50%;
      background:#00ff88; box-shadow:0 0 6px #00ff88; margin-right:5px;
      animation:blink 2s ease-in-out infinite; }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.15} }
  </style>
  <script>
    function pad(n){ return String(n).padStart(2,'0'); }
    function tick(){
      var d=new Date();
      document.getElementById('ts').textContent=
        pad(d.getDate())+'.'+pad(d.getMonth()+1)+'.'+d.getFullYear()+
        '  '+pad(d.getHours())+':'+pad(d.getMinutes())+':'+pad(d.getSeconds());
    }
    document.addEventListener('DOMContentLoaded',function(){ tick(); setInterval(tick,1000); });

    setInterval(()=>{
      fetch('/stats').then(r=>r.json()).then(d=>{
        document.getElementById('fps').textContent=d.fps.toFixed(1);
        var pb=document.getElementById('person-badge');
        document.getElementById('person-cnt').textContent=d.person;
        pb.className='badge'+(d.person>0?' alert':'');
        document.getElementById('car-cnt').textContent=d.car;
        document.getElementById('total').textContent=d.total;
      });
    },800);

    var activeCode=null;
    function spd(){ return document.getElementById('spd').value; }
    function ptzCall(action,code,extra){
      var url='/api/ptz?action='+action+'&code='+code+'&speed='+spd();
      if(extra) url+=extra;
      fetch(url).catch(()=>{});
      document.getElementById('ptz-status').textContent=action.toUpperCase()+' ▶ '+code;
    }
    function ptzDown(btn,code){ activeCode=code; btn.classList.add('pressed'); ptzCall('start',code); }
    function ptzUp(btn,code){ btn.classList.remove('pressed'); if(activeCode===code){ activeCode=null; ptzCall('stop',code); } }
    function ptzStopAll(){
      if(activeCode){ ptzCall('stop',activeCode); activeCode=null; }
      document.querySelectorAll('.ptz-btn,.zoom-btn').forEach(b=>b.classList.remove('pressed'));
      document.getElementById('ptz-status').textContent='STOP';
    }
    function gotoPreset(n){
      fetch('/api/ptz?action=start&code=GotoPreset&preset='+n+'&speed='+spd()).catch(()=>{});
      document.getElementById('ptz-status').textContent='GOTO PRESET '+n;
    }
    function takeSnapshot(){
      var a=document.createElement('a'); a.href='/api/snapshot';
      a.download='snap_'+Date.now()+'.jpg'; a.click();
    }
  </script>
</head>
<body>
  <header>
    <div class="logo">
      <div class="logo-top">KASIRGA</div>
      <div class="logo-sub">Kara Sistemleri İzleme ve Raporlama Görüntü Analizi V 2.0</div>
    </div>
    <div class="badges">
      <div class="badge"><span class="lbl">FRAME RATE</span><span id="fps">--</span> FPS</div>
      <div class="badge" id="person-badge"><span class="lbl">PERSONS</span><span id="person-cnt">--</span> DETECTED</div>
      <div class="badge"><span class="lbl">VEHICLES</span><span id="car-cnt">--</span> DETECTED</div>
      <div class="badge"><span class="lbl">TOTAL LOG</span><span id="total">--</span> HITS</div>
    </div>
  </header>

  <div class="feed-wrap">
    <div class="corner tr"></div><div class="corner bl"></div>
    <div class="cam-label">LIVE // CAM-01 // SURVEILLANCE ACTIVE</div>
    <div class="ts" id="ts"></div>
    <img src="/video_feed" alt="live feed">
  </div>

  <div class="ctrl-panel">
    <div class="ctrl-section" data-label="PAN / TILT">
      <div class="ptz-grid">
        <button class="ptz-btn" onmousedown="ptzDown(this,'LeftUp')"   onmouseup="ptzUp(this,'LeftUp')"   ontouchstart="ptzDown(this,'LeftUp')"   ontouchend="ptzUp(this,'LeftUp')">&#8598;</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'Up')"       onmouseup="ptzUp(this,'Up')"       ontouchstart="ptzDown(this,'Up')"       ontouchend="ptzUp(this,'Up')">&#8593;</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'RightUp')"  onmouseup="ptzUp(this,'RightUp')"  ontouchstart="ptzDown(this,'RightUp')"  ontouchend="ptzUp(this,'RightUp')">&#8599;</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'Left')"     onmouseup="ptzUp(this,'Left')"     ontouchstart="ptzDown(this,'Left')"     ontouchend="ptzUp(this,'Left')">&#8592;</button>
        <button class="ptz-btn ptz-center" onclick="ptzStopAll()">STOP</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'Right')"    onmouseup="ptzUp(this,'Right')"    ontouchstart="ptzDown(this,'Right')"    ontouchend="ptzUp(this,'Right')">&#8594;</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'LeftDown')" onmouseup="ptzUp(this,'LeftDown')" ontouchstart="ptzDown(this,'LeftDown')" ontouchend="ptzUp(this,'LeftDown')">&#8601;</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'Down')"     onmouseup="ptzUp(this,'Down')"     ontouchstart="ptzDown(this,'Down')"     ontouchend="ptzUp(this,'Down')">&#8595;</button>
        <button class="ptz-btn" onmousedown="ptzDown(this,'RightDown')" onmouseup="ptzUp(this,'RightDown')" ontouchstart="ptzDown(this,'RightDown')" ontouchend="ptzUp(this,'RightDown')">&#8600;</button>
      </div>
      <div class="ptz-status" id="ptz-status">STANDBY</div>
    </div>

    <div class="ctrl-section" data-label="ZOOM / FOCUS">
      <div class="zoom-col">
        <button class="zoom-btn" onmousedown="ptzDown(this,'ZoomTele')"  onmouseup="ptzUp(this,'ZoomTele')"  ontouchstart="ptzDown(this,'ZoomTele')"  ontouchend="ptzUp(this,'ZoomTele')">[+] TELE</button>
        <button class="zoom-btn" onmousedown="ptzDown(this,'ZoomWide')"  onmouseup="ptzUp(this,'ZoomWide')"  ontouchstart="ptzDown(this,'ZoomWide')"  ontouchend="ptzUp(this,'ZoomWide')">[-] WIDE</button>
        <button class="zoom-btn" onmousedown="ptzDown(this,'FocusNear')" onmouseup="ptzUp(this,'FocusNear')" ontouchstart="ptzDown(this,'FocusNear')" ontouchend="ptzUp(this,'FocusNear')">[&#9685;] NEAR</button>
        <button class="zoom-btn" onmousedown="ptzDown(this,'FocusFar')"  onmouseup="ptzUp(this,'FocusFar')"  ontouchstart="ptzDown(this,'FocusFar')"  ontouchend="ptzUp(this,'FocusFar')">[&#8734;] FAR</button>
      </div>
    </div>

    <div class="ctrl-section" data-label="SPEED / PRESETS">
      <div class="speed-block">
        <div>
          <span class="ctrl-label">PTZ SPEED <span class="speed-val" id="spd-val">4</span> / 8</span>
          <input type="range" id="spd" min="1" max="8" value="4"
                 oninput="document.getElementById('spd-val').textContent=this.value">
        </div>
        <div>
          <span class="ctrl-label">GOTO PRESET</span>
          <div class="preset-grid">
            <button class="preset-btn" onclick="gotoPreset(1)">PRE-1</button>
            <button class="preset-btn" onclick="gotoPreset(2)">PRE-2</button>
            <button class="preset-btn" onclick="gotoPreset(3)">PRE-3</button>
            <button class="preset-btn" onclick="gotoPreset(4)">PRE-4</button>
            <button class="preset-btn" onclick="gotoPreset(5)">PRE-5</button>
            <button class="preset-btn" onclick="gotoPreset(6)">PRE-6</button>
          </div>
        </div>
      </div>
    </div>

    <div class="ctrl-section" data-label="ACTIONS">
      <div class="action-col">
        <button class="act-btn snap" onclick="takeSnapshot()">&#9737; SNAPSHOT</button>
        <button class="act-btn" onclick="ptzCall('start','AutoFocus','')">AF AUTO FOCUS</button>
        <button class="act-btn" onclick="ptzCall('start','IrisAuto','')">IR AUTO IRIS</button>
      </div>
    </div>
  </div>

  <footer>
    <span><span class="dot"></span>SYSTEM ONLINE</span>
    <span>PERSON: YOLO11M-COCO(""" + str(PERSON_CONF) + """) &nbsp;|&nbsp; CAR: YOLO11S-FLIR(""" + str(CAR_CONF) + """)</span>
    <span>localhost:5000</span>
  </footer>
</body>
</html>"""


# ── Capture loop ───────────────────────────────────────────────────────────────
def capture_loop():
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("HATA: Kameraya bağlanılamadı!")
        return
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Kamera bağlandı: {w}x{h}")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            pass
        frame_queue.put(frame)


# ── Inference loop ─────────────────────────────────────────────────────────────
def inference_loop():
    global output_frame
    prev_time = time.time()

    while True:
        frame = frame_queue.get()
        orig_h, orig_w = frame.shape[:2]
        dw, dh = DISPLAY_SIZE
        fh = orig_h

        display = cv2.resize(frame, DISPLAY_SIZE)
        sx, sy  = dw / orig_w, dh / orig_h
        n_person = n_car = 0

        # ── İnsan: yolo11m COCO + 3 katmanlı filtre ─────────────────────────
        res_person = model_person(frame, conf=PERSON_CONF, classes=[0],
                                  verbose=False, imgsz=IMGSZ, iou=0.45)

        raw_persons = []  # bu frame'deki ham geçerli tespitler
        if res_person[0].boxes is not None:
            for row in res_person[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, _ = row
                bh = y2 - y1
                bw = x2 - x1
                if bh / fh < MIN_PERSON_H:          continue  # çok küçük
                ar = bh / max(bw, 1)
                if ar < PERSON_AR_MIN or ar > PERSON_AR_MAX: continue  # yatık/çok ince kutu
                raw_persons.append((x1, y1, x2, y2, conf))

        # Temporal consistency: bu frame'deki her tespiti geçmiş frame'lerle karşılaştır
        person_history.append([(x1,y1,x2,y2) for x1,y1,x2,y2,_ in raw_persons])

        def iou(a, b):
            ix1,iy1 = max(a[0],b[0]), max(a[1],b[1])
            ix2,iy2 = min(a[2],b[2]), min(a[3],b[3])
            inter = max(0,ix2-ix1)*max(0,iy2-iy1)
            ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
            return inter / max(ua+ub-inter, 1)

        confirmed_persons = []
        for det in raw_persons:
            box = det[:4]
            # Geçmiş frame'lerin en az birinde IoU>0.3 ile eşleşmeli
            confirmed = False
            for past_frame in list(person_history)[:-1]:  # son frame hariç
                if any(iou(box, pb) > 0.30 for pb in past_frame):
                    confirmed = True
                    break
            if confirmed or len(person_history) < TEMPORAL_WIN:
                confirmed_persons.append(det)

        for (x1, y1, x2, y2, conf) in confirmed_persons:
            n_person += 1
            draw_box(display, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                     f"person {conf:.2f}", COLORS[0])

        # ── Araç: FLIR özel model (araç tespitinde daha spesifik) ────────────
        res_car = model_car(frame, conf=CAR_CONF, classes=[2],
                            verbose=False, imgsz=IMGSZ)
        if res_car[0].boxes is not None:
            for row in res_car[0].boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, _ = row
                if (y2 - y1) / fh < MIN_CAR_H:
                    continue
                n_car += 1
                draw_box(display, int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy),
                         f"car {conf:.2f}", COLORS[2])

        now      = time.time()
        fps      = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        stats["fps"]    = round(fps, 1)
        stats["person"] = n_person
        stats["car"]    = n_car
        stats["total"] += n_person + n_car

        if n_person or n_car:
            print(f"[{fps:.1f} FPS] person:{n_person}  car:{n_car}")

        with lock:
            output_frame = display


# ── Flask routes ───────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    def gen():
        while True:
            with lock:
                if output_frame is None:
                    time.sleep(0.01)
                    continue
                frame_copy = output_frame
            _, buf = cv2.imencode(".jpg", frame_copy, [cv2.IMWRITE_JPEG_QUALITY, 78])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
            time.sleep(0.02)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stats")
def stats_api():
    return jsonify(stats)


@app.route("/api/ptz")
def api_ptz():
    action = request.args.get("action", "start")
    code   = request.args.get("code",   "Up")
    speed  = request.args.get("speed",  "4")
    preset = request.args.get("preset", "1")
    arg1   = preset if code == "GotoPreset" else "0"
    url    = (f"http://{CAM_IP}/cgi-bin/ptz.cgi"
              f"?action={action}&channel=1&code={code}"
              f"&arg1={arg1}&arg2={speed}&arg3=0")
    try:
        return requests.get(url, auth=CAM_AUTH, timeout=3).text
    except Exception as e:
        return str(e), 500


@app.route("/api/snapshot")
def api_snapshot():
    try:
        r = requests.get(f"http://{CAM_IP}/cgi-bin/snapshot.cgi",
                         auth=CAM_AUTH, timeout=5, stream=True)
        return Response(
            r.iter_content(chunk_size=8192),
            content_type=r.headers.get("Content-Type", "image/jpeg"),
            headers={"Content-Disposition": "attachment; filename=snapshot.jpg"}
        )
    except Exception as e:
        return str(e), 500


# ── Başlat ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    threading.Thread(target=capture_loop,  daemon=True).start()
    threading.Thread(target=inference_loop, daemon=True).start()
    print(f"\n{'='*50}")
    print(f"  Tarayıcıda aç: http://localhost:5000")
    print(f"{'='*50}\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
