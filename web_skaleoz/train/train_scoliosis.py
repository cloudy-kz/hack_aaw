"""
skaleoz — обучаемый пайплайн скрининга сколиоза (вдохновлён MGScoliosis).

Выход в стиле MGScoliosis: joint ordinal regression степени + регрессия угла.
Лёгкий бэкбон (чистый torch, без torchvision) → экспорт ONNX для onnxruntime-web.

ВАЖНО (честно): по умолчанию обучается на СИНТЕТИКЕ (нет публичного
размеченного датасета фото спины + Cobb). Синтетика доказывает, что пайплайн
работает end-to-end и даёт браузерный .onnx — но НЕ имеет клинической ценности.
Для реальной модели: положите реальные фото в data/ (имя файла кодирует угол,
как в MGScoliosis: `*-<angle>-*.jpg`) и запустите с --data data/.
"""
import argparse, os, math, numpy as np, torch, torch.nn as nn
import cv2

H = W = 128
THRESHOLDS = [10, 25, 40]          # границы степеней: норма<10<лёгкая<25<средняя<40<тяжёлая
K = len(THRESHOLDS) + 1            # 4 класса степени

# ─────────────────────────── синтетические данные ───────────────────────────
def draw_back(angle, rng):
    """Спина со сколиотической кривой ~угол. С вариативностью: масштаб (человек не
    целиком), сдвиг/кроп, боковое освещение, поворот камеры, флип, яркость, шум —
    чтобы модель держала разные ракурсы/свет/кадрирование."""
    base = rng.uniform(0.82, 0.96)
    img = np.full((H, W, 3), base, np.float32)
    # боковой градиент освещения
    gx = np.linspace(rng.uniform(-0.12, 0.02), rng.uniform(-0.02, 0.12), W)[None, :, None]
    img = np.clip(img + gx, 0, 1)
    sc = rng.uniform(0.55, 1.05)                 # размер человека в кадре
    cx = W/2 + rng.uniform(-16, 16)
    top = 22*sc + rng.uniform(-34, 18)           # вертикальный сдвиг → частичный кадр
    bot = top + (H-40)*sc + rng.uniform(-8, 26)
    sh = rng.uniform(0.70, 0.86)
    torso = np.array([[cx-34*sc,top+10],[cx-30*sc,bot],[cx+30*sc,bot],[cx+34*sc,top+10]], np.int32)
    cv2.fillConvexPoly(img, torso, (sh-0.05, sh, sh-0.05))
    cv2.circle(img, (int(cx), int(top-8*sc)), max(3,int(12*sc)), (sh+0.03,sh+0.05,sh+0.03), -1, cv2.LINE_AA)
    span = max(1, bot-top-18)
    amp = angle/50.0*15.0 + rng.normal(0, 0.6)   # угол масштаб-инвариантен (не *sc)
    ys = np.linspace(top+12, bot-6, 24)
    xs = cx + amp*np.sin((ys-top-12)/span*math.pi*1.4) + rng.normal(0, 0.5, ys.shape)
    cv2.polylines(img, [np.stack([xs,ys],1).astype(np.int32)], False, (0.35,0.4,0.55), max(1,int(2*sc)), cv2.LINE_AA)
    tilt = angle/50.0*10 + rng.normal(0, 1.2); dy = math.tan(math.radians(tilt))*24*sc
    cv2.line(img,(int(cx-24*sc),int(top+22*sc-dy)),(int(cx+24*sc),int(top+22*sc+dy)),(0.2,0.6,0.5),2,cv2.LINE_AA)
    cv2.line(img,(int(cx-20*sc),int(bot-14*sc+dy*0.6)),(int(cx+20*sc),int(bot-14*sc-dy*0.6)),(0.2,0.5,0.7),2,cv2.LINE_AA)
    if rng.random() < 0.5: img = np.ascontiguousarray(img[:, ::-1])   # флип (магнитуда угла та же)
    rot = rng.uniform(-6, 6)                                          # поворот камеры
    M = cv2.getRotationMatrix2D((W/2, H/2), rot, 1.0)
    img = cv2.warpAffine(img, M, (W, H), borderValue=(float(base),)*3)
    img = np.clip(img*rng.uniform(0.75, 1.22) + rng.uniform(-0.08, 0.08), 0, 1)  # яркость/контраст
    img += rng.normal(0, 0.025, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)

def make_synth(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n,3,H,W), np.float32); A = np.zeros(n, np.float32)
    for i in range(n):
        a = rng.random()*50
        X[i] = draw_back(a, rng).transpose(2,0,1); A[i] = a
    return torch.from_numpy(X), torch.from_numpy(A)

# ─────────────────────────── реальные данные (если есть) ───────────────────────────
def letterbox(im):
    """Вписать в WxH с сохранением пропорций (серые поля) — устойчиво к разным
    размерам/кадрированию. Тот же препроцесс, что в браузере (onnxruntime-web)."""
    h,w = im.shape[:2]; s = min(W/w, H/h); nw,nh = int(w*s), int(h*s)
    out = np.full((H,W,3), 127, np.uint8)
    r = cv2.resize(im, (nw,nh))
    out[(H-nh)//2:(H-nh)//2+nh, (W-nw)//2:(W-nw)//2+nw] = r
    return out

def load_real(folder):
    import glob, re
    xs, ang = [], []
    for p in glob.glob(os.path.join(folder, "**", "*.*"), recursive=True):
        if not p.lower().endswith((".jpg",".jpeg",".png")): continue
        m = re.search(r"-(\d+(?:\.\d+)?)-", os.path.basename(p))  # ...-<angle>-...
        if not m: continue
        im = cv2.imread(p)
        if im is None: continue
        im = cv2.cvtColor(letterbox(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        xs.append(im.transpose(2,0,1)); ang.append(float(m.group(1)))
    if not xs: raise SystemExit(f"В {folder} нет размеченных фото (имя вида *-18-*.jpg)")
    return torch.tensor(np.stack(xs)), torch.tensor(ang, dtype=torch.float32)

# ─────────────────────────── модель ───────────────────────────
class Net(nn.Module):
    """Лёгкий CNN + две головы: ordinal степень (K-1 порогов) + угол."""
    def __init__(self, k=K):
        super().__init__()
        def blk(i,o): return nn.Sequential(
            nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o,o,3,1,1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(3,16), blk(16,32), blk(32,64), blk(64,96))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.ord = nn.Linear(96, k-1)   # кумулятивные пороги (ordinal regression)
        self.ang = nn.Linear(96, 1)     # угол
    def forward(self, x):
        f = self.pool(self.f(x)).flatten(1)
        return self.ord(f), self.ang(f).squeeze(-1)

def ordinal_targets(angle):
    return torch.stack([(angle >= t).float() for t in THRESHOLDS], 1)  # [N, K-1]

def severity_from_ord(logits):
    return (torch.sigmoid(logits) > 0.5).sum(1)  # 0..K-1

# ─────────────────────────── обучение ───────────────────────────
def run(args):
    torch.manual_seed(0)
    if args.data:
        X, A = load_real(args.data); print(f"Реальные данные: {len(A)} фото")
        n_val = max(1, len(A)//5); Xtr,Atr,Xva,Ava = X[n_val:],A[n_val:],X[:n_val],A[:n_val]
    else:
        print("СИНТЕТИКА (нет реального датасета) — пайплайн-пруф, не клиника")
        Xtr, Atr = make_synth(args.n, 1); Xva, Ava = make_synth(args.n//4, 99)
    dev = "cpu"
    net = Net().to(dev)
    print(f"Параметров: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    import copy
    opt = torch.optim.Adam(net.parameters(), 1.5e-4, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(); mse = nn.MSELoss()
    N = len(Atr); best=-1; best_state=copy.deepcopy(net.state_dict())
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(N); tot=0
        for i in range(0, N, args.bs):
            idx = perm[i:i+args.bs]
            xb, ab = Xtr[idx].to(dev), Atr[idx].to(dev)
            ol, al = net(xb)
            loss = bce(ol, ordinal_targets(ab)) + 1.0*mse(al, ab/50.0)   # угол нормирован в [0,1]
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
        # eval
        net.eval()
        with torch.no_grad():
            ol, al = net(Xva.to(dev))
            mae = (al.cpu()*50.0-Ava).abs().mean().item()   # обратно из [0,1] в градусы
            sev_pred = severity_from_ord(ol).cpu()
            sev_true = ordinal_targets(Ava).sum(1).long()
            acc = (sev_pred==sev_true).float().mean().item()
        # чекпоинт по (acc - штраф за MAE): берём стабильно лучший
        sc_metric = acc - mae/100.0
        if sc_metric > best: best = sc_metric; best_state = copy.deepcopy(net.state_dict())
        print(f"эп {ep+1:02d}/{args.epochs}  loss {tot/N:.3f}  угол MAE {mae:4.1f}°  степень acc {acc*100:4.1f}%")
    net.load_state_dict(best_state); print("[best checkpoint loaded]")
    return net

# ─────────────────────────── экспорт ONNX ───────────────────────────
def export(net, path):
    net.eval(); dummy = torch.zeros(1,3,H,W)
    torch.onnx.export(net, dummy, path, opset_version=17,
        input_names=["image"], output_names=["ordinal","angle"],
        dynamic_axes={"image":{0:"b"},"ordinal":{0:"b"},"angle":{0:"b"}})
    # верификация: torch vs onnxruntime
    import onnxruntime as ort
    with torch.no_grad(): to,ta = net(dummy)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    oo, oa = sess.run(None, {"image": dummy.numpy()})
    ok = np.allclose(to.numpy(), oo, atol=1e-4) and np.allclose(ta.numpy(), oa, atol=1e-4)
    sz = os.path.getsize(path)/1e6
    print(f"ONNX сохранён: {path}  ({sz:.2f} МБ)  верификация torch==ort: {'OK' if ok else 'FAIL'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="папка с реальными фото (имя *-<angle>-*.jpg)")
    ap.add_argument("--n", type=int, default=600, help="синтетических train-примеров")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--out", default="model.onnx")
    a = ap.parse_args()
    net = run(a)
    export(net, os.path.join(os.path.dirname(__file__), a.out))
