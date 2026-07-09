"""
skaleoz — рентген-режим: регрессия 3 углов Кобба по X-ray снимку.

Данные: Spinal-AI2024 (github.com/Ernestchenchen/Spinal-AI2024) — 20k синтетически
сгенерированных (но реалистичных, калиброванных под реальные снимки) рентгенов
позвоночника с разметкой Cobb. Датасет без явной лицензии, "publicly released"
в статье CurvNet (arXiv:2411.12604) — используем как честную демонстрацию
пайплайна для отдельного режима «У меня есть рентген», не для клинической эксплуатации.

Формат GT: `<file>.jpg,angle1,angle2,angle3` (три сегмента Cobb, 0..90°).
"""
import argparse, os, copy, numpy as np, torch, torch.nn as nn
import cv2

H = W = 160  # рентген детальнее фото спины — берём чуть крупнее вход

def letterbox(im, size=W):
    h,w = im.shape[:2]; s = min(size/w, size/h); nw,nh = max(1,int(w*s)), max(1,int(h*s))
    out = np.full((size,size,3), 20, np.uint8)  # рентген тёмный фон — не серый, а почти чёрный
    r = cv2.resize(im, (nw,nh))
    out[(size-nh)//2:(size-nh)//2+nh, (size-nw)//2:(size-nw)//2+nw] = r
    return out

def load_split(img_dir, gt_file, limit=None):
    rows = []
    with open(gt_file, encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            fn=parts[0]; angs=[float(x) for x in parts[1:4]]
            rows.append((fn, angs))
    if limit: rows = rows[:limit]
    xs, ys = [], []
    for fn, angs in rows:
        p = os.path.join(img_dir, fn)
        if not os.path.exists(p): continue
        im = cv2.imread(p)
        if im is None: continue
        im = cv2.cvtColor(letterbox(im), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        xs.append(im.transpose(2,0,1)); ys.append(angs)
    if not xs: raise SystemExit(f"Нет изображений в {img_dir} с разметкой из {gt_file}")
    return torch.tensor(np.stack(xs)), torch.tensor(np.array(ys, dtype=np.float32))

class CobbNet(nn.Module):
    """Лёгкий CNN, три регрессионных выхода (proximal thoracic / main thoracic /
    thoracolumbar Cobb, как в разметке AASCE/Spinal-AI2024)."""
    def __init__(self):
        super().__init__()
        def blk(i,o): return nn.Sequential(
            nn.Conv2d(i,o,3,2,1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o,o,3,1,1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.f = nn.Sequential(blk(3,20), blk(20,40), blk(40,80), blk(80,120), blk(120,160))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(160, 3)
    def forward(self, x):
        f = self.pool(self.f(x)).flatten(1)
        return self.head(f)  # нормировано [0,1] снаружи ×90

def run(args):
    torch.manual_seed(0)
    base = os.path.dirname(__file__)
    Xtr, Ytr = load_split(os.path.join(base,'xray_data','img'), os.path.join(base,'xray_data','gt_train.txt'), args.limit)
    Xva, Yva = load_split(os.path.join(base,'xray_data','img_val'), os.path.join(base,'xray_data','gt_test.txt'), args.val_limit)
    print(f"train={len(Ytr)} val={len(Yva)}")
    net = CobbNet()
    print(f"Параметров: {sum(p.numel() for p in net.parameters())/1e6:.2f}M")
    opt = torch.optim.Adam(net.parameters(), 2e-4, weight_decay=1e-4)
    mse = nn.MSELoss()
    N = len(Ytr); best=1e9; best_state=copy.deepcopy(net.state_dict())
    Ytr_n, Yva_n = Ytr/90.0, Yva/90.0
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(N); tot=0
        for i in range(0, N, args.bs):
            idx = perm[i:i+args.bs]
            xb, yb = Xtr[idx], Ytr_n[idx]
            pred = net(xb)
            loss = mse(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
        net.eval()
        with torch.no_grad():
            pred = net(Xva)
            mae = (pred*90 - Yva).abs().mean().item()
            mae_per = (pred*90 - Yva).abs().mean(0).tolist()
        if mae < best: best = mae; best_state = copy.deepcopy(net.state_dict())
        print(f"эп {ep+1:02d}/{args.epochs}  loss {tot/N:.4f}  MAE {mae:5.2f}°  (по углам: {mae_per[0]:.1f}/{mae_per[1]:.1f}/{mae_per[2]:.1f})")
    net.load_state_dict(best_state); print(f"[best MAE {best:.2f}° loaded]")
    return net

def export(net, path):
    net.eval(); dummy = torch.zeros(1,3,H,W)
    torch.onnx.export(net, dummy, path, opset_version=17,
        input_names=["image"], output_names=["cobb"],
        dynamic_axes={"image":{0:"b"},"cobb":{0:"b"}})
    import onnxruntime as ort
    with torch.no_grad(): t_out = net(dummy)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o_out = sess.run(None, {"image": dummy.numpy()})[0]
    ok = np.allclose(t_out.numpy(), o_out, atol=1e-4)
    sz = os.path.getsize(path)/1e6
    print(f"ONNX сохранён: {path}  ({sz:.2f} МБ)  верификация torch==ort: {'OK' if ok else 'FAIL'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=900)
    ap.add_argument("--val-limit", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--out", default="model_xray.onnx")
    a = ap.parse_args()
    net = run(a)
    export(net, os.path.join(os.path.dirname(__file__), a.out))
