"""BiLSTM (category + priority) and autoencoder (anomaly) -> outputs/predictions.csv"""
import re, json, numpy as np, pandas as pd, torch, torch.nn as nn
from collections import Counter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
torch.manual_seed(0); np.random.seed(0)

df = pd.read_csv("data/tickets.csv")
tr, va, te = (df.split == s for s in ("train", "val", "test"))
CATS = sorted(df.category.unique()); PRIO = ["Low", "Medium", "High", "Critical"]
tok = lambda s: re.findall(r"[a-z]+", s.lower())

# vocab from TRAIN only (no leakage)
cnt = Counter(w for t in df[tr].description for w in tok(t))
vocab = {w: i + 2 for i, w in enumerate([w for w, c in cnt.items() if c >= 2])}
MAXLEN = 20
def enc(t):
    ids = [vocab.get(w, 1) for w in tok(t)][:MAXLEN]
    return ids + [0] * (MAXLEN - len(ids))
X = torch.tensor([enc(t) for t in df.description])
yc = torch.tensor(df.category.map(CATS.index).values); yp = torch.tensor(df.priority.map(PRIO.index).values)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(len(vocab) + 2, 48, padding_idx=0)
        s.lstm = nn.LSTM(48, 48, batch_first=True, bidirectional=True)
        s.drop = nn.Dropout(0.5)
        s.hc = nn.Linear(96, len(CATS)); s.hp = nn.Linear(96, 4)
    def forward(s, x):
        h, _ = s.lstm(s.drop(s.emb(x)))
        h = s.drop(h.max(1).values)
        return s.hc(h), s.hp(h)

net = Net(); opt = torch.optim.Adam(net.parameters(), 2e-3, weight_decay=1e-3); ce = nn.CrossEntropyLoss()
idx = {k: torch.tensor(np.where(m)[0]) for k, m in (("tr", tr), ("va", va), ("te", te))}
def loss_on(i):
    a, b = net(X[i]); return ce(a, yc[i]) + ce(b, yp[i])
best, bad, state = 9e9, 0, None
for ep in range(40):
    net.train(); perm = idx["tr"][torch.randperm(len(idx["tr"]))]
    for i in perm.split(64):
        opt.zero_grad(); loss_on(i).backward(); opt.step()
    net.eval()
    with torch.no_grad(): v = loss_on(idx["va"]).item()
    print(f"epoch {ep+1:2d} val_loss {v:.3f}")
    if v < best - 1e-3: best, bad, state = v, 0, {k: t.clone() for k, t in net.state_dict().items()}
    else:
        bad += 1
        if bad >= 4: break
net.load_state_dict(state); net.eval()
with torch.no_grad(): lc, lp = net(X)
df["pred_category"] = [CATS[i] for i in lc.argmax(1)]; df["pred_priority"] = [PRIO[i] for i in lp.argmax(1)]

# Autoencoder on TF-IDF (fit on train only); threshold = 95th pct of val error
tf = TfidfVectorizer(max_features=400, min_df=3).fit(df[tr].description)
T = torch.tensor(tf.transform(df.description).toarray(), dtype=torch.float32)
ae = nn.Sequential(nn.Linear(T.shape[1], 32), nn.ReLU(), nn.Linear(32, 8), nn.ReLU(), nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, T.shape[1]))
o2 = torch.optim.Adam(ae.parameters(), 1e-3); Ttr = T[idx["tr"]]
for ep in range(60):
    for i in torch.randperm(len(Ttr)).split(64):
        o2.zero_grad(); nn.functional.mse_loss(ae(Ttr[i]), Ttr[i]).backward(); o2.step()
with torch.no_grad(): err = ((ae(T) - T) ** 2).mean(1).numpy()
thr = np.percentile(err[va.values], 95)
df["anomaly_score"] = err; df["is_anomaly"] = (err > thr).astype(int)

m = {}
for n, mask in (("val", va), ("test", te)):
    d = df[mask]
    m[n] = dict(category_acc=round(accuracy_score(d.category, d.pred_category), 3),
                category_f1=round(f1_score(d.category, d.pred_category, average="macro"), 3),
                priority_acc=round(accuracy_score(d.priority, d.pred_priority), 3),
                priority_f1=round(f1_score(d.priority, d.pred_priority, average="macro"), 3))
d = df[te]
m["anomaly_test"] = dict(flagged_pct=round(d.is_anomaly.mean() * 100, 1),
                         recall_on_synthetic_outliers=round(d[d.true_outlier == 1].is_anomaly.mean(), 3),
                         precision=round(d[d.is_anomaly == 1].true_outlier.mean(), 3))
print(json.dumps(m, indent=1)); json.dump(m, open("outputs/metrics.json", "w"), indent=1)
df.to_csv("outputs/predictions.csv", index=False)
