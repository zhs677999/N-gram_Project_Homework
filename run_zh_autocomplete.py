import os
import sys
import json
import argparse
import numpy as np

# =========================
# Utils
# =========================
def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def tokenize_zhchar(text: str):
    # 汉字/标点/空格/换行：逐字符
    return list(text)

def build_vocab(tokens, min_freq=1, max_vocab=0):
    from collections import Counter
    cnt = Counter(tokens)
    vocab = ["<UNK>"]

    items = cnt.most_common()
    if max_vocab and max_vocab > 1:
        items = items[: max_vocab - 1]

    for tok, c in items:
        if c >= min_freq and tok != "<UNK>":
            vocab.append(tok)

    stoi = {t: i for i, t in enumerate(vocab)}
    itos = {i: t for t, i in stoi.items()}
    return vocab, stoi, itos

def read_corpus_from_dir(data_dir, encoding="utf-8", skip_first_line=True):
    texts = []
    n_files = 0
    for root, _, files in os.walk(data_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                path = os.path.join(root, fn)
                try:
                    with open(path, "r", encoding=encoding, errors="ignore") as f:
                        if skip_first_line:
                            # 保留原始换行：splitlines(True) 会把 '\n' 保留在每行末尾
                            lines = f.read().splitlines(True)
                            content = "".join(lines[1:]) if len(lines) > 1 else ""
                        else:
                            content = f.read()
                    if content.strip():
                        texts.append(content)
                        n_files += 1
                except Exception as e:
                    print(f"[WARN] skip {path}: {e}")

    if n_files == 0:
        raise RuntimeError(f"No .txt files found in {data_dir}")

    return "\n\n".join(texts), n_files

def ids_from_tokens(tokens, stoi):
    return [stoi.get(t, 0) for t in tokens]  # 0=<UNK>

# =========================
# Adam (handwritten)
# =========================
class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, grads):
        self.t += 1
        b1, b2 = self.b1, self.b2
        lr, eps = self.lr, self.eps
        t = self.t

        for k in self.params.keys():
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)

            mhat = self.m[k] / (1 - (b1 ** t))
            vhat = self.v[k] / (1 - (b2 ** t))
            self.params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    def state_dict(self):
        return {
            "t": np.array(self.t, dtype=np.int64),
            "lr": np.array(self.lr, dtype=np.float32),
            "b1": np.array(self.b1, dtype=np.float32),
            "b2": np.array(self.b2, dtype=np.float32),
            "eps": np.array(self.eps, dtype=np.float32),
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state):
        self.t = int(state["t"])
        self.lr = float(state["lr"])
        self.b1 = float(state["b1"])
        self.b2 = float(state["b2"])
        self.eps = float(state["eps"])
        self.m = state["m"]
        self.v = state["v"]

# =========================
# Model: GRU (handwritten)
# =========================
class ZHCharGRU:
    def __init__(self, vocab_size, embed_size=64, hidden_size=256, seed=42):
        np.random.seed(seed)
        V, D, H = vocab_size, embed_size, hidden_size
        self.V, self.D, self.H = V, D, H

        self.E = (np.random.randn(D, V) * 0.01).astype(np.float32)

        self.Wxz = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Whz = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.bz  = np.zeros((H, 1), dtype=np.float32)

        self.Wxr = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Whr = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.br  = np.zeros((H, 1), dtype=np.float32)

        self.Wxh = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Whh = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.bh  = np.zeros((H, 1), dtype=np.float32)

        self.Why = (np.random.randn(V, H) * 0.01).astype(np.float32)
        self.by  = np.zeros((V, 1), dtype=np.float32)

        self.params = {
            "E": self.E,
            "Wxz": self.Wxz, "Whz": self.Whz, "bz": self.bz,
            "Wxr": self.Wxr, "Whr": self.Whr, "br": self.br,
            "Wxh": self.Wxh, "Whh": self.Whh, "bh": self.bh,
            "Why": self.Why, "by": self.by
        }

    def forward(self, inputs, targets, hprev):
        xs, hs = {}, {}
        zs, rs, hts = {}, {}, {}
        ps = {}
        hs[-1] = hprev.copy()
        loss = 0.0

        for t in range(len(inputs)):
            idx = inputs[t]
            x = self.E[:, idx:idx+1]
            xs[t] = (idx, x)

            z = sigmoid(self.Wxz @ x + self.Whz @ hs[t-1] + self.bz)
            r = sigmoid(self.Wxr @ x + self.Whr @ hs[t-1] + self.br)
            h_tilde = np.tanh(self.Wxh @ x + self.Whh @ (r * hs[t-1]) + self.bh)
            h = (1 - z) * hs[t-1] + z * h_tilde

            zs[t], rs[t], hts[t], hs[t] = z, r, h_tilde, h

            y = self.Why @ h + self.by
            p = softmax(y)
            ps[t] = p
            loss += -np.log(p[targets[t], 0] + 1e-12)

        cache = (xs, hs, zs, rs, hts, ps, inputs, targets)
        return loss, cache

    def backward(self, cache):
        xs, hs, zs, rs, hts, ps, inputs, targets = cache
        H = self.H
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dhnext = np.zeros((H, 1), dtype=np.float32)

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0

            grads["Why"] += dy @ hs[t].T
            grads["by"]  += dy

            dh = self.Why.T @ dy + dhnext

            z = zs[t]
            r = rs[t]
            h_tilde = hts[t]
            hprev = hs[t-1]
            idx, x = xs[t]

            dh_tilde = dh * z
            dz = dh * (h_tilde - hprev)
            dhprev = dh * (1 - z)

            da_h = (1 - h_tilde * h_tilde) * dh_tilde
            grads["Wxh"] += da_h @ x.T
            grads["Whh"] += da_h @ (r * hprev).T
            grads["bh"]  += da_h

            dx = self.Wxh.T @ da_h
            d_rhprev = self.Whh.T @ da_h
            dr = d_rhprev * hprev
            dhprev += d_rhprev * r

            da_r = dr * r * (1 - r)
            grads["Wxr"] += da_r @ x.T
            grads["Whr"] += da_r @ hprev.T
            grads["br"]  += da_r
            dx += self.Wxr.T @ da_r
            dhprev += self.Whr.T @ da_r

            da_z = dz * z * (1 - z)
            grads["Wxz"] += da_z @ x.T
            grads["Whz"] += da_z @ hprev.T
            grads["bz"]  += da_z
            dx += self.Wxz.T @ da_z
            dhprev += self.Whz.T @ da_z

            grads["E"][:, idx:idx+1] += dx
            dhnext = dhprev

        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads, dhnext

    def step_state(self, idx, h):
        x = self.E[:, idx:idx+1]
        z = sigmoid(self.Wxz @ x + self.Whz @ h + self.bz)
        r = sigmoid(self.Wxr @ x + self.Whr @ h + self.br)
        h_tilde = np.tanh(self.Wxh @ x + self.Whh @ (r * h) + self.bh)
        h = (1 - z) * h + z * h_tilde
        y = self.Why @ h + self.by
        return h, y

# =========================
# Model: LSTM (handwritten)
# =========================
class ZHCharLSTM:
    def __init__(self, vocab_size, embed_size=64, hidden_size=256, seed=42):
        np.random.seed(seed)
        V, D, H = vocab_size, embed_size, hidden_size
        self.V, self.D, self.H = V, D, H

        self.E = (np.random.randn(D, V) * 0.01).astype(np.float32)

        self.Wxi = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Whi = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.bi  = np.zeros((H, 1), dtype=np.float32)

        self.Wxf = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Whf = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.bf  = np.zeros((H, 1), dtype=np.float32)

        self.Wxo = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Who = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.bo  = np.zeros((H, 1), dtype=np.float32)

        self.Wxg = (np.random.randn(H, D) * 0.01).astype(np.float32)
        self.Whg = (np.random.randn(H, H) * 0.01).astype(np.float32)
        self.bg  = np.zeros((H, 1), dtype=np.float32)

        self.Why = (np.random.randn(V, H) * 0.01).astype(np.float32)
        self.by  = np.zeros((V, 1), dtype=np.float32)

        self.params = {
            "E": self.E,
            "Wxi": self.Wxi, "Whi": self.Whi, "bi": self.bi,
            "Wxf": self.Wxf, "Whf": self.Whf, "bf": self.bf,
            "Wxo": self.Wxo, "Who": self.Who, "bo": self.bo,
            "Wxg": self.Wxg, "Whg": self.Whg, "bg": self.bg,
            "Why": self.Why, "by": self.by
        }

    def forward(self, inputs, targets, hprev, cprev):
        xs, hs, cs = {}, {}, {}
        is_, fs, os, gs = {}, {}, {}, {}
        ps = {}

        hs[-1] = hprev.copy()
        cs[-1] = cprev.copy()
        loss = 0.0

        for t in range(len(inputs)):
            idx = inputs[t]
            x = self.E[:, idx:idx+1]
            xs[t] = (idx, x)

            h_prev = hs[t-1]
            c_prev = cs[t-1]

            i = sigmoid(self.Wxi @ x + self.Whi @ h_prev + self.bi)
            f = sigmoid(self.Wxf @ x + self.Whf @ h_prev + self.bf)
            o = sigmoid(self.Wxo @ x + self.Who @ h_prev + self.bo)
            g = np.tanh(self.Wxg @ x + self.Whg @ h_prev + self.bg)

            c = f * c_prev + i * g
            h = o * np.tanh(c)

            is_[t], fs[t], os[t], gs[t] = i, f, o, g
            cs[t], hs[t] = c, h

            y = self.Why @ h + self.by
            p = softmax(y)
            ps[t] = p
            loss += -np.log(p[targets[t], 0] + 1e-12)

        cache = (xs, hs, cs, is_, fs, os, gs, ps, inputs, targets)
        return loss, cache

    def backward(self, cache):
        xs, hs, cs, is_, fs, os, gs, ps, inputs, targets = cache
        H = self.H

        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        dhnext = np.zeros((H, 1), dtype=np.float32)
        dcnext = np.zeros((H, 1), dtype=np.float32)

        for t in reversed(range(len(inputs))):
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0

            grads["Why"] += dy @ hs[t].T
            grads["by"]  += dy

            dh = self.Why.T @ dy + dhnext

            c = cs[t]
            cprev = cs[t-1]
            hprev = hs[t-1]
            i, f, o, g = is_[t], fs[t], os[t], gs[t]

            tanh_c = np.tanh(c)

            do = dh * tanh_c
            dc = dh * o * (1 - tanh_c * tanh_c) + dcnext
            df = dc * cprev
            dcprev = dc * f
            di = dc * g
            dg = dc * i

            dai = di * i * (1 - i)
            daf = df * f * (1 - f)
            dao = do * o * (1 - o)
            dag = dg * (1 - g * g)

            idx, x = xs[t]

            grads["Wxi"] += dai @ x.T
            grads["Whi"] += dai @ hprev.T
            grads["bi"]  += dai

            grads["Wxf"] += daf @ x.T
            grads["Whf"] += daf @ hprev.T
            grads["bf"]  += daf

            grads["Wxo"] += dao @ x.T
            grads["Who"] += dao @ hprev.T
            grads["bo"]  += dao

            grads["Wxg"] += dag @ x.T
            grads["Whg"] += dag @ hprev.T
            grads["bg"]  += dag

            dx = (self.Wxi.T @ dai + self.Wxf.T @ daf + self.Wxo.T @ dao + self.Wxg.T @ dag)
            dhprev = (self.Whi.T @ dai + self.Whf.T @ daf + self.Who.T @ dao + self.Whg.T @ dag)

            grads["E"][:, idx:idx+1] += dx

            dhnext = dhprev
            dcnext = dcprev

        for k in grads:
            np.clip(grads[k], -5, 5, out=grads[k])

        return grads, dhnext, dcnext

    def step_state(self, idx, h, c):
        x = self.E[:, idx:idx+1]
        i = sigmoid(self.Wxi @ x + self.Whi @ h + self.bi)
        f = sigmoid(self.Wxf @ x + self.Whf @ h + self.bf)
        o = sigmoid(self.Wxo @ x + self.Who @ h + self.bo)
        g = np.tanh(self.Wxg @ x + self.Whg @ h + self.bg)
        c = f * c + i * g
        h = o * np.tanh(c)
        y = self.Why @ h + self.by
        return h, c, y

# =========================
# Checkpoint
# =========================
def save_ckpt(path, model, opt, meta: dict):
    pack = {}
    for k, v in model.params.items():
        pack[f"param__{k}"] = v

    st = opt.state_dict()
    pack["opt__t"] = st["t"]
    pack["opt__lr"] = st["lr"]
    pack["opt__b1"] = st["b1"]
    pack["opt__b2"] = st["b2"]
    pack["opt__eps"] = st["eps"]
    for k, v in st["m"].items():
        pack[f"opt_m__{k}"] = v
    for k, v in st["v"].items():
        pack[f"opt_v__{k}"] = v

    pack["meta__vocab"] = np.array(meta["vocab"], dtype=object)
    pack["meta__config_json"] = np.array(json.dumps(meta["config"], ensure_ascii=False), dtype=object)
    pack["meta__progress_json"] = np.array(json.dumps(meta["progress"], ensure_ascii=False), dtype=object)

    np.savez_compressed(path, **pack)

def load_ckpt(path):
    z = np.load(path, allow_pickle=True)
    vocab = list(z["meta__vocab"].tolist())
    config = json.loads(str(z["meta__config_json"].item()))
    progress = json.loads(str(z["meta__progress_json"].item()))
    return z, vocab, config, progress

def build_model(config, vocab_size):
    if config["cell"] == "lstm":
        return ZHCharLSTM(vocab_size, embed_size=config["embed"], hidden_size=config["hidden"], seed=config["seed"])
    return ZHCharGRU(vocab_size, embed_size=config["embed"], hidden_size=config["hidden"], seed=config["seed"])

def restore_model_and_opt(z, config, vocab_size):
    model = build_model(config, vocab_size)
    opt = Adam(model.params, lr=config["lr"], betas=(config["b1"], config["b2"]), eps=config["eps"])
    # params
    for k in model.params.keys():
        model.params[k][...] = z[f"param__{k}"]
    # opt
    state = {
        "t": int(z["opt__t"]),
        "lr": float(z["opt__lr"]),
        "b1": float(z["opt__b1"]),
        "b2": float(z["opt__b2"]),
        "eps": float(z["opt__eps"]),
        "m": {},
        "v": {},
    }
    for k in model.params.keys():
        state["m"][k] = z[f"opt_m__{k}"]
        state["v"][k] = z[f"opt_v__{k}"]
    opt.load_state_dict(state)
    return model, opt

# =========================
# Sampling
# =========================
def sample_continuation(model, itos, stoi, prompt, gen_len=200, temperature=0.9, cell="gru"):
    H = model.H
    h = np.zeros((H, 1), dtype=np.float32)
    c = np.zeros((H, 1), dtype=np.float32)

    prompt_tokens = tokenize_zhchar(prompt)
    prompt_ids = ids_from_tokens(prompt_tokens, stoi)

    # warm up with prompt
    for idx in prompt_ids:
        if cell == "lstm":
            h, c, _ = model.step_state(idx, h, c)
        else:
            h, _ = model.step_state(idx, h)

    cur = prompt_ids[-1] if prompt_ids else stoi.get("。", 0)
    out_ids = []

    for _ in range(gen_len):
        if cell == "lstm":
            h, c, y = model.step_state(cur, h, c)
        else:
            h, y = model.step_state(cur, h)
        p = softmax(y / max(temperature, 1e-6))
        cur = int(np.random.choice(range(model.V), p=p.ravel()))
        out_ids.append(cur)

    return prompt + "".join(itos[i] for i in out_ids)

# =========================
# Train + Interactive
# =========================
def train_or_load_and_run(args):
    ckpt_exists = os.path.exists(args.ckpt)

    # 1) 如果 ckpt 存在且用户没要求强制重训，则直接加载
    if ckpt_exists and (not args.force_retrain):
        z, vocab, config, progress = load_ckpt(args.ckpt)
        stoi = {t:i for i,t in enumerate(vocab)}
        itos = {i:t for i,t in enumerate(vocab)}
        model, opt = restore_model_and_opt(z, config, len(vocab))
        print(f"[LOAD] {args.ckpt} | cell={config['cell']} vocab={len(vocab)} iter={progress.get('iter',0)}")
        interactive_loop(model, itos, stoi, config["cell"], args)
        return

    # 2) 否则：读取文件夹语料、建词表、训练、保存、进入交互
    text, n_files = read_corpus_from_dir(args.data_dir, encoding=args.encoding, skip_first_line=args.skip_first_line)
    tokens = tokenize_zhchar(text)

    if args.max_chars > 0 and len(tokens) > args.max_chars:
        tokens = tokens[:args.max_chars]
        print(f"[INFO] truncate tokens to max_chars={args.max_chars}")

    vocab, stoi, itos = build_vocab(tokens, min_freq=args.min_freq, max_vocab=args.max_vocab)
    V = len(vocab)
    print(f"[DATA] files={n_files} tokens={len(tokens)} vocab={V} (min_freq={args.min_freq}, max_vocab={args.max_vocab})")

    config = {
        "cell": args.cell,
        "embed": args.embed,
        "hidden": args.hidden,
        "seq": args.seq,
        "lr": args.lr,
        "b1": args.b1,
        "b2": args.b2,
        "eps": args.eps,
        "seed": args.seed,
    }

    model = build_model(config, V)
    opt = Adam(model.params, lr=args.lr, betas=(args.b1, args.b2), eps=args.eps)

    data_ids = ids_from_tokens(tokens, stoi)
    T = args.seq
    H = args.hidden

    hprev = np.zeros((H, 1), dtype=np.float32)
    cprev = np.zeros((H, 1), dtype=np.float32)

    smooth_loss = -np.log(1.0 / V) * T
    pos = 0

    print("[TRAIN] start ...")
    for it in range(1, args.iters + 1):
        if pos + T + 1 >= len(data_ids):
            pos = 0
            hprev[...] = 0
            cprev[...] = 0

        inputs = data_ids[pos:pos+T]
        targets = data_ids[pos+1:pos+T+1]

        if args.cell == "lstm":
            loss, cache = model.forward(inputs, targets, hprev, cprev)
            grads, dh, dc = model.backward(cache)
            opt.step(grads)
            _, hs, cs, *_ = cache
            hprev = hs[T-1].copy()
            cprev = cs[T-1].copy()
        else:
            loss, cache = model.forward(inputs, targets, hprev)
            grads, dh = model.backward(cache)
            opt.step(grads)
            _, hs, *_ = cache
            hprev = hs[T-1].copy()

        smooth_loss = smooth_loss * 0.999 + loss * 0.001

        if it % args.sample_every == 0:
            demo = sample_continuation(
                model, itos, stoi,
                prompt=args.demo_prompt,
                gen_len=args.sample_len,
                temperature=args.temp,
                cell=args.cell
            )
            print(f"\niter {it}/{args.iters} | smooth_loss={smooth_loss:.4f} | pos={pos}/{len(data_ids)}")
            print("---- demo ----")
            print(demo)
            print("-------------")

        if it % args.save_every == 0 or it == args.iters:
            meta = {
                "vocab": vocab,
                "config": config,
                "progress": {"iter": it, "pos": pos, "smooth_loss": float(smooth_loss)}
            }
            save_ckpt(args.ckpt, model, opt, meta)
            print(f"[SAVE] {args.ckpt} (iter={it})")

        pos += T

    print("[TRAIN] done. Enter interactive mode.")
    interactive_loop(model, itos, stoi, args.cell, args)

def interactive_loop(model, itos, stoi, cell, args):
    print("\n============================")
    print("交互续写模式：输入一段话，回车后自动续写")
    print("退出请输入：/exit 或 /quit")
    print("============================\n")

    while True:
        try:
            prompt = input(">>> ").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\n[EXIT]")
            break

        if prompt.strip() in ["/exit", "/quit"]:
            print("[EXIT]")
            break
        if not prompt.strip():
            continue

        out = sample_continuation(
            model, itos, stoi,
            prompt=prompt,
            gen_len=args.gen_len,
            temperature=args.temp,
            cell=cell
        )
        print(out)
        print()

# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser("Chinese text continuation (handwritten GRU/LSTM)")
    p.add_argument("--data_dir", type=str, required=True, help="语料txt文件夹（包含500个txt）")
    p.add_argument("--ckpt", type=str, default="zh_ckpt.npz", help="checkpoint保存路径")
    p.add_argument("--cell", type=str, default="gru", choices=["gru", "lstm"])
    p.add_argument("--embed", type=int, default=64)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--seq", type=int, default=128, help="BPTT 截断长度")
    p.add_argument("--iters", type=int, default=5000, help="训练步数（第一次训练建议先跑5000~30000）")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--b1", type=float, default=0.9)
    p.add_argument("--b2", type=float, default=0.999)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_first_line", type=int, default=1, help="1=每个txt跳过第一行，从第二行开始读")

    p.add_argument("--min_freq", type=int, default=1)
    p.add_argument("--max_vocab", type=int, default=0, help="0=不截断词表（字符表）")
    p.add_argument("--max_chars", type=int, default=0, help="0=不截断，>0 则最多读取这么多字符训练（调试用）")
    p.add_argument("--encoding", type=str, default="utf-8")

    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--sample_len", type=int, default=200)
    p.add_argument("--demo_prompt", type=str, default="从前有座山，山里有座庙。")

    p.add_argument("--temp", type=float, default=0.9, help="采样温度，越高越随机")
    p.add_argument("--gen_len", type=int, default=200, help="交互续写长度")

    p.add_argument("--force_retrain", action="store_true", help="即使存在ckpt也强制重新训练")
    return p.parse_args()

def main():
    args = parse_args()
    train_or_load_and_run(args)

if __name__ == "__main__":
    main()
