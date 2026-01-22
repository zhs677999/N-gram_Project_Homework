#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One-click Markov Writer (stdlib-only) - Enhanced Coherence Edition
=================================================================
Features:
- Recursively trains from a folder of .txt files
- Multi-order (2..n) n-gram Markov with backoff for better coherence
- Prompt bias to stay on-topic + repetition penalty to reduce loops
- Caches model to pickle for fast restart; auto-invalidates if corpus changes
- Interactive REPL: input a prompt -> auto-continue

Quick start:
1) Put all .txt into ./corpus/  (can include subfolders)
2) python markov_writer.py --retrain
"""

import argparse
import os
import pickle
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# -----------------------------
# Tokenization / Detokenization
# -----------------------------

_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+|[。！？；，,.!?;:()（）“”\"'—\-…]+|\n")

ZH_PUNCT = set("。！")
EN_PUNCT = set(".!")
ALL_PUNCT = ZH_PUNCT | EN_PUNCT | set("()（）“”\"'—-…")

def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)

def detokenize(tokens: List[str]) -> str:
    out: List[str] = []
    prev = ""
    for t in tokens:
        if t == "\n":
            out.append("\n")
            prev = "\n"
            continue

        is_zh = bool(re.fullmatch(r"[\u4e00-\u9fff]", t))
        is_word = bool(re.fullmatch(r"[A-Za-z0-9]+", t))
        is_punct = all(ch in ALL_PUNCT for ch in t)

        if not out:
            out.append(t)
        else:
            if is_punct:
                if out[-1].endswith(" "):
                    out[-1] = out[-1].rstrip()
                out.append(t)
            elif is_zh:
                out.append(t)
            elif is_word:
                if prev and (re.fullmatch(r"[A-Za-z0-9]+", prev) or prev in EN_PUNCT or prev in {")", "）", "\"", "”", "'"}):
                    out.append(" " + t)
                else:
                    out.append(t)
            else:
                out.append(t)

        prev = t

    s = "".join(out)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# -----------------------------
# Model / Cache metadata
# -----------------------------

@dataclass
class CorpusMeta:
    corpus_dir: str
    file_count: int
    total_bytes: int
    latest_mtime: float
    n: int

def scan_corpus_meta(corpus_dir: str) -> CorpusMeta:
    file_count = 0
    total_bytes = 0
    latest_mtime = 0.0
    for dirpath, _, filenames in os.walk(corpus_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                file_count += 1
                total_bytes += int(st.st_size)
                if st.st_mtime > latest_mtime:
                    latest_mtime = float(st.st_mtime)

    return CorpusMeta(
        corpus_dir=os.path.abspath(corpus_dir),
        file_count=file_count,
        total_bytes=total_bytes,
        latest_mtime=latest_mtime,
        n=0,
    )

def meta_matches(a: CorpusMeta, b: CorpusMeta) -> bool:
    return (
        os.path.abspath(a.corpus_dir) == os.path.abspath(b.corpus_dir)
        and a.file_count == b.file_count
        and a.total_bytes == b.total_bytes
        and abs(a.latest_mtime - b.latest_mtime) < 1e-6
        and a.n == b.n
    )


# -----------------------------
# Markov multi-order backoff model
# -----------------------------

class MarkovModel:
    def __init__(self, n: int = 4, seed: Optional[int] = 42):
        self.n = max(2, min(6, int(n)))
        self.rng = random.Random(seed)

        # order -> {context_tuple: Counter(next_token)}
        self.maps: Dict[int, Dict[Tuple[str, ...], Counter]] = {k: {} for k in range(2, self.n + 1)}
        self.keys_by_order: Dict[int, List[Tuple[str, ...]]] = {k: [] for k in range(2, self.n + 1)}

    def update_from_text(self, text: str) -> None:
        toks = tokenize(text)
        if len(toks) < 3:
            return

        for order in range(2, self.n + 1):
            if len(toks) <= order:
                continue
            mp = self.maps[order]
            for i in range(len(toks) - order):
                key = tuple(toks[i:i + order])
                nxt = toks[i + order]
                if key not in mp:
                    mp[key] = Counter()
                mp[key][nxt] += 1

    def finalize(self) -> None:
        for order in range(2, self.n + 1):
            self.keys_by_order[order] = list(self.maps[order].keys())

    def _weighted_choice(self, counter: Counter) -> str:
        items = list(counter.items())
        population = [k for k, _ in items]
        weights = [w for _, w in items]
        return self.rng.choices(population, weights=weights, k=1)[0]

    def _get_counter_with_backoff(self, ctx: List[str]) -> Optional[Counter]:
        for order in range(min(self.n, len(ctx)), 1, -1):
            key = tuple(ctx[-order:])
            counter = self.maps[order].get(key)
            if counter:
                return counter
        return None

    def _pick_context_near_prompt(self, anchors: Counter, want_order: int) -> List[str]:
        keys = self.keys_by_order.get(want_order, [])
        if not keys:
            return []
        if anchors:
            for _ in range(250):
                cand = self.rng.choice(keys)
                if any(tok in anchors for tok in cand):
                    return list(cand)
        return list(self.rng.choice(keys))

    def continue_from_prompt(
        self,
        prompt: str,
        min_sentences: int = 4,
        max_sentences: int = 8,
        max_new_tokens: int = 600,
        prompt_bias: float = 2.0,
        repetition_penalty: float = 0.75,
    ) -> str:
        p_tokens = tokenize(prompt)

        anchors = Counter([t for t in p_tokens if t.strip() and t != "\n"])
        for k in list(anchors.keys()):
            if all(ch in ALL_PUNCT for ch in k):
                anchors.pop(k, None)

        ctx = p_tokens[-self.n:].copy()
        if len(ctx) < 2:
            ctx = self._pick_context_near_prompt(anchors, want_order=self.n) or self._pick_context_near_prompt(anchors, want_order=2)
            if not ctx:
                return ""

        out: List[str] = []
        sentences = 0
        recent_window: List[str] = []
        target_sentences = self.rng.randint(min_sentences, max_sentences)

        for _ in range(max_new_tokens):
            counter = self._get_counter_with_backoff(ctx)
            if not counter:
                ctx = self._pick_context_near_prompt(anchors, want_order=self.n) or self._pick_context_near_prompt(anchors, want_order=2)
                if not ctx:
                    break
                continue

            adjusted = Counter(counter)

            if anchors:
                for tok in list(adjusted.keys()):
                    if tok in anchors:
                        adjusted[tok] = max(1, int(adjusted[tok] * prompt_bias))

            for tok in recent_window:
                if tok in adjusted:
                    adjusted[tok] = max(1, int(adjusted[tok] * repetition_penalty))

            nxt = self._weighted_choice(adjusted)

            out.append(nxt)
            ctx.append(nxt)

            if nxt.strip():
                recent_window.append(nxt)
                if len(recent_window) > 50:
                    recent_window.pop(0)

            if nxt in ZH_PUNCT or nxt in EN_PUNCT:
                sentences += 1
                if sentences >= target_sentences:
                    break

        return detokenize(out)


# -----------------------------
# IO helpers
# -----------------------------

def read_text_file(fp: str) -> str:
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

def iter_txt_files(corpus_dir: str):
    for dirpath, _, filenames in os.walk(corpus_dir):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                yield os.path.join(dirpath, fn)

def load_cache(cache_path: str):
    with open(cache_path, "rb") as f:
        return pickle.load(f)

def save_cache(cache_path: str, meta: CorpusMeta, model: MarkovModel) -> None:
    payload = {
        "meta": meta,
        "n": model.n,
        "maps": model.maps,
        "keys_by_order": model.keys_by_order,
    }
    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, cache_path)


# -----------------------------
# Build / Load model
# -----------------------------

def build_or_load_model(
    corpus_dir: str,
    n: int,
    cache_path: str,
    seed: int,
    retrain: bool = False,
) -> MarkovModel:
    corpus_dir = os.path.abspath(corpus_dir)
    if not os.path.isdir(corpus_dir):
        print(f"[ERR] corpus_dir not found: {corpus_dir}")
        sys.exit(1)

    meta_now = scan_corpus_meta(corpus_dir)
    meta_now.n = n

    if (not retrain) and os.path.exists(cache_path):
        try:
            payload = load_cache(cache_path)
            meta_old: CorpusMeta = payload.get("meta")
            if meta_old and meta_matches(meta_old, meta_now):
                if "maps" in payload and "keys_by_order" in payload:
                    model = MarkovModel(n=n, seed=seed)
                    model.maps = payload["maps"]
                    model.keys_by_order = payload["keys_by_order"]
                    print(f"[OK] Loaded cache: {cache_path} (files={meta_now.file_count}, n={n})")
                    return model
                print("[INFO] Old cache format detected. Please retrain with --retrain (or delete cache file).")
            else:
                print("[INFO] Cache exists but corpus changed (or n changed). Retraining...")
        except Exception as e:
            print(f"[WARN] Failed to load cache, retraining. ({e})")

    model = MarkovModel(n=n, seed=seed)
    files = list(iter_txt_files(corpus_dir))
    if not files:
        print(f"[ERR] No .txt files found under: {corpus_dir}")
        sys.exit(1)

    print(f"[INFO] Training Markov backoff model (n={n}) from {len(files)} txt files under: {corpus_dir}")
    t0 = time.time()
    for idx, fp in enumerate(files, 1):
        text = read_text_file(fp)
        if text:
            model.update_from_text(text)

        if idx % 25 == 0 or idx == len(files):
            elapsed = time.time() - t0
            print(f"  - {idx}/{len(files)} files processed, elapsed {elapsed:.1f}s")

    model.finalize()

    has_any = any(model.keys_by_order[order] for order in range(2, model.n + 1))
    if not has_any:
        print("[ERR] Training produced empty model (maybe files too short).")
        sys.exit(1)

    save_cache(cache_path, meta_now, model)
    states = sum(len(model.keys_by_order[o]) for o in range(2, model.n + 1))
    print(f"[OK] Trained + cached: {cache_path} (states={states})")

    return model


# -----------------------------
# REPL
# -----------------------------

def repl(
    model: MarkovModel,
    max_new_tokens: int,
    min_sentences: int,
    max_sentences: int,
    prompt_bias: float,
    repetition_penalty: float,
):
    print("\n===== Markov 续写器已就绪 =====")
    print("输入一段话回车续写；输入 :q 退出；输入 :help 查看指令。\n")

    while True:
        try:
            prompt = input("你：").rstrip("\n")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not prompt.strip():
            continue

        cmd = prompt.strip()
        if cmd in {":q", ":quit", ":exit"}:
            print("Bye.")
            return
        if cmd == ":help":
            print(
                "\n指令：\n"
                "  :q / :quit / :exit   退出\n"
                "  :help                帮助\n\n"
                "建议：提示词越长（>=2-3句），续写越连贯。\n"
            )
            continue

        cont = model.continue_from_prompt(
            prompt,
            min_sentences=min_sentences,
            max_sentences=max_sentences,
            max_new_tokens=max_new_tokens,
            prompt_bias=prompt_bias,
            repetition_penalty=repetition_penalty,
        )

        if not cont:
            print("续写：<生成失败：模型为空或提示词无法匹配>\n")
            continue

        print("\n续写：")
        print(cont)
        print("\n合并：")
        print(prompt + cont)
        print("\n" + "-" * 30 + "\n")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser(description="One-click Markov Writer (folder of txt -> train -> interactive continue)")
    ap.add_argument("--corpus_dir", default="corpus", help="txt语料文件夹（默认 ./corpus）")
    ap.add_argument("--n", type=int, default=5, help="n-gram n（2-6，中文建议4或5）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    ap.add_argument("--cache", default="markov_cache.pkl", help="缓存文件路径")
    ap.add_argument("--retrain", action="store_true", help="强制重新训练（忽略缓存）")

    ap.add_argument("--max_new_tokens", type=int, default=600, help="最多续写token数")
    ap.add_argument("--min_sentences", type=int, default=8, help="最少句子数")
    ap.add_argument("--max_sentences", type=int, default=10, help="最多句子数")

    ap.add_argument("--prompt_bias", type=float, default=2.0, help="提示词粘性（>1 越大越不跑题）")
    ap.add_argument("--repetition_penalty", type=float, default=0.75, help="重复惩罚（0.5~0.9 越小越抑制重复）")

    args = ap.parse_args()

    n = max(2, min(6, args.n))
    model = build_or_load_model(
        corpus_dir=args.corpus_dir,
        n=n,
        cache_path=args.cache,
        seed=args.seed,
        retrain=args.retrain,
    )

    repl(
        model,
        max_new_tokens=args.max_new_tokens,
        min_sentences=args.min_sentences,
        max_sentences=args.max_sentences,
        prompt_bias=args.prompt_bias,
        repetition_penalty=args.repetition_penalty,
    )

if __name__ == "__main__":
    main()
