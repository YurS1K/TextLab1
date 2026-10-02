# extract_features.py
# Запуск: python extract_features.py
# На входе:  learning_corpus.csv
# На выходе: папка features/ со всеми матрицами признаков

import re
import json
import numpy as np
import pandas as pd
import spacy
from collections import Counter
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz

IN = "learning_corpus.csv"
OUT = Path("features")
OUT.mkdir(exist_ok=True)

# ============================================================
# 0. Загрузка и модель
# ============================================================
df = pd.read_csv(IN).fillna("")
print(f"Загружено текстов: {len(df)}")
print("Жанры:", df["genre"].value_counts().to_dict() if "genre" in df else "—")

nlp = spacy.load("ru_core_news_sm")

# ============================================================
# 1. НИЗКОУРОВНЕВЫЕ: уровень символов
# ============================================================
def char_features(text: str) -> dict:
    letters  = re.findall(r"[А-Яа-яЁёA-Za-z]", text)
    upper    = re.findall(r"[А-ЯЁA-Z]", text)
    digits   = re.findall(r"\d", text)
    punct    = re.findall(r"[^\w\s]", text)
    return {
        "n_chars":         len(text),
        "n_letters":       len(letters),
        "n_upper":         len(upper),
        "n_digits":        len(digits),
        "n_punct":         len(punct),
        "n_unique_chars":  len(set(text)),
        "share_punct":     len(punct) / max(len(text), 1),
        "share_upper":     len(upper) / max(len(letters), 1),
    }

# ============================================================
# 2. НИЗКОУРОВНЕВЫЕ: уровень слов
# ============================================================
def word_features(doc, text: str) -> dict:
    sents  = list(doc.sents)
    words  = [t for t in doc if t.is_alpha]
    lemmas = [t.lemma_.lower() for t in words]
    cnt    = Counter(lemmas)
    hapax  = sum(1 for w, c in cnt.items() if c == 1)

    sent_words = [sum(1 for t in s if t.is_alpha) for s in sents]
    sent_chars = [len(s.text) for s in sents]
    wl         = [len(t.text) for t in words]

    return {
        "n_sentences":          len(sents),
        "n_words":              len(words),
        "n_unique_words":       len(set(lemmas)),
        "ttr":                  len(set(lemmas)) / max(len(words), 1),
        "n_hapax":              hapax,
        "share_hapax":          hapax / max(len(set(lemmas)), 1),
        "avg_sent_len_words":   float(np.mean(sent_words)) if sent_words else 0.0,
        "std_sent_len_words":   float(np.std(sent_words))  if sent_words else 0.0,
        "min_sent_len_words":   int(min(sent_words)) if sent_words else 0,
        "max_sent_len_words":   int(max(sent_words)) if sent_words else 0,
        "avg_sent_len_chars":   float(np.mean(sent_chars)) if sent_chars else 0.0,
        "std_sent_len_chars":   float(np.std(sent_chars))  if sent_chars else 0.0,
        "avg_word_len":         float(np.mean(wl)) if wl else 0.0,
        "std_word_len":         float(np.std(wl))  if wl else 0.0,
        "commas_per_sent":      text.count(",") / max(len(sents), 1),
        "dashes_per_sent":      (text.count("—") + text.count(" - ")) / max(len(sents), 1),
        "questions_per_sent":   text.count("?") / max(len(sents), 1),
        "exclam_per_sent":      text.count("!") / max(len(sents), 1),
    }

# ============================================================
# 3. НИЗКОУРОВНЕВЫЕ: уровень структуры — POS
# ============================================================
POS_LIST = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "ADP",
            "CCONJ", "SCONJ", "PART", "DET", "NUM", "PROPN",
            "INTJ", "AUX"]

def pos_features(doc) -> dict:
    counts = Counter(t.pos_ for t in doc if not t.is_space and not t.is_punct)
    total  = sum(counts.values()) or 1
    return {f"pos_{p.lower()}_share": counts.get(p, 0) / total for p in POS_LIST}

def morph_features(doc) -> dict:
    verbs_total = 0
    past = pres = fut = 0
    noun_sg = noun_pl = 0
    adj_short = 0
    adj_total = 0
    for t in doc:
        if t.pos_ == "VERB":
            verbs_total += 1
            tense = t.morph.get("Tense")
            if "Past" in tense:  past += 1
            elif "Pres" in tense: pres += 1
            elif "Fut" in tense:  fut += 1
        if t.pos_ == "NOUN":
            num = t.morph.get("Number")
            if "Sing" in num:  noun_sg += 1
            elif "Plur" in num: noun_pl += 1
        if t.pos_ == "ADJ":
            adj_total += 1
            if "Short" in t.morph.get("Variant"): adj_short += 1
    return {
        "verb_past_share":  past / max(verbs_total, 1),
        "verb_pres_share":  pres / max(verbs_total, 1),
        "verb_fut_share":   fut  / max(verbs_total, 1),
        "noun_sing_share":  noun_sg / max(noun_sg + noun_pl, 1),
        "adj_short_share":  adj_short / max(adj_total, 1),
    }

# ============================================================
# 4. ВЫСОКОУРОВНЕВЫЕ: синтаксис
# ============================================================
def syntax_features(doc, text: str) -> dict:
    sents = list(doc.sents)
    n_direct_speech = 0
    n_complex       = 0
    depths          = []

    for s in sents:
        # прямая речь: «...», "...", предложение, начинающееся с тире
        if re.search(r"[«\"].+?[»\"]", s.text):
            n_direct_speech += 1
        if s.text.strip().startswith(("—", "–")):
            n_direct_speech += 1

        # сложное предложение: есть подчинительный союз или mark-зависимость
        if any(t.pos_ == "SCONJ" for t in s) or any(t.dep_ == "mark" for t in s):
            n_complex += 1

        # глубина дерева зависимостей
        d_max = 0
        for t in s:
            d = 0
            cur = t
            while cur.head != cur and d < 100:
                d += 1
                cur = cur.head
            d_max = max(d_max, d)
        depths.append(d_max)

    n_quotes = len(re.findall(r"[«\"]", text))
    return {
        "n_direct_speech_sents": n_direct_speech,
        "share_direct_speech":   n_direct_speech / max(len(sents), 1),
        "share_complex_sents":   n_complex       / max(len(sents), 1),
        "avg_tree_depth":        float(np.mean(depths)) if depths else 0.0,
        "max_tree_depth":        int(max(depths))       if depths else 0,
        "quotes_per_sent":       n_quotes / max(len(sents), 1),
    }

# ============================================================
# 5. ВЫСОКОУРОВНЕВЫЕ: Universal Dependencies
# ============================================================
DEP_LIST = ["nsubj", "obj", "iobj", "amod", "advmod", "obl",
            "conj", "cc", "mark", "acl", "relcl", "advcl",
            "xcomp", "nmod", "appos", "parataxis"]

def dep_features(doc) -> dict:
    counts = Counter(t.dep_ for t in doc if not t.is_space and not t.is_punct)
    total  = sum(counts.values()) or 1
    return {f"dep_{d}_share": counts.get(d, 0) / total for d in DEP_LIST}

# ============================================================
# 6. ПРОГОН ПО КОРПУСУ
# ============================================================
rows = []
docs_texts = []            # (doc, original_text)
lemma_all_texts = []       # леммы без фильтра — для Delta Burrows
lemma_filtered_texts = []  # леммы без стоп-слов — для тематических n-gram
pos_texts = []             # POS-последовательности — для POS n-gram

print("Обрабатываю тексты через spaCy…")
for i, r in df.iterrows():
    text = str(r["text"])
    doc  = nlp(text)

    feats = {"id": i, "genre": r.get("genre", ""), "title": r.get("title", "")}
    feats.update(char_features(text))
    feats.update(word_features(doc, text))
    feats.update(pos_features(doc))
    feats.update(morph_features(doc))
    feats.update(syntax_features(doc, text))
    feats.update(dep_features(doc))
    rows.append(feats)

    docs_texts.append((doc, text))
    lemma_all_texts.append(" ".join(
        t.lemma_.lower() for t in doc if t.is_alpha
    ))
    lemma_filtered_texts.append(" ".join(
        t.lemma_.lower() for t in doc if t.is_alpha and not t.is_stop
    ))
    pos_texts.append(" ".join(
        t.pos_ for t in doc if not t.is_space and not t.is_punct
    ))

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(df)}")

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUT / "corpus_numeric_features.csv",
               index=False, encoding="utf-8-sig")
print(f"Числовые признаки: {feat_df.shape}")

# ============================================================
# 7. ТЕМАТИЧЕСКИЕ: word 1–2-граммы (TF-IDF)
# ============================================================
tf_word = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                          max_features=3000, sublinear_tf=True)
X_word = tf_word.fit_transform(lemma_filtered_texts)
save_npz(OUT / "tfidf_word12.npz", X_word)
with open(OUT / "tfidf_word12_vocab.json", "w", encoding="utf-8") as f:
    json.dump(tf_word.get_feature_names_out().tolist(), f, ensure_ascii=False)
print(f"word 1-2-граммы: {X_word.shape}")

# ============================================================
# 8. НИЗКОУРОВНЕВЫЕ: char 3–5-граммы (TF-IDF)
# ============================================================
tf_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                          min_df=3, max_features=3000, sublinear_tf=True)
X_char = tf_char.fit_transform([t for _, t in docs_texts])
save_npz(OUT / "tfidf_char35.npz", X_char)
with open(OUT / "tfidf_char35_vocab.json", "w", encoding="utf-8") as f:
    json.dump(tf_char.get_feature_names_out().tolist(), f, ensure_ascii=False)
print(f"char 3-5-граммы: {X_char.shape}")

# ============================================================
# 9. СТРУКТУРА: POS 1–3-граммы (TF-IDF)
# ============================================================
tf_pos = TfidfVectorizer(ngram_range=(1, 3), min_df=2,
                         max_features=2000, sublinear_tf=True)
X_pos = tf_pos.fit_transform(pos_texts)
save_npz(OUT / "tfidf_pos13.npz", X_pos)
with open(OUT / "tfidf_pos13_vocab.json", "w", encoding="utf-8") as f:
    json.dump(tf_pos.get_feature_names_out().tolist(), f, ensure_ascii=False)
print(f"POS 1-3-граммы: {X_pos.shape}")

# ============================================================
# 10. ДЕЛЬТА БАРРОУЗА: топ-N слов + z-оценки
# ============================================================
TOP_N = 200
all_cnt = Counter()
for t in lemma_all_texts:
    all_cnt.update(t.split())

top_words = [w for w, _ in all_cnt.most_common(TOP_N)]

fw = np.zeros((len(lemma_all_texts), TOP_N))
for i, t in enumerate(lemma_all_texts):
    tokens = t.split()
    n = len(tokens)
    if n == 0:
        continue
    c = Counter(tokens)
    for j, w in enumerate(top_words):
        fw[i, j] = c.get(w, 0) / n

mu  = fw.mean(axis=0)
sd  = fw.std(axis=0); sd[sd == 0] = 1
fw_z = (fw - mu) / sd

fw_df = pd.DataFrame(fw_z, columns=[f"fw_{w}" for w in top_words])
fw_df.to_csv(OUT / "function_words_z.csv",
             index=False, encoding="utf-8-sig")
with open(OUT / "top_function_words.json", "w", encoding="utf-8") as f:
    json.dump(top_words, f, ensure_ascii=False)
print(f"Delta Burrows: {fw_z.shape}")

# ============================================================
# 11. ЕДИНАЯ ТАБЛИЦА (числовые + Delta Burrows)
# ============================================================
combined = pd.concat(
    [feat_df.drop(columns=["title"]), fw_df],
    axis=1
)
combined.to_csv(OUT / "corpus_features_full.csv",
                index=False, encoding="utf-8-sig")
print(f"Итоговая таблица признаков: {combined.shape}")

print("\nГотово. Файлы в папке:", OUT.resolve())
for p in sorted(OUT.iterdir()):
    print("  ", p.name)