import re
import numpy as np
import pandas as pd
from collections import Counter
from pathlib import Path
import stanza


IN = "corpuses/learning_corpus.csv"
OUT = Path("features")
OUT.mkdir(exist_ok=True)

LIMIT = None
nlp = stanza.Pipeline(
    lang="ru",
    processors="tokenize,pos,lemma,depparse",
    use_gpu=False,
    verbose=False,
    tokenize_pretokenized=False,
)

# Загрузка
df = pd.read_csv(IN).fillna("")
if LIMIT:
    df = df.head(LIMIT).copy()
print(f"Текстов: {len(df)}")
if "genre" in df.columns:
    print("Жанры:", df["genre"].value_counts().to_dict())

texts = df["text"].astype(str).tolist()


# Прогон через Stanza
print("Запускаю Stanza…")
docs = nlp.bulk_process(texts)
print(f"Готово: {len(docs)} документов")

# Стоп-слова — короткий список самых частых служебных слов
RU_STOP = set("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы
по только ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг
ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
ничего ей может они тут где есть надо ней для мы тебя их чем был сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой
совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех
никогда можно при наконец два об другой хоть после над больше тот через эти нас
про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между
""".split())

# Извлечение признаков

def parse_feats(feats_str):
    """'Case=Nom|Number=Sing' → {'Case':'Nom','Number':'Sing'}"""
    if not feats_str:
        return {}
    d = {}
    for part in feats_str.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            d[k] = v
    return d


def char_features(text: str) -> dict:
    letters = re.findall(r"[А-Яа-яЁёA-Za-z]", text)
    upper   = re.findall(r"[А-ЯЁA-Z]", text)
    punct   = re.findall(r"[^\w\s]", text)
    return {
        # Общее число символов в тексте. Простой и грубый «объём».
        "n_chars":        len(text),
        # Число букв (кириллица + латиница). Отделяется от цифр и пунктуации.
        "n_letters":      len(letters),
        # Число заглавных букв. Косвенно ловит аббревиатуры, имена, НАЗВАНИЯ.
        "n_upper":        len(upper),
        # Число знаков пунктуации. Жанровая «плотность» запятых, тире, кавычек.
        "n_punct":        len(punct),
        # Доля пунктуации от всех символов. Нормированный вариант n_punct.
        "share_punct":    len(punct) / max(len(text), 1),
    }


def word_features(doc, text: str) -> dict:
    sents = doc.sentences
    words = [w for s in sents for w in s.words if w.upos not in ("PUNCT", "SYM", "X")]
    lemmas = [w.lemma.lower() for w in words if w.lemma]
    cnt = Counter(lemmas)

    # Гапаксы: слова, встретившиеся ровно один раз. Чем их больше — тем «свежее» лексика.
    hapax = sum(1 for _, c in cnt.items() if c == 1)

    sent_words = [sum(1 for w in s.words
                       if w.upos not in ("PUNCT", "SYM", "X"))
                  for s in sents]
    sent_chars = [len(s.text) for s in sents]
    wl         = [len(w.text) for w in words]

    n_sents = len(sents) or 1
    return {
        # Общее число предложений. Один из базовых признаков объёма.
        "n_sentences":         len(sents),

        # Общее число слов (без пунктуации).
        "n_words":             len(words),

        # Число уникальных лемм. Лексическое разнообразие в абсолютных числах.
        "n_unique_words":      len(set(lemmas)),

        # Type-Token Ratio = уникальные леммы / все слова. Чем ближе к 1 — тем разнообразнее.
        "ttr":                 len(set(lemmas)) / max(len(words), 1),

        # Доля гапаксов от уникальных слов. Характеристика богатства и «неповторимости».
        "share_hapax":         hapax / max(len(set(lemmas)), 1),

        # Средняя длина предложения в словах. Ключевой жанровый признак.
        "avg_sent_len_words":  float(np.mean(sent_words)) if sent_words else 0.0,

        # Средняя длина слова в символах. Ловит длинные термины / короткие бытовые слова.
        "avg_word_len":        float(np.mean(wl)) if wl else 0.0,

        # Запятых на предложение. Плотность обособлений — стилевой признак.
        "commas_per_sent":     text.count(",") / n_sents,

        # Тире и дефисов на предложение. У спортивных текстов часто много тире.
        "dashes_per_sent":     (text.count("—") + text.count(" - ")) / n_sents,

        # Вопросительных знаков на предложение. Косвенно — диалогичность.
        "questions_per_sent":  text.count("?") / n_sents,

        # Восклицательных знаков на предложение. Эмоциональность.
        "exclam_per_sent":     text.count("!") / n_sents,
    }


POS_LIST = ["NOUN","VERB","ADJ","ADV","PRON","ADP","CCONJ","SCONJ",
            "PART","DET","NUM","PROPN","INTJ","AUX"]


def pos_features(doc) -> dict:
    counts = Counter(w.upos for s in doc.sentences
                     for w in s.words
                     if w.upos not in ("PUNCT", "SYM", "X"))
    total = sum(counts.values()) or 1
    return {
        # Доля существительных. У news обычно выше, чем у аналитики.
        f"pos_{p.lower()}_share": counts.get(p, 0) / total for p in POS_LIST
        # Раскрывается в:
        # pos_noun_share  — существительные
        # pos_verb_share  — глаголы
        # pos_adj_share   — прилагательные
        # pos_adv_share   — наречия
        # pos_pron_share  — местоимения
        # pos_adp_share   — предлоги
        # pos_cconj_share — сочинительные союзы (и, но, а)
        # pos_sconj_share — подчинительные союзы (что, чтобы, если)
        # pos_part_share  — частицы (не, же, ли)
        # pos_det_share   — определители
        # pos_num_share   — числительные
        # pos_propn_share — имена собственные
        # pos_intj_share  — междометия
        # pos_aux_share   — вспомогательные глаголы (быть)
    }


def morph_features(doc) -> dict:
    verbs = past = pres = fut = 0
    noun_sg = noun_pl = 0
    adj_short = adj_total = 0
    case_counts = Counter()
    for s in doc.sentences:
        for w in s.words:
            f = parse_feats(w.feats)
            if w.upos == "VERB":
                verbs += 1
                t = f.get("Tense")
                if t == "Past":  past += 1
                elif t == "Pres": pres += 1
                elif t == "Fut":  fut += 1
            if w.upos == "NOUN":
                n = f.get("Number")
                if n == "Sing":  noun_sg += 1
                elif n == "Plur": noun_pl += 1
                case_counts[f.get("Case")] += 1
            if w.upos == "ADJ":
                adj_total += 1
                if f.get("Variant") == "Short" or w.xpos == "KR":
                    adj_short += 1

    noun_total = noun_sg + noun_pl or 1
    case_total = sum(case_counts.values()) or 1
    return {
        # Доля глаголов в прошедшем времени. В репортаже ↑, в новостях — фон.
        "verb_past_share": past / max(verbs, 1),

        # Доля глаголов в настоящем времени. Новостной стиль «сейчас сообщает».
        "verb_pres_share": pres / max(verbs, 1),

        # Доля глаголов в будущем времени. Прогнозы/анонсы.
        "verb_fut_share":  fut  / max(verbs, 1),

        # Доля существительных в единственном числе. Обратное — множественное.
        "noun_sing_share": noun_sg / noun_total,

        # Доля кратких прилагательных. Признак более книжного/официального стиля.
        "adj_short_share": adj_short / max(adj_total, 1),

        # Доли падежей среди существительных.
        # Именительный: подлежащие — повествовательная плотность.
        "case_nom_share":  case_counts.get("Nom", 0) / case_total,

        # Родительный: определения, принадлежность — характерно для аналитики.
        "case_gen_share":  case_counts.get("Gen", 0) / case_total,

        # Дательный: адресат — реже, но показателен для интервью.
        "case_dat_share":  case_counts.get("Dat", 0) / case_total,

        # Винительный: прямые объекты, действие направлено на кого-то.
        "case_acc_share":  case_counts.get("Acc", 0) / case_total,

        # Творительный: инструмент, действующее лицо.
        "case_ins_share":  case_counts.get("Ins", 0) / case_total,

        # Предложный: «о чём», «в чём» — тематическая привязка.
        "case_loc_share":  case_counts.get("Loc", 0) / case_total,
    }

rows = []
lemma_all, lemma_no_stop, pos_seq = [], [], []

print("Извлекаю признаки…")
for i, (doc, text) in enumerate(zip(docs, texts)):
    feats = {"id": i, "genre": df.iloc[i].get("genre", ""),
             "title": df.iloc[i].get("title", "")}
    feats.update(char_features(text))
    feats.update(word_features(doc, text))
    feats.update(pos_features(doc))
    feats.update(morph_features(doc))
    rows.append(feats)

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{len(docs)}")

feat_df = pd.DataFrame(rows)
feat_df.to_csv(OUT / "corpus_numeric_features.csv",
               index=False, encoding="utf-8-sig")
print(f"Числовые признаки: {feat_df.shape}")
print("Файлы в:", OUT.resolve())
for p in sorted(OUT.iterdir()):
    print("  ", p.name)