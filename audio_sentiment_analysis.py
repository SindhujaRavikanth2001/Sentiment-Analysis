#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline
from textblob import TextBlob
from sklearn.metrics import cohen_kappa_score, confusion_matrix

# =============================
# SETTINGS & PATHS
# =============================
INPUT_DIR = r"C:\Materials\Research\Scrappy\webscraper\webscraper\spiders\Poll\OneDrive_2025-04-30\Quant+Qual Project Data\Audio Question Zip Files\Q4_Speech Recognitions"
LOCAL_MODELS_DIR = r"C:\Materials\Research\Scrappy\webscraper\webscraper\spiders\Poll\LLM"
RESULTS_DIR = r"C:\Materials\Research\Scrappy\webscraper\webscraper\spiders\Poll\LLM Sentiment\Q4"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "audio_sentiment_results.csv")
SUMMARY_XLSX = os.path.join(RESULTS_DIR, "audio_sentiment_summary.xlsx")

os.makedirs(RESULTS_DIR, exist_ok=True)

# Topics keywords for sentiment-by-topic charts
topics = {
    "Immigration": ["immigration","migrant","refugee","border","asylum"],
    "Abortion": ["abortion","pro-choice","pro-life","reproductive","roe v wade","right to life"],
    "Election": ["election","vote","candidate","ballot","campaign","polling","electoral"],
    "Healthcare": ["healthcare","medical","insurance","hospital","doctor","patient","prescription"],
    "Protest": ["protest","demonstration","riot","march","activist","civil rights"],
    "Coronavirus/Covid-19": ["coronavirus","covid","covid-19","pandemic","vaccine","mask","quarantine"],
    "Mail-in Ballot": ["mail-in ballot","absentee voting","mail vote","postal vote"],
    "Early Voting": ["early voting","advance voting","pre-election voting"],
    "Economy": ["economy","economic","jobs","unemployment","inflation","stock market","wage","tax"],
    "Black Lives Matter/BLM": ["black lives matter","blm","racial justice"],
    "Race Relations": ["race relations","racial equality","racism","discrimination","prejudice"],
    "Community": ["community","neighborhood","local","town","city","civic"]
}

# Map HF labels to human-readable
HF_LABEL_MAP = {
    "LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive",
    0: "Negative", 1: "Neutral", 2: "Positive",
    "NEGATIVE": "Negative", "NEUTRAL": "Neutral", "POSITIVE": "Positive"
}

# Map TextBlob polarity to category
def map_textblob(pol):
    if pol > 0.1:
        return "Positive"
    if pol < -0.1:
        return "Negative"
    return "Neutral"

# Check keywords in text
def check_keywords(text):
    low = text.lower()
    return {t: any(kw in low for kw in kws) for t, kws in topics.items()}

# Load speech transcripts
def load_texts(root_dir):
    recs = []
    for fn in os.listdir(root_dir):
        if fn.lower().endswith('.txt'):
            fid = os.path.splitext(fn)[0]
            path = os.path.join(root_dir, fn)
            try:
                txt = open(path, encoding='utf-8').read().strip()
            except:
                txt = ''
            if txt:
                recs.append({'file_id': fid, 'text': txt})
    return pd.DataFrame(recs)

# Main processing and visualizations
def main():
    df = load_texts(INPUT_DIR)

    # Initialize pipelines
    tw_pipe = pipeline("sentiment-analysis",
                       model=os.path.join(LOCAL_MODELS_DIR, "twitter_roberta"),
                       tokenizer=os.path.join(LOCAL_MODELS_DIR, "twitter_roberta"))
    em_pipe = pipeline("text-classification",
                       model=os.path.join(LOCAL_MODELS_DIR, "distilbert_emotion"),
                       tokenizer=os.path.join(LOCAL_MODELS_DIR, "distilbert_emotion"))
    db_pipe = pipeline("sentiment-analysis",
                       model=os.path.join(LOCAL_MODELS_DIR, "deberta_sentiment"),
                       tokenizer=os.path.join(LOCAL_MODELS_DIR, "deberta_sentiment"))

    # Process transcripts
    results = []
    for _, r in tqdm(df.iterrows(), total=df.shape[0], desc="Analyzing transcripts"):
        fid, txt = r['file_id'], r['text']
        tw_res = tw_pipe(txt[:1000])[0]
        db_res = db_pipe(txt[:1000])[0]
        em_res = em_pipe(txt[:1000])[0]
        tb_pol = TextBlob(txt).sentiment.polarity

        rec = {
            'file_id': fid,
            'twitter_label': HF_LABEL_MAP.get(tw_res['label']), 'twitter_score': tw_res.get('score'),
            'deberta_label': HF_LABEL_MAP.get(db_res['label']), 'deberta_score': db_res.get('score'),
            'emotion_label': em_res.get('label'), 'emotion_score': em_res.get('score'),
            'textblob_polarity': tb_pol, 'textblob_label': map_textblob(tb_pol)
        }
        rec.update({f"keyword_{t}": v for t, v in check_keywords(txt).items()})
        results.append(rec)

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

    labels = ["Negative", "Neutral", "Positive"]

    # 1. Confusion matrices with axis labels
    cm_pairs = [
        ("Twitter_vs_DeBERTa", 'twitter_label', 'deberta_label'),
        ("Twitter_vs_TextBlob", 'twitter_label', 'textblob_label'),
        ("DeBERTa_vs_TextBlob", 'deberta_label', 'textblob_label')
    ]
    for name, c1, c2 in cm_pairs:
        cm = confusion_matrix(df_res[c1], df_res[c2], labels=labels)
        k = cohen_kappa_score(df_res[c1], df_res[c2])
        model1 = c1.replace('_label','').capitalize()
        model2 = c2.replace('_label','').capitalize()
        plt.figure()
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.title(f"{name} ConfMatrix κ={k:.2f}")
        plt.xlabel(f"{model2} Predicted")
        plt.ylabel(f"{model1} True")
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, cm[i, j], ha='center', va='center',
                         color='white' if cm[i, j] > cm.max()/2 else 'black')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"conf_{name}.png"))
        plt.close()

    # 2. Overall sentiment distribution
    twc = df_res.twitter_label.value_counts().reindex(labels, fill_value=0)
    dbc = df_res.deberta_label.value_counts().reindex(labels, fill_value=0)
    tbc = df_res.textblob_label.value_counts().reindex(labels, fill_value=0)
    x = np.arange(len(labels)); w = 0.25
    plt.figure()
    plt.bar(x-w, twc, w, label="Twitter")
    plt.bar(x,   dbc, w, label="DeBERTa")
    plt.bar(x+w, tbc, w, label="TextBlob")
    plt.xticks(x, labels)
    plt.title("Overall Sentiment Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "overall_sent_dist.png"))
    plt.close()

    # 3. Sentiment by topic
    for col, name in [("twitter_label","Twitter"),("deberta_label","DeBERTa"),("textblob_label","TextBlob")]:
        data = np.array([
            df_res[df_res[f"keyword_{t}"]][col].value_counts().reindex(labels, fill_value=0).values
            for t in topics.keys()
        ])
        x = np.arange(len(topics)); w = 0.25
        plt.figure(figsize=(12,6))
        for i, lbl in enumerate(labels):
            plt.bar(x+(i-1)*w, data[:,i], w, label=lbl)
        plt.xticks(x, list(topics.keys()), rotation=45, ha='right')
        plt.title(f"{name} Sentiment by Topic")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{name.lower()}_sent_by_topic.png"))
        plt.close()

    # 4. Emotion distribution
    ec = df_res.emotion_label.value_counts()
    plt.figure()
    ec.plot(kind='bar', color='skyblue')
    plt.title("Emotion Distribution")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emotion_distribution.png"))
    plt.close()

    # Build summary including predictions
    summary_records = []
    same3 = df_res[df_res[['twitter_label','deberta_label','textblob_label']].nunique(axis=1)==1]
    diff3 = df_res[df_res[['twitter_label','deberta_label','textblob_label']].nunique(axis=1)==3]
    pairs = [('twitter_label','deberta_label'),('twitter_label','textblob_label'),('deberta_label','textblob_label')]

    def append_records(df_subset, group_name, n=5):
        for _, row in df_subset.head(n).iterrows():
            summary_records.append({
                'group': group_name,
                'file_id': row.file_id,
                'twitter_label': row.twitter_label,
                'deberta_label': row.deberta_label,
                'textblob_label': row.textblob_label
            })

    append_records(same3, 'all_three_same')
    append_records(diff3, 'all_three_diff')
    for m1, m2 in pairs:
        same2 = df_res[df_res[m1]==df_res[m2]]
        diff2 = df_res[df_res[m1]!=df_res[m2]]
        name1, name2 = m1.split('_')[0], m2.split('_')[0]
        append_records(same2, f'{name1}_{name2}_same')
        append_records(diff2, f'{name1}_{name2}_diff')

    summary_df = pd.DataFrame(summary_records)

    # Save summary as Excel or CSV
    try:
        import openpyxl
        summary_df.to_excel(SUMMARY_XLSX, sheet_name='summary', index=False)
        print(f"Excel summary saved to {SUMMARY_XLSX}")
    except ImportError:
        csv_out = SUMMARY_XLSX.replace('.xlsx','.csv')
        summary_df.to_csv(csv_out, index=False)
        print(f"openpyxl not available; summary CSV saved to {csv_out}")

    # STATISTICAL ANALYSIS
    NUMERIC_MAP = {'Negative': -1, 'Neutral': 0, 'Positive': 1}
    df_res['numeric_twitter'] = df_res['twitter_label'].map(NUMERIC_MAP)
    df_res['numeric_deberta'] = df_res['deberta_label'].map(NUMERIC_MAP)
    df_res['numeric_textblob'] = df_res['textblob_label'].map(NUMERIC_MAP)

        # Repeated measures ANOVA (using provided pattern)
    try:
        import statsmodels.api as sm
        # Prepare for repeated measures ANOVA
        df_an = df_res.dropna(subset=["twitter_label", "deberta_label", "textblob_label"])
        df_long = pd.melt(
            df_an,
            id_vars=["file_id"],
            value_vars=["twitter_label", "deberta_label", "textblob_label"],
            var_name="model",
            value_name="score"
        )
        # Convert categorical labels to numeric scores
        df_long["score"] = df_long["score"].map(NUMERIC_MAP)
        # Map model codes to readable names
        df_long["model"] = df_long["model"].map({
            "twitter_label": "Twitter-roberta",
            "deberta_label": "DeBERTa-v3",
            "textblob_label": "TextBlob"
        })
        # Fit repeated measures ANOVA model
        ols = sm.OLS.from_formula('score ~ C(model) + C(file_id)', data=df_long).fit()
        anova_table = sm.stats.anova_lm(ols, typ=2)
        print("Repeated Measures ANOVA results (based on melted data):")
        print(anova_table)
        anova_csv = os.path.join(RESULTS_DIR, 'anova_results.csv')
        anova_table.to_csv(anova_csv)
        print(f"ANOVA table saved to {anova_csv}")
    except ImportError:
        print("statsmodels not installed; skipping repeated measures ANOVA")

    # One-way ANOVA
    try:
        from scipy.stats import f_oneway
        f_stat, p_val = f_oneway(df_res['numeric_twitter'], df_res['numeric_deberta'], df_res['numeric_textblob'])
        print(f"One-way ANOVA: F={f_stat:.2f}, p={p_val:.4f}")
        with open(os.path.join(RESULTS_DIR, 'oneway_anova.txt'), 'w') as f:
            f.write(f"F={f_stat:.2f}\np={p_val:.4f}")
    except ImportError:
        print("scipy not installed; skipping one-way ANOVA")

if __name__ == '__main__':
    main()
