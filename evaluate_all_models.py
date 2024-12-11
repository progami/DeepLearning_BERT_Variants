#!/usr/bin/env python
import os
import json
import nltk
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from statistics import mean
from datasets import load_dataset

# Ensure required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def normalize_answer(s):
    import string
    def remove_articles(text):
        return " ".join([w for w in text.split() if w.lower() not in ('a', 'an', 'the')])
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(s.lower())))

def get_tokens(s):
    if not s:
        return []
    return nltk.word_tokenize(normalize_answer(s))

def exact_match_score(prediction, ground_truths):
    normalized_prediction = normalize_answer(prediction)
    return any(normalized_prediction == normalize_answer(gt) for gt in ground_truths)

def f1_score(prediction, ground_truths):
    pred_toks = get_tokens(prediction)
    scores = []
    for gt in ground_truths:
        gt_toks = get_tokens(gt)
        common = set(pred_toks) & set(gt_toks)
        num_common = sum(min(pred_toks.count(w), gt_toks.count(w)) for w in common)
        if len(pred_toks) == 0 or len(gt_toks) == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_toks) if len(pred_toks) else 0
            recall = num_common / len(gt_toks) if len(gt_toks) else 0
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)
        scores.append(f1)
    return max(scores) if scores else 0.0

def load_predictions(predictions_path):
    with open(predictions_path, "r") as f:
        preds = json.load(f)
    return {str(item["id"]): item["prediction_text"] for item in preds}

def load_squad_references():
    squad = load_dataset("squad")
    ground_truths = {}
    for item in squad['validation']:
        qid = str(item['id'])
        refs = item['answers']['text']
        if not refs:
            refs = ['']
        ground_truths[qid] = refs
    return ground_truths

def main():
    print("Evaluating using SQuAD dataset references...")
    ground_truths = load_squad_references()

    current_dir = os.getcwd()
    dirs = [d for d in os.listdir(current_dir) if os.path.isdir(d)]
    model_dirs = []
    for d in dirs:
        predictions_path = os.path.join(d, "predictions.json")
        if os.path.exists(predictions_path):
            model_dirs.append(d)

    if not model_dirs:
        print("No model directories with predictions.json found.")
        return

    with open("results_metrics.txt", "w") as outfile:
        outfile.write("Evaluation Results for All Models:\n\n")

        for model_dir in model_dirs:
            predictions_path = os.path.join(model_dir, "predictions.json")
            predictions = load_predictions(predictions_path)

            em_scores = []
            f1_scores = []
            filtered_refs = []
            filtered_hyps = []

            for qid, pred_ans in predictions.items():
                if qid in ground_truths:
                    refs = ground_truths[qid]
                    em = 1.0 if exact_match_score(pred_ans, refs) else 0.0
                    f1 = f1_score(pred_ans, refs)
                    em_scores.append(em)
                    f1_scores.append(f1)
                    filtered_refs.append(refs[0])
                    filtered_hyps.append(pred_ans)

            avg_em = mean(em_scores)*100 if em_scores else 0.0
            avg_f1 = mean(f1_scores)*100 if f1_scores else 0.0

            # Compute BLEU
            bleu_score = 0.0
            if filtered_refs and filtered_hyps:
                tokenized_hyps = [nltk.word_tokenize(h) for h in filtered_hyps]
                tokenized_refs = [[nltk.word_tokenize(r)] for r in filtered_refs]
                # Ensure no division by zero
                if all(len(ref[0]) > 0 for ref in tokenized_refs) and all(len(h) > 0 for h in tokenized_hyps):
                    try:
                        bleu_score = nltk.translate.bleu_score.corpus_bleu(tokenized_refs, tokenized_hyps)
                    except ZeroDivisionError:
                        bleu_score = 0.0

            # ROUGE
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge1_scores = []
            rouge2_scores = []
            rougel_scores = []
            for hyp, ref in zip(filtered_hyps, filtered_refs):
                scores = scorer.score(ref, hyp)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougel_scores.append(scores['rougeL'].fmeasure)
            avg_rouge1 = mean(rouge1_scores) if rouge1_scores else 0.0
            avg_rouge2 = mean(rouge2_scores) if rouge2_scores else 0.0
            avg_rougel = mean(rougel_scores) if rougel_scores else 0.0

            # BERTScore
            bert_f1 = 0.0
            if filtered_refs and filtered_hyps:
                P, R, F1 = bert_score(filtered_hyps, filtered_refs, lang="en")
                bert_f1 = float(F1.mean()) if len(F1) > 0 else 0.0

            print("\nModel:", model_dir)
            print(f"Exact Match (EM): {avg_em:.2f}%")
            print(f"F1 Score: {avg_f1:.2f}%")
            print(f"BLEU: {bleu_score:.4f}")
            print(f"ROUGE-1 (F1): {avg_rouge1:.4f}")
            print(f"ROUGE-2 (F1): {avg_rouge2:.4f}")
            print(f"ROUGE-L (F1): {avg_rougel:.4f}")
            print(f"BERTScore F1: {bert_f1:.4f}")

            outfile.write(f"Model: {model_dir}\n")
            outfile.write(f"Exact Match (EM): {avg_em:.2f}%\n")
            outfile.write(f"F1 Score: {avg_f1:.2f}%\n")
            outfile.write(f"BLEU: {bleu_score:.4f}\n")
            outfile.write(f"ROUGE-1 (F1): {avg_rouge1:.4f}\n")
            outfile.write(f"ROUGE-2 (F1): {avg_rouge2:.4f}\n")
            outfile.write(f"ROUGE-L (F1): {avg_rougel:.4f}\n")
            outfile.write(f"BERTScore F1: {bert_f1:.4f}\n\n")

        print("\nAll model results have been saved to results_metrics.txt")
        outfile.write("All model results have been saved here.\n")

if __name__ == "__main__":
    main()

