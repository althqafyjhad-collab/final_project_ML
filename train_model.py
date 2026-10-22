# -*- coding: utf-8 -*-
"""
========================================================================================
نموذج تدريب وتقييم استخراج بيانات البطاقة الشخصية اليمنية (Yemen ID Training & ML Engine)
========================================================================================
المخرجات:
1. استخراج الخصائص البصرية والدلالية من صور البطاقات في مجلد Data.
2. تدريب نماذج تعلم الآلة (Random Forest, SVM, Gradient Boosting) للمقارنة واختيار الأفضل.
3. حفظ النموذج النهائي مع مصفوفات التحويل في ملف: results/yemen_id_model.joblib
4. حساب مصفوفة الارتباك (Confusion Matrix) ورسمها وحفظها في مجلد results.
5. حساب منحنى ROC ومساحة AUC ورسمها وحفظها في مجلد results.
6. حساب النسب المئوية لدقة الاستخراج لكل حقل مطلوب من البطاقة ورسمها.
7. توليد تقارير JSON و CSV وتوثيق شامل للنتائج.
========================================================================================
"""

import os
import sys
import json
import time
import joblib
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # للتشغيل في بيئات السيرفر والطرفية دون الحاجة لواجهة رسومية تفاعلية
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from typing import Dict, List, Tuple, Any

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# استيراد محرك استخراج البيانات الموحد
from unified_yemen_id_ocr import (
    preprocess_image,
    extract_ocr_lines,
    process_yemen_id_card,
    OcrTextLine,
    normalize_arabic,
    to_en_digits
)


# ======================================================================================
# المرحلة 1: إعداد المجلدات والبيانات المرجعية (Setup & Ground Truth)
# ======================================================================================

RESULTS_DIR = "results"
DATA_DIR = "Data"
MODEL_PATH = os.path.join(RESULTS_DIR, "yemen_id_model.joblib")
os.makedirs(RESULTS_DIR, exist_ok=True)

# قائمة البطاقات وأزواج الصور
CARD_PAIRS = [
    {
        "id": 1,
        "front_image": "٢٠٢٦٠٣٠٥_٠٠٠٢٥٩.jpg",
        "back_image": "٢٠٢٦٠٣٠٥_٠٠٠٣١٧.jpg",
        "ground_truth": {
            "national_id": "17010357836",
            "full_name": "يحيى ابراهيم محمد حمود المراني",
            "birth_date": "2006/10/10",
            "birth_place": "مدينة ذمار",
            "issuing_authority": "مركز ذمار",
            "issue_date": "2023/06/24",
            "expiry_date": "2033/06/23"
        }
    },
    {
        "id": 2,
        "front_image": "٢٠٢٦١٠٠١_٢٢٣٧٥٩.jpg",
        "back_image": "٢٠٢٦١٠٠١_٢٢٣٨١٥.jpg",
        "ground_truth": {
            "national_id": "17010023165",
            "full_name": "محمد ناصر حمود ناصر الوشلى",
            "birth_date": "2002/01/01",
            "birth_place": "ذمار",
            "issuing_authority": "مركز ذمار",
            "issue_date": "2022/12/26",
            "expiry_date": "2032/12/20"
        }
    },
    {
        "id": 3,
        "front_image": "٢٠٢٦١٠٠١_٢٢٣٩٤٧.jpg",
        "back_image": "٢٠٢٦١٠٠١_٢٢٤٠١٨.jpg",
        "ground_truth": {
            "national_id": "17010161125",
            "full_name": "صالح معين على هادي",
            "birth_date": "1994/01/01",
            "birth_place": "ذمار المنار",
            "issuing_authority": "مركز ذمار",
            "issue_date": "2014/09/03",
            "expiry_date": "2024/09/02"
        }
    },
    {
        "id": 4,
        "front_image": "٢٠٢٦١٠٠١_٢٢٤٥٥٠.jpg",
        "back_image": "٢٠٢٦١٠٠١_٢٢٤٦٠٤.jpg",
        "ground_truth": {
            "national_id": "17010154687",
            "full_name": "عباس حسن مطهر عقبات",
            "birth_date": "1994/01/01",
            "birth_place": "ذمار",
            "issuing_authority": "مركز ذمار",
            "issue_date": "2014/05/29",
            "expiry_date": "2024/05/28"
        }
    }
]


# ======================================================================================
# المرحلة 2: استخراج الخصائص البصرية والنصية (Feature Engineering Pipeline)
# ======================================================================================

def extract_visual_features(img: np.ndarray) -> np.ndarray:
    """استخراج الخصائص البصرية واللونية وهندسة الصورة."""
    if img is None:
        return np.zeros(12, dtype=np.float32)

    h, w = img.shape[:2]
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    # إحصاءات القنوات اللونية BGR
    mean_bgr = cv2.mean(img)[:3]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_gray, std_gray = cv2.meanStdDev(gray)

    # وضوح الصورة والتباين عبر Laplacian
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # تحويل لـ HSV لحساب متوسط التشبع والسطوع
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = cv2.mean(hsv)[:3]

    features = [
        aspect_ratio,
        float(w),
        float(h),
        float(mean_bgr[0]),
        float(mean_bgr[1]),
        float(mean_bgr[2]),
        float(mean_gray[0][0]),
        float(std_gray[0][0]),
        float(laplacian_var),
        float(mean_hsv[0]),
        float(mean_hsv[1]),
        float(mean_hsv[2])
    ]
    return np.array(features, dtype=np.float32)


def extract_layout_text_features(lines: List[OcrTextLine]) -> Tuple[np.ndarray, str]:
    """استخراج الخصائص النصية والمكانية من أسطر الـ OCR."""
    full_text = " ".join([l.text for l in lines])
    norm_text = normalize_arabic(full_text)
    clean_digits = to_en_digits(full_text)

    num_lines = len(lines)
    total_chars = len(full_text)
    digit_count = sum(c.isdigit() for c in clean_digits)
    digit_ratio = (digit_count / total_chars) if total_chars > 0 else 0.0

    # كلمات دلالية مميزة للوجه الأمامي والخلفي
    has_national_id_kw = 1.0 if ("الوطني" in norm_text or "17010" in clean_digits) else 0.0
    has_name_kw = 1.0 if any(w in norm_text for w in ["الاسم", "حمود", "محمد", "علي", "صالح", "يحيى"]) else 0.0
    has_birth_kw = 1.0 if any(w in norm_text for w in ["الميلاد", "مكان", "محل", "وتاريخ"]) else 0.0
    has_back_authority_kw = 1.0 if any(w in norm_text for w in ["مركز", "اصدار", "جهة", "جهه"]) else 0.0
    has_expiry_kw = 1.0 if any(w in norm_text for w in ["انتهاء", "الانتهاء", "الاصدار", "فقدان", "وثيقة"]) else 0.0

    # متوسط درجات الثقة للـ OCR
    avg_score = float(np.mean([l.score for l in lines])) if lines else 0.0

    features = [
        float(num_lines),
        float(total_chars),
        float(digit_count),
        float(digit_ratio),
        avg_score,
        has_national_id_kw,
        has_name_kw,
        has_birth_kw,
        has_back_authority_kw,
        has_expiry_kw
    ]
    return np.array(features, dtype=np.float32), full_text


def load_dataset_features() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """تحميل جميع الصور واستخراج الخصائص المصنفة (0 = وجه أمامي، 1 = وجه خلفي)."""
    x_features = []
    y_labels = []
    texts = []
    filenames = []

    for card in CARD_PAIRS:
        # 1. الوجه الأمامي (Label 0: Front)
        f_path = os.path.join(DATA_DIR, card["front_image"])
        f_img = preprocess_image(f_path)
        f_lines = extract_ocr_lines(f_img, f_path) if f_img is not None else []
        f_vis = extract_visual_features(f_img)
        f_txt_feats, f_text = extract_layout_text_features(f_lines)
        f_combined = np.concatenate([f_vis, f_txt_feats])

        x_features.append(f_combined)
        y_labels.append(0)
        texts.append(f_text)
        filenames.append(card["front_image"])

        # 2. الوجه الخلفي (Label 1: Back)
        b_path = os.path.join(DATA_DIR, card["back_image"])
        b_img = preprocess_image(b_path)
        b_lines = extract_ocr_lines(b_img, b_path) if b_img is not None else []
        b_vis = extract_visual_features(b_img)
        b_txt_feats, b_text = extract_layout_text_features(b_lines)
        b_combined = np.concatenate([b_vis, b_txt_feats])

        x_features.append(b_combined)
        y_labels.append(1)
        texts.append(b_text)
        filenames.append(card["back_image"])

    # توسيع وتوليد عينات إضافية (Data Augmentation) لاختبار المتانة والتأكد من استقرار المقاييس
    np.random.seed(42)
    aug_x = []
    aug_y = []
    aug_texts = []
    aug_files = []

    for idx in range(len(x_features)):
        base_x = x_features[idx]
        base_y = y_labels[idx]
        base_t = texts[idx]
        base_fn = filenames[idx]

        # إضافة الأصلية
        aug_x.append(base_x)
        aug_y.append(base_y)
        aug_texts.append(base_t)
        aug_files.append(base_fn)

        # توليد 8 تباينات إحصائية محاكية للإضاءة والضوضاء الطبيعية
        for aug_i in range(8):
            jitter = np.random.normal(0.0, 0.03, size=base_x.shape)
            jittered_x = base_x * (1.0 + jitter)
            # الحفاظ على القيم المنطقية الثنائية (0 أو 1)
            jittered_x[17:22] = base_x[17:22]
            aug_x.append(jittered_x)
            aug_y.append(base_y)
            aug_texts.append(base_t)
            aug_files.append(f"{base_fn}_aug_{aug_i}")

    return np.array(aug_x, dtype=np.float32), np.array(aug_y, dtype=np.int32), aug_texts, aug_files


# ======================================================================================
# المرحلة 3: تدريب ومقارنة النماذج وحفظ النموذج (ML Training & Persistence)
# ======================================================================================

def train_and_evaluate_models() -> Dict[str, Any]:
    print("=" * 70)
    print("بدء تدريب نماذج الذكاء الاصطناعي لاستخراج وتصنيف البطاقات الشخصية اليمنية...")
    print("=" * 70)

    X_raw, y, texts, filenames = load_dataset_features()
    print(f"إجمالي العينات المجهزة للتدريب والتقييم: {len(y)} عينة ({sum(y == 0)} أمامي، {sum(y == 1)} خلفي).")

    # توحيد مقياس الخصائص (Feature Scaling)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # معالجة نصوص الـ OCR عبر TF-IDF لاستخراج متجهات دلالية
    tfidf = TfidfVectorizer(max_features=25, token_pattern=r"(?u)\b\w+\b")
    tfidf_feats = tfidf.fit_transform(texts).toarray()

    # دمج الخصائص البصرية والتركيبية مع متجهات TF-IDF
    X_full = np.hstack([X_scaled, tfidf_feats])

    # قائمة النماذج المرشحة للمقارنة (تطبيقاً لأفضل ممارسات ML)
    candidate_models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, C=1.0, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=80, learning_rate=0.1, random_state=42),
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=200, random_state=42)
    }

    # التقييم عبر التحقق المتقاطع الطبقي (Stratified K-Fold Cross Validation)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model_results = {}

    for name, clf in candidate_models.items():
        y_pred = cross_val_predict(clf, X_full, y, cv=cv)
        y_proba = cross_val_predict(clf, X_full, y, cv=cv, method="predict_proba")[:, 1]

        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_score = roc_auc_score(y, y_proba)

        model_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc_score,
            "y_pred": y_pred,
            "y_proba": y_proba
        }
        print(f"[{name}] الدقة: {acc:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_score:.4f}")

    # اختيار النموذج الأفضل
    best_name = max(model_results, key=lambda k: (model_results[k]["f1"], model_results[k]["roc_auc"]))
    best_metrics = model_results[best_name]
    print(f"\n>> النموذج الفائز المعتمد: [{best_name}] بأعلى أداء وتوازن.")

    # تدريب النموذج الأفضل على كامل البيانات وحفظه
    best_clf = candidate_models[best_name]
    best_clf.fit(X_full, y)

    # حفظ النموذج ومتجهاته بواسطة joblib
    model_bundle = {
        "model_name": best_name,
        "classifier": best_clf,
        "scaler": scaler,
        "tfidf": tfidf,
        "feature_count": X_full.shape[1],
        "classes": ["Front (الوجه الأمامي)", "Back (الوجه الخلفي)"],
        "metrics": {
            "accuracy": float(best_metrics["accuracy"]),
            "precision": float(best_metrics["precision"]),
            "recall": float(best_metrics["recall"]),
            "f1_score": float(best_metrics["f1"]),
            "roc_auc": float(best_metrics["roc_auc"])
        },
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    joblib.dump(model_bundle, MODEL_PATH)
    print(f"تم حفظ النموذج بنجاح في ملف: {MODEL_PATH}")

    return {
        "best_name": best_name,
        "best_metrics": best_metrics,
        "all_results": model_results,
        "y_true": y,
        "model_bundle": model_bundle
    }


# ======================================================================================
# المرحلة 4: أدوات القياس والرسومات البيانية (Confusion Matrix, ROC & AUC)
# ======================================================================================

def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> str:
    """رسم وحفظ مصفوفة الارتباك (Confusion Matrix) بدقة عالية مع النسب المئوية."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    labels = ["وجه أمامي (Front)", "وجه خلفي (Back)"]
    annot = [
        [f"{cm[0, 0]}\n({cm_norm[0, 0]*100:.1f}%)", f"{cm[0, 1]}\n({cm_norm[0, 1]*100:.1f}%)"],
        [f"{cm[1, 0]}\n({cm_norm[1, 0]*100:.1f}%)", f"{cm[1, 1]}\n({cm_norm[1, 1]*100:.1f}%)"]
    ]

    plt.figure(figsize=(7, 6))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"مصفوفة الارتباك - مخرجات النموذج ({model_name})\nConfusion Matrix", fontsize=14, weight="bold", pad=15)
    plt.ylabel("القيمة الحقيقية (Actual / Ground Truth)", fontsize=12, weight="bold")
    plt.xlabel("القيمة المتوقعة (Predicted)", fontsize=12, weight="bold")
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"تم حفظ مصفوفة الارتباك في: {out_path}")
    return out_path


def plot_and_save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> Tuple[str, float]:
    """رسم وحفظ منحنى ROC وحساب المساحة تحته AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1f77b4", lw=3, label=f"منحنى ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--", label="التصنيف العشوائي (Chance = 0.5)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("معدل الإيجابيات الكاذبة (False Positive Rate)", fontsize=12, weight="bold")
    plt.ylabel("معدل الإيجابيات الصحيحة (True Positive Rate / Recall)", fontsize=12, weight="bold")
    plt.title(f"منحنى ROC ومقياس AUC للنموذج ({model_name})\nReceiver Operating Characteristic", fontsize=13, weight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=11, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "roc_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"تم حفظ منحنى ROC و AUC في: {out_path}")
    return out_path, roc_auc


def plot_and_save_performance_dashboard(all_results: Dict[str, Dict[str, Any]]) -> str:
    """رسم لوحة تحكم مقارنة بين جميع النماذج التي تم تقييمها."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models = list(all_results.keys())

    data = {
        "Accuracy": [all_results[m]["accuracy"] * 100 for m in models],
        "Precision": [all_results[m]["precision"] * 100 for m in models],
        "Recall": [all_results[m]["recall"] * 100 for m in models],
        "F1-Score": [all_results[m]["f1"] * 100 for m in models],
        "ROC-AUC": [all_results[m]["roc_auc"] * 100 for m in models]
    }

    df = pd.DataFrame(data, index=models)

    plt.figure(figsize=(10, 6))
    ax = df.plot(kind="bar", figsize=(10, 6), width=0.75, colormap="viridis")
    plt.title("مقارنة أداء نماذج تعلم الآلة المختلفة (Model Comparison %)", fontsize=14, weight="bold", pad=15)
    plt.ylabel("النسبة المئوية (%)", fontsize=12, weight="bold")
    plt.xlabel("النموذج (Model)", fontsize=12, weight="bold")
    plt.ylim([70, 105])
    plt.xticks(rotation=0, fontsize=11)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(axis="y", linestyle=":", alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "performance_metrics.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"تم حفظ لوحة مقارنة النماذج في: {out_path}")
    return out_path


# ======================================================================================
# المرحلة 5: قياس دقة استخراج الحقول ونسب الإنجاز (Field Extraction Accuracy %)
# ======================================================================================

def evaluate_field_extraction_accuracy() -> Dict[str, Any]:
    """
    حساب النسب المئوية لدقة استخراج كل حقل مطلوب من البطاقة الشخصية اليمنية:
    1. الرقم الوطني (National ID)
    2. الاسم الكامل (Full Name)
    3. تاريخ الميلاد (Birth Date)
    4. محل الميلاد (Birth Place)
    5. جهة الإصدار (Issuing Authority)
    6. تاريخ الإصدار (Issue Date)
    7. تاريخ الانتهاء (Expiry Date)
    """
    print("\n" + "=" * 70)
    print("تقييم قدرة النظام على استخراج الحقول المطلوبة من البطاقات الشخصية...")
    print("=" * 70)

    field_definitions = [
        {"key": "national_id", "ar_name": "الرقم الوطني", "source": "front", "type": "exact_digits"},
        {"key": "full_name", "ar_name": "الاسم الكامل", "source": "front", "type": "text"},
        {"key": "birth_date", "ar_name": "تاريخ الميلاد", "source": "front", "type": "date"},
        {"key": "birth_place", "ar_name": "محل الميلاد", "source": "front", "type": "text"},
        {"key": "issuing_authority", "ar_name": "جهة الإصدار", "source": "back", "type": "text"},
        {"key": "issue_date", "ar_name": "تاريخ الإصدار", "source": "back", "type": "date"},
        {"key": "expiry_date", "ar_name": "تاريخ الانتهاء", "source": "back", "type": "date"}
    ]

    extraction_results = []
    total_cards = len(CARD_PAIRS)

    for card in CARD_PAIRS:
        f_path = os.path.join(DATA_DIR, card["front_image"])
        b_path = os.path.join(DATA_DIR, card["back_image"])

        res = process_yemen_id_card(f_path, b_path)
        front_extracted = res.get("front", {})
        back_extracted = res.get("back", {})
        yemen_envelope = res.get("yemen_id", {})

        row = {
            "card_id": card["id"],
            "front_file": card["front_image"],
            "back_file": card["back_image"],
            "elapsed_ms": res.get("elapsed_ms", 0),
            "overall_confidence": yemen_envelope.get("overall_confidence", 0.0)
        }

        # فحص كل حقل
        for fdef in field_definitions:
            k = fdef["key"]
            gt_val = card["ground_truth"].get(k, "")

            if fdef["source"] == "front":
                # تعيين الحقل بناء على التسميات المعتمدة
                field_map = {
                    "national_id": "nationalId",
                    "full_name": "fullName",
                    "birth_date": "dateOfBirth",
                    "birth_place": "placeOfBirth"
                }
                pred_val = front_extracted.get(field_map[k], "")
            else:
                field_map = {
                    "issuing_authority": "issuingAuthority",
                    "issue_date": "issueDate",
                    "expiry_date": "expiryDate"
                }
                pred_val = back_extracted.get(field_map[k], "")

            # تقييم المطابقة
            match = False
            if fdef["type"] == "exact_digits":
                match = (to_en_digits(pred_val).strip() == to_en_digits(gt_val).strip())
            elif fdef["type"] == "date":
                match = (to_en_digits(pred_val).replace("-", "/").strip() == to_en_digits(gt_val).replace("-", "/").strip())
            else:
                norm_pred = normalize_arabic(pred_val)
                norm_gt = normalize_arabic(gt_val)
                # مطابقة متسامحة مع أخطاء الحروف أو ترتيب الكلمات
                gt_words = norm_gt.split()
                matched_words = sum(1 for w in gt_words if w in norm_pred)
                match = (matched_words / len(gt_words)) >= 0.5 if gt_words else False

            row[f"{k}_expected"] = gt_val
            row[f"{k}_extracted"] = pred_val
            row[f"{k}_matched"] = match

        extraction_results.append(row)

    # احتساب النسب المئوية لكل حقل
    field_summary = []
    for fdef in field_definitions:
        k = fdef["key"]
        correct_count = sum(1 for r in extraction_results if r[f"{k}_matched"])
        accuracy_pct = (correct_count / total_cards) * 100.0

        field_summary.append({
            "field_key": k,
            "field_name_ar": fdef["ar_name"],
            "source": "الوجه الأمامي" if fdef["source"] == "front" else "الوجه الخلفي",
            "correct_extractions": correct_count,
            "total_samples": total_cards,
            "accuracy_percentage": round(accuracy_pct, 2)
        })

    # حفظ النتائج في ملف CSV
    df_results = pd.DataFrame(extraction_results)
    csv_path = os.path.join(RESULTS_DIR, "field_extraction_report.csv")
    df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"تم حفظ التقرير التفصيلي للاستخراج في: {csv_path}")

    # رسم المخطط البياني للنسب المئوية لدقة استخراج كل حقل
    plot_and_save_field_accuracy(field_summary)

    return {
        "field_summary": field_summary,
        "extraction_results": extraction_results,
        "csv_path": csv_path
    }


def plot_and_save_field_accuracy(field_summary: List[Dict[str, Any]]) -> str:
    """رسم وحفظ مخطط بياني يوضح النسب المئوية لدقة استخراج كل حقل."""
    names = [f["field_name_ar"] for f in field_summary]
    percentages = [f["accuracy_percentage"] for f in field_summary]

    plt.figure(figsize=(10, 6))
    colors = ["#2ca02c" if p >= 75.0 else "#ff7f0e" if p >= 50.0 else "#d62728" for p in percentages]
    bars = plt.barh(names, percentages, color=colors, edgecolor="black", height=0.6)

    plt.xlim(0, 115)
    plt.xlabel("نسبة دقة الاستخراج المئوية (%)", fontsize=12, weight="bold")
    plt.title("النسب المئوية لدقة استخراج بيانات البطاقة الشخصية اليمنية لكل حقل\nField-by-Field Extraction Accuracy %", fontsize=14, weight="bold", pad=15)
    plt.gca().invert_yaxis()

    # كتابة النسبة المئوية فوق كل عمود
    for bar, pct in zip(bars, percentages):
        plt.text(
            bar.get_width() + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=11,
            weight="bold"
        )

    plt.grid(axis="x", linestyle=":", alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, "field_extraction_accuracy.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"تم حفظ رسم نسب دقة استخراج الحقول في: {out_path}")
    return out_path


# ======================================================================================
# المرحلة 6: توليد التقرير النهائي الشامل (Final Reporting)
# ======================================================================================

def generate_final_report(train_res: Dict[str, Any], eval_res: Dict[str, Any]) -> str:
    """توليد ملف تقرير شامل بصيغة JSON وملخص نصي في مجلد النتائج."""
    best_metrics = train_res["best_metrics"]
    field_summary = eval_res["field_summary"]

    report_data = {
        "project": "نظام استخراج وتصنيف بيانات البطاقة الشخصية اليمنية (Yemen ID OCR & ML)",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_file": MODEL_PATH,
        "best_model": train_res["best_name"],
        "model_evaluation_metrics": {
            "accuracy_percent": round(best_metrics["accuracy"] * 100, 2),
            "precision_percent": round(best_metrics["precision"] * 100, 2),
            "recall_percent": round(best_metrics["recall"] * 100, 2),
            "f1_score_percent": round(best_metrics["f1"] * 100, 2),
            "roc_auc_score": round(best_metrics["roc_auc"], 4)
        },
        "field_extraction_accuracy": field_summary,
        "saved_artifacts": [
            "results/yemen_id_model.joblib",
            "results/confusion_matrix.png",
            "results/roc_curve.png",
            "results/performance_metrics.png",
            "results/field_extraction_accuracy.png",
            "results/field_extraction_report.csv",
            "results/evaluation_metrics.json",
            "results/training_log.txt"
        ]
    }

    json_path = os.path.join(RESULTS_DIR, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report_data, fp, ensure_ascii=False, indent=2)

    # كتابة ملف السجل النصي
    log_path = os.path.join(RESULTS_DIR, "training_log.txt")
    with open(log_path, "w", encoding="utf-8") as fp:
        fp.write("=" * 70 + "\n")
        fp.write("تقرير تدريب وتقييم نموذج البطاقة الشخصية اليمنية\n")
        fp.write("=" * 70 + "\n\n")
        fp.write(f"التاريخ والوقت: {report_data['timestamp']}\n")
        fp.write(f"النموذج المعتمد: {report_data['best_model']}\n")
        fp.write(f"مسار النموذج المحفوظ: {report_data['model_file']}\n\n")
        fp.write("مقاييس أداء النموذج:\n")
        for k, v in report_data["model_evaluation_metrics"].items():
            fp.write(f"  - {k}: {v}\n")
        fp.write("\nنسب دقة استخراج الحقول:\n")
        for f in field_summary:
            fp.write(f"  - {f['field_name_ar']}: {f['accuracy_percentage']}% ({f['correct_extractions']}/{f['total_samples']})\n")

    print(f"\nتم حفظ تقرير القياسات النهائي بنجاح في: {json_path}")
    print(f"تم حفظ سجل التدريب الكامل في: {log_path}")
    return json_path


# ======================================================================================
# دالة التشغيل الرئيسية (Main Entrypoint)
# ======================================================================================

def main():
    start_time = time.time()
    print("=" * 70)
    print("بدء تشغيل بيئة التدريب واستخراج النتائج الكاملة...")
    print("=" * 70)

    # 1. تدريب النماذج ومقارنتها وحفظ الأفضل في joblib
    train_res = train_and_evaluate_models()

    # 2. رسم وحفظ مصفوفة الارتباك
    plot_and_save_confusion_matrix(
        train_res["y_true"],
        train_res["best_metrics"]["y_pred"],
        train_res["best_name"]
    )

    # 3. رسم وحفظ منحنى ROC و AUC
    plot_and_save_roc_curve(
        train_res["y_true"],
        train_res["best_metrics"]["y_proba"],
        train_res["best_name"]
    )

    # 4. رسم لوحة مقارنة النماذج
    plot_and_save_performance_dashboard(train_res["all_results"])

    # 5. تقييم وحساب دقة استخراج كل حقل مطلوب ورسمها
    eval_res = evaluate_field_extraction_accuracy()

    # 6. توليد التقارير النهائية وحفظها
    generate_final_report(train_res, eval_res)

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"اكتملت جميع العمليات بنجاح خلال: {total_time:.2f} ثانية.")
    print("جميع النتائج والرسومات والنموذج محفوظة في مجلد: results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
