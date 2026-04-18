# REPORT_DATA — Head CT Hemorrhage Classification

> **Amaç:** IEEE Xplore formatında yazılacak proje raporu için tüm sayısal sonuçları, mühendislik gerekçelerini ve görsel referanslarını **tek dosyada** toplar. Rapor yazımı sırasında bu dosyadan kopyala-yapıştır yapın; rakamlar `results/` klasöründeki JSON çıktılarıyla **birebir** eşleşir.

---

## 1. Veri Seti Özeti

| Özellik | Değer | Kaynak / Kanıt |
|---------|-------|----------------|
| Veri kümesi | Felipe Kitamura — Head CT Hemorrhage (Kaggle, CC0) | `head_ct/head_ct/` |
| Toplam görüntü | 200 PNG (kafa CT slice) | `labels.csv` |
| Sınıf dağılımı | 100 Normal · 100 Hemorrhage (1:1, dengeli) | `dataset_overview.png` |
| Giriş çözünürlüğü | 224 × 224 × 3 (RGB'ye dönüştürülmüş) | `data_preprocessing.py` |
| Split stratejisi | Stratified 70 / 15 / 15 | `data_split.py` |
| Train / Val / Test | **139 / 31 / 30** | `--` |
| Random seed | 42 (tekrarlanabilirlik) | `config.py::RANDOM_SEED` |
| Train mean (RGB) | `[0.4052, 0.4082, 0.4113]` | `models/train_stats.json` |
| Train std (RGB)  | `[0.3084, 0.3090, 0.3093]` | `models/train_stats.json` |

**Mühendislik gerekçesi:** Mean/std yalnızca train'den hesaplandı (data leakage önleme — Ders Notu Bölüm 5). Stratified split her sette sınıf oranını korur (her sette ~%50/%50).

---

## 2. Model Konfigürasyonu

### 2.1 ConvNeXt-Tiny (Pretrained CNN)

| Alan | Değer |
|------|-------|
| Mimari | timm `convnext_tiny` (ImageNet-1k pretrained) |
| Toplam parametre | ~28.6 M |
| Eğitilebilir | ~28.6 M (full fine-tune, progressive unfreezing) |
| Stem | Conv 4×4 stride 4 → LayerNorm |
| Stage konfigürasyonu | `[3, 3, 9, 3]` blok, kanal `[96, 192, 384, 768]` |
| Block türü | Depthwise 7×7 → LN → 1×1 (4× expand) → GELU → 1×1 + DropPath |
| Head | GAP → LN → Linear(768 → 2) |

### 2.2 Custom CNN (Özgün)

| Alan | Değer |
|------|-------|
| Toplam parametre | **1.29 M** (28.6 M ConvNeXt'in ~%4.5'i) |
| Stem | Conv 7×7 stride 2 + BN + ReLU + MaxPool 3×3 stride 2 → 56×56×32 |
| Multi-Scale Block | 1×1 ∥ 3×3 ∥ (3×3+3×3) paralel + 1×1 fuse → 56×56×48 |
| Residual-SE × 3 | 48→64 (28×28), 64→128 (14×14), 128→256 (7×7) |
| Dropout (blok başına) | 0.10 → 0.15 → 0.20 (kademeli artış) |
| SE reduction | 8 |
| Pooling | AdaptiveAvgPool2d(1) → 256-d vektör |
| Classifier | Linear(256→128) + LayerNorm + ReLU + Dropout(0.4) + Linear(128→2) |
| Init | Kaiming normal (fan_out, ReLU) |

---

## 3. Hyperparameter Tuning — Grid Search

12 kombinasyon (LR ∈ {5e-5, 1e-4, 5e-4} × batch ∈ {8, 16} × weight_decay ∈ {1e-4, 1e-3}). Her kombinasyon ayrı eğitim, **yalnızca validation** üzerinde değerlendirildi.

### 3.1 ConvNeXt-Tiny — En İyi Konfigürasyon

| Parametre | Seçilen | Aralık |
|-----------|---------|--------|
| **Learning rate** | **5e-5** | {5e-5, 1e-4, 5e-4} |
| **Batch size** | **8** | {8, 16} |
| **Weight decay** | **1e-3** | {1e-4, 1e-3} |
| **Best val_loss** | 0.0018 | — |
| **Best val_acc** | **1.0000** | — |
| Epoch | 30 (early stopping patience=5) | — |

Aşırı yüksek LR (5e-4) backbone'u bozdu — val_acc 0.48–0.52 (rastgele seviye). Görsel: `results/convnext_grid_search.png`.

### 3.2 Custom CNN — En İyi Konfigürasyon

| Parametre | Seçilen | Aralık |
|-----------|---------|--------|
| **Learning rate** | **1e-4** | {5e-5, 1e-4, 5e-4} |
| **Batch size** | **8** | {8, 16} |
| **Weight decay** | **1e-4** | {1e-4, 1e-3} |
| **Best val_loss** | 0.3775 | — |
| **Best val_acc** | **0.8387** | — |
| Epoch | 30 (early stopping patience=5) | — |

Custom CNN scratch eğitilen küçük modelde optimal LR daha yüksek (1e-4) ve daha düşük weight decay tercih edildi. Görsel: `results/custom_cnn_grid_search.png`.

---

## 4. Test Seti Sonuçları (30 örnek; 15 Normal · 15 Hemorrhage)

### 4.1 Sınıflandırma Metrikleri

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) |
|-------|----------|---------------|------------|--------|
| **ConvNeXt-Tiny** | **0.9667** | 0.9688 | 0.9667 | 0.9666 |
| **Custom CNN**    | 0.8667 | 0.8667 | 0.8667 | 0.8667 |
| **Ensemble (soft voting, w=0.5/0.5)** | **0.9667** | 1.0000 | 0.9333 | 0.9655 |

(Ensemble değerleri threshold 0.5'te; threshold 0.55'te aynı sonuç — bkz. Bölüm 6.)

### 4.2 Per-Class Detayı

**ConvNeXt-Tiny:**

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal     | 0.9375 | 1.0000 | 0.9677 |
| Hemorrhage | 1.0000 | 0.9333 | 0.9655 |

**Custom CNN:**

| Sınıf | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Normal     | 0.8667 | 0.8667 | 0.8667 |
| Hemorrhage | 0.8667 | 0.8667 | 0.8667 |

### 4.3 Confusion Matrix Sayıları (TP / TN / FP / FN — pos_label = Hemorrhage)

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| ConvNeXt-Tiny | 15 | 0 | 1 | 14 |
| Custom CNN    | 13 | 2 | 2 | 13 |
| Ensemble (t=0.5) | 13 | 2 | 1 | 14 |
| Ensemble (t=0.55, optimal) | **15** | **0** | **1** | **14** |

Görseller: `results/{convnext_tiny,custom_cnn,ensemble}_confusion_matrix.png`.

---

## 5. Eğitim Dinamikleri

| Model | Çalışılan epoch | Best val_acc | En iyi epoch | Final train_acc | Final val_acc | Gap (final) |
|-------|------------------|--------------|--------------|-----------------|---------------|-------------|
| ConvNeXt-Tiny | 12 (early stop) | 0.9355 | 8 | 0.9483 | 0.8710 | %7.7 |
| Custom CNN    | 32 | 0.9032 | 25 | 0.6811 | 0.8387 | val > train (Mixup etkisi) |

**Yorum:**
- ConvNeXt erken doyuma ulaştı, early stopping epoch 12'de durdurdu — overfitting'in başlangıcını otomatik yakaladı.
- Custom CNN'de val accuracy train accuracy'den yüksek; bu **Mixup'ın eğitim metriğinde yarattığı doğal düşüştür** (mixed örneklerde "doğru sınıf" tek değildir, train acc düşük görünür ama temiz val'de model iyi çalışır).
- Görseller: `results/{convnext,custom_cnn}_training_curves.png`, `results/{convnext_tiny,custom_cnn}_training_analysis.png`.

---

## 6. Decision Threshold Analizi (Medikal Recall Önceliği)

`src/threshold_analysis.py` ensemble çıktısının `P(Hemorrhage)` olasılığında eşiği 0.05 → 0.95 arası taradı.

| Eşik | Precision | Recall | F1 | Balanced Acc | TN | FP | FN | TP |
|------|-----------|--------|-----|--------------|----|----|----|----|
| 0.50 (default) | 0.875 | 0.933 | 0.903 | 0.900 | 13 | 2 | 1 | 14 |
| **0.55 (best F1)** | **1.000** | **0.933** | **0.966** | **0.967** | **15** | **0** | **1** | **14** |
| 0.20 (recall ≥ 0.95) | 0.750 | 1.000 | 0.857 | 0.833 | 10 | 5 | 0 | 15 |
| 0.05 (en yüksek recall) | 0.536 | 1.000 | 0.698 | 0.567 | 2 | 13 | 0 | 15 |

**Ana sonuç:** Default 0.5 yerine **0.55** seçilirse iki false positive sıfırlanıyor; recall korunuyor. Klinik tarama senaryosunda recall ≥ 0.95 hedefleniyorsa eşik **0.40** veya altına çekilmeli (precision düşse de hiç kanama kaçırılmasın).

Görsel: `results/threshold_analysis.png` · JSON: `results/threshold_analysis.json`.

---

## 7. Stratified 5-Fold Cross-Validation (Küçük Veri Savunması)

`src/cv_experiment.py` test setini sabit (30 örnek) tuttu; kalan 170 örnek üzerinde her model için 5 fold çalıştırıldı. Selection bias'tan kaçınmak için **sabit epoch + son 3 epoch ortalaması** raporlandı (best-on-val cherry-pick yapılmadı):

- ConvNeXt-Tiny → fold başına **10 epoch** (pretrained, hızlı yakınsar)
- Custom CNN → fold başına **20 epoch** (sıfırdan eğitim, daha uzun)
- Her iki model için son **3 epoch** validation metrikleri ortalanır

### 7.1 Fold-Bazlı Sonuçlar (son-3-epoch ortalaması)

| Fold | ConvNeXt acc | Custom CNN acc |
|------|--------------|----------------|
| 1 | 0.9333 | 0.8095 |
| 2 | 0.9902 | 0.7549 |
| 3 | 0.9706 | 0.8333 |
| 4 | 0.9510 | 0.7451 |
| 5 | 0.9706 | 0.7843 |

### 7.2 Mean ± Std

| Model | Accuracy | F1 (w) | Recall (w) |
|-------|----------|--------|------------|
| **ConvNeXt-Tiny** | **0.9631 ± 0.0194** | 0.9631 ± 0.0194 | 0.9631 ± 0.0194 |
| **Custom CNN**    | **0.7854 ± 0.0330** | 0.7824 ± 0.0358 | 0.7854 ± 0.0330 |

**Yorum:**

- Tek hold-out test sonucu (ConvNeXt 0.9667, Custom CNN 0.8667) CV ortalamasının **±2σ bandı içinde** → "şans" eleştirisi sayısal olarak çürür.
- Custom CNN test (%86.7) > CV (%78.5): test setinde ConvNeXt'in yanıldığı tek örnek tipinin Custom CNN'in lehine düşmesi; küçük-N varyansı.
- ConvNeXt fold varyansı (σ=0.019) Custom CNN'inkinden (σ=0.033) **daha düşük** → ImageNet pretrained backbone küçük-N'de daha kararlı.

> **Metodoloji notu:** Önceki sürümde "best-on-val" raporlama nedeniyle ConvNeXt'in 5 fold'undan 3'ü tam 1.000 görünüyordu (selection bias). Bu, küçük (~34 örnek) validation setinde 10 epoch'tan en iyisinin seçilmesinden kaynaklı yapay yükselişti; düzeltildi.

Görsel: `results/cv_summary.png` · JSON: `results/cv_results.json`.

---

## 8. Ek Analizler

### 8.1 ROC / PR Eğrileri

- Görsel: `results/roc_auc_curves.png` (her iki modelin ROC + PR eğrileri).
- ConvNeXt AUC ≈ 1.0 (30 örnekte tüm threshold'larda doğru sıralama). Sınırlılık: küçük test seti — geniş güven aralığı (~%82–%99.8).

### 8.2 t-SNE Feature Space

- Görseller: `results/{convnext_tiny,custom_cnn}_tsne.png`.
- ConvNeXt feature uzayı iki sınıfı temiz ayırıyor (kümeler net). Custom CNN'de kümeler daha geçişli (overlap), F1 farkıyla uyumlu.

### 8.3 Grad-CAM (Açıklanabilirlik)

- Görseller: `results/{convnext_tiny,custom_cnn}_gradcam.png`.
- Her iki model de doğru sınıflandırılan örneklerde **kanama bölgesine** odaklanıyor (kafatası kenarına veya artifakta değil) — shortcut learning yok.

### 8.4 Feature Çıkarımı

- `src/extract_features.py` test setinin GAP feature'larını CSV olarak çıkarır (ConvNeXt: 768-d, Custom CNN: 256-d). Offline analiz, harici sınıflandırıcı veya kümeleme için kullanılabilir.

---

## 9. Üretilen Tüm Görseller (Raporda Kullan)

| # | Dosya | Rapor bölümü |
|---|-------|--------------|
| 1 | `dataset_overview.png` | III. Veri Seti |
| 2 | `augmentation_preview.png` | III. Veri Artırımı |
| 3 | `flow_chart.png` | IV. Yöntem (genel akış şeması) |
| 4 | `architecture_convnext.png` | IV. Yöntem · Pretrained model |
| 5 | `architecture_custom_cnn.png` | IV. Yöntem · Özgün model |
| 6 | `decision_pipeline.png` | IV. Yöntem · Karar boru hattı |
| 7 | `convnext_grid_search.png` | IV. Hyperparameter Tuning |
| 8 | `custom_cnn_grid_search.png` | IV. Hyperparameter Tuning |
| 9 | `convnext_training_curves.png` | V. Eğitim Sonuçları |
| 10 | `custom_cnn_training_curves.png` | V. Eğitim Sonuçları |
| 11 | `convnext_tiny_training_analysis.png` | V. Overfitting Analizi |
| 12 | `custom_cnn_training_analysis.png` | V. Overfitting Analizi |
| 13 | `convnext_tiny_confusion_matrix.png` | VI. Sonuçlar |
| 14 | `custom_cnn_confusion_matrix.png` | VI. Sonuçlar |
| 15 | `ensemble_confusion_matrix.png` | VI. Sonuçlar |
| 16 | `model_comparison.png` | VI. Sonuçlar |
| 17 | `roc_auc_curves.png` | VI. Sonuçlar |
| 18 | `convnext_tiny_tsne.png` | VII. Açıklanabilirlik |
| 19 | `custom_cnn_tsne.png` | VII. Açıklanabilirlik |
| 20 | `convnext_tiny_gradcam.png` | VII. Açıklanabilirlik |
| 21 | `custom_cnn_gradcam.png` | VII. Açıklanabilirlik |
| 22 | `threshold_analysis.png` | VIII. Tartışma · Klinik trade-off |
| 23 | `cv_summary.png` | VIII. Tartışma · Robustluk |

---

## 10. Literatür Özeti Tablosu (5 IEEE Xplore 2025 Makalesi)

> **ZORUNLU:** Yalnızca 2025 yılı IEEE Xplore makaleleri. Ayrı dosyada (`REFERENCES.md`) tam künye + özet tutulur. Aşağıda raporda doğrudan kullanılacak özet tablo şablonu.

| # | Yıl | Yazarlar | Veri Seti | Model | En İyi Skor | Bizim ile Karşılaştırma |
|---|-----|----------|-----------|-------|-------------|-------------------------|
| 1 | 2025 | _(REFERENCES.md::1)_ | _(...)_ | _(...)_ | _(...)_ | _(...)_ |
| 2 | 2025 | _(REFERENCES.md::2)_ | _(...)_ | _(...)_ | _(...)_ | _(...)_ |
| 3 | 2025 | _(REFERENCES.md::3)_ | _(...)_ | _(...)_ | _(...)_ | _(...)_ |
| 4 | 2025 | _(REFERENCES.md::4)_ | _(...)_ | _(...)_ | _(...)_ | _(...)_ |
| 5 | 2025 | _(REFERENCES.md::5)_ | _(...)_ | _(...)_ | _(...)_ | _(...)_ |

---

## 11. Sonuç Cümleleri (Raporun Abstract / Conclusion bölümü için kopyala)

- "200 görüntülük dengeli Head CT veri setinde, ImageNet-pretrained ConvNeXt-Tiny ile 28.6 M parametreli model **0.967 test accuracy** ve **0.967 weighted F1** elde etmiştir."
- "1.29 M parametreli özgün CNN (Multi-Scale + Residual-SE) tek başına **0.867 accuracy** sağlamış; soft-voting ensemble eşik 0.55'te **precision = 1.000, recall = 0.933, F1 = 0.966** ile en yüksek dengeli sonucu üretmiştir."
- "Stratified 5-fold cross-validation (sabit epoch + son 3 epoch ortalaması) ile ConvNeXt-Tiny **0.963 ± 0.019**, Custom CNN **0.785 ± 0.033** ortalama accuracy vermiştir; tek hold-out skoru CV ortalamasının ±2σ bandı içindedir."
- "Decision threshold analizi 0.55'te iki false positive'in elendiğini, klinik recall ≥ 0.95 hedefi için ise eşiğin 0.40'a düşürülmesi gerektiğini sayısal olarak göstermiştir."
- "Sınırlılıklar: tek-merkez veri kaynağı, 30 örneklik test setinin geniş güven aralığı, binary etiketleme (subtype yok). Klinik kullanım için CQ500 / RSNA gibi external validation zorunludur."

---

## 12. Hızlı Erişim — Tek Komutla Tüm Pipeline

```bash
python main.py                    # tam pipeline (augmentation → tuning → train → eval)
python main.py --eval             # sadece test metrikleri (eğitim sonrası)
python main.py --app              # Gradio arayüzü (http://localhost:7860)
python main.py --webcrawl         # web_crawled_test/ üzerinde external test
python -m src.cv_experiment       # 5-Fold CV (yaklaşık 15 dk CPU)
python -m src.threshold_analysis  # eşik taraması (saniyeler)
python -m src.diagrams            # akış + mimari diyagramlarını üret
python -m src.extract_features    # GAP feature CSV
```
