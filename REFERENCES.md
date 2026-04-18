# REFERENCES — IEEE Xplore 2025 Literatür Listesi

> **Şartname (Proje2.docx):** "5 adet SADECE ve SADECE 2025 yılı IEEE Xplore makalesi" + literatür özet tablosu + raporun referanslar bölümü.
>
> **Önemli — okumadan kullanma:** Aşağıdaki 5 makalenin ilki **doğrulanmış 2025** (DOI: 10.1109/ACCESS.2025.3626224). Diğer 4 aday IEEE Xplore document numarası 11M+ aralığında olduğu için **çok büyük olasılıkla 2025**, ama final teslim öncesi her birini IEEE Xplore'da açıp **"Date of Publication" alanı 2025 mi** kontrol edin. Eğer biri 2024 çıkarsa, dosya altındaki **"Yedek Adaylar"** listesinden değiştirin.

---

## Literatür Özeti Tablosu (Raporda Kullanılacak)

| # | Yazarlar (kısaltılmış) | Başlık | Yayın | Veri Seti | Model / Yöntem | Performans | Bizim Çalışma ile Karşılaştırma |
|---|------------------------|--------|-------|-----------|----------------|------------|--------------------------------|
| 1 | A. Chaudhary, Y. Gaur, A. Abraham, H. Singh | Multiclass Intracranial Hemorrhage Detection and Confidence Aware Triage via Deep Ensemble Learning | IEEE Access, 2025 | RSNA ICH (5 subtype) | Deep ensemble + MC Dropout uncertainty estimation, triage skoru | Multiclass, large-scale | 5-sınıf vs bizim 2-sınıf; triage senaryosunda **ensemble + uncertainty** ortak; biz binary + 200-örnek scope'undayız |
| 2 | _(IEEE Xplore: doc 11436839)_ | Ensemble Deep Learning Approaches for Brain Hemorrhage Detection from CT Images | IEEE Conf., 2025 | (RSNA / public CT) | Inception + ResNet + EfficientNet ensemble, augmentation, feature extraction | Yüksek accuracy (CT subtype) | Aynı ensemble felsefesi; biz ConvNeXt-Tiny + Custom CNN ile 2-model soft voting ve eşik optimizasyonu ekledik |
| 3 | _(IEEE Xplore: doc 11394250)_ | Intracranial Hemorrhage Detection from CT Scan Using Deep Learning | IEEE Conf., 2025 | RSNA / public head CT | End-to-end DL pipeline + online web app (klinik kullanım) | Klinik akış entegrasyonu | Bizimki Gradio arayüzü + Grad-CAM ile benzer (UI + olasılık + açıklanabilirlik); web-app yerine local app |
| 4 | _(IEEE Xplore: doc 11073367)_ | Acute Intracranial Hemorrhage Detection and Classification using AI | IEEE Conf., 2025 | Multi-source ICH CT | Transformer-based feature extraction + GAN ile minority class (epidural) augmentasyonu | İmbalance çözümü | Bizim veri 1:1 dengeli olduğu için GAN gereksiz; ama Custom CNN'imizin Multi-Scale Block'u küçük subdural kanamayı yakalamada benzer hedef |
| 5 | _(IEEE Xplore: doc 11218836)_ | Multiclass Intracranial Hemorrhage Detection and Confidence Aware Triage via Deep Ensemble Learning | IEEE Access, 2025 | RSNA ICH | (1 numaralı çalışmanın IEEE Xplore document linkı — aynı makale) | — | Bu doc# 1 numara ile aynı makale; **5. sıraya farklı bir 2025 makalesi seçin** (aşağıdaki yedeklerden) |

> **#5 düzeltme uyarısı:** Aday listesi içinde 11218836 numarası, 1 numaradaki makalenin IEEE Xplore record'u. Yani doğrulanmış aday sayısı 4 oldu. **5. makale için aşağıdaki "Yedek Adaylar" listesinden bir tanesini IEEE Xplore'da 2025 yılı doğrulamasıyla seçin** ve tabloya geçirin.

---

## IEEE Xplore Doğrudan Linkler (Erişim için)

1. **Multiclass ICH Detection — Deep Ensemble Learning + Confidence Triage**
   - [https://ieeexplore.ieee.org/document/11218836/](https://ieeexplore.ieee.org/document/11218836/)
   - DOI: `10.1109/ACCESS.2025.3626224`
   - Yazarlar: Abhay Chaudhary, Yana Gaur, Ajith Abraham, Harmandeep Singh
   - Yayın: IEEE Access, 2025

2. **Ensemble Deep Learning Approaches for Brain Hemorrhage Detection**
   - [https://ieeexplore.ieee.org/document/11436839/](https://ieeexplore.ieee.org/document/11436839/)

3. **Intracranial Hemorrhage Detection from CT Scan Using Deep Learning**
   - [https://ieeexplore.ieee.org/document/11394250/](https://ieeexplore.ieee.org/document/11394250/)

4. **Acute Intracranial Hemorrhage Detection and Classification using AI**
   - [https://ieeexplore.ieee.org/document/11073367/](https://ieeexplore.ieee.org/document/11073367/)

5. _(yedek adaylardan birini seçin → linkini buraya yazın)_

---

## Yedek Adaylar (5. makale için — IEEE Xplore'da 2025 yılı doğrulayın)

Bu adaylar konu/yöntem olarak uygun; her birinin **Date of Publication** alanını IEEE Xplore'da kontrol edin. 2025 olanı 5. sıraya alın.

| Doc # | Olası Başlık | Not |
|-------|--------------|-----|
| 10822353 | Brain Hemorrhage CT Image Detection and Classification using Deep Learning Methods | Aralık 2024 — IEEE 10M aralığı; 2025 değilse atlayın |
| 10691276 | Automated Intracranial Hemorrhage Detection Using Deep Learning in Medical Image Analysis | Yıl doğrulanmalı |
| 10635323 | Multi-Stage Transformer Fusion for Efficient Intracranial Hemorrhage Subtype Classification | Vision Transformer, transformer fusion |
| 10391388 | A Transformer-Based Deep Learning Architecture for Accurate Intracranial Hemorrhage Detection and Classification | Transformer baseline |

**Ek arama yöntemi (5. makaleyi siz seçeceksiniz):**

IEEE Xplore Advanced Search → şu sorguları çalıştırın ve sonuçları "Year: 2025" filtresiyle daraltın:

```
("intracranial hemorrhage" OR "brain hemorrhage" OR "head CT") AND ("deep learning" OR "CNN" OR "transformer") AND classification
```

İlk sayfada en az 3-5 sonuç çıkacak; 1-4 numaralarınızla aynı yöntemi kullanmayan birini seçerseniz literatür çeşitliliği artar (öneri: Vision Transformer veya self-supervised learning makalesi).

---

## IEEE Stili Referans Formatı (Raporun "References" bölümü için)

Aşağıdaki formatı doğrudan kopyalayıp final yazar/sayfa bilgilerini IEEE Xplore'dan tamamlayın:

```
[1] A. Chaudhary, Y. Gaur, A. Abraham, and H. Singh, "Multiclass Intracranial
    Hemorrhage Detection and Confidence Aware Triage via Deep Ensemble
    Learning," IEEE Access, vol. 13, pp. XXXX-XXXX, 2025,
    doi: 10.1109/ACCESS.2025.3626224.

[2] [Yazar(lar)], "Ensemble Deep Learning Approaches for Brain Hemorrhage
    Detection from CT Images," in Proc. [Konferans Adi], 2025, pp. XXX-XXX,
    doi: 10.1109/[Konferans-DOI].

[3] [Yazar(lar)], "Intracranial Hemorrhage Detection from CT Scan Using
    Deep Learning," in Proc. [Konferans Adi], 2025, pp. XXX-XXX,
    doi: 10.1109/[Konferans-DOI].

[4] [Yazar(lar)], "Acute Intracranial Hemorrhage Detection and Classification
    using AI," in Proc. [Konferans Adi], 2025, pp. XXX-XXX,
    doi: 10.1109/[Konferans-DOI].

[5] [Yazar(lar)], "[5. makalenin başlığı]," in [Yayın türü], 2025,
    doi: 10.1109/[DOI].
```

**Ek referanslar (literatür tablosu dışında, yöntem kısmında atıf gerektiren):**

```
[6] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie,
    "A ConvNet for the 2020s," in Proc. IEEE/CVF CVPR, 2022, pp. 11976-11986,
    doi: 10.1109/CVPR52688.2022.01167.        ← ConvNeXt orijinal makalesi

[7] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in
    Proc. IEEE/CVF CVPR, 2018, pp. 7132-7141,
    doi: 10.1109/CVPR.2018.00745.              ← SE Block orijinal makalesi

[8] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz, "mixup: Beyond
    Empirical Risk Minimization," in Proc. ICLR, 2018.   ← Mixup makalesi

[9] R. R. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization," in Proc. IEEE ICCV, 2017, pp. 618-626,
    doi: 10.1109/ICCV.2017.74.                 ← Grad-CAM orijinal makalesi
```

> **Not:** [6]-[9] **2025 dışı** atıflardır; literatür özet tablosunda değil, sadece raporun yöntem kısmında "ConvNeXt mimarisini [6] referansından aldık" şeklinde teknik atıf olarak kullanılır. Şartnamenin "5 adet 2025 IEEE makalesi" maddesi yalnızca tablo için geçerli; yöntem atıfları bunun dışında değerlendirilir (rapor formatı bunu zaten ayırır).

---

## Kontrol Listesi (Teslim Öncesi)

- [ ] 5 makalenin her birinin IEEE Xplore "Date of Publication" yıl bilgisi 2025 mi? (her birini açıp doğrula)
- [ ] DOI numaraları kopyalandı mı?
- [ ] Yazar isimleri tam yazıldı mı (sadece doc# değil)?
- [ ] Literatür tablosundaki "Performans" sütunu makalelerin gerçek raporladığı sayılarla dolduruldu mu?
- [ ] "Bizim Çalışma ile Karşılaştırma" sütununda kendi 0.967 accuracy / 1.29M parametre / 5-fold CV değerlerinizden bahsedildi mi?
- [ ] Referanslar bölümünde IEEE formatında sıralı (köşeli parantezli) yazıldı mı?
