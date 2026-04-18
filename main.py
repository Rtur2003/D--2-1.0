# =========================================================================
# MAIN PIPELINE - Tum Proje Adimlari
# =========================================================================
# BM 480 Derin Ogrenme - Proje 2
# Head CT Hemorrhage Classification
#
# Kullanim:
#   python main.py              -> Tam pipeline (tune + train + eval + webcrawl)
#   python main.py --train      -> Sadece egitim
#   python main.py --eval       -> Sadece degerlendirme
#   python main.py --tune       -> Sadece hiperparametre tuning
#   python main.py --app        -> Arayuzu baslat
#   python main.py --webcrawl   -> Web-crawled goruntuleri test et
#   python main.py --augpreview -> Augmentation onizleme
#   python main.py --diagrams   -> Akis semasi + mimari diyagramlari uret
#   python main.py --threshold  -> Decision threshold analizi (medikal)
#   python main.py --cv         -> Stratified 5-Fold Cross-Validation
#   python main.py --features   -> GAP feature CSV cikarimi
# =========================================================================

import sys
import os
import io

# Windows cp1254 encoding sorununu coz - UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# src/ klasorunu path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DEVICE, RESULTS_DIR, MODELS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="BM 480 - Head CT Hemorrhage Classification Pipeline"
    )
    parser.add_argument("--train", action="store_true",
                        help="Modelleri egit")
    parser.add_argument("--eval", action="store_true",
                        help="Test degerlendirmesi + Grad-CAM + t-SNE + ROC")
    parser.add_argument("--tune", action="store_true",
                        help="Hiperparametre tuning (Grid Search)")
    parser.add_argument("--app", action="store_true",
                        help="Gradio arayuzunu baslat")
    parser.add_argument("--webcrawl", action="store_true",
                        help="Web-crawled test goruntuleri")
    parser.add_argument("--augpreview", action="store_true",
                        help="Augmentation onizleme")
    parser.add_argument("--diagrams", action="store_true",
                        help="Akis semasi + mimari diyagramlari uret")
    parser.add_argument("--threshold", action="store_true",
                        help="Decision threshold analizi (medikal)")
    parser.add_argument("--cv", action="store_true",
                        help="Stratified 5-Fold Cross-Validation")
    parser.add_argument("--features", action="store_true",
                        help="GAP feature CSV cikarimi")
    parser.add_argument("--all", action="store_true",
                        help="Tum pipeline (tune + train + eval)")

    args = parser.parse_args()

    # Hiçbir flag verilmezse --all kabul et
    if not any(vars(args).values()):
        args.all = True

    print("=" * 70)
    print("BM 480 DERIN OGRENME - PROJE 2")
    print("Head CT Hemorrhage Classification")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Models dir: {MODELS_DIR}")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 70)

    # ── Augmentation Preview ───────────────────────────────────────
    if args.augpreview or args.all:
        print("\n[STEP 1/5] Augmentation Onizleme...")
        from data_augmentation import preview_augmentations
        from config import DATA_DIR
        preview_augmentations(
            str(DATA_DIR / "001.png"),
            save_path=str(RESULTS_DIR / "augmentation_preview.png")
        )

    # ── Hyperparameter Tuning ──────────────────────────────────────
    if args.tune or args.all:
        print("\n[STEP 2/5] Hiperparametre Tuning (Grid Search)...")
        from hyperparameter_tuning import run_hyperparameter_tuning
        run_hyperparameter_tuning()

    # ── Training ───────────────────────────────────────────────────
    if args.train or args.all:
        print("\n[STEP 3/5] Model Egitimi...")
        from train import run_training
        run_training(augment=True)

    # ── Evaluation (+ Grad-CAM, t-SNE, ROC, Ensemble) ─────────────
    if args.eval or args.all:
        print("\n[STEP 4/5] Test Degerlendirmesi + Gorsellestirmeler...")
        from evaluate import run_evaluation
        run_evaluation()

    # ── Web Crawled Test ───────────────────────────────────────────
    if args.webcrawl or args.all:
        print("\n[STEP 5/5] Web-Crawled Test Goruntuleri...")
        from web_crawler import test_web_crawled_images
        test_web_crawled_images()

    # ── Diagrams (akis semasi + mimari) ────────────────────────────
    if args.diagrams:
        print("\n[EXTRA] Akis semasi + mimari diyagramlari...")
        from diagrams import make_all
        make_all()

    # ── Decision Threshold Analizi ────────────────────────────────
    if args.threshold:
        print("\n[EXTRA] Decision Threshold Analizi (medikal)...")
        from threshold_analysis import run_threshold_analysis
        run_threshold_analysis()

    # ── 5-Fold Cross Validation ────────────────────────────────────
    if args.cv:
        print("\n[EXTRA] Stratified 5-Fold Cross-Validation...")
        from cv_experiment import run_cv
        run_cv()

    # ── Feature Extraction (GAP CSV) ──────────────────────────────
    if args.features:
        print("\n[EXTRA] GAP Feature CSV cikarimi...")
        from extract_features import extract_features_to_csv
        extract_features_to_csv()

    # ── Interface ──────────────────────────────────────────────────
    if args.app:
        print("\n[APP] Gradio Arayuzu Baslatiliyor...")
        from app import launch_interface
        launch_interface()

    if not args.app:
        print("\n" + "=" * 70)
        print("PIPELINE TAMAMLANDI!")
        print("=" * 70)
        print(f"\nCiktilar:")
        print(f"  Modeller     : {MODELS_DIR}/")
        print(f"  Grafikler    : {RESULTS_DIR}/")
        print(f"\nOlusturulan grafikler:")
        expected_files = [
            "augmentation_preview.png",
            "dataset_overview.png",
            "convnext_training_curves.png",
            "custom_cnn_training_curves.png",
            "convnext_tiny_training_analysis.png",
            "custom_cnn_training_analysis.png",
            "convnext_tiny_confusion_matrix.png",
            "custom_cnn_confusion_matrix.png",
            "ensemble_confusion_matrix.png",
            "model_comparison.png",
            "roc_auc_curves.png",
            "convnext_tiny_tsne.png",
            "custom_cnn_tsne.png",
            "convnext_tiny_gradcam.png",
            "custom_cnn_gradcam.png",
            "convnext_grid_search.png",
            "custom_cnn_grid_search.png",
            "flow_chart.png",
            "architecture_convnext.png",
            "architecture_custom_cnn.png",
            "decision_pipeline.png",
            "threshold_analysis.png",
            "cv_summary.png",
        ]
        for f in expected_files:
            path = RESULTS_DIR / f
            status = "OK" if path.exists() else "--"
            print(f"  [{status}] {f}")

        print(f"\nArayuzu baslatmak icin:")
        print(f"  python main.py --app")
        print("=" * 70)


if __name__ == "__main__":
    main()
