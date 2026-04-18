"""
Head CT Hemorrhage Classifier — Clinical-style decision-support UI.

Tasarim felsefesi:
- Minimal text, gorsel hierarcheye guven
- Tek bakista karar (banner + olasilik bar)
- Tum gelismis ayarlar accordion icinde
- Inter / JetBrains Mono tipografi
"""
import io
import json
import os
import socket
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import gradio as gr
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from config import CLASS_NAMES, DATA_DIR, DEVICE, IMG_SIZE, MODELS_DIR
from custom_cnn import get_custom_cnn
from ensemble import EnsembleModel
from gradcam import GradCAM, get_target_layer, overlay_cam_on_image
from pretrained_model import get_convnext_model

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_cache = {}

MODEL_KEYS = {
    "Ensemble": "ensemble",
    "ConvNeXt-Tiny": "convnext",
    "Custom CNN": "custom",
}


# ── Theme (clinical, calm, monochrome with single teal accent) ──────────

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"),
    font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#fafbfc",
    background_fill_primary="#ffffff",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#e5e7eb",
    block_radius="14px",
    block_shadow="0 1px 2px rgba(15, 23, 42, 0.04)",
    button_primary_background_fill="#0d9488",
    button_primary_background_fill_hover="#0f766e",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f1f5f9",
    button_secondary_background_fill_hover="#e2e8f0",
    button_secondary_text_color="#0f172a",
    input_background_fill="#ffffff",
    input_border_color_focus="#0d9488",
)

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; padding: 24px 16px !important; }

#brand-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
}
#brand-bar .brand {
    display: flex; align-items: center; gap: 14px;
}
#brand-bar .logo {
    width: 36px; height: 36px; border-radius: 9px;
    background: linear-gradient(135deg, #0d9488 0%, #134e4a 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.02em;
}
#brand-bar .titles h1 {
    margin: 0; font-size: 1.05rem; font-weight: 600; letter-spacing: -0.01em;
    color: #0f172a;
}
#brand-bar .titles .sub {
    font-size: 0.75rem; color: #64748b; margin-top: 1px;
}
#brand-bar .meta {
    display: flex; gap: 16px; font-size: 0.72rem; color: #64748b;
    font-family: 'JetBrains Mono', monospace;
}
#brand-bar .meta b { color: #0f172a; font-weight: 600; }

.dropzone-card .image-frame { border-style: dashed !important; }
button.lg { font-size: 0.9rem !important; }

#result-banner > div {
    border-radius: 12px; padding: 18px 22px;
    display: flex; align-items: center; justify-content: space-between;
    transition: all 0.2s ease;
}
#result-banner .label {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em;
    opacity: 0.85;
}
#result-banner .verdict {
    font-size: 1.6rem; font-weight: 600; letter-spacing: -0.01em; margin-top: 3px;
}
#result-banner .conf {
    text-align: right; font-family: 'JetBrains Mono', monospace;
}
#result-banner .conf .val {
    font-size: 1.6rem; font-weight: 700;
}
#result-banner .conf .lbl {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.12em; opacity: 0.85;
}

.prob-bars { padding: 4px 2px; }
.prob-bars .row { margin-bottom: 10px; }
.prob-bars .row:last-child { margin-bottom: 0; }
.prob-bars .head {
    display: flex; justify-content: space-between; font-size: 0.78rem;
    color: #475569; margin-bottom: 4px; font-weight: 500;
}
.prob-bars .head .v { font-family: 'JetBrains Mono', monospace; color: #0f172a; }
.prob-bars .track {
    height: 6px; background: #e2e8f0; border-radius: 999px; overflow: hidden;
}
.prob-bars .fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }

#detail-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    font-size: 0.78rem; color: #475569; padding: 4px;
}
#detail-grid .k { color: #64748b; }
#detail-grid .v { font-family: 'JetBrains Mono', monospace; color: #0f172a; text-align: right; }

.muted { color: #64748b; font-size: 0.78rem; }
.disclaimer-mini {
    font-size: 0.7rem; color: #94a3b8; padding-top: 12px; border-top: 1px solid #e2e8f0;
    margin-top: 16px;
}
"""


# ── Model loading ───────────────────────────────────────────────────────

def _load_models():
    if "convnext" in _cache:
        return
    with open(str(MODELS_DIR / "train_stats.json"), "r") as f:
        stats = json.load(f)

    convnext = get_convnext_model(pretrained=False)
    ckpt = torch.load(str(MODELS_DIR / "convnext_tiny_best.pth"), map_location=DEVICE, weights_only=False)
    convnext.load_state_dict(ckpt["model_state_dict"])
    convnext.eval().to(DEVICE)

    custom = get_custom_cnn()
    ckpt = torch.load(str(MODELS_DIR / "custom_cnn_best.pth"), map_location=DEVICE, weights_only=False)
    custom.load_state_dict(ckpt["model_state_dict"])
    custom.eval().to(DEVICE)

    ensemble = EnsembleModel(convnext, custom, weight1=0.5, weight2=0.5)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])

    _cache.update({
        "convnext": convnext, "custom": custom,
        "ensemble": ensemble, "transform": transform, "stats": stats,
    })


def _tta(model, x):
    views = [x, torch.flip(x, [3]), torch.flip(x, [2]), torch.flip(x, [2, 3])]
    with torch.no_grad():
        return torch.stack([torch.softmax(model(v), 1) for v in views], 0).mean(0)


# ── Rendering helpers ──────────────────────────────────────────────────

def _empty_banner():
    return (
        "<div id='result-banner'><div style='background:#f8fafc;color:#94a3b8;border:1px dashed #cbd5e1;'>"
        "<div><div class='label'>Bekleniyor</div>"
        "<div class='verdict' style='font-size:1.05rem;font-weight:500;'>Goruntu yukleyin</div></div>"
        "</div></div>"
    )


def _banner(pred: str, conf: float, threshold: float):
    is_hem = pred == "Hemorrhage"
    bg = (
        "linear-gradient(135deg,#7f1d1d 0%,#991b1b 100%)" if is_hem
        else "linear-gradient(135deg,#065f46 0%,#0f766e 100%)"
    )
    label = "Kanama Suphesi" if is_hem else "Normal Bulgu"
    sub = f"Esik: {threshold:.2f}"
    return (
        f"<div id='result-banner'><div style='background:{bg};color:#fff;'>"
        f"<div><div class='label'>Tahmin · {sub}</div>"
        f"<div class='verdict'>{label}</div></div>"
        f"<div class='conf'><div class='lbl'>Guven</div>"
        f"<div class='val'>{conf*100:.1f}%</div></div>"
        f"</div></div>"
    )


def _prob_bars(probs: dict):
    rows = []
    for cls, p in probs.items():
        is_hem = cls == "Hemorrhage"
        color = "#dc2626" if is_hem else "#0d9488"
        rows.append(
            f"<div class='row'><div class='head'><span>{cls}</span>"
            f"<span class='v'>{p*100:.2f}%</span></div>"
            f"<div class='track'><div class='fill' style='width:{p*100:.1f}%;background:{color};'></div></div></div>"
        )
    return f"<div class='prob-bars'>{''.join(rows)}</div>"


def _detail(items: list):
    rows = []
    for k, v in items:
        rows.append(f"<div class='k'>{k}</div><div class='v'>{v}</div>")
    return f"<div id='detail-grid'>{''.join(rows)}</div>"


def _ensemble_table(scores1, scores2, ens):
    def cell(v, accent=False):
        c = "#0f766e" if accent else "#0f172a"
        return f"<td style='padding:7px 10px;text-align:right;font-family:JetBrains Mono,monospace;color:{c};font-weight:{600 if accent else 400};'>{v:.3f}</td>"
    rows = "".join(
        f"<tr><td style='padding:7px 10px;font-weight:500;'>{c}</td>"
        f"{cell(scores1[c])}{cell(scores2[c])}{cell(ens[c], accent=True)}</tr>"
        for c in CLASS_NAMES
    )
    return (
        "<table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>"
        "<thead><tr style='background:#f1f5f9;color:#475569;'>"
        "<th style='padding:8px 10px;text-align:left;font-weight:600;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;'>Sinif</th>"
        "<th style='padding:8px 10px;text-align:right;font-weight:600;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;'>ConvNeXt</th>"
        "<th style='padding:8px 10px;text-align:right;font-weight:600;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;'>Custom</th>"
        "<th style='padding:8px 10px;text-align:right;font-weight:600;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;color:#0f766e;'>Ensemble</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


# ── Inference ───────────────────────────────────────────────────────────

def predict(image, model_label: str, use_tta: bool, threshold: float, show_cam: bool):
    if image is None:
        return _empty_banner(), _prob_bars({"Normal": 0.0, "Hemorrhage": 0.0}), None, "", _detail([])

    _load_models()
    transform = _cache["transform"]

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    key = MODEL_KEYS.get(model_label, "ensemble")

    ensemble_html = ""
    if key == "ensemble":
        if use_tta:
            p1 = _tta(_cache["convnext"], x)[0].cpu().numpy()
            p2 = _tta(_cache["custom"], x)[0].cpu().numpy()
        else:
            with torch.no_grad():
                p1 = torch.softmax(_cache["convnext"](x), 1)[0].cpu().numpy()
                p2 = torch.softmax(_cache["custom"](x), 1)[0].cpu().numpy()
        probs_arr = 0.5 * p1 + 0.5 * p2
        s1 = {CLASS_NAMES[i]: float(p1[i]) for i in range(2)}
        s2 = {CLASS_NAMES[i]: float(p2[i]) for i in range(2)}
        se = {CLASS_NAMES[i]: float(probs_arr[i]) for i in range(2)}
        ensemble_html = _ensemble_table(s1, s2, se)
        cam_model, cam_name = _cache["convnext"], "convnext"
    else:
        model = _cache[key]
        cam_model, cam_name = model, key
        if use_tta:
            probs_arr = _tta(model, x)[0].cpu().numpy()
        else:
            with torch.no_grad():
                probs_arr = torch.softmax(model(x), 1)[0].cpu().numpy()

    probs = {CLASS_NAMES[i]: float(probs_arr[i]) for i in range(2)}
    # threshold-based decision: hemorrhage if P(hem) >= threshold else Normal
    pred = "Hemorrhage" if probs["Hemorrhage"] >= threshold else "Normal"
    conf = probs[pred]

    cam_img = None
    if show_cam:
        try:
            target = get_target_layer(cam_model, cam_name)
            cam, _, _ = GradCAM(cam_model, target).generate(x.clone())
            base = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
            cam_img = (overlay_cam_on_image(base, cam, alpha=0.45) * 255).astype(np.uint8)
        except Exception as e:
            cam_img = None

    detail_items = [
        ("Model", model_label),
        ("TTA", "4-view" if use_tta else "kapali"),
        ("Karar esigi", f"{threshold:.2f}"),
        ("Zaman", datetime.now().strftime("%H:%M:%S")),
        ("P(Normal)", f"{probs['Normal']:.4f}"),
        ("P(Hemorrhage)", f"{probs['Hemorrhage']:.4f}"),
    ]

    return (
        _banner(pred, conf, threshold),
        _prob_bars(probs),
        cam_img,
        ensemble_html,
        _detail(detail_items),
    )


def _examples():
    candidates = [
        ("001.png", "Hemorrhage"),
        ("015.png", "Hemorrhage"),
        ("042.png", "Hemorrhage"),
        ("105.png", "Normal"),
        ("130.png", "Normal"),
        ("172.png", "Normal"),
    ]
    return [[str(DATA_DIR / f)] for f, _ in candidates if (DATA_DIR / f).exists()]


# ── UI ──────────────────────────────────────────────────────────────────

def create_interface():
    with gr.Blocks(title="Head CT — Hemorrhage Classifier", theme=THEME, css=CSS) as demo:
        gr.HTML(
            "<div id='brand-bar'>"
            "<div class='brand'>"
            "<div class='logo'>CT</div>"
            "<div class='titles'>"
            "<h1>Head CT — Hemorrhage Classifier</h1>"
            "<div class='sub'>Karar destek prototipi · BM 480 Derin Ogrenme</div>"
            "</div></div>"
            "<div class='meta'>"
            "<span><b>2 model</b> + ensemble</span>"
            "<span>200 CT goruntu</span>"
            "<span>Stratified 70/15/15</span>"
            "</div></div>"
        )

        with gr.Row(equal_height=False):
            # Left column — input
            with gr.Column(scale=5, min_width=380):
                image_input = gr.Image(
                    type="pil", height=440, sources=["upload", "clipboard"],
                    show_label=False, elem_classes=["dropzone-card"],
                )
                with gr.Row():
                    predict_btn = gr.Button("Analiz Et", variant="primary", scale=3)
                    clear_btn = gr.Button("Temizle", variant="secondary", scale=1)

                with gr.Accordion("Ayarlar", open=False):
                    model_choice = gr.Radio(
                        choices=list(MODEL_KEYS.keys()),
                        value="Ensemble",
                        label="Model",
                        info=None,
                    )
                    threshold = gr.Slider(
                        minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                        label="Karar esigi (P(Hemorrhage))",
                        info="Dusuk esik = daha fazla pozitif (recall artar). Medikal sistemlerde 0.40 sik tercih edilir.",
                    )
                    use_tta = gr.Checkbox(label="Test-time augmentation (4-view)", value=True)
                    show_cam = gr.Checkbox(label="Grad-CAM uret", value=True)

                gr.HTML(
                    "<div class='disclaimer-mini'>"
                    "Yalnizca arastirma amaclidir. Klinik karar yerine gecmez."
                    "</div>"
                )

            # Right column — result
            with gr.Column(scale=7, min_width=420):
                banner = gr.HTML(_empty_banner(), elem_id="result-banner-wrap")
                bars = gr.HTML(_prob_bars({"Normal": 0.0, "Hemorrhage": 0.0}))

                with gr.Tabs():
                    with gr.Tab("Grad-CAM"):
                        cam_image = gr.Image(height=320, show_label=False, interactive=False)
                    with gr.Tab("Ensemble Detay"):
                        ensemble_html = gr.HTML(
                            "<div class='muted' style='padding:8px;'>Ensemble secildiginde her iki modelin ayri skorlari ve birlesim sonucu burada listelenir.</div>"
                        )
                    with gr.Tab("Tahmin Detayi"):
                        detail_html = gr.HTML(_detail([]))

        with gr.Accordion("Hazir CT Ornekleri", open=True):
            gr.Examples(
                examples=_examples(),
                inputs=image_input,
                label="",
                examples_per_page=6,
            )

        predict_btn.click(
            fn=predict,
            inputs=[image_input, model_choice, use_tta, threshold, show_cam],
            outputs=[banner, bars, cam_image, ensemble_html, detail_html],
        )
        clear_btn.click(
            fn=lambda: (
                None, _empty_banner(),
                _prob_bars({"Normal": 0.0, "Hemorrhage": 0.0}),
                None,
                "<div class='muted' style='padding:8px;'>Ensemble secildiginde her iki modelin ayri skorlari ve birlesim sonucu burada listelenir.</div>",
                _detail([]),
            ),
            inputs=[],
            outputs=[image_input, banner, bars, cam_image, ensemble_html, detail_html],
        )

    return demo


def _is_port_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((bind_host, port))
        except OSError:
            return False
    return True


def _pick_port(host: str, preferred: int, max_tries: int = 20) -> int:
    for offset in range(max_tries):
        candidate = preferred + offset
        if _is_port_available(host, candidate):
            return candidate
    raise OSError(f"{preferred}-{preferred + max_tries - 1} araliginda bos port yok.")


def launch_interface():
    host = os.getenv("GRADIO_SERVER_NAME", "127.0.0.1")
    port = _pick_port(host, int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    return create_interface().launch(server_name=host, server_port=port, share=False)


if __name__ == "__main__":
    launch_interface()
