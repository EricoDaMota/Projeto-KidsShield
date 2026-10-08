from __future__ import annotations

from io import BytesIO
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    CANVAS_AVAILABLE = False

from src.detector import FaceDetector, draw_face_boxes
from src.processor import apply_blur, image_to_png_bytes

st.set_page_config(
    page_title="Rosto Seguro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

STEPS = [
    "Início",
    "Carregar",
    "Detecção",
    "Proteção",
    "Revisão",
    "Confirmação",
    "Exportação",
]


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --teal:#0d7f76;
            --teal-dark:#075f59;
            --teal-soft:#e9f8f5;
            --mint:#bfeee6;
            --ink:#0b172a;
            --muted:#687386;
            --line:#dfe5e8;
            --bg:#f5f8f7;
            --card:#ffffff;
            --warning:#fffaf0;
            --warning-line:#f2d985;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: var(--ink);
        }
        .stApp { background: var(--bg); }
        [data-testid="stHeader"], [data-testid="stToolbar"], footer { visibility:hidden; height:0; }
        #MainMenu { visibility:hidden; }
        .block-container {
            max-width: 1220px;
            padding-top: 0.85rem;
            padding-bottom: 3.2rem;
        }

        /* Top bar */
        .topbar {
            display:flex; align-items:center; justify-content:space-between;
            background:#fff; border-bottom:1px solid var(--line);
            padding: 0.55rem 0.15rem 0.75rem 0.15rem;
            margin-bottom: 1.45rem;
        }
        .brand { display:flex; align-items:center; gap:.7rem; }
        .brand-icon {
            width:34px; height:34px; border-radius:10px; background:var(--teal);
            display:grid; place-items:center; color:white; font-size:17px;
            box-shadow:0 4px 10px rgba(13,127,118,.18);
        }
        .brand-name {font-size:.92rem; font-weight:800; line-height:1.05; color:var(--ink);}
        .brand-sub {font-size:.62rem; color:var(--muted); margin-top:.16rem;}
        .local-pill {
            display:inline-flex; align-items:center; gap:.35rem;
            border:1px solid #d9f0eb; background:#f4fbf9; color:#176f68;
            border-radius:999px; padding:.35rem .7rem; font-size:.65rem; font-weight:700;
        }

        /* Stepper */
        .steps-wrap { display:flex; justify-content:flex-end; margin-bottom:1.45rem; }
        .steps { display:flex; align-items:center; gap:0; }
        .step-item { display:flex; align-items:center; color:#9aa3af; font-size:.68rem; white-space:nowrap; }
        .step-dot {
            width:22px; height:22px; border-radius:999px; border:1px solid #cfd7dc;
            display:grid; place-items:center; margin-right:.36rem; background:#fff;
            font-size:.62rem; font-weight:700;
        }
        .step-item.done, .step-item.active { color:var(--teal); font-weight:700; }
        .step-item.done .step-dot { background:var(--teal); border-color:var(--teal); color:#fff; }
        .step-item.active .step-dot { border:1.5px solid var(--teal); color:var(--teal); }
        .step-line { width:36px; height:1px; background:#d3dadd; margin:0 .45rem; }
        .step-line.done { background:var(--teal); }

        /* Typography */
        .eyebrow {
            font-size:.67rem; letter-spacing:.18em; font-weight:800; color:var(--teal-dark);
            text-transform:uppercase; margin-bottom:.45rem;
        }
        .page-title { font-size:2rem; line-height:1.08; font-weight:850; letter-spacing:-.035em; color:var(--ink); margin:0; }
        .page-sub { color:var(--muted); margin:.5rem 0 1.2rem 0; font-size:.9rem; }
        .hero-title { font-size:3.25rem; line-height:1.02; font-weight:900; letter-spacing:-.05em; margin:.65rem 0 1rem 0; max-width:600px; }
        .hero-copy { color:#536172; font-size:1rem; line-height:1.65; max-width:610px; }
        .privacy-label {
            display:inline-flex; align-items:center; gap:.42rem; border:1px solid #9ddfd5; color:#0c756e;
            border-radius:999px; padding:.36rem .72rem; font-size:.68rem; font-weight:800; letter-spacing:.12em;
        }

        /* Cards */
        .surface {
            background:#fff; border:1px solid var(--line); border-radius:18px;
            box-shadow:0 10px 28px rgba(20,41,50,.055);
        }
        .image-shell {
            background:#fff; border:1px solid var(--line); border-radius:18px;
            padding:12px; box-shadow:0 10px 28px rgba(20,41,50,.055);
        }
        .mini-stat {
            display:flex; align-items:center; gap:.8rem; padding:.8rem 1rem;
            background:#fff; border:1px solid var(--line); border-radius:14px;
        }
        .stat-icon {
            width:34px;height:34px;border-radius:10px;background:#effaf7;color:var(--teal);
            display:grid;place-items:center;font-size:1rem;
        }
        .stat-title {font-weight:800;font-size:.82rem;color:var(--ink);}
        .stat-sub {font-size:.68rem;color:var(--muted);margin-top:.1rem;}
        .safety-box {
            border:1px solid var(--warning-line); background:var(--warning); border-radius:14px;
            padding:1rem 1.15rem; display:flex; gap:.75rem; align-items:center;
        }
        .safety-ico { width:34px;height:34px;border-radius:10px;background:#fff2bf;display:grid;place-items:center;color:#9a6b00; }
        .safety-title {font-size:.82rem;font-weight:800;color:#2c3440;}
        .safety-text {font-size:.69rem;color:#707783;margin-top:.13rem;}
        .result-note {font-size:.72rem;color:var(--teal-dark);font-weight:700;margin-top:.55rem;}

        /* Home simples e direta */
        .home-wrap {
            max-width:780px;
            margin:4.2rem auto 0 auto;
            text-align:center;
        }
        .home-wrap .hero-title {max-width:760px; margin-left:auto; margin-right:auto;}
        .home-wrap .hero-copy {max-width:650px; margin-left:auto; margin-right:auto;}
        .home-actions {max-width:310px; margin:1.6rem auto 0 auto;}
        .home-points {
            display:grid; grid-template-columns:repeat(3,1fr); gap:.75rem;
            margin:2rem auto 0 auto; max-width:760px;
        }
        .home-point {
            background:#fff; border:1px solid #d7e1e3; border-radius:14px;
            padding:1rem .9rem; color:#354354; font-size:.78rem; line-height:1.4;
            box-shadow:0 6px 18px rgba(20,41,50,.035);
        }
        .home-point b {display:block;color:var(--ink);font-size:.82rem;margin-bottom:.2rem;}

        /* Streamlit widgets — contraste alto inclusive no hover */
        div.stButton > button, div.stDownloadButton > button {
            border-radius:10px !important; min-height:2.75rem; font-weight:750 !important;
            border:1px solid #9daeb4 !important; box-shadow:none !important;
            background:#ffffff !important; color:#16313a !important;
            transition:background .15s ease,border-color .15s ease,color .15s ease,transform .15s ease !important;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            background:#e5f4f1 !important; color:#075f59 !important;
            border-color:#0d7f76 !important;
        }
        div.stButton > button:focus, div.stDownloadButton > button:focus {
            color:#075f59 !important; border-color:#0d7f76 !important;
            box-shadow:0 0 0 3px rgba(13,127,118,.14) !important;
        }
        div.stButton > button p, div.stDownloadButton > button p { color:inherit !important; }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
            background:var(--teal) !important; border-color:var(--teal) !important; color:white !important;
        }
        div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button[kind="primary"]:hover,
        div.stButton > button[kind="primary"]:focus, div.stDownloadButton > button[kind="primary"]:focus {
            background:var(--teal-dark) !important; border-color:var(--teal-dark) !important; color:#ffffff !important;
        }
        div.stButton > button:disabled, div.stDownloadButton > button:disabled {
            background:#e8ecee !important; color:#7b858d !important; border-color:#cbd3d7 !important;
            opacity:1 !important;
        }
        [data-testid="stFileUploader"] button {
            background:#ffffff !important; color:#075f59 !important; border:1px solid #0d7f76 !important;
        }
        [data-testid="stFileUploader"] button:hover {
            background:#dff3ef !important; color:#064f4a !important; border-color:#075f59 !important;
        }
        [data-testid="stCheckbox"] label { color:#172b34 !important; font-weight:700 !important; }
        [data-testid="stCheckbox"] div[role="checkbox"] { border-color:#6e858d !important; }
        [data-testid="stNumberInput"] input,
        [data-testid="stTextInput"] input {
            background:#ffffff !important; color:#10252d !important; border:1px solid #9fb0b6 !important;
        }
        [data-testid="stNumberInput"] input:focus,
        [data-testid="stTextInput"] input:focus { border-color:#0d7f76 !important; box-shadow:0 0 0 2px rgba(13,127,118,.12) !important; }
        [data-testid="stFileUploader"] {
            background:#fff; border:1px dashed #9fd9d1; border-radius:14px; padding:.35rem;
        }
        [data-testid="stFileUploader"] section { background:#fbfefd; border:none; }
        [data-testid="stImage"] img { border-radius:12px; }
        [data-testid="stCheckbox"] label {font-weight:650;}
        div[data-testid="stExpander"] {border:1px solid var(--line);border-radius:14px;background:#fff;overflow:hidden;}
        div[data-testid="stSlider"] {padding-top:.1rem;}
        div[data-testid="stAlert"] {border-radius:12px;}

        .face-list-title { display:flex;justify-content:space-between;align-items:center;font-weight:800;font-size:.85rem;margin-bottom:.75rem; }
        .counter-pill { background:#eefaf7;color:var(--teal-dark);padding:.22rem .45rem;border-radius:999px;font-size:.62rem;font-weight:800; }
        .privacy-foot {color:#687386;font-size:.68rem;margin-top:.7rem;display:flex;gap:.35rem;align-items:center;}

        @media (max-width: 800px) {
            .hero-title{font-size:2.3rem}.home-wrap{margin-top:2rem;text-align:left}.home-actions{margin-left:0}.home-points{grid-template-columns:1fr}.steps-wrap{justify-content:flex-start;overflow-x:auto}.step-line{width:18px}.block-container{padding-left:1rem;padding-right:1rem}.page-title{font-size:1.65rem}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def defaults() -> dict:
    return {
        "step": 0,
        "image_rgb": None,
        "file_name": "",
        "faces": [],
        "selected": [],
        "manual_boxes": [],
        "manual_tool": "Desenhar nova área",
        "manual_canvas_rev": 0,
        "manual_canvas_seed": [],
        "manual_first": False,
        "processed": None,
        "blur_strength": 48,
        "margin_percent": 10,
        "detected_once": False,
        "safety_confirm": False,
    }


def init_state() -> None:
    for key, value in defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_all() -> None:
    clear_face_widget_state()
    for key, value in defaults().items():
        st.session_state[key] = value
    st.rerun()


def goto(step: int) -> None:
    st.session_state.step = max(0, min(step, len(STEPS) - 1))
    if st.session_state.step == 3:
        # Ao (re)entrar na tela de proteção, o editor manual recomeça a partir
        # das áreas já salvas, inclusive ao voltar da revisão para corrigir.
        bump_manual_canvas()
    st.rerun()


FACE_WIDGET_PREFIX = "face_select_"


def face_widget_key(index: int) -> str:
    """Chave estável para o seletor de cada rosto durante toda a sessão."""
    return f"{FACE_WIDGET_PREFIX}{index}"


def clear_face_widget_state() -> None:
    """Remove estados antigos dos checkboxes ao trocar/reanalisar a imagem."""
    for key in list(st.session_state.keys()):
        if str(key).startswith(FACE_WIDGET_PREFIX):
            del st.session_state[key]


def on_face_selection_change(index: int) -> None:
    """Sincroniza UM rosto por vez; os demais permanecem intocados."""
    flags = selected_flags()
    if 0 <= index < len(flags):
        flags[index] = bool(st.session_state.get(face_widget_key(index), flags[index]))
        st.session_state.selected = flags
        # Qualquer mudança de seleção invalida uma imagem processada anterior.
        st.session_state.processed = None
        st.session_state.safety_confirm = False


def set_all_face_selections(value: bool) -> None:
    """Seleciona/desmarca todos e mantém os widgets sincronizados."""
    flags = [bool(value)] * len(all_boxes())
    st.session_state.selected = flags
    for i, flag in enumerate(flags):
        st.session_state[face_widget_key(i)] = flag
    st.session_state.processed = None
    st.session_state.safety_confirm = False


def topbar() -> None:
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <div class="brand-icon">♢</div>
            <div><div class="brand-name">Rosto Seguro</div><div class="brand-sub">Proteção de identidade</div></div>
          </div>
          <div class="local-pill">⌾ &nbsp; Processamento privado e local</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def visual_step_index() -> int:
    step = st.session_state.step
    if step <= 1:
        return 0
    if step == 2:
        return 1
    if step == 3:
        return 2
    if step == 4:
        return 3
    return 4


def stepper() -> None:
    if st.session_state.step == 0:
        return
    labels = ["Imagem", "Detecção", "Proteção", "Revisão", "Exportação"]
    current = visual_step_index()
    parts = []
    for i, label in enumerate(labels):
        cls = "done" if i < current else ("active" if i == current else "")
        mark = "✓" if i < current else str(i + 1)
        parts.append(f'<div class="step-item {cls}"><span class="step-dot">{mark}</span>{label}</div>')
        if i < len(labels) - 1:
            line_cls = "done" if i < current else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')
    st.markdown('<div class="steps-wrap"><div class="steps">' + ''.join(parts) + '</div></div>', unsafe_allow_html=True)


def all_boxes() -> List[Tuple[int, int, int, int]]:
    return list(st.session_state.faces) + list(st.session_state.manual_boxes)


def selected_flags() -> List[bool]:
    boxes = all_boxes()
    values = list(st.session_state.selected)
    if len(values) < len(boxes):
        values.extend([True] * (len(boxes) - len(values)))
        st.session_state.selected = values
    elif len(values) > len(boxes):
        values = values[: len(boxes)]
        st.session_state.selected = values
    return values


def fit_canvas_size(image_rgb: np.ndarray, max_width: int = 860) -> tuple[int, int, float]:
    height, width = image_rgb.shape[:2]
    if width <= max_width:
        return width, height, 1.0
    scale = max_width / float(width)
    return int(width * scale), int(height * scale), scale


def manual_boxes_to_canvas_objects(boxes: List[Tuple[int, int, int, int]], scale: float) -> list[dict]:
    objects: list[dict] = []
    for x, y, w, h in boxes:
        objects.append({
            "type": "Rect",
            # O Fabric.js 7 usa "center" como origem padrão; sem isso a caixa
            # reaparece deslocada meia largura/altura após cada atualização.
            "originX": "left",
            "originY": "top",
            "left": round(x * scale, 2),
            "top": round(y * scale, 2),
            "width": round(w * scale, 2),
            "height": round(h * scale, 2),
            "scaleX": 1,
            "scaleY": 1,
            "fill": "rgba(13, 127, 118, 0.18)",
            "stroke": "#0d7f76",
            "strokeWidth": 3,
            "rx": 8,
            "ry": 8,
        })
    return objects


def canvas_objects_to_manual_boxes(objects: list[dict] | None, scale: float, image_shape: tuple[int, ...]) -> List[Tuple[int, int, int, int]]:
    if not objects:
        return []

    height, width = image_shape[:2]
    origin_offset = {"left": 0.0, "top": 0.0, "center": 0.5, "right": 1.0, "bottom": 1.0}
    boxes: list[Tuple[int, int, int, int]] = []
    for obj in objects:
        # O streamlit-drawable-canvas 0.13 (Fabric.js 7) devolve "Rect" com R maiúsculo.
        # Ao arrastar um canto no modo Editar, o quadrante vira "Polygon"; nesse caso
        # o blur cobre o retângulo que envolve o polígono, para a área não sumir.
        if str(obj.get("type", "")).lower() not in ("rect", "polygon"):
            continue
        obj_w = abs(float(obj.get("width", 0)) * float(obj.get("scaleX", 1)))
        obj_h = abs(float(obj.get("height", 0)) * float(obj.get("scaleY", 1)))
        if obj_w <= 2 or obj_h <= 2:
            continue
        left = float(obj.get("left", 0)) - origin_offset.get(obj.get("originX", "left"), 0.0) * obj_w
        top = float(obj.get("top", 0)) - origin_offset.get(obj.get("originY", "top"), 0.0) * obj_h

        # Recorta a caixa nos limites da imagem: uma área parcialmente fora da
        # foto (ex.: rosto cortado na borda) continua válida para o blur.
        x1 = max(0, min(width, int(round(left / scale))))
        y1 = max(0, min(height, int(round(top / scale))))
        x2 = max(0, min(width, int(round((left + obj_w) / scale))))
        y2 = max(0, min(height, int(round((top + obj_h) / scale))))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def trim_face_widget_state(total_boxes: int) -> None:
    for key in list(st.session_state.keys()):
        key_str = str(key)
        if not key_str.startswith(FACE_WIDGET_PREFIX):
            continue
        try:
            idx = int(key_str.replace(FACE_WIDGET_PREFIX, ""))
        except ValueError:
            continue
        if idx >= total_boxes:
            del st.session_state[key]


def bump_manual_canvas() -> None:
    """Recria o editor manual partindo das áreas manuais atuais."""
    st.session_state.manual_canvas_seed = list(st.session_state.manual_boxes)
    st.session_state.manual_canvas_rev = int(st.session_state.get("manual_canvas_rev", 0)) + 1


def sync_manual_boxes(new_manual_boxes: List[Tuple[int, int, int, int]]) -> bool:
    """Atualiza as áreas manuais; retorna True quando algo mudou."""
    old_manual_boxes = list(st.session_state.manual_boxes)
    if old_manual_boxes == list(new_manual_boxes):
        return False

    auto_count = len(st.session_state.faces)
    current_flags = list(st.session_state.selected)
    auto_flags = current_flags[:auto_count]
    manual_flags = current_flags[auto_count : auto_count + len(old_manual_boxes)]

    if len(manual_flags) < len(old_manual_boxes):
        manual_flags.extend([True] * (len(old_manual_boxes) - len(manual_flags)))

    if len(new_manual_boxes) > len(manual_flags):
        manual_flags.extend([True] * (len(new_manual_boxes) - len(manual_flags)))

    manual_flags = manual_flags[: len(new_manual_boxes)]

    st.session_state.manual_boxes = list(new_manual_boxes)
    st.session_state.selected = auto_flags + manual_flags
    st.session_state.processed = None
    st.session_state.safety_confirm = False
    trim_face_widget_state(auto_count + len(new_manual_boxes))
    return True


def open_uploaded_image(uploaded) -> np.ndarray:
    data = uploaded.getvalue()
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("A imagem é maior que 20 MB.")
    img = Image.open(BytesIO(data))
    img.verify()
    img = Image.open(BytesIO(data)).convert("RGB")
    width, height = img.size
    if width < 64 or height < 64:
        raise ValueError("A imagem é pequena demais para análise.")
    if width * height > 40_000_000:
        raise ValueError("A resolução é muito alta. Use uma imagem de até 40 megapixels.")
    return np.array(img)


def page_heading(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div><h1 class="page-title">{title}</h1><div class="page-sub">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def home_screen() -> None:
    st.markdown(
        """
        <div class="home-wrap">
          <div class="privacy-label">♢ &nbsp; PRIVACIDADE EM PRIMEIRO LUGAR</div>
          <h1 class="hero-title">Proteja identidades antes de compartilhar.</h1>
          <div class="hero-copy">
            Carregue uma foto, confira os rostos encontrados, escolha quem deve ser protegido e revise o resultado antes de salvar.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="home-actions">', unsafe_allow_html=True)
    if st.button("Carregar imagem  →", type="primary", use_container_width=True):
        goto(1)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="home-points">
          <div class="home-point"><b>1. Carregue</b>Selecione uma imagem JPG, JPEG ou PNG.</div>
          <div class="home-point"><b>2. Proteja</b>Confira e selecione as pessoas que devem ser desfocadas.</div>
          <div class="home-point"><b>3. Revise</b>Compare o resultado antes de baixar a imagem.</div>
        </div>
        <div class="privacy-foot" style="justify-content:center;margin-top:1.25rem">♙ &nbsp; Suas imagens são processadas localmente e não são enviadas a terceiros.</div>
        """,
        unsafe_allow_html=True,
    )


def upload_screen() -> None:
    page_heading("IMAGEM", "Carregue a imagem", "Escolha uma foto JPG, JPEG ou PNG. O processamento será feito localmente.")
    uploaded = st.file_uploader("Imagem", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            image_rgb = open_uploaded_image(uploaded)
            previous_name = st.session_state.file_name
            if uploaded.name != previous_name or st.session_state.image_rgb is None:
                st.session_state.image_rgb = image_rgb
                st.session_state.file_name = uploaded.name
                clear_face_widget_state()
                st.session_state.faces = []
                st.session_state.selected = []
                st.session_state.manual_boxes = []
                st.session_state.processed = None
                st.session_state.detected_once = False
                st.session_state.safety_confirm = False
            h, w = image_rgb.shape[:2]
            st.markdown('<div class="image-shell">', unsafe_allow_html=True)
            st.image(image_rgb, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.caption(f"{uploaded.name}  ·  {w} × {h}px")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            st.error(f"Não foi possível carregar esta imagem. {exc}")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if st.button("←  Voltar", use_container_width=True):
            goto(0)
    with c3:
        disabled = st.session_state.image_rgb is None
        if st.button("Continuar  →", type="primary", disabled=disabled, use_container_width=True):
            goto(2)


def detection_screen() -> None:
    page_heading("DETECÇÃO DE ROSTOS", "Confira as pessoas encontradas", "Verifique se cada pessoa foi identificada antes de continuar.")
    image_rgb = st.session_state.image_rgb
    if image_rgb is None:
        st.warning("Nenhuma imagem carregada.")
        if st.button("Escolher imagem"):
            goto(1)
        return

    if not st.session_state.detected_once:
        with st.spinner("Analisando imagem..."):
            detector = FaceDetector()
            found = detector.detect(image_rgb)
            clear_face_widget_state()
            st.session_state.faces = [f.as_tuple() for f in found]
            st.session_state.selected = [True] * len(found)
            st.session_state.manual_boxes = []
            st.session_state.detected_once = True

    faces = st.session_state.faces
    preview = draw_face_boxes(image_rgb, faces) if faces else image_rgb
    st.markdown('<div class="image-shell">', unsafe_allow_html=True)
    st.image(preview, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if faces:
        stat_left, stat_right = st.columns([3, 1])
        with stat_left:
            st.markdown(
                f'<div class="mini-stat"><div class="stat-icon">⌗</div><div><div class="stat-title">{len(faces)} rosto(s) detectado(s)</div><div class="stat-sub">Todos estão delimitados na imagem</div></div></div>',
                unsafe_allow_html=True,
            )
        with stat_right:
            if st.button("Confirmar detecção  →", type="primary", use_container_width=True):
                st.session_state.manual_first = False
                goto(3)
    else:
        st.warning("Nenhum rosto foi detectado. Você ainda pode continuar e adicionar uma área manual na próxima etapa.")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Tentar novamente", use_container_width=True):
                st.session_state.detected_once = False
                st.rerun()
        with c2:
            if st.button("Continuar  →", type="primary", use_container_width=True):
                st.session_state.manual_first = False
                goto(3)

    # Borrão manual disponível logo após a detecção, com ou sem rostos encontrados.
    st.write("")
    st.markdown("### Borrão manual")
    manual_left, manual_right = st.columns([3, 1])
    with manual_left:
        st.markdown(
            '<div class="mini-stat"><div class="stat-icon">✎</div><div><div class="stat-title">Marque qualquer área com o mouse</div>'
            '<div class="stat-sub">Use quando um rosto não foi detectado, aparece cortado na borda da foto, de perfil ou parcialmente coberto.</div></div></div>',
            unsafe_allow_html=True,
        )
    with manual_right:
        if st.button("Usar borrão manual  →", use_container_width=True, key="open_manual_blur"):
            st.session_state.manual_first = True
            goto(3)

    st.write("")
    if st.button("←  Voltar"):
        goto(1)


def manual_blur_section(image_rgb: np.ndarray) -> None:
    """Editor de borrão manual: quadrantes desenhados com o mouse sobre a imagem."""
    st.markdown("### Blur manual personalizado")
    st.caption(
        "A detecção automática continua funcionando normalmente. Se algum rosto não for reconhecido, "
        "crie um quadrante manual em qualquer ponto da imagem e ajuste com o mouse."
    )

    if CANVAS_AVAILABLE:
        guide1, guide2, guide3 = st.columns(3)
        with guide1:
            st.markdown(
                '<div class="mini-stat"><div class="stat-icon">＋</div><div><div class="stat-title">1. Criar quadrante</div>'
                '<div class="stat-sub">Clique e arraste diretamente sobre qualquer região da imagem.</div></div></div>',
                unsafe_allow_html=True,
            )
        with guide2:
            st.markdown(
                '<div class="mini-stat"><div class="stat-icon">↔</div><div><div class="stat-title">2. Mover ou aumentar</div>'
                '<div class="stat-sub">Na barra do editor, ative Editar e arraste a caixa ou suas alças.</div></div></div>',
                unsafe_allow_html=True,
            )
        with guide3:
            st.markdown(
                '<div class="mini-stat"><div class="stat-icon">⌫</div><div><div class="stat-title">3. Apagar</div>'
                '<div class="stat-sub">Use os botões abaixo para remover uma área ou limpar todas.</div></div></div>',
                unsafe_allow_html=True,
            )

        st.info(
            "No editor abaixo, o mouse sempre cria um novo quadrante. Para mover, aumentar ou diminuir um quadrante já criado, "
            "use o botão **Editar** da barra do próprio editor e depois clique na caixa desejada."
        )

        canvas_width, canvas_height, canvas_scale = fit_canvas_size(image_rgb)
        canvas_bg = Image.fromarray(image_rgb).resize((canvas_width, canvas_height))
        initial_drawing = {
            "version": "7.0.0",
            # O desenho inicial só muda quando o editor é recriado (nova chave). Se ele
            # acompanhasse cada atualização, o editor voltaria ao estado anterior e
            # os quadrantes ficariam alternando entre a versão antiga e a nova.
            "objects": manual_boxes_to_canvas_objects(st.session_state.manual_canvas_seed, canvas_scale),
        }

        # streamlit-drawable-canvas >= 0.12 removeu os modos transform/edit.
        # A edição (mover/redimensionar) agora é feita pelo botão Editar da barra do canvas.
        canvas_result = st_canvas(
            fill_color="rgba(13, 127, 118, 0.18)",
            stroke_width=3,
            stroke_color="#0d7f76",
            background_image=canvas_bg,
            background_image_fit="stretch",
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect",
            initial_drawing=initial_drawing,
            key=f"manual_canvas_v4_{st.session_state.manual_canvas_rev}",
        )

        if canvas_result is not None and canvas_result.json_data is not None:
            new_manual_boxes = canvas_objects_to_manual_boxes(
                (canvas_result.json_data or {}).get("objects"),
                canvas_scale,
                image_rgb.shape,
            )
            if sync_manual_boxes(new_manual_boxes):
                # Atualiza já a lista de áreas e o preview com o quadrante recém-desenhado.
                st.rerun()

        current_manual = list(st.session_state.manual_boxes)
        if current_manual:
            st.success(
                f"{len(current_manual)} quadrante(s) manual(is) ativo(s). Eles serão incluídos no blur final."
            )
            st.markdown("**Quadrantes manuais criados**")
            for idx, box in enumerate(current_manual, start=1):
                info_col, action_col = st.columns([4, 1])
                with info_col:
                    x, y, bw, bh = box
                    st.caption(
                        f"Quadrante {idx} · tamanho {bw}×{bh}px · posição atual ajustável pelo mouse no editor."
                    )
                with action_col:
                    if st.button(
                        f"Apagar {idx}",
                        key=f"delete_manual_{idx}",
                        use_container_width=True,
                    ):
                        remaining = list(st.session_state.manual_boxes)
                        del remaining[idx - 1]
                        sync_manual_boxes(remaining)
                        bump_manual_canvas()
                        st.rerun()

            del_col, clear_col = st.columns(2)
            with del_col:
                if st.button("Apagar última área manual", use_container_width=True):
                    sync_manual_boxes(st.session_state.manual_boxes[:-1])
                    bump_manual_canvas()
                    st.rerun()
            with clear_col:
                if st.button("Limpar todas as áreas manuais", use_container_width=True):
                    sync_manual_boxes([])
                    bump_manual_canvas()
                    st.rerun()
        else:
            st.info("Nenhum quadrante manual criado. Clique e arraste sobre a imagem acima para criar um.")

        st.write("")
        manual_apply_boxes = all_boxes()
        manual_apply_flags = selected_flags()
        manual_chosen = [box for i, box in enumerate(manual_apply_boxes) if manual_apply_flags[i]]
        if st.button(
            "Aplicar proteção e revisar  →",
            type="primary",
            use_container_width=True,
            key="apply_after_manual_editor",
        ):
            if not manual_chosen:
                st.error("Nenhuma área está selecionada. Selecione um rosto detectado ou crie um quadrante manual.")
            else:
                with st.spinner(f"Aplicando desfoque em {len(manual_chosen)} área(s)..."):
                    st.session_state.processed = apply_blur(
                        image_rgb,
                        manual_chosen,
                        strength=st.session_state.blur_strength,
                        margin_percent=st.session_state.margin_percent,
                    )
                goto(4)
    else:
        st.warning(
            "O editor manual com mouse não foi carregado. Feche o programa, execute INICIAR novamente e aguarde a atualização das dependências."
        )


def selection_screen() -> None:
    page_heading(
        "ESCOLHA DE PROTEÇÃO",
        "Quem deve ter a identidade protegida?",
        "Marque somente os rostos que deverão receber o desfoque. Cada rosto funciona de forma independente.",
    )
    image_rgb = st.session_state.image_rgb
    if image_rgb is None:
        goto(1)
        return

    if st.session_state.manual_first:
        manual_blur_section(image_rgb)
        st.markdown("### Rostos e áreas selecionados")

    boxes = all_boxes()
    flags = selected_flags()

    # O preview sempre usa o estado canônico de seleção.
    preview = draw_face_boxes(image_rgb, boxes, flags) if boxes else image_rgb

    img_col, side_col = st.columns([3.15, 1], gap="medium")
    with img_col:
        st.markdown('<div class="image-shell">', unsafe_allow_html=True)
        st.image(preview, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        if boxes:
            st.caption("Verde = será desfocado · Cinza = permanecerá visível")

    with side_col:
        if boxes:
            selected_count = sum(flags)
            st.markdown(
                f'<div class="face-list-title"><span>Rostos</span><span class="counter-pill">{selected_count} de {len(boxes)}</span></div>',
                unsafe_allow_html=True,
            )
            auto_count = len(st.session_state.faces)

            for i, _box in enumerate(boxes):
                key = face_widget_key(i)
                # Inicializa apenas uma vez. Depois disso, o próprio widget é a entrada
                # e o callback atualiza a lista canônica st.session_state.selected.
                if key not in st.session_state:
                    st.session_state[key] = bool(flags[i])

                label = f"Rosto {i + 1}" if i < auto_count else f"Área manual {i - auto_count + 1}"
                st.checkbox(
                    label,
                    key=key,
                    on_change=on_face_selection_change,
                    args=(i,),
                    help="Marcado: este rosto será desfocado. Desmarcado: ficará visível.",
                )

            st.markdown(
                f'<div class="result-note">{selected_count} rosto(s) selecionado(s) para desfocar</div>',
                unsafe_allow_html=True,
            )
            st.write("")

            # O processamento lê SOMENTE a lista canônica atual, na mesma ordem das caixas.
            if st.button("Aplicar proteção  →", type="primary", use_container_width=True):
                current_flags = selected_flags()
                chosen = [box for i, box in enumerate(boxes) if current_flags[i]]
                if not chosen:
                    st.error("Nenhum rosto foi selecionado. Marque pelo menos um rosto para aplicar o desfoque.")
                else:
                    with st.spinner(f"Aplicando desfoque em {len(chosen)} rosto(s)..."):
                        st.session_state.processed = apply_blur(
                            image_rgb,
                            chosen,
                            strength=st.session_state.blur_strength,
                            margin_percent=st.session_state.margin_percent,
                        )
                    goto(4)
        else:
            st.info("Nenhum rosto foi detectado. Use o borrão manual para adicionar uma área.")

    if boxes:
        a, b, _ = st.columns([1, 1, 3])
        with a:
            st.button(
                "Selecionar todos",
                use_container_width=True,
                on_click=set_all_face_selections,
                args=(True,),
            )
        with b:
            st.button(
                "Desmarcar todos",
                use_container_width=True,
                on_click=set_all_face_selections,
                args=(False,),
            )

    if not st.session_state.manual_first:
        manual_blur_section(image_rgb)

    st.session_state.blur_strength = st.slider("Intensidade do desfoque", 20, 100, int(st.session_state.blur_strength), 5)
    st.session_state.margin_percent = st.slider("Margem extra ao redor do rosto (%)", 0, 30, int(st.session_state.margin_percent), 1)

    st.write("")
    if st.button("←  Voltar"):
        goto(2)


def review_screen() -> None:
    page_heading("REVISÃO OBRIGATÓRIA", "Revise antes de continuar", "Compare as versões e confira com atenção se todas as identidades necessárias foram protegidas.")
    if st.session_state.processed is None:
        st.warning("Ainda não existe uma imagem processada.")
        if st.button("Voltar para seleção"):
            goto(3)
        return

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div style="font-size:.75rem;font-weight:800;margin-bottom:.4rem">Imagem original <span style="float:right;color:#9aa3af;font-weight:500">Referência</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="image-shell">', unsafe_allow_html=True)
        st.image(st.session_state.image_rgb, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        count = sum(selected_flags())
        st.markdown(f'<div style="font-size:.75rem;font-weight:800;margin-bottom:.4rem">Imagem protegida <span style="float:right;color:#0d7f76;font-weight:700">♢ {count} protegido(s)</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="image-shell">', unsafe_allow_html=True)
        st.image(st.session_state.processed, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    sleft, sright = st.columns([3, 1.25])
    with sleft:
        st.markdown('<div class="safety-box"><div class="safety-ico">⌗</div><div><div class="safety-title">Faça uma última verificação visual</div><div class="safety-text">Caso algo esteja incorreto, volte e ajuste a seleção. A exportação só será liberada após sua confirmação.</div></div></div>', unsafe_allow_html=True)
    with sright:
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Voltar e corrigir", use_container_width=True):
                goto(3)
        with b2:
            if st.button("Está correto  →", type="primary", use_container_width=True):
                st.session_state.safety_confirm = False
                goto(5)


def confirm_screen() -> None:
    if st.session_state.processed is None:
        goto(3)
        return

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="image-shell">', unsafe_allow_html=True)
        st.image(st.session_state.processed, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-note">♢ &nbsp; Imagem final com {sum(selected_flags())} identidade(s) protegida(s)</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div style="width:42px;height:42px;border-radius:12px;background:#0d7f76;color:white;display:grid;place-items:center;font-size:1.15rem;margin-bottom:1rem">♙</div>', unsafe_allow_html=True)
        page_heading("CONFIRMAÇÃO DE SEGURANÇA", "Antes de exportar", "Sua atenção é a última camada de proteção. Confirme que revisou cuidadosamente o resultado.")
        st.session_state.safety_confirm = st.checkbox(
            "Você verificou todos os rostos que deveriam ser protegidos?",
            value=st.session_state.safety_confirm,
            help="Confirme que revisou a imagem e que as identidades necessárias estão desfocadas.",
        )
        st.write("")
        if st.button("Continuar para exportação  →", type="primary", disabled=not st.session_state.safety_confirm, use_container_width=True):
            goto(6)
        if st.button("←  Revisar novamente", use_container_width=True):
            goto(4)


def result_screen() -> None:
    if st.session_state.processed is None:
        goto(3)
        return

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown('<div class="image-shell">', unsafe_allow_html=True)
        st.image(st.session_state.processed, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-note">✓ &nbsp; Imagem final com {sum(selected_flags())} identidade(s) protegida(s)</div>', unsafe_allow_html=True)

    with right:
        page_heading("EXPORTAÇÃO", "Imagem protegida com sucesso", "O arquivo está pronto para ser salvo no computador.")
        png = image_to_png_bytes(st.session_state.processed)
        stem = (st.session_state.file_name.rsplit(".", 1)[0] or "imagem") if st.session_state.file_name else "imagem"
        st.download_button(
            "↓  Baixar imagem",
            data=png,
            file_name=f"{stem}_protegida.png",
            mime="image/png",
            type="primary",
            use_container_width=True,
        )
        st.caption("O compartilhamento continua sob seu controle. Nenhuma rede social é acionada automaticamente.")
        st.write("")
        if st.button("⟳  Processar outra imagem", use_container_width=True):
            reset_all()


def main() -> None:
    inject_css()
    init_state()
    topbar()
    stepper()
    screens = [home_screen, upload_screen, detection_screen, selection_screen, review_screen, confirm_screen, result_screen]
    screens[st.session_state.step]()


if __name__ == "__main__":
    main()
