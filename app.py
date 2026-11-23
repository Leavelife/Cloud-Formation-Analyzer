import streamlit as st
from components.styles import inject_styles
from components.navbar import render_navbar
from pages.overview import render_overview
from pages.image_input import render_image_input
from pages.processing import render_processing
from pages.segmentation import render_segmentation
from pages.analysis import render_analysis

st.set_page_config(page_title="Cloud Formation Analyzer", page_icon="☁️", layout="wide", initial_sidebar_state="collapsed")
inject_styles()

if "active_page" not in st.session_state:
    st.session_state.active_page = "Overview"
if "image_rgb" not in st.session_state:
    st.session_state.image_rgb = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None
if "processed" not in st.session_state:
    st.session_state.processed = None
if "roi" not in st.session_state:
    st.session_state.roi = None
if "params" not in st.session_state:
    st.session_state.params = {
        "h_min": 0, "h_max": 180, "s_min": 0, "s_max": 85,
        "v_min": 150, "v_max": 255,
        "lab_l_min": 170, "lab_chroma_max": 28,
        "blur": 1, "method": "Adaptive fusion", "combined_mode": "AND",
        "morph": "Opening", "kernel": 3, "erosion": 0, "dilation": 0,
        "argd_k": 0.9, "argd_sunlight_k": 0.6, "argd_threshold": -0.15,
        "nsv_threshold": -0.3, "hyta_fixed_threshold": 0.0,
        "hyta_std_threshold": 0.12, "superpixel_count": 400,
        "superpixel_compactness": 10.0, "local_threshold_scale": 0.25,
        "local_feature_ceiling": -0.65, "hybrid_w_argd": 0.35,
        "hybrid_w_nsv": 0.25, "hybrid_w_ratio": 0.2, "hybrid_w_local": 0.2,
        "hybrid_threshold": 0.55, "ambiguity_width": 0.12,
        "sunlight_penalty": 0.35, "sunlight_mode": "Warm-bright + user region",
        "texture_weight": 0.0,
        "texture_window": 9, "texture_bins": 16,
        "segmentation_method": "SLIC",
        "structure_w_argd": 0.30, "structure_w_nsv": 0.25,
        "structure_w_ratio": 0.25, "structure_w_texture": 0.20,
        "structure_threshold": 0.55, "continuity_margin": 0.12,
        "continuity_bonus": 0.08,
        "adaptive_threshold_scale": 0.25,
        "thin_contrast_threshold": 0.20, "thin_score_threshold": 0.48,
        "thin_color_floor": 0.28,
        "dark_region_threshold": 0.48, "dark_saturation_ceiling": 0.48,
        "dark_texture_floor": 0.16, "dark_continuity_bonus": 0.10,
        "fusion_w_baseline": 0.45, "fusion_w_thin": 0.55,
        "fusion_w_dark": 0.45, "fusion_threshold": 0.52,
    }

render_navbar()
page = st.session_state.active_page
if page == "Overview":
    render_overview()
elif page == "Image Input":
    render_image_input()
elif page == "Processing":
    render_processing()
elif page == "Segmentation":
    render_segmentation()
elif page == "Analysis":
    render_analysis()
