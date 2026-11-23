
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import hashlib

def _draw_roi_preview(image_rgb, roi):
    preview = Image.fromarray(image_rgb.copy())
    draw = ImageDraw.Draw(preview)
    x1, y1, x2, y2 = roi
    draw.rectangle((x1, y1, x2, y2), outline=(57, 169, 232), width=max(3, image_rgb.shape[1] // 300))
    return preview

def render_image_input():
    st.markdown('<div class="eyebrow">STEP 01 / ACQUISITION</div><h1>Image input</h1><p>Upload a sky photograph, then define the precise rectangular and optional polygon region. Only valid pixels inside this area contribute to coverage and evaluation.</p>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop a sky image here", type=["jpg", "jpeg", "png"])
    if uploaded:
        try:
            image_bytes = uploaded.getvalue()
            image_digest = hashlib.sha256(image_bytes).hexdigest()
            if st.session_state.get("image_digest") != image_digest:
                st.session_state.processed = None
                st.session_state.roi = None
                for key in ("roi_x1", "roi_x2", "roi_y1", "roi_y2", "polygon_points_text"):
                    st.session_state.pop(key, None)
            image = Image.open(uploaded).convert("RGB")
            arr = np.array(image)
            st.session_state.image_rgb = arr
            st.session_state.file_name = uploaded.name
            st.session_state.image_digest = image_digest
        except Exception as e:
            st.error(f"Could not read image: {e}")

    if st.session_state.image_rgb is None:
        st.info("Upload an image to begin. For the best baseline result, use a photo that shows a clear sky area with minimal buildings, trees, or foreground objects.")
        return

    arr = st.session_state.image_rgb
    h, w = arr.shape[:2]
    if st.session_state.roi is None:
        st.session_state.roi = (0, 0, w, h)
    x1, y1, x2, y2 = st.session_state.roi

    st.markdown("### Select the sky analysis region")
    st.caption("Adjust the four boundaries. The blue rectangle marks the selected region; everything outside it is excluded.")
    preview_col, info_col = st.columns([1.5, 1], gap="large")
    with preview_col:
        st.image(_draw_roi_preview(arr, (x1, y1, x2, y2)), caption="Selected region preview", use_container_width=True)
    with info_col:
        st.markdown('<div class="panel"><div class="panel-title">Image properties</div>', unsafe_allow_html=True)
        st.metric("Image dimensions", f"{w} × {h} px")
        st.metric("Selected region", f"{max(0,x2-x1)} × {max(0,y2-y1)} px")
        st.markdown(f'<div class="small-muted">File: {st.session_state.file_name}</div></div>', unsafe_allow_html=True)

    st.markdown("#### Region boundaries")
    c1, c2 = st.columns(2)
    with c1:
        nx1 = st.slider("Left (X1)", 0, max(1, w-1), min(x1, w-1), key="roi_x1")
        ny1 = st.slider("Top (Y1)", 0, max(1, h-1), min(y1, h-1), key="roi_y1")
    with c2:
        nx2 = st.slider("Right (X2)", 1, w, max(1, min(x2, w)), key="roi_x2")
        ny2 = st.slider("Bottom (Y2)", 1, h, max(1, min(y2, h)), key="roi_y2")

    valid = nx2 > nx1 and ny2 > ny1
    if valid:
        next_roi = (nx1, ny1, nx2, ny2)
        if st.session_state.roi != next_roi:
            st.session_state.processed = None
        st.session_state.roi = next_roi
        st.success("Analysis region is valid. Only this region will be processed.")
        cropped = arr[ny1:ny2, nx1:nx2]
        st.image(cropped, caption="Exact region sent to the processing pipeline", use_container_width=True)
        with st.expander("Optional polygon exclusion / sky validity mask"):
            st.caption("Enter full-image coordinates to create a polygon inside the rectangle. Format: x1,y1; x2,y2; x3,y3")
            polygon_text = st.text_input(
                "Polygon points",
                value=st.session_state.get("polygon_points_text", ""),
                key="polygon_points_text",
                placeholder=f"{nx1},{ny1}; {nx2},{ny1}; {nx2},{ny2}; {nx1},{ny2}",
            )
            previous = st.session_state.processed
            if previous is not None and previous.get("polygon_text") != polygon_text:
                st.session_state.processed = None
            st.caption("Use this to exclude buildings, trees, or other non-sky material. A polygon is an approximation and must be checked visually.")
    else:
        st.error("Invalid region: right must be greater than left, and bottom must be greater than top.")

    if st.button("Continue to processing →", type="primary", disabled=not valid):
        st.session_state.processed = None
        st.session_state.active_page = "Processing"
        st.rerun()
