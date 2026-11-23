import cv2
import numpy as np
import streamlit as st


def _feature_preview(feature: np.ndarray) -> np.ndarray:
    finite = np.isfinite(feature)
    if not np.any(finite):
        return np.zeros((*feature.shape[:2], 3), dtype=np.uint8)
    values = np.asarray(feature, dtype=np.float32)
    low, high = float(np.min(values[finite])), float(np.max(values[finite]))
    if high - low < 1e-8:
        normalized = np.zeros(values.shape, dtype=np.uint8)
    else:
        normalized = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


def render_segmentation():
    st.markdown(
        '<div class="eyebrow">STEP 03 / METHOD COMPARISON</div><h1>Segmentation results</h1>'
        '<p>All methods use the same uploaded image, crop, and valid-sky mask. Compare the existing structure-aware baseline with thin-cloud, dark-cloud, and fused results.</p>',
        unsafe_allow_html=True,
    )
    result = st.session_state.processed
    required_methods = {
        "Structure-aware", "Thin-cloud enhancement",
        "Dark-cloud enhancement", "Adaptive fusion",
    }
    if result is None or not required_methods.issubset(result.get("methods", {})):
        st.warning("Run the new comparison pipeline from Processing first.")
        if st.button("Open Processing"):
            st.session_state.active_page = "Processing"
            st.rerun()
        return

    st.caption(
        f"Selected method: {result['selected_method']} · "
        f"ROI: {result['roi'][2]-result['roi'][0]} × {result['roi'][3]-result['roi'][1]} px · "
        "time excludes image upload and Streamlit rendering."
    )
    st.markdown("### Thin and dark cloud comparison")
    overview = st.columns(3)
    overview_items = (
        ("source", st.session_state.image_rgb, "Original uploaded image"),
        ("hsv", result["methods"]["Structure-aware"]["refined"], "Existing structure-aware baseline"),
        ("thin", result["methods"]["Thin-cloud enhancement"]["refined"], "Thin-cloud enhancement"),
        ("dark", result["methods"]["Dark-cloud enhancement"]["refined"], "Dark-cloud enhancement"),
        ("fusion", result["methods"]["Adaptive fusion"]["refined"], "Final adaptive fusion"),
        ("ambiguous", result["methods"]["Adaptive fusion"]["ambiguous"], "Ambiguous fusion regions"),
    )
    for index, (key, image, caption) in enumerate(overview_items):
        with overview[index % 3]:
            st.image(
                image,
                caption=caption,
                clamp=key == "ambiguous",
                use_container_width=True,
            )

    names = [
        "HSV", "LAB", "Otsu", "ARGD", "NSV", "HYTA", "Superpixel",
        "Texture + SLIC", "Structure-aware", "Thin-cloud enhancement",
        "Dark-cloud enhancement", "Adaptive fusion", "Hybrid",
    ]
    tabs = st.tabs(names)
    for tab, name in zip(tabs, names):
        method = result["methods"][name]
        with tab:
            left, right = st.columns(2)
            with left:
                st.image(_feature_preview(method["feature"]), caption=f"{name} feature map", use_container_width=True)
                st.image(method["mask"], caption="Binary candidate mask", clamp=True, use_container_width=True)
            with right:
                st.image(method["refined"], caption="Morphologically refined mask", clamp=True, use_container_width=True)
                st.image(method["overlay"], caption="Overlay", use_container_width=True)
            st.caption(
                f"Threshold: {method['threshold']} · "
                f"Processing: {method['processing_ms']:.2f} ms"
            )
            if method["diagnostics"]:
                display_diagnostics = {
                    key: value
                    for key, value in method["diagnostics"].items()
                    if not isinstance(value, np.ndarray)
                }
                st.json(display_diagnostics, expanded=False)
            if name == "HYTA":
                st.line_chart(
                    {"pixel count": method["diagnostics"]["histogram"]},
                    height=180,
                )
                st.caption("Normalized blue/red feature histogram")
            if name == "Hybrid":
                st.image(method["ambiguous"], caption="Ambiguous-score band (not asserted as cloud)", clamp=True, use_container_width=True)
            if name in (
                "Texture + SLIC", "Structure-aware",
                "Thin-cloud enhancement", "Dark-cloud enhancement", "Adaptive fusion",
            ):
                st.image(method["ambiguous"], caption="Ambiguous regions (heuristic score band)", clamp=True, use_container_width=True)

    st.markdown("### Supporting feature diagnostics")
    features = result["features"]
    feature_tabs = st.tabs(["RGB / HSV", "ARGD weights", "Illumination", "Texture", "Superpixel boundaries"])
    with feature_tabs[0]:
        rgb_cols = st.columns(3)
        for index, channel in enumerate(("R", "G", "B")):
            with rgb_cols[index]:
                st.image(_feature_preview(features["rgb"][:, :, index]), caption=f"Normalized RGB {channel}", use_container_width=True)
        hsv_cols = st.columns(3)
        for index, channel in enumerate(("H", "S", "V")):
            with hsv_cols[index]:
                st.image(_feature_preview(features["hsv"][:, :, index].astype(np.float32)), caption=f"HSV {channel}", use_container_width=True)
    with feature_tabs[1]:
        st.image(_feature_preview(features["argd_weights"]), caption="Per-pixel k weight", use_container_width=True)
        st.caption("The base red weight applies globally; the alternate weight applies only to the explicitly user-marked sunlight rectangle.")
    with feature_tabs[2]:
        cols = st.columns(3)
        for col, key, label in zip(
            cols,
            ("brightness", "warmth", "warm_bright_candidates"),
            ("Brightness", "Warmth cue (R−B)", "Warm-bright candidates"),
        ):
            with col:
                image = features["illumination"][key]
                st.image(_feature_preview(image.astype(np.float32)), caption=label, use_container_width=True)
        st.caption("Warm-bright candidates are an image-derived cue, not a detected sun position.")
    with feature_tabs[3]:
        cols = st.columns(3)
        for col, key, label in zip(
            cols,
            ("local_std", "local_entropy", "gradient"),
            ("Local standard deviation", "Local entropy", "Gradient magnitude"),
        ):
            with col:
                st.image(_feature_preview(features["texture"][key]), caption=label, use_container_width=True)
    with feature_tabs[4]:
        st.image(features["structure_boundaries"], caption="Structure-aware superpixel boundaries", clamp=True, use_container_width=True)
        diagnostics = result["methods"]["Structure-aware"]["diagnostics"]
        st.caption(
            f"{diagnostics.get('implementation', 'Unknown implementation')} · "
            f"{diagnostics.get('region_count', 0)} coherent regions"
        )
        st.image(features["superpixel_boundaries"], caption="Local NSV method boundaries", clamp=True, use_container_width=True)

    st.caption("White pixels are cloud candidates; black pixels are non-cloud. The warm-sky penalty and hybrid weighting are tunable experiment choices, not validated universal rules.")
    if st.button("View analysis →", type="primary"):
        st.session_state.active_page = "Analysis"
        st.rerun()
