import streamlit as st
from core.pipeline import process_image

PARAM_DEFAULTS = {
    "h_min":0,"h_max":180,"s_min":0,"s_max":85,"v_min":150,"v_max":255,
    "lab_l_min":170,"lab_chroma_max":28,"blur":1,"method":"Adaptive fusion",
    "combined_mode":"AND","morph":"None","kernel":3,"erosion":0,"dilation":0,
    "argd_k":0.9,"argd_sunlight_k":0.6,"argd_threshold":-0.15,
    "nsv_threshold":-0.3,"hyta_fixed_threshold":0.0,"hyta_std_threshold":0.12,
    "superpixel_count":400,"superpixel_compactness":10.0,
    "local_threshold_scale":0.25,"local_feature_ceiling":-0.65,
    "hybrid_w_argd":0.35,"hybrid_w_nsv":0.25,"hybrid_w_ratio":0.2,"hybrid_w_local":0.2,
    "hybrid_threshold":0.55,"ambiguity_width":0.12,"sunlight_penalty":0.35,
    "sunlight_mode":"Warm-bright + user region",
    "texture_weight":0.0,"texture_window":9,"texture_bins":16,
    "segmentation_method":"SLIC",
    "structure_w_argd":0.30,"structure_w_nsv":0.25,
    "structure_w_ratio":0.25,"structure_w_texture":0.20,
    "structure_threshold":0.55,"continuity_margin":0.12,"continuity_bonus":0.08,
    "adaptive_threshold_scale":0.25,
    "thin_contrast_threshold":0.20,"thin_score_threshold":0.48,
    "thin_color_floor":0.28,
    "dark_region_threshold":0.48,"dark_saturation_ceiling":0.48,
    "dark_texture_floor":0.16,"dark_continuity_bonus":0.10,
    "fusion_w_baseline":0.45,"fusion_w_thin":0.55,
    "fusion_w_dark":0.45,"fusion_threshold":0.52,
}


def render_processing():
    st.markdown('<div class="eyebrow">STEP 02 / PROCESSING WORKSPACE</div><h1>Configure segmentation</h1><p>Compare classical color, texture, superpixel, and regional methods on the same ROI, then choose the result used for analysis.</p>', unsafe_allow_html=True)
    if st.session_state.image_rgb is None:
        st.warning("Upload an image first.")
        if st.button("Go to Image Input"):
            st.session_state.active_page = "Image Input"; st.rerun()
        return
    left, right = st.columns([1.1, .9], gap="large")
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Source image</div><div class="small-muted">Current uploaded frame</div></div>', unsafe_allow_html=True)
        st.image(st.session_state.image_rgb, use_container_width=True)
    with right:
        mode = st.radio("Processing mode", ["Automatic baseline", "Manual experiment"], horizontal=True)
        p = {**PARAM_DEFAULTS, **st.session_state.params}
        if mode == "Manual experiment":
            method_names = [
                "HSV", "LAB", "Otsu", "ARGD", "NSV", "HYTA", "Superpixel",
                "Texture + SLIC", "Structure-aware", "Thin-cloud enhancement",
                "Dark-cloud enhancement", "Adaptive fusion", "Hybrid",
            ]
            p["method"] = st.selectbox(
                "Method used for final analysis",
                method_names,
                index=method_names.index(p.get("method", "Adaptive fusion"))
                if p.get("method", "Adaptive fusion") in method_names else 11,
            )
            st.markdown("**HSV baseline**")
            p["h_min"],p["h_max"] = st.slider("Hue",0,180,(int(p["h_min"]),int(p["h_max"])), help="OpenCV hue uses 0–180.")
            p["s_min"],p["s_max"] = st.slider("Saturation",0,255,(int(p["s_min"]),int(p["s_max"])), help="Low saturation favors neutral pixels.")
            p["v_min"],p["v_max"] = st.slider("Value",0,255,(int(p["v_min"]),int(p["v_max"])), help="High values favor bright pixels.")
            st.markdown("**Additional LAB baseline**")
            p["lab_l_min"] = st.slider("LAB minimum lightness",0,255,int(p["lab_l_min"]))
            p["lab_chroma_max"] = st.slider("LAB maximum chroma",0,150,int(p["lab_chroma_max"]))
            st.markdown("**ARGD sunlight-aware feature**")
            p["argd_k"] = st.slider("Base red weight k",0.0,2.0,float(p["argd_k"]),0.01)
            p["argd_sunlight_k"] = st.slider("User-marked sunlight-region k",0.0,2.0,float(p["argd_sunlight_k"]),0.01)
            p["argd_threshold"] = st.slider("ARGD cloud threshold", -1.0, 1.0, float(p["argd_threshold"]), 0.01)
            p["nsv_threshold"] = st.slider("NSV cloud threshold", -1.0, 1.0, float(p["nsv_threshold"]), 0.01)
            p["hyta_fixed_threshold"] = st.slider("HYTA fixed threshold", -1.0, 1.0, float(p["hyta_fixed_threshold"]), 0.01)
            p["hyta_std_threshold"] = st.slider("HYTA unimodal/bimodal std boundary",0.0,0.5,float(p["hyta_std_threshold"]),0.005)
            st.markdown("**Superpixel region method**")
            p["segmentation_method"] = st.selectbox(
                "Segmentation method",
                ["SLIC", "Watershed"],
                index=["SLIC", "Watershed"].index(p.get("segmentation_method", "SLIC")),
                help="SLIC respects color boundaries and is preferred. Watershed is an explicit fallback option.",
            )
            p["superpixel_count"] = st.slider("Requested superpixels",50,2000,int(p["superpixel_count"]),50)
            p["superpixel_compactness"] = st.slider("SLIC compactness",1.0,40.0,float(p["superpixel_compactness"]),0.5, help="Higher values favor more compact, spatially regular regions.")
            p["local_threshold_scale"] = st.slider("Local median-deviation scale",0.0,2.0,float(p["local_threshold_scale"]),0.05)
            p["local_feature_ceiling"] = st.slider("Local NSV candidate ceiling",-1.0,0.0,float(p["local_feature_ceiling"]),0.01)
            with st.expander("Structure-aware weights, threshold, and texture"):
                p["structure_w_argd"] = st.slider("ARGD evidence weight",0.0,1.0,float(p["structure_w_argd"]),0.01)
                p["structure_w_nsv"] = st.slider("NSV evidence weight",0.0,1.0,float(p["structure_w_nsv"]),0.01)
                p["structure_w_ratio"] = st.slider("Normalized B/R evidence weight",0.0,1.0,float(p["structure_w_ratio"]),0.01)
                p["structure_w_texture"] = st.slider("Texture evidence weight",0.0,1.0,float(p["structure_w_texture"]),0.01)
                p["structure_threshold"] = st.slider("Classification threshold",0.0,1.0,float(p["structure_threshold"]),0.01, help="Higher values select fewer superpixel regions.")
                p["continuity_margin"] = st.slider("Neighbor continuity margin",0.0,0.4,float(p["continuity_margin"]),0.01, help="Allows only near-threshold regions adjacent to selected cloud regions to receive a bounded score bonus.")
                p["continuity_bonus"] = st.slider("Neighbor continuity bonus",0.0,0.25,float(p["continuity_bonus"]),0.01)
                st.caption("Weights are normalized internally. Texture summarizes local standard deviation, entropy, and gradient; the resulting score is heuristic, not a probability.")
                p["hybrid_w_argd"] = st.slider("ARGD weight",0.0,1.0,float(p["hybrid_w_argd"]),0.01)
                p["hybrid_w_nsv"] = st.slider("NSV weight",0.0,1.0,float(p["hybrid_w_nsv"]),0.01)
                p["hybrid_w_ratio"] = st.slider("Normalized B/R weight",0.0,1.0,float(p["hybrid_w_ratio"]),0.01)
                p["hybrid_w_local"] = st.slider("Local-region weight",0.0,1.0,float(p["hybrid_w_local"]),0.01)
                p["hybrid_threshold"] = st.slider("Hybrid cloud-score threshold",0.0,1.0,float(p["hybrid_threshold"]),0.01)
                p["ambiguity_width"] = st.slider("Ambiguous-score band half-width",0.0,0.4,float(p["ambiguity_width"]),0.01)
                p["sunlight_penalty"] = st.slider("Warm-bright / marked-sun penalty",0.0,1.0,float(p["sunlight_penalty"]),0.01)
                p["texture_weight"] = st.slider("Texture support weight",0.0,1.0,float(p["texture_weight"]),0.01)
                p["texture_window"] = st.select_slider("Texture window",options=[3,5,7,9,11,15],value=int(p["texture_window"]))
                p["adaptive_threshold_scale"] = st.slider(
                    "Adaptive local threshold scale",0.0,1.5,float(p["adaptive_threshold_scale"]),0.05,
                    help="Higher values require stronger local contrast for thin-cloud support.",
                )
                p["sunlight_mode"] = st.selectbox(
                    "Sunlight-interference mode",
                    ["None", "Warm-bright cue", "User region", "Warm-bright + user region"],
                    index=["None", "Warm-bright cue", "User region", "Warm-bright + user region"].index(
                        p.get("sunlight_mode", "Warm-bright + user region")
                    ),
                    help="Warm-bright pixels are an image-derived cue; CFA does not estimate solar position.",
                )
            with st.expander("Thin, dark, and fusion settings"):
                p["thin_contrast_threshold"] = st.slider("Thin-cloud local contrast threshold",0.0,2.0,float(p["thin_contrast_threshold"]),0.05)
                p["thin_score_threshold"] = st.slider("Thin-cloud evidence threshold",0.0,1.0,float(p["thin_score_threshold"]),0.01)
                p["thin_color_floor"] = st.slider("Thin-cloud minimum color support",0.0,1.0,float(p["thin_color_floor"]),0.01)
                p["dark_region_threshold"] = st.slider("Dark-region evidence threshold",0.0,1.0,float(p["dark_region_threshold"]),0.01)
                p["dark_saturation_ceiling"] = st.slider("Dark-region saturation limit",0.0,1.0,float(p["dark_saturation_ceiling"]),0.01)
                p["dark_texture_floor"] = st.slider("Dark-region texture minimum",0.0,1.0,float(p["dark_texture_floor"]),0.01)
                p["dark_continuity_bonus"] = st.slider("Dark-region neighbor support",0.0,0.3,float(p["dark_continuity_bonus"]),0.01)
                p["fusion_w_baseline"] = st.slider("Baseline contribution",0.0,1.0,float(p["fusion_w_baseline"]),0.01)
                p["fusion_w_thin"] = st.slider("Thin-cloud contribution",0.0,1.0,float(p["fusion_w_thin"]),0.01)
                p["fusion_w_dark"] = st.slider("Dark-cloud contribution",0.0,1.0,float(p["fusion_w_dark"]),0.01)
                p["fusion_threshold"] = st.slider("Adaptive fusion threshold",0.0,1.0,float(p["fusion_threshold"]),0.01)
                st.caption("Enhancement candidates require regional support. Fusion preserves the baseline and score-gates additions; all scores are heuristic.")
            st.markdown("**Image refinement**")
            p["blur"] = st.select_slider("Gaussian blur kernel", options=[1,3,5,7,9], value=int(p["blur"]), help="Optional smoothing. 1 preserves the original pixels.")
            p["morph"] = st.selectbox("Morphological operation", ["None","Opening","Closing","Opening + Closing","Erosion","Dilation"], index=["None","Opening","Closing","Opening + Closing","Erosion","Dilation"].index(p["morph"]))
            p["kernel"] = st.select_slider("Morphology kernel", options=[1,3,5,7,9], value=int(p["kernel"]), help="Larger kernels affect larger regions and should be used cautiously.")
            p["erosion"] = st.number_input("Erosion iterations", 0, 5, int(p["erosion"]), help="Shrinks selected regions and removes thin noise.")
            p["dilation"] = st.number_input("Dilation iterations", 0, 5, int(p["dilation"]), help="Expands selected regions and closes small gaps.")
        else:
            p = {**p, "method":"Adaptive fusion","h_min":0,"h_max":180,"s_min":0,"s_max":85,"v_min":150,"v_max":255,"blur":1,"morph":"None","kernel":3,"erosion":0,"dilation":0}
            st.info("Automatic processing compares the baseline with thin- and dark-cloud enhancements, then selects adaptive fusion. It remains an experiment, not a universal classifier.")
        roi = st.session_state.roi or (0, 0, st.session_state.image_rgb.shape[1], st.session_state.image_rgb.shape[0])
        roi_width, roi_height = roi[2] - roi[0], roi[3] - roi[1]
        st.markdown("**Optional user-marked sunlight-interference region**")
        st.caption("This rectangle is in ROI-local coordinates. CFA does not infer actual solar position.")
        sun_x = st.slider("Sunlight region X boundaries",0,roi_width,(0,0),key="sun_region_x")
        sun_y = st.slider("Sunlight region Y boundaries",0,roi_height,(0,0),key="sun_region_y")
        sunlight_region = (sun_x[0],sun_y[0],sun_x[1],sun_y[1]) if sun_x[1] > sun_x[0] and sun_y[1] > sun_y[0] else None
        st.session_state.params = p
        previous_result = st.session_state.processed
        active_sunlight_region = (
            sunlight_region
            if p.get("sunlight_mode") in ("User region", "Warm-bright + user region")
            else None
        )
        if previous_result is not None and (
            previous_result.get("params") != p
            or previous_result.get("sunlight_region") != active_sunlight_region
            or previous_result.get("polygon_text") != st.session_state.get("polygon_points_text", "")
        ):
            st.session_state.processed = None
        if st.button("Run processing", type="primary", use_container_width=True):
            try:
                polygon_points = None
                polygon_text = st.session_state.get("polygon_points_text", "")
                if polygon_text.strip():
                    try:
                        polygon_points = [tuple(map(int, pair.split(","))) for pair in polygon_text.split(";") if pair.strip()]
                    except Exception as exc:
                        raise ValueError("Polygon format must be x1,y1; x2,y2; ...") from exc
                st.session_state.processed = process_image(
                    st.session_state.image_rgb,
                    p,
                    sky_roi=st.session_state.roi,
                    polygon_points=polygon_points,
                    sunlight_region=sunlight_region,
                )
                st.session_state.processed["polygon_text"] = polygon_text
                st.success("Processing completed.")
            except Exception as e:
                st.error(f"Processing failed: {e}")
        if st.button("Reset parameters", use_container_width=True):
            st.session_state.params = PARAM_DEFAULTS.copy()
            st.session_state.processed = None
            st.rerun()
    if st.session_state.processed:
        st.divider()
        st.markdown("### Quick result")
        st.image(st.session_state.processed["overlay"], caption="Candidate cloud regions highlighted", use_container_width=True)
        if st.button("Inspect segmentation →"):
            st.session_state.active_page = "Segmentation"; st.rerun()
