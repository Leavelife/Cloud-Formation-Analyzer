import csv
import hashlib
import io

import numpy as np
import streamlit as st
from PIL import Image

from core.analysis import (
    analyze_cloud_characteristics,
    compare_method_coverage,
    coverage_summary,
    rank_methods_by_metric,
    recommend_method_without_ground_truth,
)
from core.evaluation import (
    evaluate_methods,
    make_comparison_map,
    prepare_ground_truth,
)
from core.pipeline import make_overlay


CONDITIONS = [
    "Clear blue sky",
    "White clouds / midday",
    "Warm-colored clear sky",
    "Golden-hour sky with clouds",
    "Sunset clouds",
    "Hazy sky",
    "Thin cirrus",
    "Clouds near strong sunlight",
]

METRICS = {
    "F1-Score": "f1",
    "IoU": "iou",
    "Precision": "precision",
    "Recall": "recall",
    "Pixel Accuracy": "accuracy",
}


def _csv_bytes(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _method_rows_for_display(rows: list[dict]) -> list[dict]:
    return [
        {
            "Method": row["method"],
            "Cloud pixels": row["cloud_pixels"],
            "Sky pixels": row["sky_pixels"],
            "Valid ROI pixels": row["valid_roi_pixels"],
            "Cloud coverage (%)": round(row["cloud_coverage"], 2),
            "Sky coverage (%)": round(row["sky_coverage"], 2),
        }
        for row in rows
    ]


def _show_cloud_characteristics(summary: dict) -> None:
    st.markdown("## Cloud Characteristics")
    if not summary["brightness_available"]:
        st.info("Brightness feature data is unavailable for this processed result.")
    else:
        st.markdown("#### Brightness distribution")
        edges = summary["brightness"]["bin_edges"]
        counts = summary["brightness"]["histogram"]
        histogram_rows = [
            {
                "Brightness bin": f"{edges[index]:.2f}–{edges[index + 1]:.2f}",
                "Cloud pixels": counts["cloud"][index],
                "Sky pixels": counts["sky"][index],
            }
            for index in range(len(edges) - 1)
        ]
        st.bar_chart(histogram_rows, x="Brightness bin", y=["Cloud pixels", "Sky pixels"])

    color_rows = []
    for space, channels in summary["color"].items():
        for channel, values in channels.items():
            color_rows.append({
                "Space": space.upper(),
                "Channel": channel,
                "Cloud mean": values["cloud"]["mean"],
                "Cloud median": values["cloud"]["median"],
                "Sky mean": values["sky"]["mean"],
                "Sky median": values["sky"]["median"],
            })
    if color_rows:
        st.markdown("#### Color distribution")
        st.dataframe(color_rows, use_container_width=True, hide_index=True)
        rgb_channels = summary["color"].get("rgb", {})
        if rgb_channels:
            edges = next(iter(rgb_channels.values()))["histogram"]["bin_edges"]
            rgb_histogram = []
            for index in range(len(edges) - 1):
                row = {"Normalized channel value": f"{edges[index]:.2f}–{edges[index + 1]:.2f}"}
                for channel in ("R", "G", "B"):
                    if channel in rgb_channels:
                        row[f"{channel} cloud"] = rgb_channels[channel]["histogram"]["cloud"][index]
                        row[f"{channel} sky"] = rgb_channels[channel]["histogram"]["sky"][index]
                rgb_histogram.append(row)
            st.bar_chart(
                rgb_histogram,
                x="Normalized channel value",
                y=[f"{channel} {region}" for channel in ("R", "G", "B") for region in ("cloud", "sky")],
            )

    texture_rows = []
    for feature_name, values in summary["texture"].items():
        texture_rows.append({
            "Feature": feature_name.replace("_", " ").title(),
            "Cloud mean": values["cloud"]["mean"],
            "Cloud p90": values["cloud"]["p90"],
            "Sky mean": values["sky"]["mean"],
            "Sky p90": values["sky"]["p90"],
        })
    if texture_rows:
        st.markdown("#### Local texture and edge distribution")
        st.dataframe(texture_rows, use_container_width=True, hide_index=True)
        st.caption(
            f"Cloud edge pixels above the valid-ROI 75th-percentile gradient "
            f"({summary['gradient']['edge_threshold_p75']!s}): "
            f"{summary['gradient']['cloud_edge_pixels']:,}."
        )

    st.markdown("#### Spatial distribution")
    st.caption("Each cell is cloud coverage within that part of the valid ROI.")
    st.dataframe(
        np.round(summary["spatial_distribution"], 1),
        use_container_width=True,
        hide_index=True,
    )

    region_columns = st.columns(4)
    region_columns[0].metric("Connected cloud regions", summary["regions"]["region_count"])
    region_columns[1].metric(
        "Median region size",
        "—" if summary["regions"]["median_region_pixels"] is None
        else f'{summary["regions"]["median_region_pixels"]:.0f} px',
    )
    region_columns[2].metric("Thin candidate pixels", summary["thin_candidates"]["pixel_count"])
    region_columns[3].metric("Dark candidate pixels", summary["dark_candidates"]["pixel_count"])
    st.caption(
        f"Thin candidates: {summary['thin_candidates']['region_count']} connected regions; "
        f"dark candidates: {summary['dark_candidates']['region_count']} connected regions. "
        "These are feature-based observations, not cloud-type classifications."
    )


def _show_ambiguity(methods: dict, valid_mask: np.ndarray, ambiguity_width: float) -> None:
    st.markdown("## Ambiguous Region Analysis")
    method = methods.get("Adaptive fusion")
    if not isinstance(method, dict):
        st.info("Adaptive-fusion evidence is unavailable for this result.")
        return
    score = method.get("score")
    threshold = method.get("threshold")
    if not isinstance(score, np.ndarray) or not isinstance(threshold, (int, float, np.number)):
        st.info("This method does not provide a numeric heuristic evidence score.")
        return
    if score.shape != valid_mask.shape:
        st.info("The evidence map does not match the valid ROI dimensions.")
        return

    valid = np.asarray(valid_mask, dtype=bool)
    high_cloud = (score >= float(threshold) + ambiguity_width) & valid
    high_sky = (score <= float(threshold) - ambiguity_width) & valid
    ambiguous = np.asarray(method.get("ambiguous", np.zeros(score.shape))) > 0
    ambiguous &= valid
    columns = st.columns(3)
    for column, mask, label in zip(
        columns,
        (high_cloud, high_sky, ambiguous),
        ("High cloud evidence", "High sky evidence", "Ambiguous regions"),
    ):
        with column:
            st.image(
                mask.astype(np.uint8) * 255,
                caption=f"{label} · {np.count_nonzero(mask):,} px",
                clamp=True,
                use_container_width=True,
            )
    st.caption(
        f"Evidence uses the adaptive-fusion score and threshold {float(threshold):.3f} "
        f"with a ±{ambiguity_width:.3f} ambiguity band. Scores are heuristic, not calibrated probabilities."
    )


def _show_evaluation(
    result: dict,
    methods: dict,
    valid_mask: np.ndarray,
    selected_name: str,
) -> None:
    st.markdown("## Evaluation Metrics")
    st.caption(
        "Evaluation requires a binary annotation for this same photograph and the exact displayed ROI. "
        "Dimensions are checked; CFA cannot independently prove image correspondence or annotation alignment."
    )
    gt_file = st.file_uploader(
        "Upload matching ground-truth mask",
        type=["png", "jpg", "jpeg"],
        key="ground_truth_upload",
    )
    if gt_file is None:
        st.warning("Upload a ground-truth mask to enable Evaluation Mode metrics.")
        return

    x1, y1, x2, y2 = result["roi"]
    st.caption(
        f"Required mask size: {x2 - x1} × {y2 - y1} pixels. "
        "White (255) = cloud; black (0) = non-cloud. The mask is not resized."
    )
    file_digest = hashlib.sha256(gt_file.getvalue()).hexdigest()
    alignment_confirmed = st.checkbox(
        "I confirm this mask represents the same uploaded image and is aligned pixel-for-pixel to the selected ROI.",
        key=f"ground_truth_alignment_confirmed_{file_digest}",
    )
    if not alignment_confirmed:
        st.info("Confirm image correspondence and ROI alignment before scoring.")
        return

    try:
        ground_truth_image = np.array(Image.open(gt_file).convert("RGB"))
        ground_truth = prepare_ground_truth(ground_truth_image, result["mask"].shape)
        compatible_methods = {
            name: method
            for name, method in methods.items()
            if isinstance(method, dict)
            and isinstance(method.get("refined", method.get("mask")), np.ndarray)
            and method.get("refined", method.get("mask")).shape == valid_mask.shape
        }
        metrics_by_method = evaluate_methods(compatible_methods, ground_truth, valid_mask)
    except (ValueError, OSError) as exc:
        st.error(f"Ground-truth validation failed: {exc}")
        return

    metric_rows = [
        {
            "Method": name,
            "IoU (%)": round(metrics["iou"], 3),
            "Precision (%)": round(metrics["precision"], 3),
            "Recall (%)": round(metrics["recall"], 3),
            "F1-Score (%)": round(metrics["f1"], 3),
            "Pixel Accuracy (%)": round(metrics["accuracy"], 3),
            "TP": metrics["tp"],
            "FP": metrics["fp"],
            "FN": metrics["fn"],
            "TN": metrics["tn"],
        }
        for name, metrics in metrics_by_method.items()
    ]
    st.markdown("### Method Performance Comparison")
    st.dataframe(metric_rows, use_container_width=True, hide_index=True)
    st.bar_chart(metric_rows, x="Method", y="F1-Score (%)")
    st.bar_chart(metric_rows, x="Method", y="IoU (%)")
    st.bar_chart(metric_rows, x="Method", y=["Precision (%)", "Recall (%)"])

    ranking_label = st.selectbox("Rank methods by", list(METRICS), index=0)
    ranking = rank_methods_by_metric(metrics_by_method, METRICS[ranking_label])
    st.markdown("### Best Method for This Image")
    if ranking["score"] is None:
        st.info("No comparable metric values are available.")
    elif len(ranking["methods"]) == 1:
        best_name = ranking["methods"][0]
        st.success(
            f"{best_name} has the highest {ranking_label} for this image: "
            f"{ranking['score']:.3f}%."
        )
        st.caption(
            "Selection uses the requested metric against this one aligned ground-truth mask. "
            "It is not a claim of overall or dataset-level superiority."
        )
    else:
        st.info(
            f"Tie: {', '.join(ranking['methods'])} share the highest "
            f"{ranking_label} ({ranking['score']:.3f}%)."
        )

    selected_metrics = metrics_by_method.get(selected_name)
    if selected_metrics is None:
        st.info("The selected method has no compatible mask for a confusion matrix.")
        return
    st.markdown(f"### Confusion Matrix · {selected_name}")
    st.dataframe(
        [
            {
                "Ground truth": "Cloud",
                "Predicted cloud": selected_metrics["tp"],
                "Predicted non-cloud": selected_metrics["fn"],
            },
            {
                "Ground truth": "Non-cloud",
                "Predicted cloud": selected_metrics["fp"],
                "Predicted non-cloud": selected_metrics["tn"],
            },
        ],
        use_container_width=True,
        hide_index=True,
    )
    selected_method = methods[selected_name]
    comparison = make_comparison_map(
        selected_method.get("refined", selected_method.get("mask")),
        ground_truth,
        valid_mask=valid_mask,
    )
    map_columns = st.columns(3)
    with map_columns[0]:
        st.image(ground_truth * 255, caption="Ground truth", clamp=True, use_container_width=True)
    with map_columns[1]:
        st.image(
            selected_method.get("refined", selected_method.get("mask")),
            caption=f"{selected_name} prediction",
            clamp=True,
            use_container_width=True,
        )
    with map_columns[2]:
        st.image(comparison, caption="TP green · FP red · FN blue · excluded gray", use_container_width=True)

    condition = st.selectbox("Image condition", CONDITIONS, key="evaluation_condition")
    evaluation_rows = [
        {
            "image": st.session_state.file_name or "uploaded image",
            "condition": condition,
            "Method": name,
            **{
                "IoU (%)": round(metrics["iou"], 3),
                "Precision (%)": round(metrics["precision"], 3),
                "Recall (%)": round(metrics["recall"], 3),
                "F1-Score (%)": round(metrics["f1"], 3),
                "Pixel Accuracy (%)": round(metrics["accuracy"], 3),
                "TP": metrics["tp"],
                "FP": metrics["fp"],
                "FN": metrics["fn"],
                "TN": metrics["tn"],
            },
        }
        for name, metrics in metrics_by_method.items()
    ]
    st.download_button(
        "Download per-image method comparison (CSV)",
        data=_csv_bytes(evaluation_rows),
        file_name="method_comparison.csv",
        mime="text/csv",
    )
    comparison_buffer = io.BytesIO()
    Image.fromarray(comparison).save(comparison_buffer, format="PNG")
    st.download_button(
        "Download selected method comparison map (PNG)",
        data=comparison_buffer.getvalue(),
        file_name="evaluation_comparison.png",
        mime="image/png",
    )

    if "evaluation_log" not in st.session_state:
        st.session_state.evaluation_log = []
    if st.button("Add this image to evaluation log", type="primary"):
        for row in evaluation_rows:
            st.session_state.evaluation_log = [
                existing for existing in st.session_state.evaluation_log
                if not (
                    existing["image"] == row["image"]
                    and existing["condition"] == condition
                    and existing["Method"] == row["Method"]
                )
            ]
            st.session_state.evaluation_log.append(row)
        st.success("Per-image metrics recorded in this session.")
    if st.session_state.evaluation_log:
        st.markdown("### Evaluation Log")
        st.dataframe(st.session_state.evaluation_log, use_container_width=True, hide_index=True)
        st.download_button(
            "Download evaluation log (CSV)",
            data=_csv_bytes(st.session_state.evaluation_log),
            file_name="cloud_dataset_evaluation.csv",
            mime="text/csv",
        )

    st.caption(
        "No thin-cloud or dark-cloud region metrics are shown because no separate region annotations were supplied. "
        "Inspect those masks as visual observations instead."
    )


def render_analysis():
    st.markdown(
        '<div class="eyebrow">STEP 04 / ANALYSIS & EVALUATION</div>'
        "<h1>Cloud analysis dashboard</h1>"
        "<p>Review coverage and visual evidence immediately; compare measured performance when an aligned annotation is available.</p>",
        unsafe_allow_html=True,
    )
    result = st.session_state.processed
    if result is None:
        st.warning("No processed result available.")
        if st.button("Go to Processing"):
            st.session_state.active_page = "Processing"
            st.rerun()
        return

    valid_mask = np.asarray(result.get("valid_mask"))
    if valid_mask.ndim != 2:
        st.error("The processed result does not contain a valid ROI mask. Run processing again.")
        return
    methods = result.get("methods", {})
    method_rows = compare_method_coverage(methods, valid_mask)
    available_methods = [row["method"] for row in method_rows]
    if not available_methods:
        st.error("No segmentation methods provide masks aligned with the valid ROI.")
        return

    mode = st.radio(
        "Dashboard mode",
        ("Analysis Mode", "Evaluation Mode"),
        horizontal=True,
        key="analysis_dashboard_mode",
    )
    previous_selection = st.session_state.get("analysis_selected_method")
    default_index = (
        available_methods.index(previous_selection)
        if previous_selection in available_methods
        else available_methods.index(result.get("selected_method"))
        if result.get("selected_method") in available_methods
        else 0
    )
    selected_name = st.selectbox(
        "Segmentation method for selected-result analysis",
        available_methods,
        index=default_index,
        key="analysis_selected_method",
    )
    selected_method = methods[selected_name]
    selected_mask = selected_method.get("refined", selected_method.get("mask"))
    if not isinstance(selected_mask, np.ndarray) or selected_mask.shape != valid_mask.shape:
        st.error("The selected method mask does not match the valid ROI dimensions.")
        return

    try:
        selected_coverage = coverage_summary(selected_mask, valid_mask)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.markdown("## Analysis Overview")
    coverage_columns = st.columns(5)
    coverage_columns[0].metric("Cloud Coverage", f'{selected_coverage["cloud_coverage"]:.2f}%')
    coverage_columns[1].metric("Sky Coverage", f'{selected_coverage["sky_coverage"]:.2f}%')
    coverage_columns[2].metric("Cloud Pixels", f'{selected_coverage["cloud_pixels"]:,}')
    coverage_columns[3].metric("Sky Pixels", f'{selected_coverage["sky_pixels"]:,}')
    coverage_columns[4].metric("Valid ROI Pixels", f'{selected_coverage["valid_roi_pixels"]:,}')
    st.caption(
        f"Cloud coverage + sky coverage = "
        f"{selected_coverage['cloud_coverage'] + selected_coverage['sky_coverage']:.2f}% "
        "of valid ROI pixels. Pixels outside the rectangular/polygon ROI are excluded."
    )

    st.markdown("## Selected Segmentation Result")
    st.caption(f"Method: **{selected_name}**")
    source = np.asarray(result.get("source"))
    if source.ndim != 3 or source.shape[:2] != valid_mask.shape:
        st.error("The original ROI image is missing or does not match the segmentation dimensions.")
        return
    overlay = selected_method.get("overlay")
    if not isinstance(overlay, np.ndarray) or overlay.shape != source.shape:
        overlay = make_overlay(source, selected_mask)
    selected_columns = st.columns(3)
    with selected_columns[0]:
        st.image(source, caption="Original image · selected ROI", use_container_width=True)
    with selected_columns[1]:
        st.image(selected_mask, caption="Binary cloud mask", clamp=True, use_container_width=True)
    with selected_columns[2]:
        st.image(overlay, caption=f"Overlay · {selected_name}", use_container_width=True)
    mask_buffer = io.BytesIO()
    Image.fromarray((selected_mask > 0).astype(np.uint8) * 255).save(mask_buffer, format="PNG")
    st.download_button(
        "Download selected cloud mask (PNG)",
        data=mask_buffer.getvalue(),
        file_name="predicted_cloud_mask.png",
        mime="image/png",
    )

    st.markdown("## Method Comparison")
    st.caption(
        "Coverage differences describe output area only; they do not establish which method is more accurate."
    )
    st.dataframe(_method_rows_for_display(method_rows), use_container_width=True, hide_index=True)
    method_tabs = st.tabs(available_methods)
    for tab, name in zip(method_tabs, available_methods):
        method = methods[name]
        method_mask = method.get("refined", method.get("mask"))
        method_overlay = method.get("overlay")
        with tab:
            image_columns = st.columns(2)
            with image_columns[0]:
                st.image(method_mask, caption=f"{name} · binary mask", clamp=True, use_container_width=True)
            with image_columns[1]:
                if isinstance(method_overlay, np.ndarray) and method_overlay.shape == source.shape:
                    st.image(method_overlay, caption=f"{name} · overlay", use_container_width=True)
                else:
                    st.info("Overlay is unavailable for this method.")
            row = next(item for item in method_rows if item["method"] == name)
            st.caption(
                f'{row["cloud_coverage"]:.2f}% cloud · '
                f'{row["cloud_pixels"]:,} cloud pixels of {row["valid_roi_pixels"]:,} valid pixels'
            )

    st.markdown("## Cloud Coverage Analysis")
    coverage_chart_rows = [
        {"Method": row["method"], "Cloud coverage (%)": row["cloud_coverage"]}
        for row in method_rows
    ]
    st.bar_chart(coverage_chart_rows, x="Method", y="Cloud coverage (%)")

    features = result.get("features", {})
    try:
        characteristics = analyze_cloud_characteristics(
            selected_mask,
            valid_mask,
            features,
            thin_candidates=features.get("thin_candidates"),
            dark_candidates=features.get("dark_candidates"),
            ambiguous_mask=selected_method.get("ambiguous"),
        )
    except ValueError as exc:
        st.error(f"Could not summarize cloud characteristics: {exc}")
        characteristics = None
    if characteristics is not None:
        _show_cloud_characteristics(characteristics)

    ambiguity_width = float(result.get("params", {}).get("ambiguity_width", 0.12))
    _show_ambiguity(methods, valid_mask, ambiguity_width)

    if mode == "Evaluation Mode":
        _show_evaluation(result, methods, valid_mask, selected_name)
    else:
        st.markdown("## Method Recommendation and Limitations")
        recommendation = recommend_method_without_ground_truth(methods, valid_mask)
        st.info(recommendation["message"])
        st.caption(recommendation["reason"])
        st.dataframe(
            [
                {"Evidence check": name.replace("_", " ").title(), "Observed": str(value)}
                for name, value in recommendation["signals"].items()
            ],
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Thin- and dark-cloud candidate masks are feature-based observations, not cloud-type labels. "
        "Internal scores are heuristic, not calibrated probabilities. Per-image evaluation does not establish "
        "the best overall method across a separate dataset."
    )
