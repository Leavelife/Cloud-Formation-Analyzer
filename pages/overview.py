import streamlit as st

def render_overview():
    st.markdown('<div class="hero"><div class="eyebrow">DIGITAL IMAGE PROCESSING · ANALYSIS WORKSPACE</div><h1>Understand the sky.<br>Measure the clouds.</h1><p>Turn a sky photograph into measurable visual information. Explore classical color, texture, superpixel, and regional cloud evidence through one transparent pipeline.</p></div>', unsafe_allow_html=True)
    a,b = st.columns([1.55,1])
    with a:
        st.markdown('<div class="panel"><div class="eyebrow">THE OBJECTIVE</div><div class="panel-title">From pixels to cloud coverage</div><p>Cloud Formation Analyzer combines baseline color methods with SLIC regional analysis, thin/dark-cloud evidence, adaptive fusion, and uncertainty visualization.</p><p class="small-muted">Coverage is measured only across valid sky pixels in the selected ROI. This is an experimental image estimate, not a weather forecast or meteorological measurement.</p></div>', unsafe_allow_html=True)
        if st.button("Start an analysis  →", type="primary"):
            st.session_state.active_page = "Image Input"
            st.rerun()
    with b:
        st.markdown('<div class="panel"><div class="eyebrow">PROCESS OVERVIEW</div><div class="panel-title">A simple, inspectable workflow</div><ol><li>Upload a sky photograph</li><li>Define the valid sky ROI</li><li>Configure and run segmentation</li><li>Compare masks and feature evidence</li><li>Analyze coverage or evaluate ground truth</li></ol></div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("### Analysis workflow")
    cols = st.columns(4)
    steps = [("01","Acquire","Load a JPG, JPEG, or PNG image."),("02","Segment","Compare baseline, regional, thin, and dark methods."),("03","Refine","Inspect morphology and ambiguous evidence."),("04","Measure","Calculate coverage or compare against ground truth.")]
    for col,(n,title,desc) in zip(cols,steps):
        with col:
            st.markdown(f'<div class="metric workflow-card"><div class="eyebrow">{n}</div><div class="panel-title">{title}</div><div class="small-muted">{desc}</div></div>', unsafe_allow_html=True)
