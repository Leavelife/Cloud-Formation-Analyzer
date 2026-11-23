import streamlit as st

PAGES = ["Overview", "Image Input", "Processing", "Segmentation", "Analysis"]

def render_navbar():
    st.markdown(
        '<div class="topbar">'
        '<div class="brand"><span>CF</span> / CLOUD FORMATION ANALYZER</div>'
        '<div class="small-muted">CLASSICAL COMPUTER VISION · SKY IMAGING</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns([1,1,1,1,1,2])
    for i, name in enumerate(PAGES):
        with cols[i]:
            if st.button(name, key=f"nav_{name}", type="primary" if st.session_state.active_page == name else "secondary", use_container_width=True):
                st.session_state.active_page = name
                st.rerun()
    with cols[5]:
        st.markdown('<div style="text-align:right;padding-top:10px" class="small-muted">PCD PROJECT · MVP 01</div>', unsafe_allow_html=True)
