import streamlit as st

st.markdown(
    """
    <style>
    /* Remove uploader container */
    div[data-testid="stFileUploader"] {
        background: transparent;
        border: none;
        padding: 0;
    }

    div[data-testid="stFileUploader"] section {
        padding: 0;
        border: none;
        background: transparent;
    }

    /* Hide help text (200MB per file) */
    div[data-testid="stFileUploader"] small {
        display: none;
    }

    /* Remove drag/drop text */
    div[data-testid="stFileUploader"] section > div > div {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Put uploader on the right
left, right = st.columns([20, 0.2])

with st.container(border=True, height=500):
    with right:
        uploaded_file = st.file_uploader(
            "",
            type="csv",
            label_visibility="collapsed"
        )

        product_data = {
        "Content": ["Smartphone", "Smartwatch", "Smart Home Bundle"],
        "Topics": [":blue[Electronics]", ":green[IoT]", ":violet[Bundle]"],
        "Status": ["🟢 COMPLETE", "🟡 IN PROGRESS", "🔴 NOT STARTED"],
        "Links": [1247, 892, 654],
    }
    st.table(product_data, border="horizontal")