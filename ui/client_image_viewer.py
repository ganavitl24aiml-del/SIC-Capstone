
import textwrap
import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import base64

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="SecureFL | Client Network",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLIENTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "chest_xray"
    / "client_visualization"
)

DISEASES = [
    "Pneumonia",
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Nodule",
    "Infiltration"
]

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 10%,
            rgba(30, 136, 229, 0.12),
            transparent 28%
        ),
        radial-gradient(
            circle at 10% 85%,
            rgba(0, 188, 212, 0.08),
            transparent 25%
        ),
        #07111f;
    color: #e8eef7;
}

/* Remove Streamlit top padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1450px;
}

/* Hide default menu/footer */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

/* Header */
.hero {
    padding: 28px 32px;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 24px;
    background:
        linear-gradient(
            135deg,
            rgba(20, 42, 67, 0.90),
            rgba(8, 20, 35, 0.90)
        );
    box-shadow: 0 20px 60px rgba(0,0,0,0.28);
    margin-bottom: 28px;
}

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 30px;
    font-weight: 700;
    letter-spacing: -1px;
}

.brand span {
    color: #38bdf8;
}

.subtitle {
    color: #8fa3b8;
    margin-top: 6px;
    font-size: 14px;
}

.status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    margin-top: 18px;
    padding: 7px 13px;
    border-radius: 30px;
    background: rgba(34,197,94,0.10);
    border: 1px solid rgba(34,197,94,0.20);
    color: #86efac;
    font-size: 12px;
    font-weight: 600;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px #22c55e;
}

/* Section */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 600;
    margin-top: 30px;
    margin-bottom: 6px;
}

.section-description {
    color: #8094aa;
    font-size: 13px;
    margin-bottom: 20px;
}

/* Client card */
.client-card {
    position: relative;
    min-height: 180px;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.07);
    background:
        linear-gradient(
            145deg,
            rgba(21,40,62,0.90),
            rgba(9,23,39,0.90)
        );
    transition: all 0.25s ease;
}

.client-card:hover {
    transform: translateY(-4px);
    border-color: rgba(56,189,248,0.35);
    box-shadow: 0 18px 45px rgba(0,0,0,0.30);
}

.client-number {
    color: #38bdf8;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.client-name {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 600;
    margin-top: 5px;
}

.client-images {
    font-size: 28px;
    font-weight: 700;
    margin-top: 20px;
}

.client-label {
    color: #758aa0;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Metrics */
.metric {
    padding: 20px;
    border-radius: 17px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
}

.metric-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 25px;
    font-weight: 700;
}

.metric-label {
    color: #8195aa;
    font-size: 11px;
    margin-top: 5px;
}

/* Disease cards */
.disease-card {
    padding: 18px;
    border-radius: 16px;
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 10px;
}

.disease-name {
    font-weight: 600;
    font-size: 13px;
}

.disease-count {
    color: #38bdf8;
    font-weight: 700;
}

/* Image gallery */
.image-card {
    border-radius: 18px;
    overflow: hidden;
    background: #0d1b2a;
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 16px;
}

.image-caption {
    padding: 12px 14px;
}

.image-id {
    font-size: 11px;
    color: #7f94a9;
    overflow: hidden;
    text-overflow: ellipsis;
}

.image-labels {
    font-size: 11px;
    color: #38bdf8;
    margin-top: 5px;
}

/* Divider */
.glow-line {
    height: 1px;
    background: linear-gradient(
        90deg,
        transparent,
        rgba(56,189,248,0.4),
        transparent
    );
    margin: 30px 0;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(56,189,248,0.20);
    background: rgba(56,189,248,0.08);
    color: #c9eaff;
    font-weight: 600;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(56,189,248,0.55);
    background: rgba(56,189,248,0.15);
}

/* Selectboxes */
div[data-baseweb="select"] > div {
    background: #0d1b2a;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #07111f;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def load_client(client_number):

    client_path = CLIENTS_DIR / f"client_{client_number}"

    csv_candidates = [
        client_path / "labels.csv",
        client_path / "train.csv",
        client_path / "client_data.csv"
    ]

    csv_path = None

    for path in csv_candidates:
        if path.exists():
            csv_path = path
            break

    if csv_path is None:
        return None, None

    df = pd.read_csv(csv_path)

    image_dirs = [
        client_path / "images",
        client_path
    ]

    image_dir = None

    for directory in image_dirs:
        if directory.exists():
            image_dir = directory
            break

    return df, image_dir


def get_labels(row):

    labels = []

    for disease in DISEASES:
        if disease in row and row[disease] == 1:
            labels.append(disease)

    return labels


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
<div class="hero">

    <div class="brand">
        Secure<span>FL</span>
    </div>

    <div class="subtitle">
        Privacy-Preserving Multi-Disease Clinical Decision Support
    </div>

    <div class="status">
        <div class="status-dot"></div>
        FEDERATED CLIENT NETWORK ONLINE
    </div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# LOAD ALL CLIENT DATA
# ============================================================

client_data = {}

for i in range(1, 6):

    df, image_dir = load_client(i)

    if df is not None:
        client_data[i] = {
            "df": df,
            "image_dir": image_dir
        }


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_images = sum(
    len(data["df"])
    for data in client_data.values()
)

st.markdown(
    '<div class="section-title">Federated Client Network</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Explore the distributed medical imaging data across the five simulated hospital clients.'
    '</div>',
    unsafe_allow_html=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown("""
    <div class="metric">
        <div class="metric-value">05</div>
        <div class="metric-label">Simulated Hospitals</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="metric">
        <div class="metric-value">06</div>
        <div class="metric-label">Target Diseases</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric">
        <div class="metric-value">{total_images:,}</div>
        <div class="metric-label">Distributed X-rays</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown("""
    <div class="metric">
        <div class="metric-value">FedAvg</div>
        <div class="metric-label">Aggregation Method</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CLIENT CARDS
# ============================================================

st.markdown(
    '<div class="glow-line"></div>',
    unsafe_allow_html=True
)

cols = st.columns(5)

for i, col in enumerate(cols, start=1):

    if i not in client_data:
        continue

    df = client_data[i]["df"]

    with col:

        st.markdown(
            f"""
<div class="client-card">

    <div class="client-number">
        CLIENT 0{i}
    </div>

    <div class="client-name">
        Hospital 0{i}
    </div>

    <div class="client-images">
        {len(df):,}
    </div>

    <div class="client-label">
        X-ray images
    </div>

</div>
""",
            unsafe_allow_html=True
        )

        st.write("")

        if st.button(
            f"Explore Client 0{i}",
            key=f"client_{i}",
            use_container_width=True
        ):
            st.session_state["selected_client"] = i


# ============================================================
# DEFAULT CLIENT
# ============================================================

if "selected_client" not in st.session_state:
    st.session_state["selected_client"] = 1

selected_client = st.session_state["selected_client"]

# Make sure the value is always an integer
if not isinstance(selected_client, int):
    try:
        selected_client = int(selected_client)
        st.session_state["selected_client"] = selected_client
    except (ValueError, TypeError):
        selected_client = 1
        st.session_state["selected_client"] = 1

df = client_data[selected_client]["df"]
image_dir = client_data[selected_client]["image_dir"]


# ============================================================
# CLIENT EXPLORER
# ============================================================

st.markdown(
    '<div class="glow-line"></div>',
    unsafe_allow_html=True
)

client_number = selected_client

st.markdown(
    f'<div class="section-title">Hospital 0{client_number} — Data Explorer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Inspect disease distribution and the actual X-ray images assigned to this client.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# DISEASE DISTRIBUTION
# ============================================================

disease_counts = {
    disease: int(df[disease].sum())
    for disease in DISEASES
    if disease in df.columns
}

disease_cols = st.columns(6)

for col, disease in zip(disease_cols, DISEASES):

    count = disease_counts.get(disease, 0)

    with col:

        st.markdown(f"""
        <div class="disease-card">

            <div class="disease-name">
                {disease}
            </div>

            <div class="disease-count">
                {count:,}
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FILTER
# ============================================================

st.markdown(
    '<div class="section-title">X-ray Gallery</div>',
    unsafe_allow_html=True
)

left, right = st.columns([2, 1])

with left:

    selected_disease = st.selectbox(
        "Disease",
        DISEASES,
        key="disease_selector"
    )

with right:

    number_to_show = st.selectbox(
        "Images",
        [8, 16, 24, 32],
        index=1
    )


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = df[
    df[selected_disease] == 1
].copy()

filtered_df = filtered_df.head(number_to_show)


st.markdown(
    f"""
    <div style="
        color:#8fa3b8;
        font-size:13px;
        margin:10px 0 20px 0;
    ">
        Showing <b style="color:#38bdf8">
        {len(filtered_df)}
        </b>
        images labelled
        <b style="color:#38bdf8">
        {selected_disease}
        </b>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# IMAGE GALLERY
# ============================================================

gallery_cols = st.columns(4)

for index, (_, row) in enumerate(filtered_df.iterrows()):

    image_name = row["Image Index"]

    image_path = image_dir / image_name

    if not image_path.exists():
        continue

    try:

        image = Image.open(image_path)

        labels = get_labels(row)

        with gallery_cols[index % 4]:

            st.markdown(
                '<div class="image-card">',
                unsafe_allow_html=True
            )

            st.image(
                image,
                use_container_width=True
            )

            st.markdown(
                f"""
                <div class="image-caption">

                    <div class="image-id">
                        {image_name}
                    </div>

                    <div class="image-labels">
                        {' • '.join(labels)}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )

    except Exception:
        pass


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div style="
    text-align:center;
    color:#536a80;
    font-size:11px;
    margin-top:50px;
">
    SecureFL • Federated Retrieval-Augmented Generation
    • Clinical Decision Support
</div>
""", unsafe_allow_html=True)