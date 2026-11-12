"""
Object Counting Challenge: leaderboard and submissions (Streamlit Community Cloud).

    streamlit run streamlit_app.py

The settings (HF_TOKEN, DATASET_REPO, RANK_BY, ...) come from the secrets of the app (Settings -> Secrets on
Streamlit Community Cloud, or .streamlit/secrets.toml locally) or from environment variables: see leaderboard.py.
"""
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Object Counting Challenge", page_icon="🔢", layout="wide")

# root-level secrets -> environment variables, read by leaderboard.py when it is imported
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(_key, str(_value))
except Exception:  # no secrets file: environment variables or defaults
    pass

import leaderboard as lb  # noqa: E402  (after the secrets)

APP_DIR = Path(__file__).parent
EXAMPLES_IMAGE = APP_DIR / "assets" / "examples.jpg"
DATASET_URL = os.environ.get(
    "DATASET_URL",
    "https://unipg-my.sharepoint.com/:u:/g/personal/alberto_dionigi_unipg_it/"
    "IQD_OsjIcN88RZ87OBJ9nRnRAZBsfOoM8-OntKEQzFSdB1E?e=6iLbnx")
COURSE = "Deep Learning and Robot Perception (Laboratory) · University of Perugia"


@st.cache_resource(show_spinner="Loading the leaderboard…")
def get_board():
    """One shared state for every visitor: ground truth, submissions and teams, loaded from the dataset"""
    return lb.Leaderboard()


board = get_board()

# --------------------------------------------------------------------------- style

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Work+Sans:wght@400;500;600;700&display=swap');
.block-container {max-width: 1220px; padding-top: 3.6rem;}
[data-testid="stMarkdownContainer"] h2 {font-size: 1.55rem; margin-top: 0.6rem;}
[data-testid="stMarkdownContainer"] h4 {font-size: 1.15rem;}
h1, h2, h3, h4, .hero, .chip, .badge {font-family: 'Work Sans', sans-serif !important;}
.hero {background: linear-gradient(135deg, #27348B 0%, #3D4DA6 60%, #4A5BB5 100%);
       border-radius: 18px; padding: 28px 34px 22px; margin-bottom: 18px;}
.hero, .hero * {color: #fff !important;}
.hero h1 {font-size: 2.3rem; font-weight: 700; margin: 2px 0 8px; padding: 0; letter-spacing: -0.5px;}
.hero-course {text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.78rem; opacity: 0.8;}
.hero-tagline {font-size: 1.08rem; opacity: 0.92; max-width: 820px; margin: 0 0 16px;}
.chips {display: flex; flex-wrap: wrap; gap: 8px;}
.chip {background: rgba(255,255,255,0.13); border: 1px solid rgba(255,255,255,0.28); border-radius: 999px;
       padding: 5px 13px; font-size: 0.88rem; white-space: nowrap;}
.status {margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px;}
.badge {border-radius: 8px; padding: 5px 12px; font-size: 0.88rem; font-weight: 600;
        background: rgba(255,255,255,0.15);}
.badge-open {background: #1E9E5A !important;}
.badge-closed {background: #C0392B !important;}
.examples img {width: 100%; border-radius: 14px; display: block; margin-bottom: 8px;}
.footer {text-align: center; opacity: 0.6; font-size: 0.85rem; margin-top: 28px;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- header


def status_html():
    if lb.DEADLINE is None:
        return '<span class="badge badge-open">● Submissions open</span>'
    if lb.now() > lb.DEADLINE:
        return '<span class="badge badge-closed">● Challenge closed</span>'
    left = lb.DEADLINE - lb.now()
    days, hours = left.days, left.seconds // 3600
    remaining = f"{days} day{'s' if days != 1 else ''} {hours} h left" if days else f"{hours} h left"
    return (f'<span class="badge badge-open">● Open until {lb.DEADLINE:%d %b %Y, %H:%M} (Rome)</span>'
            f'<span class="badge">{remaining}</span>')


st.markdown(f"""
<div class="hero">
  <div class="hero-course">{COURSE}</div>
  <h1>{lb.TITLE}</h1>
  <p class="hero-tagline">How many objects are in the image? Train a network on synthetic scenes rendered with Unreal
  Engine 5 and predict one number per test image.</p>
  <div class="chips">
    <span class="chip">🖼️ 12,000 images · 512×512</span>
    <span class="chip">🔷 4 classes: cube, cylinder, cone, sphere</span>
    <span class="chip">🔢 0 – 15 objects per image</span>
    <span class="chip">📏 Ranked by {lb.RANK_NAME} ({lb.RANK_DIRECTION})</span>
    <span class="chip">📤 {lb.MAX_PER_DAY} submissions per day</span>
  </div>
  <div class="status">{status_html()}</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- texts

scope = "the public part of the test set" if board.has_private else "the test set"

rules = [
    f"The leaderboard is ranked by **{lb.RANK_NAME}** ({lb.RANK_DIRECTION}) and shows the **best submission** of "
    "each team; on equal scores, the team that got there first is ahead.",
    f"Each team can submit at most **{lb.MAX_PER_DAY}** files per day; the counter resets at midnight "
    "(Italian time). Rejected files are not counted.",
    "**Accuracy** is the share of images whose prediction, rounded to the nearest integer, equals the true count.",
]
if board.has_private:
    rules.append(f"During the challenge the score is computed on about **{board.public_share * 100:.0f}%** of the "
                 "test images; the **final ranking** uses the other images, so do not overfit the public "
                 "leaderboard.")
if lb.DEADLINE:
    rules.append(f"Submissions close on **{lb.DEADLINE:%d %B %Y at %H:%M}** (Italian time).")

FORMAT_HELP = f"""
#### 📄 File format
A CSV with the header `{lb.ID_COL},{lb.PRED_COL}` and **one row per test image** ({board.size:,} rows, any
order). `{lb.ID_COL}` is the name of the image in `test/images/`; `{lb.PRED_COL}` is your prediction, integer or
decimal.

```
{lb.ID_COL},{lb.PRED_COL}
000000.png,8
000001.png,3.4
000002.png,0
```
Files saved by Excel with `;` and a decimal comma are accepted too.
"""

RANK_TIP = {
    "RMSE": "Since the ranking uses RMSE, **do not round** your predictions: rounding usually *increases* the RMSE.",
    "MAE": "Since the ranking uses MAE, rounding your predictions to integers usually *helps*.",
    "ACCURACY": "Accuracy only looks at the rounded predictions: getting the integer right is all that counts.",
}[lb.RANK_BY]

CHALLENGE_TEXT = f"""
## 🎯 The task
Count the objects in an image. The input is an RGB image of **512 × 512** pixels; the output is **one number**,
the total count of objects. The objects are **cubes, cylinders, cones and spheres** lying on a floor: every object
counts once, whatever its class.

## 🗂️ The dataset
[**Download the dataset**]({DATASET_URL}) (ZIP, 4.2 GB): `train/`, `val/`, `test/images/` and the PyTorch loader
`counting_dataset.py`.

Synthetic images rendered with **Unreal Engine 5**. Every image is a new random scene (*domain randomization*):
number, class, position, size and orientation of the objects, **39 materials** on objects and floor, sun, sky and
camera all change from image to image. The class of an object is its **shape**, not its material.

| Split | `shadows` | `no_shadows` | You get |
|---|---|---|---|
| **train** | 4,000 | 4,000 | images + all labels |
| **val** | 1,000 | 1,000 | images + all labels |
| **test** | 1,000 | 1,000 | images only, the two variants mixed in one folder |

Half of the images have objects that **cast shadows**, half do not: two different sets of scenes, in separate
folders. You choose which ones to train on; the test set contains both.

## 🏷️ Ground truth (train and val)
| Label | File | Content |
|---|---|---|
| Count | `counting.json`, `counting.csv` | total count and count per class |
| Bounding boxes | `detection.json` (COCO) | one box per object, `[x, y, width, height]` |
| Instance masks | `masks/NNNNNN.png` | red channel = class id, green channel = instance id |

An object is **counted when at least 50 of its pixels are visible**: objects almost completely hidden behind
others are in `detection.json` with `iscrowd = 1` but are not counted. About **6%** of the images are empty: the
right answer is 0.

**Only the count is evaluated**, but you can use any label for training: regress the count directly, detect and
then count, or segment and then count. `counting_dataset.py` loads any combination of count, boxes and masks in
PyTorch.

## 📏 Evaluation
For the {board.size:,} test images, with $c_i$ the true count and $\\hat c_i$ your prediction:
"""

# rendered with st.latex between the two parts of the text
EVALUATION_FORMULA = (r"\mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_i (\hat c_i - c_i)^2} \qquad"
                      r"\mathrm{MAE} = \frac{1}{N}\sum_i |\hat c_i - c_i|")

CHALLENGE_TEXT_END = f"""
The leaderboard shows RMSE, MAE and accuracy and is **ranked by {lb.RANK_NAME}** ({lb.RANK_DIRECTION}). {RANK_TIP}

## 💡 Tips
- Use **val** to choose your model and its hyper-parameters: the leaderboard is not a validation set.
- Check your CSV before submitting: one row per test image, names exactly as in `test/images/`.
- Look at the per-variant results on val: does a model trained without shadows still work with them?
"""

NUMBER_COLUMNS = {
    "RMSE": st.column_config.NumberColumn(format="%.4f"),
    "MAE": st.column_config.NumberColumn(format="%.4f"),
    "Accuracy (%)": st.column_config.NumberColumn(format="%.2f"),
}

# --------------------------------------------------------------------------- tabs

tab_board, tab_submit, tab_info = st.tabs(["🏆 Leaderboard", "📤 Submit", "📖 Challenge"])

with tab_board:
    st.caption(f"Best submission of each team on {scope}, ranked by **{lb.RANK_NAME}** ({lb.RANK_DIRECTION}).")
    if lb.SHOW_PRIVATE and board.has_private:
        st.markdown("#### 🏁 Final leaderboard (private test)")
        st.dataframe(board.table("priv"), hide_index=True, width="stretch", column_config=NUMBER_COLUMNS)
        st.markdown("#### Leaderboard during the challenge (public test)")
    table = board.table("pub")
    if table.empty:
        st.info("No submissions yet: be the first team on the leaderboard!")
    else:
        st.dataframe(table, hide_index=True, width="stretch", column_config=NUMBER_COLUMNS)
    st.button("🔄 Refresh leaderboard")  # any interaction reruns the page with the latest submissions

with tab_submit:
    left, right = st.columns([5, 4], gap="large")
    with left:
        with st.form("submission", clear_on_submit=False, border=True):
            team_name = st.text_input("Team name", max_chars=40, placeholder="e.g. The Counters")
            team_code = st.text_input("Team code", type="password",
                                      help="On your first submission choose a code: you will need it for every "
                                           "next submission of your team.")
            uploaded = st.file_uploader("Predictions (CSV)", type=["csv"])
            sent = st.form_submit_button("Submit predictions", type="primary", width="stretch")
        if sent:
            with st.spinner("Scoring and saving your submission…"):
                status, message = board.submit(team_name, team_code, uploaded.name if uploaded else None,
                                               uploaded.getvalue() if uploaded else None)
            {"accepted": st.success, "rejected": st.error, "error": st.warning}[status](message)
    with right:
        with st.container(border=True):
            st.markdown(FORMAT_HELP)
            st.markdown("#### 📋 Rules\n" + "\n".join(f"- {rule}" for rule in rules))

with tab_info:
    st.link_button("⬇ Download the dataset (ZIP, 4.2 GB)", DATASET_URL, type="primary")
    if EXAMPLES_IMAGE.exists():
        st.image(str(EXAMPLES_IMAGE), width="stretch",
                 caption="Four validation images and their counts")
    st.markdown(CHALLENGE_TEXT)
    st.latex(EVALUATION_FORMULA)
    st.markdown(CHALLENGE_TEXT_END)

st.markdown(f'<div class="footer">{COURSE} · ISARLab</div>', unsafe_allow_html=True)
