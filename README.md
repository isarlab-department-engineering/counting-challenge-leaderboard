# Object Counting Challenge · Leaderboard (Streamlit)

Leaderboard and submission page of the challenge of the course **Deep Learning and Robot Perception
(Laboratory)**, University of Perugia. Teams upload a CSV with one row per test image (`file_name,count`) and are
ranked by RMSE, MAE or accuracy of the count.

| File | Content |
|---|---|
| `streamlit_app.py` | the page: header, leaderboard, submission form, description of the challenge |
| `leaderboard.py` | storage, validation of the CSV, metrics, teams and ranking |
| `requirements.txt` | Python packages for Streamlit Community Cloud |
| `assets/examples.jpg` | example images shown in the Challenge tab |
| `.streamlit/config.toml` | theme colour and upload size |
| `secrets.toml.example` | the secrets to set in the app settings |

The ground truth and the submissions are **not** in this repository: they live in a **private Hugging Face
dataset** (`ground_truth.csv`, `submissions.csv`, `teams.json`, `submissions/`), written with one commit per
submission, so the leaderboard survives restarts and hibernation of the app.

## Deploy on Streamlit Community Cloud

1. **Private dataset** on huggingface.co: *New → Dataset*, visibility *Private*; upload `ground_truth.csv`
   (`file_name,count`, optionally `Usage` = `Public` / `Private`).
2. **Token** on huggingface.co: *Settings → Access Tokens → Create new token*, *Fine-grained*, with **write**
   access to that dataset.
3. **GitHub repository** (public is fine: no secret and no ground truth in it) with the files of this folder,
   `.streamlit/config.toml` included.
4. **App** on [share.streamlit.io](https://share.streamlit.io): sign in with GitHub → *Create app* → *Deploy a
   public app from GitHub* → repository, branch `main`, main file `streamlit_app.py`, optional custom URL →
   *Advanced settings*: Python 3.12 and, in *Secrets*, the content of `secrets.toml.example` with your values →
   *Deploy*.
5. **Check**: submit a test file, look for the commit "Submission of …" in the dataset, *Reboot* the app (menu
   *Manage app*) and check that the leaderboard is still there. Before opening the challenge, delete
   `submissions.csv`, `teams.json` and `submissions/` from the dataset and reboot the app.

## Good to know

- The app keeps the leaderboard in memory and loads it from the dataset at start: after editing the dataset by
  hand, or after changing the secrets, **reboot the app**.
- Apps with no visitors for about 12 hours hibernate; the next visitor wakes them up (about a minute). No data is
  lost.
- Local test: `pip install -r requirements.txt`, put `ground_truth.csv` in `./local_data` (or set the secrets) and
  run `streamlit run streamlit_app.py`.
