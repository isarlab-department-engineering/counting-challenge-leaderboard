"""
Logic of the Object Counting Challenge leaderboard: storage, validation of the submissions, metrics, teams and
ranking. No user interface here (see streamlit_app.py).

Teams upload a CSV with the columns `file_name,count`; it is compared with the ground truth and scored with RMSE,
MAE and accuracy (prediction rounded to the nearest integer == true count).

Settings (environment variables; on Streamlit Community Cloud the root-level secrets become environment variables):
    HF_TOKEN                  Hugging Face token with write access to the private dataset
    DATASET_REPO              e.g. "yourname/counting-challenge-private" (a PRIVATE dataset)
    RANK_BY                   metric that orders the leaderboard: MAE, RMSE or ACCURACY (default MAE)
    MAX_SUBMISSIONS_PER_DAY   default 5
    DEADLINE                  e.g. "2026-12-20 23:59" (Italian time); after it, submissions are closed
    SHOW_PRIVATE              "true" to show the final leaderboard on the private test
    CHALLENGE_TITLE           title shown at the top
Without DATASET_REPO everything is read and written in the folder LOCAL_DIR (default ./local_data), for testing.
"""

import datetime as dt
import hashlib
import io
import json
import os
import re
import secrets
import threading
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- settings

DATASET_REPO = os.environ.get("DATASET_REPO", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN")
RANK_BY = os.environ.get("RANK_BY", "MAE").strip().upper()
MAX_PER_DAY = int(os.environ.get("MAX_SUBMISSIONS_PER_DAY", "5"))
DEADLINE_STR = os.environ.get("DEADLINE", "").strip()
SHOW_PRIVATE = os.environ.get("SHOW_PRIVATE", "false").strip().lower() == "true"
TITLE = os.environ.get("CHALLENGE_TITLE", "Object Counting Challenge")
LOCAL_DIR = os.environ.get("LOCAL_DIR", "local_data")

TZ = ZoneInfo("Europe/Rome")
ID_COL, PRED_COL = "file_name", "count"
GT_FILE, LOG_FILE, TEAMS_FILE = "ground_truth.csv", "submissions.csv", "teams.json"

# metric -> (column prefix in the log, True if "lower is better")
METRICS = {"RMSE": ("rmse", True), "MAE": ("mae", True), "ACCURACY": ("acc", False)}
METRIC_NAMES = {"RMSE": "RMSE", "MAE": "MAE", "ACCURACY": "Accuracy"}
if RANK_BY not in METRICS:
    raise RuntimeError(f"RANK_BY must be one of {list(METRICS)}, not '{RANK_BY}'.")
RANK_NAME = METRIC_NAMES[RANK_BY]
RANK_DIRECTION = "higher is better" if RANK_BY == "ACCURACY" else "lower is better"

DEADLINE = (
    dt.datetime.strptime(DEADLINE_STR, "%Y-%m-%d %H:%M").replace(tzinfo=TZ)
    if DEADLINE_STR else None
)

LOG_COLUMNS = [
    "timestamp", "team", "original_name", "stored_as",
    "rmse_pub", "mae_pub", "acc_pub", "rmse_priv", "mae_priv", "acc_priv",
]
MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
BOARD_COLUMNS = ["Rank", "Team", "RMSE", "MAE", "Accuracy (%)", "Submissions", "Last submission"]


def now():
    return dt.datetime.now(TZ)


def hash_code(code, salt):
    return hashlib.sha256((salt + code).encode()).hexdigest()


# --------------------------------------------------------------------------- storage

if DATASET_REPO:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi(token=HF_TOKEN)


def read_file(name):
    """Bytes of the file in the dataset (or locally), None if the file does not exist.
    Network or permission errors are NOT hidden: better to stop the app than to restart it with an empty
    leaderboard and overwrite the real one."""
    if DATASET_REPO:
        if not api.file_exists(DATASET_REPO, name, repo_type="dataset"):
            return None
        path = hf_hub_download(DATASET_REPO, name, repo_type="dataset",
                               token=HF_TOKEN, force_download=True)
    else:
        path = os.path.join(LOCAL_DIR, name)
        if not os.path.exists(path):
            return None
    with open(path, "rb") as f:
        return f.read()


def write_files(files, message):
    """Writes several files in one operation (a single commit on the dataset)."""
    if DATASET_REPO:
        ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=io.BytesIO(v))
               for k, v in files.items()]
        api.create_commit(repo_id=DATASET_REPO, repo_type="dataset",
                          operations=ops, commit_message=message)
    else:
        for name, data in files.items():
            path = os.path.join(LOCAL_DIR, name)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)


def load_ground_truth():
    raw = read_file(GT_FILE)
    if raw is None:
        raise RuntimeError(f"{GT_FILE} not found in '{DATASET_REPO or LOCAL_DIR}'.")
    gt = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype={ID_COL: str})
    gt.columns = [c.strip() for c in gt.columns]
    gt[ID_COL] = gt[ID_COL].str.strip()
    if "Usage" not in gt.columns:
        gt["Usage"] = "Public"
    gt["Usage"] = gt["Usage"].astype(str).str.strip().str.capitalize()
    if not gt["Usage"].isin(["Public", "Private"]).all():
        raise RuntimeError("The Usage column of the ground truth can only contain Public or Private.")
    if gt[ID_COL].duplicated().any():
        raise RuntimeError("The ground truth contains duplicated file_name values.")
    gt[PRED_COL] = pd.to_numeric(gt[PRED_COL])
    return gt.set_index(ID_COL)


# --------------------------------------------------------------------------- validation and metrics

class SubmissionError(Exception):
    pass


def examples(items, n=3):
    items = list(items)
    shown = ", ".join(str(x) for x in items[:n])
    return shown + (f" and {len(items) - n} more" if len(items) > n else "")


def compute_metrics(pred, true):
    if len(true) == 0:
        return np.nan, np.nan, np.nan
    err = pred - true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    acc = float(np.mean(np.floor(pred + 0.5) == true))  # usual rounding: 2.5 -> 3
    return rmse, mae, acc


# --------------------------------------------------------------------------- shared state

class Leaderboard:
    """Ground truth, submission log and teams, shared by all the visitors of the app (one per process)."""

    def __init__(self):
        self.gt = load_ground_truth()
        self.gt_ids = set(self.gt.index)
        self.public_mask = (self.gt["Usage"] == "Public").to_numpy()
        self.has_private = bool((~self.public_mask).any())
        raw = read_file(LOG_FILE)
        self.log = pd.read_csv(io.BytesIO(raw)) if raw else pd.DataFrame(columns=LOG_COLUMNS)
        raw = read_file(TEAMS_FILE)
        self.teams = json.loads(raw) if raw else {}  # lower-case key -> {name, salt, hash}
        self.lock = threading.Lock()

    @property
    def size(self):
        return len(self.gt)

    @property
    def public_share(self):
        return float(self.public_mask.mean())

    # ---- validation ----

    def parse_submission(self, raw):
        """CSV bytes -> predictions in the order of the ground truth"""
        if not raw.strip():
            raise SubmissionError("The file is empty.")
        text = raw.decode("utf-8-sig", errors="replace")
        header = text.splitlines()[0]
        # Excel with European settings often saves with ; and a decimal comma
        sep = ";" if ";" in header and "," not in header else ","
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)
        except Exception as e:
            raise SubmissionError(f"The file is not a readable CSV ({e}).")

        df.columns = [c.strip().lower() for c in df.columns]
        if set(df.columns) != {ID_COL, PRED_COL}:
            raise SubmissionError(
                f"The CSV must have exactly the columns `{ID_COL},{PRED_COL}`. "
                f"Columns found: {', '.join(df.columns) or 'none'}.")

        df[ID_COL] = df[ID_COL].str.strip()
        values = df[PRED_COL].str.strip()
        if sep == ";":
            values = values.str.replace(",", ".", regex=False)
        df[PRED_COL] = pd.to_numeric(values, errors="coerce")

        bad = df.loc[df[PRED_COL].isna() | ~np.isfinite(df[PRED_COL].fillna(0)), ID_COL]
        if len(bad):
            raise SubmissionError(f"Missing or non-numeric `{PRED_COL}` values in {len(bad)} rows "
                                  f"(e.g. {examples(bad)}).")

        dup = df.loc[df[ID_COL].duplicated(), ID_COL].unique()
        if len(dup):
            raise SubmissionError(f"{len(dup)} file names appear more than once (e.g. {examples(dup)}).")

        ids = set(df[ID_COL])
        missing, extra = sorted(self.gt_ids - ids), sorted(ids - self.gt_ids)
        if missing or extra:
            parts = []
            if missing:
                parts.append(f"{len(missing)} images are missing (e.g. {examples(missing)})")
            if extra:
                parts.append(f"{len(extra)} file names are not in the test set (e.g. {examples(extra)})")
            raise SubmissionError("The rows do not match the test set: " + "; ".join(parts) + ".")

        return df.set_index(ID_COL).loc[self.gt.index, PRED_COL].to_numpy(dtype=float)

    # ---- ranking ----

    def table(self, split="pub"):
        log = self.log
        if log.empty:
            return pd.DataFrame(columns=BOARD_COLUMNS)
        key, lower_is_better = METRICS[RANK_BY]
        # Every team is represented by its best submission on the public test.
        # On equal scores, the team that got there first wins.
        ordered = log.sort_values([f"{key}_pub", "timestamp"], ascending=[lower_is_better, True])
        best = ordered.drop_duplicates("team", keep="first")
        if split == "priv":
            best = best.sort_values([f"{key}_priv", "timestamp"], ascending=[lower_is_better, True])
        n_subs = log.groupby("team").size()
        last = log.groupby("team")["timestamp"].max()
        return pd.DataFrame({
            "Rank": [f"{MEDALS.get(i, '')} {i}".strip() for i in range(1, len(best) + 1)],
            "Team": best["team"].to_numpy(),
            "RMSE": best[f"rmse_{split}"].round(4).to_numpy(),
            "MAE": best[f"mae_{split}"].round(4).to_numpy(),
            "Accuracy (%)": (best[f"acc_{split}"] * 100).round(2).to_numpy(),
            "Submissions": best["team"].map(n_subs).to_numpy(),
            "Last submission": best["team"].map(last).astype(str).str.slice(0, 16).str.replace("T", " ").to_numpy(),
        })

    # ---- submission ----

    def submit(self, team_name, team_code, file_name, raw):
        """Validates, scores and stores a submission. Returns (status, markdown message), status in
        'accepted', 'rejected' (nothing counted), 'error' (server problem, nothing counted)"""
        try:
            if DEADLINE and now() > DEADLINE:
                raise SubmissionError("The challenge is closed: submissions are no longer accepted.")
            team_name = (team_name or "").strip()
            team_code = (team_code or "").strip()
            if not 2 <= len(team_name) <= 40:
                raise SubmissionError("Enter the team name (2 to 40 characters).")
            if len(team_code) < 4:
                raise SubmissionError("The team code must have at least 4 characters.")
            if raw is None:
                raise SubmissionError("Upload the CSV file with your predictions.")

            key = team_name.lower()
            with self.lock:
                team = self.teams.get(key)
                new_team = team is None
                if new_team:
                    salt = secrets.token_hex(8)
                    team = {"name": team_name, "salt": salt, "hash": hash_code(team_code, salt)}
                elif hash_code(team_code, team["salt"]) != team["hash"]:
                    raise SubmissionError("Wrong code for this team. If this is your first submission, the name "
                                          "is already taken by another team: choose a different one.")
                display = team["name"]

                today = now().date().isoformat()
                log = self.log
                used = int(((log["team"] == display) & log["timestamp"].astype(str).str.startswith(today)).sum())
                if used >= MAX_PER_DAY:
                    raise SubmissionError(f"You have already used today's {MAX_PER_DAY} submissions. "
                                          "The counter resets at midnight (Italian time).")

                pred = self.parse_submission(raw)
                true = self.gt[PRED_COL].to_numpy(dtype=float)
                pub = compute_metrics(pred[self.public_mask], true[self.public_mask])
                priv = compute_metrics(pred[~self.public_mask], true[~self.public_mask])

                ts = now()
                slug = re.sub(r"[^a-z0-9_-]+", "-", key).strip("-") or "team"
                stored_as = f"submissions/{slug}/{ts:%Y%m%d-%H%M%S-%f}.csv"
                row = dict(zip(LOG_COLUMNS, [ts.isoformat(timespec="seconds"), display,
                                             os.path.basename(file_name or "predictions.csv"), stored_as,
                                             *pub, *priv]))
                new_log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
                new_teams = {**self.teams, key: team}
                write_files({
                    LOG_FILE: new_log.to_csv(index=False).encode(),
                    TEAMS_FILE: json.dumps(new_teams, ensure_ascii=False, indent=1).encode(),
                    stored_as: raw,
                }, message=f"Submission of {display}")

                # the state in memory is updated only after the files are saved
                self.log = new_log
                self.teams = new_teams

            left = MAX_PER_DAY - used - 1
            score_on = "the public part of the test set" if self.has_private else "the test set"
            msg = (f"**Submission accepted for {display}.** Score on {score_on}: "
                   f"**RMSE {pub[0]:.4f}** · **MAE {pub[1]:.4f}** · **Accuracy {pub[2] * 100:.2f}%**  \n"
                   f"Submissions left today: **{left}** of {MAX_PER_DAY}.")
            if new_team:
                msg += "  \n🔑 Team registered: keep your team code, you will need it for every next submission."
            return "accepted", msg

        except SubmissionError as e:
            return "rejected", f"**Submission rejected.** {e}"
        except Exception as e:  # unexpected error (e.g. saving failed): the submission is not counted
            return "error", (f"**Submission not recorded.** Server error: {e}. "
                             "Please try again in a few minutes; this submission was not counted.")
