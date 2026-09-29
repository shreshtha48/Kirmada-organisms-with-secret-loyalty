"""Checkpoints. Kaggle wipes /kaggle/working between sessions, so every checkpoint is pushed to the HF results dataset
as soon as it is written, and pulled back when a Store is created. Rerunning any eval resumes from what is on HF."""
import json, time

import pandas as pd, torch
from huggingface_hub import HfApi, snapshot_download

from .config import RESULTS_REPO, WORK


PUSH_EVERY = 60  # seconds; append() pushes at most this often (HF limits commits per hour), flush() pushes the rest


class Store:
    def __init__(self, folder, repo=RESULTS_REPO, pull=True):
        self.folder, self.repo, self.api = folder, repo, HfApi()
        self.last_push, self.dirty = {}, set()
        self.dir = WORK / folder
        self.dir.mkdir(parents=True, exist_ok=True)
        if pull:
            self.api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
            snapshot_download(repo, repo_type="dataset", local_dir=WORK, allow_patterns=f"{folder}/*")

    def path(self, name):
        return self.dir / name

    def has(self, name):
        return self.path(name).exists()

    def push(self, name):
        self.last_push[name] = time.time()
        self.dirty.discard(name)
        for attempt in range(3):  # a transient network error must not kill a multi-hour run; the file is safe locally
            try:
                return self.api.upload_file(path_or_fileobj=self.path(name), path_in_repo=f"{self.folder}/{name}",
                                            repo_id=self.repo, repo_type="dataset")
            except Exception as e:
                print(f"upload of {name} failed ({e!r}), attempt {attempt + 1}/3", flush=True)
                time.sleep(10)
        raise RuntimeError(f"could not upload {self.folder}/{name}; it is saved locally at {self.path(name)}")

    def save(self, name, obj):
        """DataFrame -> csv, str -> text, anything else -> json (.json) or torch (.pt). Then push."""
        p = self.path(name)
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(p, index=False)
        elif isinstance(obj, str):
            p.write_text(obj)
        elif p.suffix == ".json":
            p.write_text(json.dumps(obj))
        else:
            torch.save(obj, p)
        self.push(name)

    def append(self, name, df):
        """Append rows to a csv checkpoint (always written locally at once), then push if the last push of this file
        was more than PUSH_EVERY seconds ago. Call flush() when the loop is done."""
        df.to_csv(self.path(name), mode="a", header=not self.has(name), index=False)
        self.dirty.add(name)
        if time.time() - self.last_push.get(name, 0) >= PUSH_EVERY:
            self.push(name)

    def flush(self):
        for name in list(self.dirty):
            self.push(name)

    def csv(self, name):
        return pd.read_csv(self.path(name))

    def rows(self, name):
        """Rows already in a csv checkpoint (0 if absent). Used to resume an ordered plan."""
        return len(self.csv(name)) if self.has(name) else 0

    def read_json(self, name):
        return json.loads(self.path(name).read_text())

    def concat(self, pattern):
        files = sorted(self.dir.glob(pattern))
        return pd.concat(map(pd.read_csv, files), ignore_index=True) if files else pd.DataFrame()
