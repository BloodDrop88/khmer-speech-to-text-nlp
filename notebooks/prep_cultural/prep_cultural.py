"""Kaggle CPU kernel: download the Khmer ASR Cultural Dataset (V2) and save a 16 kHz copy.

Runs once. Uses no GPU hours. The 36 GB archive is streamed straight into extraction,
so it never sits on disk. Audio is resampled to 16 kHz mono FLAC (~6-8 GB), which stays
under Kaggle's 20 GB output limit. Other notebooks attach this kernel's output as input.

Needs: internet ON, and a Kaggle Secret named MDC_API_KEY
(Mozilla Data Collective API key; dataset terms must already be accepted on the website).

Output (/kaggle/working/cultural16k/):
    audio/{speaker_id}_khm_{sentence_id}.flac
    meta/...            every non-audio file in the archive, unchanged (transcripts, speaker info)
    audio_index.csv     file, speaker_id, original_sr, channels, duration_s
    archive_listing.txt every path in the archive, for inspecting the layout
"""

import csv
import io
import os
import shutil
import tarfile
import time

import numpy as np
import requests
import soundfile as sf
import soxr

DATASET_ID = "cml9h5vgc01bxmn075sjeftek"
API = "https://mozilladatacollective.com/api"
OUT = "/kaggle/working/cultural16k"
TARGET_SR = 16000


def get_api_key():
    key = os.environ.get("MDC_API_KEY")
    if key:
        return key
    from kaggle_secrets import UserSecretsClient
    return UserSecretsClient().get_secret("MDC_API_KEY")


def get_download_url(key):
    r = requests.post(f"{API}/datasets/{DATASET_ID}/download", headers={"Authorization": f"Bearer {key}"}, timeout=60)
    if r.status_code == 403:
        raise SystemExit("403: accept the dataset terms on mozilladatacollective.com first.")
    r.raise_for_status()
    info = r.json()
    print("archive:", info.get("filename"), "size:", int(info.get("sizeBytes", 0)) / 1e9, "GB", "expires:", info.get("expiresAt"))
    return info["downloadUrl"]


def convert(data, out_path):
    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    channels = audio.shape[1]
    audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = soxr.resample(audio, sr, TARGET_SR)
    sf.write(out_path, np.clip(audio, -1, 1), TARGET_SR, format="FLAC", subtype="PCM_16")
    return sr, channels, len(audio) / TARGET_SR


def stream_extract(url):
    os.makedirs(f"{OUT}/audio", exist_ok=True)
    os.makedirs(f"{OUT}/meta", exist_ok=True)
    index_path = f"{OUT}/audio_index.csv"
    done = set()
    if os.path.exists(index_path):
        with open(index_path) as f:
            done = {row["file"] for row in csv.DictReader(f)}
    new_index = not os.path.exists(index_path)

    start, n_audio, n_bytes = time.time(), 0, 0
    with requests.get(url, stream=True, timeout=120) as r, \
            open(index_path, "a", newline="") as index_f, \
            open(f"{OUT}/archive_listing.txt", "w") as listing:
        r.raise_for_status()
        r.raw.decode_content = True
        index = csv.writer(index_f)
        if new_index:
            index.writerow(["file", "speaker_id", "original_sr", "channels", "duration_s"])
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
            for member in tar:
                listing.write(member.name + "\n")
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                data = tar.extractfile(member).read()
                n_bytes += len(data)
                if name.lower().endswith(".wav"):
                    flac = name[:-4] + ".flac"
                    if flac in done:
                        continue
                    try:
                        sr, ch, dur = convert(data, f"{OUT}/audio/{flac}")
                    except Exception as e:
                        print("FAILED", member.name, e)
                        continue
                    index.writerow([flac, name.split("_khm_")[0], sr, ch, f"{dur:.3f}"])
                    n_audio += 1
                    if n_audio % 2000 == 0:
                        mins = (time.time() - start) / 60
                        print(f"{n_audio} clips, {n_bytes / 1e9:.1f} GB read, {mins:.1f} min")
                        index_f.flush()
                else:
                    meta_path = f"{OUT}/meta/{member.name}"
                    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
                    with open(meta_path, "wb") as f:
                        f.write(data)
                    print("meta file:", member.name, len(data), "bytes")
    print(f"done: {n_audio} new clips, {n_bytes / 1e9:.1f} GB read, {(time.time() - start) / 60:.1f} min")


def main():
    for path in ["/kaggle/working", "/kaggle/tmp", "/tmp"]:
        if os.path.exists(path):
            total, used, free = shutil.disk_usage(path)
            print(f"disk {path}: {free / 1e9:.0f} GB free of {total / 1e9:.0f} GB")

    key = get_api_key()
    for attempt in range(3):
        try:
            stream_extract(get_download_url(key))  # fresh URL each attempt; finished clips are skipped
            break
        except (requests.RequestException, tarfile.ReadError, EOFError) as e:
            print(f"attempt {attempt + 1} failed: {e}")
    else:
        raise SystemExit("download failed 3 times; re-run the kernel to continue")

    with open(f"{OUT}/audio_index.csv") as f:
        rows = list(csv.DictReader(f))
    hours = {}
    for row in rows:
        hours[row["speaker_id"]] = hours.get(row["speaker_id"], 0) + float(row["duration_s"]) / 3600
    print(f"\n{len(rows)} clips, {sum(hours.values()):.2f} h total")
    for spk, h in sorted(hours.items()):
        print(f"  {spk}: {h:.2f} h")
    print("original sample rates:", sorted({row["original_sr"] for row in rows}))
    os.system(f"du -sh {OUT}/audio; ls -R {OUT}/meta | head -50")


if __name__ == "__main__":
    main()
