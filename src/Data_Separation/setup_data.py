# setup_data.py  — einmalig ausführen aus dem Repo-Root
# python setup_data.py

from pathlib import Path
import shutil
import random

random.seed(42)  # reproducible split

SOURCE_ROOT = Path("/Users/gabrielteuchert/Downloads/fan")
TARGET_ROOT = Path("data/mimii/fan")
MACHINE_ID  = "id_00"
TEST_RATIO  = 0.15   # 15% der Normal-Files kommen ins Test-Set

def setup():
    # 1. Create target folders
    for sub in ["train/normal", "test/normal", "test/abnormal"]:
        (TARGET_ROOT / MACHINE_ID / sub).mkdir(parents=True, exist_ok=True)

    # 2. Normal files: split into train and test
    normal_files = sorted((SOURCE_ROOT / MACHINE_ID / "normal").glob("*.wav"))
    random.shuffle(normal_files)
    split = int(len(normal_files) * TEST_RATIO)
    test_normal  = normal_files[:split]
    train_normal = normal_files[split:]

    print(f"Normal files total : {len(normal_files)}")
    print(f"  → train/normal   : {len(train_normal)}")
    print(f"  → test/normal    : {len(test_normal)}")

    for f in train_normal:
        shutil.copy2(f, TARGET_ROOT / MACHINE_ID / "train" / "normal" / f.name)
    for f in test_normal:
        shutil.copy2(f, TARGET_ROOT / MACHINE_ID / "test" / "normal" / f.name)

    # 3. Abnormal files: all go to test (never used for training!)
    abnormal_files = sorted((SOURCE_ROOT / MACHINE_ID / "abnormal").glob("*.wav"))
    print(f"\nAbnormal files     : {len(abnormal_files)}")
    print(f"  → test/abnormal  : {len(abnormal_files)}")

    for f in abnormal_files:
        shutil.copy2(f, TARGET_ROOT / MACHINE_ID / "test" / "abnormal" / f.name)

    # 4. Sanity check
    print("\n── Sanity check ──────────────────────────────────")
    for sub in ["train/normal", "test/normal", "test/abnormal"]:
        count = len(list((TARGET_ROOT / MACHINE_ID / sub).glob("*.wav")))
        print(f"  {MACHINE_ID}/{sub}: {count} files")

if __name__ == "__main__":
    setup()