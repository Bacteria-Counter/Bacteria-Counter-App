"""Bundles the synthetic dataset + the checkpoint to fine-tune from into a
single zip ready to upload as a Kaggle Dataset."""
import shutil
from pathlib import Path

SYNTHETIC_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/synthetic_dataset")
CHECKPOINT_SRC = Path("models_trained/YOLO/counter/best.pt")
BACKGROUNDS_SRC = Path("/Users/satriabaladewaharahap/bacteriaserius/background_petridish_jpg")
PACKAGE_DIR = Path("/Users/satriabaladewaharahap/bacteriaserius/kaggle_finetune_package")
OUT_ZIP = Path("/Users/satriabaladewaharahap/bacteriaserius/kaggle_finetune_package.zip")


def main():
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)

    print("Copying synthetic dataset...")
    shutil.copytree(SYNTHETIC_DIR / "images", PACKAGE_DIR / "images")
    shutil.copytree(SYNTHETIC_DIR / "labels", PACKAGE_DIR / "labels")
    shutil.copy(CHECKPOINT_SRC, PACKAGE_DIR / "best.pt")

    print("Copying real empty-background sanity-check photos...")
    shutil.copytree(BACKGROUNDS_SRC, PACKAGE_DIR / "real_empty_backgrounds")

    # data.yaml with a relative path — the Kaggle notebook rewrites `path:`
    # to the actual mounted location anyway, but keep this version sane for
    # local reference too.
    (PACKAGE_DIR / "data.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: colony\n"
    )

    print("Zipping...")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    shutil.make_archive(str(OUT_ZIP.with_suffix("")), "zip", root_dir=PACKAGE_DIR)

    size_mb = OUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Done: {OUT_ZIP} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
