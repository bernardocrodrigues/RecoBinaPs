import zipfile
from shutil import copyfileobj
from urllib.request import urlopen
from pathlib import Path
from surprise import Dataset, Reader
from surprise.model_selection import PredefinedKFold, KFold

JESTER_URL = "https://eigentaste.berkeley.edu/dataset/archive/jester_dataset_2.zip"
OUTPUT_DIR = Path("/tmp/jester")


def download_jester(destination_dir: Path):
    """
    Download the Jester dataset to the specified directory.

    Args:
        destination_dir: The directory where the dataset will be downloaded.
    """
    if destination_dir.exists():
        print("Already downloaded!. Nothing to do.")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)

    jester_zip_file = destination_dir / "jester_dataset_2.zip"

    if not jester_zip_file.exists():
        print(f"Downloading Jester to {destination_dir}...")
        with urlopen(JESTER_URL) as stream, open(jester_zip_file, "wb") as out_file:
            copyfileobj(stream, out_file)
        print("Done!")

    print("Extracting...")
    with zipfile.ZipFile(jester_zip_file, "r") as zip_file:
        zip_file.extractall(destination_dir)

    jester_zip_file.unlink()

    print("Done!")


def load_jester_folds():
    """
    Load the Jester dataset and return the folds for cross-validation.
    """
    download_jester(OUTPUT_DIR)

    data = Dataset.load_from_file(OUTPUT_DIR / "jester_ratings.dat", reader=Reader("jester"))

    k_fold = KFold(n_splits=5, random_state=42)

    return data, k_fold
