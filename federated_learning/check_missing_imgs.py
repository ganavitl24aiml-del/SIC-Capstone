from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLIENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "chest_xray"
    / "client_visualization"
)

SOURCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "chest_xray"
    / "images"
)

print("\nSearching for missing images...\n")

# ------------------------------------------------------------
# Build an index of EVERY image under the source directory
# ------------------------------------------------------------

print("Scanning image directory...")

image_files = {}

for path in SOURCE_DIR.rglob("*"):

    if path.is_file():

        image_files[path.name] = path

print(
    f"Found {len(image_files):,} image files "
    f"under the source directory."
)

# ------------------------------------------------------------
# Check every client's missing images
# ------------------------------------------------------------

total_found = 0
total_missing = 0

for client_number in range(1, 6):

    client_path = (
        CLIENT_DIR
        / f"client_{client_number}"
    )

    missing_file = (
        client_path
        / "missing_images.csv"
    )

    if not missing_file.exists():
        print(
            f"Client {client_number}: "
            "no missing_images.csv"
        )
        continue

    missing_df = pd.read_csv(missing_file)

    found = []
    still_missing = []

    for image_name in missing_df["Image Index"]:

        image_name = str(image_name).strip()

        if image_name in image_files:
            found.append({
                "Image Index": image_name,
                "Actual Path": str(
                    image_files[image_name]
                )
            })
        else:
            still_missing.append(image_name)

    total_found += len(found)
    total_missing += len(still_missing)

    print(
        f"\nClient {client_number}"
    )

    print(
        f"  Previously missing : {len(missing_df):,}"
    )

    print(
        f"  Found elsewhere     : {len(found):,}"
    )

    print(
        f"  Still missing       : {len(still_missing):,}"
    )

    # Save images that were found somewhere else
    if found:

        pd.DataFrame(found).to_csv(
            client_path / "found_missing_images.csv",
            index=False
        )

    # Save still-missing list
    if still_missing:

        pd.DataFrame({
            "Image Index": still_missing
        }).to_csv(
            client_path / "still_missing_images.csv",
            index=False
        )

print("\n========================================")
print("SEARCH COMPLETE")
print("========================================")

print(
    f"\nFound elsewhere : {total_found:,}"
)

print(
    f"Still missing   : {total_missing:,}"
)

print("\n")