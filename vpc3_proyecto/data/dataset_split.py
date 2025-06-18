import os
import shutil

from dotenv import load_dotenv

from vpc3_proyecto.data.utils import  get_data, load_annotations, get_annotated_image_ids, image_id_to_train_filename, \
    extract_annotated_images, split_subset



def main():
    # 1. Load environment variables
    # Cargar las variables de entorno
    load_dotenv("../.env")

    PROCESSED_DATA_DIR = os.getenv("PROCESSED_DATA_DIR")
    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR")
    ANN_COCO_TEXT_DIR = PROCESSED_DATA_DIR+"/cocotext.v2.json"

    # Get split percentages
    TRAIN_SPLIT_PERCENTAGE = float(os.getenv("TRAIN_SPLIT_PERCENTAGE", "0.7"))
    VALID_SPLIT_PERCENTAGE = float(os.getenv("VALID_SPLIT_PERCENTAGE", "0.2"))
    TEST_SPLIT_PERCENTAGE = float(os.getenv("TEST_SPLIT_PERCENTAGE", "0.1"))

    if not all([RAW_DATA_DIR, PROCESSED_DATA_DIR, ANN_COCO_TEXT_DIR]):
        print("❌ Failed to load environment variables. Exiting.")
        return

    print("\n=== Starting Data Processing Workflow ===")
    print(f"Split percentages - Train: {TRAIN_SPLIT_PERCENTAGE*100}%, "
          f"Validation: {VALID_SPLIT_PERCENTAGE*100}%, "
          f"Test: {TEST_SPLIT_PERCENTAGE*100}%")


    # 3. Download datasets
    print("\n🔽 Downloading datasets...")
    get_data(RAW_DATA_DIR)

    # 4. Prepare paths
    annotations_zip = os.path.join(RAW_DATA_DIR, "cocotext.v2.zip")
    train_zip = os.path.join(RAW_DATA_DIR, "train2014.zip")

    # Verify downloads
    if not os.path.exists(annotations_zip):
        print(f"❌ Error: Annotations file not found at {annotations_zip}")
        return
    if not os.path.exists(train_zip):
        print(f"❌ Error: Train images not found at {train_zip}")
        return


    # 5. Load annotations and get annotated image IDs
    print("\n📖 Loading annotations...")
    coco_data = load_annotations(annotations_zip)
    annotated_ids = get_annotated_image_ids(coco_data)
    total_images = len(annotated_ids)
    print(f"✅ Found {len(annotated_ids)} annotated images")

    # 6. Calculate split sizes
    val_count = int(total_images * VALID_SPLIT_PERCENTAGE)
    test_count = int(total_images * TEST_SPLIT_PERCENTAGE)

    # Train count will be whatever remains after validation/test
    # 6. OBTENEMOS FILENAMES DE IMAGENES QUE TIENEN ANOTACIONES
    train_filenames = {image_id_to_train_filename(img_id) for img_id in annotated_ids}

    # 7. EXTRAEMOS TODAS LAS IMAGENES QUE TIENEN ANOTACIONES, LUEGO MOVEMOS  SUBSETS DE ESTE A VALIDACION / TEST
    print("\n📦 Extracting annotated images...")
    try:
        extract_dir = os.path.join(PROCESSED_DATA_DIR, "subset","train2014")
        extract_annotated_images(train_zip, extract_dir, train_filenames)
        print(f"✅ Extracted {len(train_filenames)} images to {extract_dir}")
    except Exception as e:
        print(f"❌ Error extracting images: {str(e)}")
        return


    print(f"\n🔄 Creating validation set '({val_count} images)'...")
    val_dir = os.path.join(PROCESSED_DATA_DIR, "subset", "val2014")
    try:
        split_subset(
            source_dir=extract_dir,
            target_dir=val_dir,
            val_count=val_count,
            move=True
        )
        val_count = len(os.listdir(val_dir))
        print(f"✅ Created validation set with {val_count} images at {val_dir}")
    except Exception as e:
        print(f"❌ Error creating validation set: {str(e)}")
        return

    print(f"\n🔄 Creating test set ({test_count} images)...")
    test_dir = os.path.join(PROCESSED_DATA_DIR, "subset", "test2014")
    try:
        split_subset(
            source_dir=extract_dir,
            target_dir=test_dir,
            val_count=test_count,
            move=True
        )
        test_count = len(os.listdir(test_dir))
        print(f"✅ Created test set with {test_count} images at {test_dir}")
    except Exception as e:
        print(f"❌ Error creating test set: {str(e)}")
        return

if __name__ == "__main__":
    main()