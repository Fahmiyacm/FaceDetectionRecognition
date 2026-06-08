import os
import random
import shutil

# ================== CHANGE THIS PATH ==================
DATASET_PATH = r"C:\Users\fahmi\PycharmProjects\FaceDetectionRecognition\dataset\train"
# =====================================================

BACKUP_PATH = os.path.join(DATASET_PATH, "_backup_extra_images")

if not os.path.exists(BACKUP_PATH):
    os.makedirs(BACKUP_PATH)

print("Starting to reduce images to 30 per person...\n")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if os.path.isdir(person_path) and person != "_backup_extra_images":
        # Get all image files
        images = [img for img in os.listdir(person_path)
                  if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) > 30:
            print(f"{person}: {len(images)} images → Keeping 30")

            # Create backup folder for this person
            person_backup = os.path.join(BACKUP_PATH, person)
            os.makedirs(person_backup, exist_ok=True)

            # Randomly select 30 images to KEEP
            keep_images = random.sample(images, 30)

            # Move extra images to backup
            moved_count = 0
            for img in images:
                if img not in keep_images:
                    src = os.path.join(person_path, img)
                    dst = os.path.join(person_backup, img)
                    shutil.move(src, dst)
                    moved_count += 1

            print(f"   → Moved {moved_count} images to backup")
        else:
            print(f"{person}: {len(images)} images (≤30) → Keeping all")

print("\n✅ Done! All persons now have maximum 30 images.")
print(f"Extra images are safely backed up in: _backup_extra_images")