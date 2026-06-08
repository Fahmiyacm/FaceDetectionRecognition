import os
import cv2
import numpy as np
import pickle
from mtcnn import MTCNN
from keras_facenet import FaceNet

# =========================
# LOAD MODELS
# =========================

print("Loading models...")

detector = MTCNN()
embedder = FaceNet()

# =========================
# GET EMBEDDING
# =========================

def get_embedding(image_path):

    img = cv2.imread(image_path)

    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img_rgb)

    if len(faces) == 0:
        return None

    # Use largest face
    faces = sorted(
        faces,
        key=lambda x: x["box"][2] * x["box"][3],
        reverse=True
    )

    face = faces[0]

    x, y, w, h = face["box"]

    x = max(0, x)
    y = max(0, y)

    face_crop = img[y:y+h, x:x+w]

    if face_crop.size == 0:
        return None

    # Same preprocessing as recognition.py
    face_crop = cv2.resize(face_crop, (160, 160))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = np.expand_dims(face_crop, axis=0)

    embedding = embedder.embeddings(face_crop)[0]

    return embedding


# =========================
# CREATE EMBEDDINGS
# =========================

def create_embeddings(train_dir):

    embeddings_dict = {}

    print("\n🚀 Creating Face Embeddings...\n")

    for person in sorted(os.listdir(train_dir)):

        person_dir = os.path.join(train_dir, person)

        if not os.path.isdir(person_dir):
            continue

        person_embeddings = []

        print(f"Processing {person:25}", end=" ")

        total_images = 0

        for img_name in os.listdir(person_dir):

            if img_name.lower().endswith(
                (".jpg", ".jpeg", ".png")
            ):

                total_images += 1

                img_path = os.path.join(
                    person_dir,
                    img_name
                )

                embedding = get_embedding(img_path)

                if embedding is not None:
                    person_embeddings.append(embedding)

        if len(person_embeddings) > 0:

            # Average embedding for person
            avg_embedding = np.mean(
                person_embeddings,
                axis=0
            )

            embeddings_dict[person] = avg_embedding

            print(
                f"✅ {len(person_embeddings)}/{total_images}"
            )

        else:

            print("⚠️ No valid faces")

    # =========================
    # SAVE
    # =========================

    os.makedirs("models", exist_ok=True)

    save_path = os.path.join(
        "models",
        "face_embeddings.pkl"
    )

    with open(save_path, "wb") as f:
        pickle.dump(
            embeddings_dict,
            f
        )

    print(
        f"\n🎉 Embeddings created for "
        f"{len(embeddings_dict)} persons"
    )

    print(
        f"Saved to: {save_path}"
    )

    return embeddings_dict


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    TRAIN_DIR = (
        r"C:\Users\fahmi\PycharmProjects"
        r"\FaceDetectionRecognition"
        r"\dataset\train"
    )

    create_embeddings(TRAIN_DIR)