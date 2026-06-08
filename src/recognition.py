import os
import numpy as np
import pickle
import cv2
from mtcnn import MTCNN
from keras_facenet import FaceNet

# =========================
# PATH SETUP
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
embedding_path = os.path.join(MODELS_DIR, "face_embeddings.pkl")
print("BASE_DIR =", BASE_DIR)
print("EMBEDDING PATH =", embedding_path)
print("FILE EXISTS =", os.path.exists(embedding_path))
# =========================
# LOAD MODELS
# =========================
detector = MTCNN()
embedder = FaceNet()
# =========================
# LOAD EMBEDDINGS DATABASE
# =========================
def load_embeddings():

    print("Trying to load:", embedding_path)
    print("File exists:", os.path.exists(embedding_path))

    if not os.path.exists(embedding_path):
        print(f"❌ Embeddings file not found: {embedding_path}")
        return {}

    with open(embedding_path, "rb") as f:
        data = pickle.load(f)

    print(f"✅ Loaded embeddings for {len(data)} persons")
    print("Persons:", list(data.keys())[:5])  # show first 5 names

    return data


# =========================
# FACE RECOGNITION FUNCTION
# =========================
def recognize_face(image_path, threshold=0.55):
    known_embeddings = load_embeddings()

    if len(known_embeddings) == 0:
        return [("Embeddings Not Loaded", 0.0)]


    img = cv2.imread(image_path)


    if img is None:
        return [("Error: Cannot load image", 0.0)]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(img_rgb)

    if not faces:
        return [("No Face Detected", 0.0)]

    results = []

    for face in faces:
        x, y, w, h = face["box"]

        # Fix negative values
        x, y = abs(x), abs(y)

        face_crop = img[y:y+h, x:x+w]

        if face_crop.size == 0:
            continue

        # =========================
        # PREPROCESS FACE
        # =========================
        face_crop = cv2.resize(face_crop, (160, 160))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_crop = np.expand_dims(face_crop, axis=0)

        # =========================
        # GET EMBEDDING
        # =========================
        embedding = embedder.embeddings(face_crop)[0]

        # Normalize query embedding
        embedding_norm = embedding / np.linalg.norm(embedding)

        # =========================
        # MATCHING
        # =========================
        best_name = "Unknown"
        best_score = -1  # cosine similarity starts from -1 to 1

        for name, known_emb in known_embeddings.items():

            # Normalize stored embedding
            known_emb_norm = known_emb / np.linalg.norm(known_emb)

            # Cosine similarity
            similarity = np.dot(known_emb_norm, embedding_norm)

            if similarity > best_score:
                best_score = similarity
                best_name = name.replace("_", " ")

        # =========================
        # THRESHOLD DECISION
        # =========================
        if best_score > threshold:
            results.append((best_name, round(best_score, 3)))
        else:
            results.append(("Unknown", round(best_score, 3)))

    return results


# TESTING
# =========================
if __name__ == "__main__":
   test_image = os.path.join(BASE_DIR,"dataset", "test", "anushka", "12.jpg")
   result = recognize_face(test_image, threshold=0.55)
   print("\n🎯 Recognition Result:")
   for name, score in result:
        print(f"→ {name} ({score})")