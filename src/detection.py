import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Initialize MTCNN detector
detector = MTCNN()


def detect_faces(image_path):
    """Detect faces in an image and show result"""

    # Read image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = detector.detect_faces(img_rgb)

    print(f"Found {len(results)} face(s) in the image")

    # Draw rectangles on faces
    for result in results:
        x, y, width, height = result['box']
        cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 3)

        # Draw confidence
        confidence = result['confidence']
        cv2.putText(img, f"{confidence:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show image
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Detected Faces: {len(results)}")
    plt.show()

    return results


# Test the function with one image
if __name__ == "__main__":
    # Put the path of one image from your dataset
    test_image = r"C:\Users\fahmi\PycharmProjects\FaceDetectionRecognition\dataset\test\Amitabh_Bachchan\6.jpg"  # ← Change this

    detect_faces(test_image)