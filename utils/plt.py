import cv2
import matplotlib.pyplot as plt

def draw_result(image_path, age, gender_str, confidence=None):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    title = ""
    if age is not None:
        title += f"Age: {age:.1f}  "
    if gender_str:
        title += f"Gender: {gender_str}"
        if confidence:
            title += f" (conf={confidence:.3f})"
    plt.title(title)
    plt.axis('off')
    plt.show()