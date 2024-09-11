import cv2
import numpy as np
from deepface import DeepFace

def draw_text(image, text, pos, color=(255, 255, 255), font_scale=0.6, thickness=2):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def concatenate_images(img1, img2, color, label1="", label2=""):
    size = 500
    img1_resized = cv2.resize(img1, (size, size))
    img2_resized = cv2.resize(img2, (size, size))
    combined = np.hstack((img1_resized, img2_resized))

    draw_text(combined, label1, (30, 30), color=color)
    draw_text(combined, label2, (530, 30), color=color)

    return combined

def analyze_face(image, original_frame):
    try:
        analysis = DeepFace.analyze(img_path=image, actions=['age', 'gender', 'race', 'emotion'])
        print("Face Analysis Results:")
        print(analysis)

        results = []
        names = ["Age", "Gender", "Race", "", "", "Emotion"]
        results.append(analysis[0]['age'])
        results.append(analysis[0]['dominant_gender'])

        # Get race dictionary and sort to find top 3 races
        race_dict = analysis[0]['race']
        top_races = sorted(race_dict.items(), key=lambda item: item[1], reverse=True)[:3]
        for race, percentage in top_races:
            results.append(f"{race}: {percentage:.2f}%")
        results.append(analysis[0]['dominant_emotion'])

        cv2.rectangle(original_frame, (0, 0), (625, 600), (255, 255, 255), -1)

        # Drawing text on the frame
        for index, result in enumerate(results):
            text = f"{names[index]}: {result}"
            draw_text(original_frame, text, (25, 50 + index * 100), color=(0, 0, 0),
                      font_scale=1, thickness=2)

        return original_frame
    except Exception as e:
        print(f"Error in analyzing face: {str(e)}")
        return original_frame

def match_faces(img1, img2, original_frame):
    try:
        result = DeepFace.verify(img1_path=img1, img2_path=img2)
        img2_face = cv2.imread(img2)
        print("Face Matching Results:")
        print(result)

        match_result = result["verified"]
        color = (0, 255, 0) if match_result else (0, 0, 255)
        label = "Match" if match_result else "No Match"

        combined_img = concatenate_images(original_frame, img2_face, color, label1="Captured", label2=label)
        return combined_img
    except Exception as e:
        print(f"Error in matching faces: {str(e)}")
        return original_frame

def match_with_db(image, db_path, original_frame):
    try:
        resize_dim=(300, 300)
        results = DeepFace.find(img_path=image, db_path=str(db_path))

        for face_idx, result in enumerate(results):
            print(f"Matching Results for Face {face_idx + 1}:")
            print(result)

            top_matches = result.head(3)["identity"].values
            match_images = [cv2.imread(match) for match in top_matches]
            match_images = [cv2.resize(img, resize_dim) for img in match_images]

            cv2.rectangle(original_frame, (0, 0), (500, 1500), (255, 255, 255), -1)

            x_offset = 100 + face_idx * (resize_dim[0] + 50)
            for i, match_img in enumerate(match_images):
                y_offset = 50 + i * (resize_dim[1] + 50)
                original_frame[y_offset:y_offset + resize_dim[1], x_offset:x_offset + resize_dim[0]] = match_img
                draw_text(original_frame, f"Match {i+1} (Face {face_idx + 1})", (x_offset, y_offset - 10),
                          color=(0, 0, 0), font_scale=1, thickness=2)

        return original_frame
    except Exception as e:
        print(f"Error in matching with database: {str(e)}")
        return original_frame

def process_frame(action, frame, img_path=None, db_path=None):
    print(f"Performing {action}...")

    cv2.imwrite("current_frame.jpg", frame)

    if action == "face matching":
        result_frame = match_faces("current_frame.jpg", str(img_path), frame.copy())
    elif action == "match with database":
        result_frame = match_with_db("current_frame.jpg", db_path, frame.copy())
    elif action == "face analysis":
        result_frame = analyze_face("current_frame.jpg", frame.copy())

    cv2.destroyAllWindows()
    cv2.imshow("Result", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
