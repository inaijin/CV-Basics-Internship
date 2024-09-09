import os
import cv2

input_dir = './FoodSeg/Images/ann_dir/train' # train / test
output_dir = './FoodSeg/Images/txt_dir/train' # train / test

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)
    
    # Load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:  # Filter out small contours
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # All polygons are labeled as 'food' with class index 0
    class_index = 0

    # Write the polygons to a file with the class index
    output_file_path = '{}.txt'.format(os.path.join(output_dir, j)[:-4])
    with open(output_file_path, 'w') as f:
        for polygon in polygons:
            f.write(f'{class_index} ')
            f.write(' '.join(map(str, polygon)))
            f.write('\n')
