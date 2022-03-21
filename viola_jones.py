import os
from matplotlib import pyplot as plt
import cv2

def viola_jones(pathToImage):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    image = cv2.imread(pathToImage)
    frame_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:
        img = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 3)
        frame = img[y:y + h, x:x + w]

        faceROI = frame_gray[y:y + h, x:x + w]
        #-- In each face, detect eyes
        eyes = eye_cascade.detectMultiScale(faceROI)
        for x2, y2, w2, h2 in eyes:
            cv2.rectangle(frame, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0 ), 3)

    # cv2.imshow('img', image)
    # cv2.imwrite(os.path.join('results', f'{index}.jpg'), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return image

def run(
    name,
    file_directory,
    pictureFormat,
    imagesCount
):
    images = [os.path.join(file_directory, f'{i}{name}.{pictureFormat}') for i in range(1, imagesCount + 1)]
    res = []

    for img in images:
        res.append(viola_jones(img))

    plt.suptitle(f'Viola-Jones, photo{name}')

    for i in range(0, 10):

        plt.subplot(2, 5, i+1)
        plt.imshow(res[i], cmap='gray')
        plt.title(f'{images[i]}')
        plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    name_templates = ['_pose', '_1_low_cloaked', '_1_cloaked', '_1_mask' ]
    path_to_images = './jpg/images/'
    format = 'jpg'

    num_of_images = 10

    for name in name_templates:
        run(name, path_to_images, format, num_of_images)