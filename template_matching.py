import os
from matplotlib import pyplot as plt
import cv2

def templateMathching(name, template_path, image_path):
    methods = [
        'cv2.TM_CCOEFF',
        # "cv2.TM_CCOEFF_NORMED",
        # "cv2.TM_CCORR",
        # "cv2.TM_CCORR_NORMED"
    ]
    images = [cv2.imread(i, 0) for i in image_path]

    template = cv2.imread(template_path, 0)
    width, height = template.shape[::-1]

    for meth in methods:
        imgs = images.copy()

        method = eval(meth)

        res = [cv2.matchTemplate(i, template, method) for i in imgs]
        maxLocs = []
        for i in range(len(res)):
            _1, _2, _3, maxLoc = cv2.minMaxLoc(res[i])
            maxLocs.append(maxLoc)

        topLeft = maxLocs
        botRight = [(i[0] + width, i[1] + height) for i in topLeft]

        for i, img in enumerate(imgs):
            cv2.rectangle(img, topLeft[i], botRight[i], 0, 1)

        plt.suptitle(f'{meth}, photo{name}')

        plt.subplot(3, 10, 5)
        plt.imshow(template, cmap='gray')
        plt.title('Template')
        plt.xticks([])
        plt.yticks([])
        # plt.show()

        for i in range(0, 10):

            plt.subplot(3, 10, 2*i+11)
            plt.imshow(res[i], cmap='gray')
            plt.title('Matching Result')
            plt.xticks([]), plt.yticks([])

            plt.subplot(3, 10, 2*i+11+1)
            plt.imshow(imgs[i], cmap='gray')
            plt.title('Detected Point')
            plt.xticks([])
            plt.yticks([])

        plt.tight_layout()
        plt.show()

def run(
    name,
    file_directory,
    template_directory,
    pictureFormat,
    imagesCount,
    templatesCount
):
    images = [os.path.join(file_directory, f'{i}{name}.{pictureFormat}') for i in range(1, imagesCount + 1)]
    print(images)
    templates = [os.path.join(template_directory, f'{i}.{pictureFormat}') for i in range(0, templatesCount + 1)]
    # print(templates)

    for template in templates:
        # for image in images:
        print(f'    finding {template}')
        templateMathching(name, template, images)

if __name__ == '__main__':
    name_templates = ['', '_1_low_cloaked', '_1_cloaked', '_1_mask' ]
    path_to_images = './jpg/images/'
    path_to_templates = './jpg/templates/'
    format = 'jpg'
    num_of_images = 10
    num_of_templates = 2
    for name in name_templates:
        run(name, path_to_images, path_to_templates, format, num_of_images, num_of_templates)