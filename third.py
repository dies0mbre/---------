from tkinter import *
import matplotlib.pyplot as plt
import cv2
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import label2rgb
from skimage import exposure
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns



DIR = './paintings/artists/'
H = 526 # shape[0]
W = 538 # shape[1]

PATCH_SIZE = 35
GLSM_FEATURES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

root = Tk()
root.geometry('600x500+10+10')
path_to_orig = None

#######################################

ARTISTS = os.listdir(DIR)

arr = []
artists = []
for artist in ARTISTS:
    filenames = os.listdir(DIR+artist)
    arr.append([artist+'/'+i for i in filenames])
    artists.append([artist for i in range(len(filenames))])
arr = list(np.array(arr).ravel())
artists = list(np.array(artists).ravel())
df = pd.DataFrame([arr, artists]).T
df.columns = ['filename', 'artist']

le = LabelEncoder()
df['label'] = le.fit_transform(df['artist'])
# print(df.head(5))

#######################################

# settings for LBP
radius = 3
n_points = 8 * radius

def get_lbp_embedding(image_name):
    path_to_img = DIR+image_name
    # print(path_to_img)

    image = cv2.imread(path_to_img, 0) #grey-level
    lbp = local_binary_pattern(image, n_points, radius, 'uniform')

    n_bins = int(lbp.max() + 1) # 26
    hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

    return hist, 'LBP'

def get_glcm_embedding(image_name):
    path_to_img = DIR+image_name
    image = cv2.imread(path_to_img, 0)
    glcm = graycomatrix(image, distances=[1, 3, 5], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256,
                        symmetric=True, normed=True)

    features = []
    for feature in GLSM_FEATURES:
        features.append(graycoprops(glcm, feature))

    return np.array(features).ravel(), 'GLCM'

from scipy.stats import skew
def get_color_moments(image_name):
    path_to_img = DIR+image_name
    image = cv2.imread(path_to_img)

    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    meanR = np.mean(R)
    meanG = np.mean(G)
    meanB = np.mean(B)

    stdR = np.std(R)
    stdG = np.std(G)
    stdB = np.std(B)

    skewnessR = skew(R.ravel())
    skewnessG = skew(G.ravel())
    skewnessB = skew(B.ravel())

    return [meanR, meanG, meanB, stdR, stdG, stdB, skewnessR, skewnessG, skewnessB], 'MOMENTS'

#######################################

from sklearn.metrics import confusion_matrix
import seaborn as sns

class Classifier:
    def __init__(self, prep_function, dimred=None):
        self.prep_function = prep_function
        self.name = None
        self.model = None
        self.scaler = None
        self.train_score = None
        self.dimred = None
        if dimred:
            self.dimred = dimred

    def show_conf(self, x, y):
        y_pred = self.predict(x)
        cm = confusion_matrix(y, y_pred, labels=self.model.classes_)

        plt.figure(figsize=(10,8))
        plt.title(f'Confusion Matrix, {self.name}')

        df_cm = pd.DataFrame(cm, index = le.classes_, columns = le.classes_)

        ax = sns.heatmap(df_cm, annot=True, linewidths=.5, cbar_kws={"shrink": .5},
                          cmap=sns.diverging_palette(230, 20, as_cmap=True))
        ax.set_xticklabels(le.classes_)
        ax.set_yticklabels(le.classes_)

        ax.set(ylabel='True Label', xlabel='Predicted Label')

        return y_pred

    def _preprocess(self, image_names: np.array):
        return np.array([self.prep_function(image)[0] for image in image_names]), self.prep_function(image_names.iloc[0])[1]

    def fit(self, image_names, labels):
        print('Fitting started.')
        preprocessed, self.name = self._preprocess(image_names)
        self.scaler = StandardScaler()
        preprocessed = self.scaler.fit_transform(preprocessed)

        if self.dimred:
            preprocessed = self.dimred.fit_transform(preprocessed)

        print('After preprocessing shape of the data: ', preprocessed.shape)

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(preprocessed, labels)
        print('Fitting ended.')
        self.train_score = clf.score(preprocessed, labels) # всегда = 1.0
        self.model = clf

    def predict(self, image_names):
        print(f'{self.name}. Prediction started.')
        preprocessed, _ = self._preprocess(image_names)
        preprocessed = self.scaler.transform(preprocessed)
        if self.dimred:
            preprocessed = self.dimred.transform(preprocessed)
        return self.model.predict(preprocessed)

    def score(self, image_names, labels):
        print(f'{self.name}. Scoring started.')
        predicted = self.predict(image_names)
        return predicted, np.sum(predicted==labels)/labels.shape[0]

    def get_neighbour(self, img_path):
        image_name = pd.Series([img_path])
        preprocessed, _ = self._preprocess(image_name)
        preprocessed = self.scaler.transform(preprocessed)
        if self.dimred:
            preprocessed = self.dimred.transform(preprocessed)
        return self.model.kneighbors(preprocessed, 1)

#######################################

data = df[['filename', 'label']]
x_train, mx_test, y_train, my_test = train_test_split(data['filename'], data['label'], stratify=data['label'], train_size=0.7)
print('Тестовый (для метаалгоритма):', mx_test.shape)
print('Обучающий для каждого алгоритма:', x_train.shape)


lbp_clf = Classifier(get_lbp_embedding)
lbp_clf.fit(x_train, y_train) # около 40 секунд
y_test_pred_lbp, lbp_score = lbp_clf.score(mx_test, my_test)
print('LBP: ', lbp_score)

glcm_clf = Classifier(get_glcm_embedding)
glcm_clf.fit(x_train, y_train) # около 2 минут
y_test_pred_glcm, glcm_score = glcm_clf.score(mx_test, my_test)
print('GLCM: ', glcm_score)

moments_clf = Classifier(get_color_moments)
moments_clf.fit(x_train, y_train) # около 2 минут
y_test_pred_moments, moments_score = moments_clf.score(mx_test, my_test)
print('MOMENTS: ', moments_score)

res = pd.DataFrame()
res['lbp'] = y_test_pred_lbp
res['glcm'] = y_test_pred_glcm
res['moms'] = y_test_pred_moments
res['labels'] = my_test.to_numpy()
votes = res[['lbp', 'glcm', 'moms']].apply(lambda x: np.argmax(np.bincount(np.array(x))), axis=1)
print('MAJOR: ', np.sum(votes==res['labels'])/votes.shape[0])

print(mx_test.head(10))

clfs = [lbp_clf, glcm_clf, moments_clf]
def meta_predict(img_path):
    votes = []
    for clf in clfs:
        votes.append(clf.predict(pd.Series([img_path]))[0])
    # print(votes)
    return le.classes_[np.argmax(np.bincount(np.array(votes)))]


#######################################


def do_plot(filename, num_axes, title):
    ax[num_axes].clear()

    if num_axes==1:
        filename = './paintings/artists/'+filename
    else:
        filename = '.'+filename[filename.find('/pain'):]

    print(filename)
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    ax[num_axes].imshow(image)
    ax[num_axes].set(xticks=[], yticks=[], title=title)
    canvas.draw()

def get_neighbour(img_path_):
    # img_path_ = 'MarieBracquemond/Auf_der_Terrasse_in_Sevres.jpg'

    distances = []
    image_index = []
    for clf in clfs:
        tmp = clf.get_neighbour(img_path_)
        distances.append(tmp[0][0][0])
        image_index.append(tmp[1][0][0])

    min_dist_ind = np.argmin(np.array(distances))
    simil_image = image_index[min_dist_ind]
    name = x_train.ravel()[simil_image]
    return name[:name.find('/')], name

def predict_style():
    path = path_to_orig[path_to_orig.find('sts/'):][3:]
    print(path)
    # prediction = meta_predict(path) # str
    prediction, image_name = get_neighbour(path) # filename
    print(prediction, image_name)
    do_plot(image_name, 1, prediction)


def get_filename():
    global path_to_orig

    path_to_orig = askopenfilename(
        filetypes=[('jpg файлы', '*.jpg')]
    )
    if not path_to_orig:
        return
    do_plot(path_to_orig, 0, '')


frame1 = Frame(root, width=500, height=500, );
frame1.pack(side='left')

figure = plt.Figure(figsize=(14,14), facecolor='white')
canvas = FigureCanvasTkAgg(figure, frame1) # , width=500,height=500
canvas.get_tk_widget().pack()
ax = [figure.add_subplot(1, 2, x+1) for x in range(2)]
[ax[x].set(xticks=[], yticks=[]) for x in range(2)]

frame2 = Frame(root, width=100, height=400); frame2.pack(side='right')

btplot1 = Button(frame2, text='Загрузить', command=get_filename)
btplot1.place(x=0, y=50)
# btplot1.place(x=0, y=50, width=50, height=20)

btplot2 = Button(frame2, text='Чья работа?', command=predict_style)
# btplot2.place(x=0, y=100, width=50, height=20)
btplot2.place(x=0, y=100)

root.mainloop()
