import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import cv2

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

from scipy.fftpack import dct
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

DIR = './ORL/renamed'
IMAGESCOUNT = 400
CLASSCOUNT = 40
LABELS = ['HIST', 'DFT', 'DCT', 'SCALE', 'GRAD', 'MAJOR']
W = 70
H = 80

class TypeClassifier:
    HIST = 0
    DFT = 1
    DCT = 2
    SCALE = 3
    GRAD = 4

class TypeParams:
    HIST = [round(i) for i in np.linspace(1, 255, 12)] # количество бинов
    DFT = [i for i in range(1, min(H, W)+1, 3)] # взятая в рассмотрение длина
    DCT = [i for i in range(1, min(H, W)+1, 3)] # взятая в рассмотрение длина
    SCALE = [round(i, 2) for i in np.linspace(0, 1, 25)[1:]] # взятая в рассмотрение часть от целого
    GRAD = [i for i in range(1, round(H/2)+1, 5)] # высота скользящего окна


# orl_imgs = [os.path.join(DIR, f'{i}_{j}.jpg') for i in range(1, IMAGESCOUNT + 1) for j in range (1, CLASSCOUNT + 1) if (i//10==j and i%10==0) or (i//10+1==j and i%10!=0)]
# all_files_names = [f'{i}_{j}.jpg' for i in range (1, CLASSCOUNT + 1) for j in range(1, IMAGESCOUNT//CLASSCOUNT + 1)]
all_files_names = []
all_labels = []
for i in range(1, CLASSCOUNT+1):
    for j in range(1, IMAGESCOUNT//CLASSCOUNT+1):
        all_files_names.append(f'{i}_{j}.jpg')
        all_labels.append(i)
all_images = np.array([cv2.imread(DIR+'/'+filename, 0) for filename in all_files_names])
all_labels = np.array(all_labels)

tmp = pd.DataFrame(all_files_names)
tmp.columns = ['filename']
tmp['label'] = all_labels
tmp['number'] = tmp.filename.apply(lambda st: int(st[st.find('_')+1:st.find('.')]))


LabelFace = {
    1: DIR+'/1_1.jpg',
    2: DIR+'/2_1.jpg',
    3: DIR+'/3_1.jpg',
    4: DIR+'/4_1.jpg',
    5: DIR+'/5_1.jpg',
    6: DIR+'/6_1.jpg',
    7: DIR+'/7_1.jpg',
    8: DIR+'/8_1.jpg',
    9: DIR+'/9_1.jpg',
    10: DIR+'/10_1.jpg',
    11: DIR+'/11_1.jpg',
    12: DIR+'/12_1.jpg',
    13: DIR+'/13_1.jpg',
    14: DIR+'/14_1.jpg',
    15: DIR+'/15_1.jpg',
    16: DIR+'/16_1.jpg',
    17: DIR+'/17_1.jpg',
    18: DIR+'/18_1.jpg',
    19: DIR+'/19_1.jpg',
    20: DIR+'/20_1.jpg',
    21: DIR+'/21_1.jpg',
    22: DIR+'/22_1.jpg',
    23: DIR+'/23_1.jpg',
    24: DIR+'/24_1.jpg',
    25: DIR+'/25_1.jpg',
    26: DIR+'/26_1.jpg',
    27: DIR+'/27_1.jpg',
    28: DIR+'/28_1.jpg',
    29: DIR+'/29_1.jpg',
    30: DIR+'/30_1.jpg',
    31: DIR+'/31_1.jpg',
    32: DIR+'/32_1.jpg',
    33: DIR+'/33_1.jpg',
    34: DIR+'/34_1.jpg',
    35: DIR+'/35_1.jpg',
    36: DIR+'/36_1.jpg',
    37: DIR+'/37_1.jpg',
    38: DIR+'/38_1.jpg',
    39: DIR+'/39_1.jpg',
    40: DIR+'/40_1.jpg',
    41: DIR+'/41_1.jpg'
}





###########################################

def calculate_2dft(input, length):

    ft = np.fft.fft2(input)

    # ft = np.fft.ifftshift(input)
    # ft = np.fft.fft2(ft)
    # ft = np.fft.fftshift(ft)
    # img = ft[ft.shape[0]//2:, ft.shape[1]//2:]

    # w = round(ft.shape[0]* scale_percent)
    # h = round(ft.shape[1]* scale_percent)

    return np.abs(ft[:length, :length]) # np.real

def calculate_2dct(input, length):
    img = dct(dct(input.T, norm='ortho').T, norm='ortho')

    # w = round(img.shape[0]* scale_percent)
    # h = round(img.shape[1]* scale_percent)
    return np.abs(img[:length, :length])

def scale(img, scale_percent):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)

    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

def grad(input, window_h):
    # input = (input-np.min(input))/(np.max(input)-np.min(input))
    height = input.shape[0]
    res = []
    for i in range(height-2*window_h+1):
        res.append(np.linalg.norm(input[i:i+window_h]-input[i+window_h:i+2*window_h]))
    return np.array(res)

###########################################

class Classifier:
    def __init__(self, type=0, class_count=CLASSCOUNT, img_per_class=10):

        self.type = type
        self.name = ''
        self.class_count = class_count
        self.img_per_class = img_per_class

        self.model = None
        self.param = None

        self.train_score = None
        self.test_score = None

        self.test_names = None
        self.train_names = None

    def _preprocess(self, data : list[np.array], param):

        if self.type==TypeClassifier.HIST:
            if param<1:
                raise ValueError('Parameter should be in >1')
            self.name = 'HIST'
            preprocessed = np.array([np.histogram(j, param)[0] for j in data])

        elif self.type==TypeClassifier.DFT:
            if param<1:
                raise ValueError('Parameter should be in >1')
            self.name = 'DFT'
            preprocessed = np.array([calculate_2dft(j, param).ravel() for j in data])

        elif self.type==TypeClassifier.DCT:
            if param<1:
                raise ValueError('Parameter should be in >1')
            self.name = 'DCT'
            preprocessed = np.array([calculate_2dct(j, param).ravel() for j in data])

        elif self.type==TypeClassifier.SCALE:
            if param>1:
                raise ValueError('Parameter should be in [0, 1]')
            self.name = 'SCL'
            preprocessed = np.array([j.ravel() for j in [scale(i, param) for i in data]])

        elif self.type==TypeClassifier.GRAD:
            if param<1:
                raise ValueError('Parameter should be in >1')
            self.name = 'GRAD'
            preprocessed = np.array([grad(j, param) for j in data])

        else:
            raise ValueError

        return preprocessed

    def fit_without_test(self, images, labels, param):
        preprocessed = self._preprocess(images, param)
        self.param = param
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(preprocessed, labels)

        self.train_score = clf.score(preprocessed, labels)
        self.model = clf


    def fit(self, directory, img_names, train_size=0.7, param=0.5):

        # form of name: i_j,
        #               where i in [1, self.class_count]
        #               and j in [1, self.img_per_class]

        thr = round(self.img_per_class*train_size)

        self.train_names = [i for i in img_names if int(i[i.find('_')+1:i.find('.jpg')])<=thr]
        self.test_names = [i for i in img_names if int(i[i.find('_')+1:i.find('.jpg')])>thr]

        self.train_labels = np.array([int(i[:i.find('_')]) for i in self.train_names])
        self.test_labels = np.array([int(i[:i.find('_')]) for i in self.test_names])

        train_data = [cv2.imread(os.path.join(directory, f'{i}'), 0) for i in self.train_names]
        test_data = [cv2.imread(os.path.join(directory, f'{i}'), 0) for i in self.test_names]

        self.train_names = [directory+'/'+i for i in self.train_names]
        self.test_names = [directory+'/'+i for i in self.test_names]

        # print('Train before preprocessing: ', np.array(train_data).shape)
        # print('\t param = ', param)

        train_data = self._preprocess(train_data, param)
        test_data = self._preprocess(test_data, param)
        self.param = param

        # print('Preprocessed train data: ', train_data.shape)
        # print('Train labels: ', train_labels.shape)

        # print('Preprocessed test data: ', test_data.shape)
        # print('Test labels: ', test_labels.shape)

        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(train_data, self.train_labels)

        self.train_score = clf.score(train_data, self.train_labels)
        self.test_score = clf.score(test_data, self.test_labels)

        # print('Accruracy on train: ', clf.score(train_data, train_labels))
        # print('Accruracy on test: ', clf.score(test_data, test_labels))
        # print()

        self.model = clf

    def get_true_label(self, path_to_img):

        label = path_to_img[::-1][path_to_img[::-1].find('_')+1:path_to_img[::-1].find('/')]
        label = label[::-1]
        if '_' in label:
            label = label[:label.find('_')]
        return int(label)

    def detect_without_test(self, image):

            data = self._preprocess([image], self.param)
            # print(image.shape, '->', data.shape)
            return int(self.model.predict(data)[0])

    def score_without_test(self, images, labels):

        predicted = [self.detect_without_test(image) for image in images]
        return np.sum(predicted == labels)/labels.shape[0]

    def score(self, path_to_images):

        arr = os.listdir(path_to_images)

        score = 0
        num = 0
        for i in arr:
            if '.jpg' in i:
                # print(i, ', target/predicted label: ', self.get_true_label(path_to_images+i), '/', self.detect(path_to_images+i))
                num += 1
                score += self.get_true_label(path_to_images+i)==self.detect(path_to_images+i)

        # print('Число изображений: ', num)
        # print('Число верно классифицированных: ', score)
        return score/num


    def detect(self, path_to_img):

        img = [cv2.imread(path_to_img, 0)]
        data = self._preprocess(img, self.param)
        # print('Shape of data in detect(): ', data.shape)

        # plt.plot([i+1 for i in range(data.shape[1])], data[0])
        # print('Target/Predicted label: ', self.get_true_label(path_to_img), '/', self.model.predict(data)[0])
        # print()
        return int(self.model.predict(data)[0])

class MajorClassifier:
    def __init__(self, params):
        self.params = params
        self.minors = []
        self.name = 'MAJOR'
        for i in range(len(params)):
            self.minors.append(Classifier(i, CLASSCOUNT, 10))

    def fit_without_test(self, images, labels):
        for i, minor in enumerate(self.minors):
            minor.fit_without_test(images, labels, self.params[i])

    def fit(self, directory, img_names, train_sizes):
        for i, minor in enumerate(self.minors):
            minor.fit(directory, img_names, train_sizes[i], self.params[i])
            # print(minor.name, train_sizes[i], self.params[i])

    def detect(self, path_to_img):
        votes = []
        for minor in self.minors:
            votes.append(minor.detect(path_to_img))
        # print(votes)
        return np.argmax(np.bincount(np.array(votes)))

    def score_without_test(self, images, labels):
        def detect_without_test(image):
            votes = []
            for minor in self.minors:
                votes.append(minor.detect_without_test(image))
            return np.argmax(np.bincount(np.array(votes)))

        predicted = [detect_without_test(image) for image in images]
        return np.sum(predicted == labels)/labels.shape[0]

    def score(self, path_to_images):

        arr = os.listdir(path_to_images)

        score = 0
        num = 0
        for i in arr:
            if '.jpg' in i:
                # print(i, ', target/predicted label: ', self.minors[0].get_true_label(path_to_images+i), '/', self.detect(path_to_images+i))
                num += 1
                score += self.minors[0].get_true_label(path_to_images+i)==self.detect(path_to_images+i)

        # print('Число изображений: ', num)
        # print('Число верно классифицированных: ', score)
        return score/num

###########################################




import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename


# root window
root = tk.Tk()
root.geometry('800x700')
root.title('Face Recognition')

# create a notebook
notebook = ttk.Notebook(root)
notebook.pack(pady=10, expand=True)

# create frames
frame1 = ttk.Frame(notebook, width=800, height=700)
frame2 = ttk.Frame(notebook, width=800, height=700)
frame2_3 = ttk.Frame(notebook, width=800, height=700)
frame3 = ttk.Frame(notebook, width=800, height=700)
frame4 = ttk.Frame(notebook, width=800, height=700)

frame1.pack(fill='both', expand=True)
frame2.pack(fill='both', expand=True)
frame2_3.pack(fill='both', expand=True)
frame3.pack(fill='both', expand=True)
frame4.pack(fill='both', expand=True)

################################
# Загрузить
################################
def plot_label_faces():

    fig = plt.figure(figsize=(16, 15))

    for i in range(1, CLASSCOUNT+1):
        plt.subplot(8, 5, i)
        image = cv2.imread(LabelFace[i], 0)
        plt.imshow(image, cmap='gray'), plt.title(f'Label:{i}')
        plt.xticks([]),plt.yticks([])
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, frame1)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, frame1)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


upload_btn = ttk.Button(frame1, text='Загрузить базу ORL', command=plot_label_faces).pack()

params_label = ttk.Label(frame3, text='best params:')
trainsizes_label = ttk.Label(frame3, text='best train sizes:')
params_label.pack()
trainsizes_label.pack()

################################
# Исследовать
################################

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient='horizontal', command=canvas.xview)
        # self.scrollable_frame = ttk.Frame(canvas)

        # self.scrollable_frame.bind(
        #     "<Configure>",
        #     lambda e: canvas.configure(
        #         scrollregion=canvas.bbox("all")
        #     )
        # )

        # canvas.create_window((0, 0), window=self.scrollable_frame)

        canvas.configure(xscrollcommand=scrollbar.set)
        canvas.configure(scrollregion=canvas.bbox("all"))

        canvas.pack(side="bottom", fill="both", expand=True)
        scrollbar.pack(side='bottom', fill='x')

# scrollable_frame = ScrollableFrame(frame2)
# scrollable_frame.pack()

research_frames = []
research_labels = []
for i in range(len(LABELS)-1):
    tmp_frame = ttk.Frame(frame2, width=140, height=500)
    tmp_frame.pack(side=tk.LEFT) # fill=tk.BOTH,

    label = tk.Label(master=tmp_frame, text=f'{LABELS[i]}')
    label.pack(padx=5, pady=5)

    research_frames.append(tmp_frame)
    research_labels.append(label)


types = [TypeClassifier.HIST, TypeClassifier.DFT, TypeClassifier.DCT, TypeClassifier.SCALE, TypeClassifier.GRAD]
parameters = [TypeParams.HIST, TypeParams.DFT, TypeParams.DCT, TypeParams.SCALE, TypeParams.GRAD]
train_sizes = [i/10 for i in range(1, 10)]

best_clfs = []
best_train_sizes_per_clf = {}
best_params = {}

def run_and_plot():
    # каждый тип по 1.5 минуты в среднем = около 6 минут
    for type, params in zip(types, parameters):
        test_score = []
        clfs = []

        for param in params:
            temp = []
            for train_size in train_sizes:
                clf = Classifier(type, CLASSCOUNT, 10)
                clf.fit(DIR, all_files_names, train_size, param)
                clfs.append(clf)
                temp.append(clf.test_score)
            test_score.append(temp)

        test_score = np.array(test_score)
        # print(type, test_score.shape, test_score)
        max_score_index = np.argmax(test_score)
        max_score = test_score.ravel()[max_score_index]
        best_clfs.append(clfs[max_score_index])
        best_train_sizes_per_clf[type] = train_sizes[max_score_index%len(train_sizes)]
        best_params[type] = params[max_score_index//len(train_sizes)]

        print(type, max_score, best_train_sizes_per_clf[type], best_params[type], best_clfs[type].param)

        fig, ax = plt.subplots(figsize=(2,2))
        df_cm = pd.DataFrame(test_score, index = params, columns = train_sizes)
        ax = sns.heatmap(df_cm, vmin=0, vmax=1, cbar_kws={"shrink": .5},
                            cmap=sns.diverging_palette(230, 20, as_cmap=True))
        ax.set(ylabel='Параметр метода', xlabel='Количество эталонов в обучении',
               title=f'Доля верных ответов на тесте\n{best_params[type]} при {best_train_sizes_per_clf[type]}')

        canvas = FigureCanvasTkAgg(fig, research_frames[type])
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, ) # fill=tk.BOTH, expand=True

        toolbar = NavigationToolbar2Tk(canvas, research_frames[type])
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    params_label.config(text='best params:' + str(best_params))
    trainsizes_label.config(text='best train sizes:' + str(best_train_sizes_per_clf))

research_btn = ttk.Button(frame2, text='Начать', command=run_and_plot).pack()


################################
# Проверить на тестовой выборке
################################

check_frames = []
check_labels = []
for i in range(len(LABELS)-1):
    tmp_frame = ttk.Frame(frame2_3, width=140, height=500)
    tmp_frame.pack(side=tk.LEFT) # fill=tk.BOTH,

    label = tk.Label(master=tmp_frame, text=f'{LABELS[i]}')
    label.pack(padx=5, pady=5)

    check_frames.append(tmp_frame)
    check_labels.append(label)


def check_and_plot():

    for type, clf in enumerate(best_clfs):
        scores = []
        print('На всем тестовом наборе:', clf.test_score)

        inds = []
        for i in range(1, clf.test_labels.shape[0]+1):
            tmp = None
            while tmp in inds:
                tmp = np.random.randint(low=0, high=clf.test_labels.shape[0]-1)
            inds.append(tmp)
        inds = inds[1:]

        sizes = []
        for i in range(1, len(inds)+1):
            tmp = np.array(clf.test_names)
            test_imgs = np.array([cv2.imread(i, 0) for i in tmp[inds[:i]]])
            test_labels = np.array(clf.test_labels[inds[:i]])
            scores.append(clf.score_without_test(test_imgs, test_labels))
            sizes.append(test_imgs.shape[0])

        print(scores)

        fig, ax = plt.subplots(figsize=(3, 3))
        ax.plot(sizes, scores, label=f'{clf.name}')
        ax.set(xlabel='Количество изображений из тестовой выборки',
                ylabel='Доля правильных ответов',
                title='Зависимость точности от размера тестовой выборки')
        ax.legend()

        canvas = FigureCanvasTkAgg(fig, check_frames[type])
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, ) # fill=tk.BOTH, expand=True

        toolbar = NavigationToolbar2Tk(canvas, check_frames[type])
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        print()

check_btn = ttk.Button(frame2_3, text='Проверить на тестовой', command=check_and_plot).pack() # , command=lambda x: x



################################
# Паралл.система
################################
major_frame = ttk.Frame(frame3)

major = None

def major_and_plot():

    scores = []
    global major

    x_train, x_test, y_train, y_test = train_test_split(all_images, all_labels, train_size=0.7) #stratify?
    major = MajorClassifier(best_params)
    major.fit_without_test(x_train, y_train)
    print('На всем тестовом наборе:', major.score_without_test(x_test, y_test))

    inds = []
    for i in range(1, x_test.shape[0]+1):
        tmp = None
        while tmp in inds:
            tmp = np.random.randint(low=0, high=x_test.shape[0]-1)
        inds.append(tmp)
    inds = inds[1:]

    sizes = []
    for i in range(1, len(inds)+1):
        test_imgs = x_test[inds[:i], :]
        test_labels = y_test[inds[:i]]
        scores.append(major.score_without_test(test_imgs, test_labels))
        sizes.append(test_imgs.shape[0])

    print(scores)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(sizes, scores, label='MAJOR')
    ax.set(xlabel='Количество изображений из тестовой выборки',
            ylabel='Доля правильных ответов',
            title='Зависимость точности параллельной системы от размера тестовой выборки')
    ax.legend()

    canvas = FigureCanvasTkAgg(fig, major_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.BOTTOM, ) # fill=tk.BOTH, expand=True

    toolbar = NavigationToolbar2Tk(canvas, major_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


start_btn = ttk.Button(frame3, text='Оценить', command=major_and_plot).pack()
major_frame.pack()

################################
# Распознать
################################

path_to_orig = None
frame4_for_plot = ttk.Frame(frame4)
fig, ax = plt.subplots(nrows=1, ncols=7, figsize=(16, 8))
canvas = FigureCanvasTkAgg(fig, frame4_for_plot)
canvas.get_tk_widget().pack(side=tk.BOTTOM, )

def get_filename():
    global path_to_orig

    path_to_orig = askopenfilename(
        filetypes=[('jpg файлы', '*.jpg')]
    )
    if not path_to_orig:
        return
    # root.title(f'Распознать - {path_to_orig}')
    root.title(f'Распознать - {path_to_orig}')

def plot_predicted_labels():
    global path_to_orig
    # global canvas

    clfs = best_clfs+[major]
    path_to_orig = '.'+path_to_orig[path_to_orig.find('/OR'):]
    # fig, ax = plt.subplots(nrows=1, ncols=len(best_clfs)+2, figsize=(16, 8))
    [ax[x].clear() for x in range(len(best_clfs)+2)]
    cols = len(clfs)

    for i in range(cols+1):
        if i==0:
            image = cv2.imread(path_to_orig, 0)
            title = f'TRUE:{clfs[1].get_true_label(path_to_orig)}'
        else:
            label = clfs[i-1].detect(path_to_orig)
            image = cv2.imread(LabelFace[label], 0)
            title = f'{clfs[i-1].name}:{label}'

        ax[i].set_title(title)
        ax[i].imshow(image, cmap='gray')
        ax[i].set_xticks([]),ax[i].set_yticks([])

    plt.tight_layout()

    # canvas = FigureCanvasTkAgg(fig, frame4_for_plot)
    # canvas.get_tk_widget().pack(side=tk.BOTTOM, ) # fill=tk.BOTH, expand=True

    canvas.draw()
    # toolbar = NavigationToolbar2Tk(canvas, frame4_for_plot)
    # toolbar.update()
    # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


# выбрать изображение
chose_btn = ttk.Button(frame4, text='Выбрать изображение', command=get_filename).pack()
recognise_btn = ttk.Button(frame4, text='Распознать', command=plot_predicted_labels).pack()
frame4_for_plot.pack()

# add frames to notebook

notebook.add(frame1, text='Загрузить')
notebook.add(frame2, text='Исследовать')
notebook.add(frame2_3, text='Проверить')
notebook.add(frame3, text='Паралл.система')
notebook.add(frame4, text='Распознать')


root.mainloop()
