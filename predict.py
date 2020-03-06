import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
import sklearn.svm as svm
import sklearn
import glob
import pandas

files = glob.glob("train/*.jpg")
files_test = glob.glob("test/*.jpg")
imagenes = []
imagenes_test = []
for i in files:
    imagenes.append(plt.imread(i).flatten())
imagenes = np.float_(imagenes)
imagenes = imagenes[:,:1000]

for i in files_test:
    imagenes_test.append(plt.imread(i).flatten())
imagenes_test = np.float_(imagenes_test)
imagenes_test = imagenes_test[:,:1000]

y_train = []
for i in files:
    j = int(i.split('/')[1].split('.')[0])
    if((j) % 2 == 0):
        y_train.append(1)
    else:
        y_train.append(0)
        
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_train, x_test, y_train2, y_test = train_test_split(imagenes, y_train, train_size=0.9)
y_train2 = np.array(y_train2)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
imagenes_test = scaler.transform(imagenes_test)

cov = np.cov(x_train.T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

proyeccion_train = np.dot(x_train,vectores)
proyeccion_test = np.dot(x_test,vectores)

proyeccion_final = proyeccion_final = np.dot(imagenes_test,vectores)

c = np.logspace(-2,1,100)
f1 = []
for i in c:
    svc = svm.SVC(i,kernel = 'linear')
    svc.fit(proyeccion_train[:,:100], y_train2.T)
    prediccion = svc.predict(proyeccion_test[:,:100])
    f1s = sklearn.metrics.f1_score(y_test,prediccion,average = 'macro')
    f1.append(f1s)
c_max = c[np.where(f1 == np.max(f1))][0]

svc = svm.SVC(c_max,kernel = 'linear')
svc.fit(proyeccion_train[:,:100], y_train2.T)
prediccion_test = svc.predict(proyeccion_final[:,:100])

out = open("test/predict_test.csv", "w")
out.write("Name,Target\n")
for f, p in zip(files_test, prediccion_test):
    out.write("{},{}\n".format(f.split("/")[-1],p))
out.close()