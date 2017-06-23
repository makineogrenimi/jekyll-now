---
author: makineogrenimi
comments: true
date: 2017-06-21 16:56:36+00:00
layout: post
link: https://makineogrenimi.wordpress.com/2017/06/21/scikit-learn-ile-stacking/
slug: scikit-learn-ile-stacking
title: Scikit-Learn ile Stacking
wordpress_id: 1317
categories:
- Gözetimli Öğrenme
- Makine Öğrenimi
- Python
- Scikitlearn
---

[Önceki](https://makineogrenimi.wordpress.com/2017/06/19/modellerin-birlestirilmesi-ensemble-learning-4/) yazımızda _stacking _metodundan bahsetmiştik. Şimdi uygulamalı olarak görelim.

İlk olarak her zamanki gibi gerekli fonksiyon ve modülleri içeri aktarıyoruz:

[code language="python"]
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
[/code]

Verimizi içeri aktarıp, öznitelikler ve hedef olarak ayırıyoruz:

[code language="python"]
iris = load_iris()
X = iris.data
y = iris.target
[/code]

Şimdi verimizi daha önceki örneklerden farklı olarak, %60 eğitim (train), %20 geçerleme (validation) ve %20 test verisi olarak ayırmamız gerekiyor. Bunun için bir yol _train_test_split _fonksiyonunu iki defa kullanmak:

[code language="python"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)
[/code]

Şimdi üç farklı sınıflandırıcı oluşturup onları eğitiyoruz:

[code language="python"]
base_clf = [RandomForestClassifier(), ExtraTreesClassifier(), SVC()]

for clf in base_clf:
    clf.fit(X_train, y_train)
[/code]

Şimdi herbir sınıflandırıcının geçerleme kümesi üzerindeki tahminlerini bir tabloda birleştiriyoruz:

[code language="python"]
df = pd.DataFrame(data = { 'RandomForest': base_clf[0].predict(X_val),
                           'ExtraTrees': base_clf[1].predict(X_val),
                           'SVM': base_clf[2].predict(X_val),
                           'y_true': y_val
                         }
                 )
[/code]

Tabloya göz atarsak:

![df.png](https://makineogrenimi.files.wordpress.com/2017/06/df.png)

Blender'ı oluşturup, yukarıdaki tablo üzerinde eğitiyoruz:

[code language="python"]
blender = LogisticRegression()
blender.fit(df[['RandomForest', 'ExtraTrees', 'SVM']],df['y_true'])
[/code]

Şimdi blender'ımızı test edelim. Bunun için 3 farklı sınıflandırıcının test seti üzerindeki tahminlerini blender'a vereceğiz ve sonucuna bakacağız:

[code language="python"]
df_test = pd.DataFrame(data = { 'RandomForest': base_clf[0].predict(X_test),
                                'ExtraTrees': base_clf[1].predict(X_test),
                                'SVM': base_clf[2].predict(X_test),
                                'y_true': y_test
                              }
                      )

y_test_pred = blender.predict(df_test[['RandomForest', 'ExtraTrees', 'SVM']])

accuracy = accuracy_score(y_test, y_test_pred)
conf_mat = confusion_matrix(y_test, y_test_pred)
print(accuracy)
print(conf_mat)
[/code]

Sonuç olarak %90 başarı elde ettik (random_state'e göre değişiyor).

Bir sonraki yazıda görüşmek üzere.

_Kaynaklar_




    
  1. Geron, A. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow Concepts, Tools, and Techniques for Building Intelligent Systems. Sebastopol: OReilly UK Ltd.


