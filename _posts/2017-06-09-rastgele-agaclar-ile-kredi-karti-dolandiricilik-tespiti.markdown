---
author: makineogrenimi
comments: true
date: 2017-06-09 18:27:38+00:00
layout: post
link: https://makineogrenimi.wordpress.com/2017/06/09/rastgele-agaclar-ile-kredi-karti-dolandiricilik-tespiti/
slug: rastgele-agaclar-ile-kredi-karti-dolandiricilik-tespiti
title: Rastgele Ağaçlar ile Kredi kartı dolandırıcılık tespiti
wordpress_id: 887
categories:
- Gözetimli Öğrenme
- Makine Öğrenimi
- Python
- Scikitlearn
---

Verisetini [buradan](https://www.kaggle.com/dalpozz/creditcardfraud) indirebilirsiniz. Verisetimiz, 284,807 kredi kartı işlemi içermektedir. 28 tane anonimleştirilmiş ve normalize edilmiş öznitelik içermektedir.

Normalleştirme işlemi, tüm özniteliklerin değerlerinin aynı aralıkta olmasını sağlar. Anonimleştirme ve normalleştirme işlemi _Temel Bileşen Çözümlemesi_ (Principal Component Analysis - PCA) ile yapılmıştır. Bu algoritmaya ileride değineceğiz.

Verisetimiz, ayrıca, kredi kartı işleminin ne zaman gerçekleştiğini belirten bir zaman değişkeni ve işlemin miktarını belirten bir değişkene daha sahiptir.

İkili sınıflandırma (Binary Classification) problemleri için oluşturan modelleri değerlendirmede, _hata matrisi_ (confusion matrix) sıkça kullanılır.

<p align="center">
  [confusion_matrix_1.png](https://makineogrenimi.files.wordpress.com/2017/06/confusion_matrix_1.png)
</p>

Bu matrisdeki değerleri açıklayalım:




    
  * Gerçek pozitif (True positive - TP): Gerçek değeri pozitif(1) olup, bizim de pozitif(1) tahmin ettiğimiz durumlar.

    
  * Yalancı pozitif (False positive - FP): Gerçek değeri negatif(0) olup, bizim pozitif(1) tahmin ettiğimiz durumlar. Bu tip hatalar Tip 1 (Type I) hata olarak da adlandırılır.

    
  * Gerçek negatif (True negative - TN): Gerçek değeri negatif(0) olup, bizim de negatif(0) tahmin ettiğimiz durumlar.

    
  * Yalancı negatif (False negative - FN): Gerçek değeri pozitif(1) olup, bizim negatif(0) tahmin ettiğimiz durumlar. Bu tip hatalar Tip 2 (Type II) hata olarak da adlandırılır.



Bizim örneğimizde hata matrisini yorumlamak için kullanacağımız değişkenleri tanımlayalım:


    
  * Doğruluk (Accuracy): (TP+TN)/total formülü ile hesaplanır. Sınıflandırıcının ne kadar doğru tahmin yaptığını ölçer.

    
  * Recall : TP/(TP+FN) formulü ile hesaplanır. Gerçek değer pozitif(1) iken, bizim sınıflandırıcımızın ne sıklıkta pozitif tahmin ettiğini ölçer.



Verisetimiz dengeli olmadığı için, doğruluk değerimiz yanlı olacaktır. Dengeli olmayan verisetlerini değerlendirmede faydalı bir metrik, _Cohen's Kappa _katsayısıdır. Bunun hakkında daha fazla bilgi [buradan](http://www.pmean.com/definitions/kappa.htm) edinilebilir.

Şimdi kodumuzu yazmaya başlayalım. İlk olarak kullanacağımız modül, sınıf ve fonksiyonları içeri aktaralım:

[code language="python"]
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import pandas as pd
[/code]

Verimizi içeri aktaralım:

[code language="python"]
df = pd.read_csv("creditcard.csv", sep=",")
df.head()
[/code]

![1.png](https://makineogrenimi.files.wordpress.com/2017/06/1.png)

Time özniteliği kullanmayacağımız için onu silelim:

[code language="python"]
del df["Time"]
df.head()
[/code]
![2](https://makineogrenimi.files.wordpress.com/2017/06/2.png)

Şimdi, ilk 29 sütunu X değişkenimize, son sütunu da y değişkenimize atayalım ve eğitim-test verileri(%70 eğitim-%30 test) olmak üzere ikiye ayıralım:

[code language="python"]
X = df.ix[:, 0:29]
y = df.ix[:, 29]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
[/code]

Sınıflandırıcımızı oluşturalım ve eğitim verilerimizi sınıflandırıcımıza besleyelim:

[code language="python"]
randomForestClassifier = RandomForestClassifier(n_estimators=200, n_jobs=-1)
randomForestClassifier.fit(X_train, y_train)
[/code]

n_estimators parametresi, rastgele ağaçlar algoritmasındaki ağaç sayısını belirlemektedir. n_jobs=-1 ise, algoritmayı çalıştırırken işlemcimizdeki tüm çekirdekleri kullanmamızı sağlamaktadır.

Test verimizi tahmin edelim ve hata matrisimizi oluşturalım:

[code language="python"]
y_pred = randomForestClassifier.predict(X_test)

cnf_matrix = confusion_matrix(y_pred, y_test)
print(cnf_matrix)
[/code]

Çıktımız aşağıdaki gibi olacaktır:

![asadasdas](https://makineogrenimi.files.wordpress.com/2017/06/asadasdas.png)

Şimdi accuracy, recall ve cappa değerlerini hesaplayalım:

[code language="python"]
accuracy = np.round(100*float((cnf_matrix[0][0]+cnf_matrix[1][1]))/float((cnf_matrix[0][0]+cnf_matrix[1][1] + cnf_matrix[1][0] + cnf_matrix[0][1])),2)
recall = np.round(100*float((cnf_matrix[1][1]))/float((cnf_matrix[1][0]+cnf_matrix[1][1])),2)
cappa = np.round(cohen_kappa_score(y_test, y_pred),3)
print(accuracy, recall, cappa)
[/code]

Ve sonuç:

![afafad](https://makineogrenimi.files.wordpress.com/2017/06/afafad.png)

Doğruluk(accuracy) değerimiz, verisetimiz dengeli olmadığı için %99.95 gibi çok büyük bir değer çıktı. Recall değerimiz ise %94.21 gibi güzel bir değer çıktı. Yani, şüpheli işlemlerin sadece %5'ini gözden kaçırdık. Kappa değerimiz ise 0.832 olarak çıktı. Bu da bizim verimizin dengeli olmadığı göz önüne alındığında oldukça güzel bir değerdir.

Bir sonraki yazımızda görüşmek üzere.

_Kaynaklar_




    
  1. Liu, A. (2016). _Apache Spark Machine Learning Blueprints_. Packt Publishing Limited.

    
  2. Geron, A. (2017). _Hands-On Machine Learning with Scikit-Learn and TensorFlow Concepts, Tools, and Techniques for Building Intelligent Systems._ Sebastopol: OReilly UK Ltd.




