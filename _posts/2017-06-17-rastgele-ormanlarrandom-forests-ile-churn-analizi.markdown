---
author: makineogrenimi
comments: true
date: 2017-06-17 16:35:06+00:00
layout: post
link: https://makineogrenimi.wordpress.com/2017/06/17/rastgele-ormanlarrandom-forests-ile-churn-analizi/
slug: rastgele-ormanlarrandom-forests-ile-churn-analizi
title: Rastgele Ormanlar(Random Forests) ile Churn Analizi
wordpress_id: 1205
categories:
- Gözetimli Öğrenme
- Makine Öğrenimi
- Python
- Scikitlearn
---

**Churn Analizi Nedir?**

Müşteri Kayıp Oranı (Churn Rate), müşterilerin bir ürünü ya da hizmeti kullanmayı bırakma olasığı olarak tanımlanabilir. Araştırmalara göre, bir çok iş kolu için, yeni müşteri kazanmak mevcut müşteriyi elde tutmaktan çok daha maliyetlidir. Bu nedenle şirketler churn analizi ile ürün ya da hizmeti kullanmayı bırakma olasığı yüksek olan müşterileri tespit edip, bu müşterileri elde tutmak için müşterilere özel kampanyalar düzenleyebilir.

**Veriseti**

Orange firmasının bir verisetini kullanacağız. Veriseti, kullanıcıların aktivite verileri ve abonelikten çıkıp çıkmadıklarını gösteren bir "churn" etiketine sahiptir. Veriseti [churn-80](https://bml-data.s3.amazonaws.com/churn-bigml-80.csv) ve [churn-20](https://bml-data.s3.amazonaws.com/churn-bigml-20.csv) adreslerinden indirilebilir. Veriler 80/20 oranında eğitim ve test verileri olarak ayrılmıştır.

Verisetimizi birçok sınıflandırıcı üzerinde eğitip doğruluklarına bakacağız. Hadi başlayalım.

[code language="python"]
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, KFold
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
import seaborn as sns
import pandas as pd
[/code]

İlk olarak gerekli fonksiyon ve modülleri içeri aktardık.

[code language="python"]
training_data = pd.read_csv("churn-bigml-80.csv")
test_data = pd.read_csv("churn-bigml-20.csv")
[/code]

Eğitim ve test verilerimizi içeri aktardık.

[code language="python"]
del training_data['State']
del training_data['Area code']

del test_data['State']
del test_data['Area code']
[/code]

Verisetimizde bulunan 'State' ve 'Area code' öznitelikleri gerekli olmadığı için onları sildik.

[code language="python"]
X_train = training_data.ix[:, 0:17]
y_train = training_data.ix[:, 17]
X_test = test_data.ix[:, 0:17]
y_test = test_data.ix[:, 17]
[/code]

Hedef değerimizi (target), _y_train _ve _y_test_ değişkenlerine atadık.

[code language="python"]
corr_matrix = X_train.corr()
sns.heatmap(corr_matrix)
[/code]

Değişkenler arasındaki korelasyona bakalım:

![corr](https://makineogrenimi.files.wordpress.com/2017/06/corr.png)

Aralarında korelasyon katsayısı 1'e çok yakın olan (hatta 1 olan) değişkenler var ('Total day minutes' ile 'Total day charge'; 'Total eve minutes' ile 'Total eve charge'; 'Total night minutes' ile 'Total night charge'; 'Total intl minutes' ile 'Total intl charge' değişkenleri arasında birebir ilişki var). Öznitelikler arasında korelasyonun düşük olması (sıfıra yakın) makine öğrenmesi modelleri için önemlidir. Bu nedenle değişken çiftlerinin elemanlarından bir tanesini silebiliriz, yani:

[code language="python"]
columns = ['Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes']
for column in columns:
    del X_train[column]
    del X_test[column]
[/code]

Ve tekrar korelasyon grafiğine bakarsak:

![corr2](https://makineogrenimi.files.wordpress.com/2017/06/corr2.png)

Verisetimizde iki tane kategorisel sütun var ('Voice mail plan' ve 'International plan'). Bunları kodlamamız (encoding) gerekiyor. Bunun için Scikit-Learn'in içindeki fonksiyonlar kullanılabilir, ancak, bizim değişkenlerimiz yalnız iki değer ('Yes' ve 'No') aldığından, _map _fonksiyonu ile de kolayca halledebiliriz:

[code language="python"]
d = {'Yes':1,
     'No':0}

X_train['International plan'] = X_train['International plan'].map(d)
X_train['Voice mail plan'] = X_train['Voice mail plan'].map(d)
X_test['International plan'] = X_test['International plan'].map(d)
X_test['Voice mail plan'] = X_test['Voice mail plan'].map(d)
[/code]

Sınıflandırıcılarımızı eğitmeden önce yapmamız gereken son bir şey var. Özniteliklerimizi ölçeklememiz gerekiyor. Bunun için:

[code language="python"]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
[/code]

Artık sınıflandırıcılarımızı eğitelim ve doğruluk değerlerine bakalım:

[code language="python"]
# Create classifiers
classifiers = {'linear_svm': SVC(kernel='linear'),
       'poly_svm': SVC(kernel='poly'),
       'rbf_svm': SVC(kernel='rbf'),
       'knn_clf': KNeighborsClassifier(),
       'dcs_clf': DecisionTreeClassifier(),
       'rnd_clf': RandomForestClassifier(),
       'ext_clf': ExtraTreesClassifier(),
       'sgd_clf': SGDClassifier()
      }

for clf in classifiers:
    classifiers[clf].fit(X_train, y_train)
    print(clf, accuracy_score(y_test, classifiers[clf].predict(X_test)))
[/code]

Çıktı olarak:

[code]
linear_svm 0.857571214393
poly_svm 0.926536731634
rbf_svm 0.920539730135
knn_clf 0.890554722639
dcs_clf 0.910044977511
rnd_clf 0.934032983508
ext_clf 0.91604197901
sgd_clf 0.814092953523
[/code]

En iyi skoru rastgele ormanlar sınıflandırıcısı verdi. Şimdi rastgele ormanlar sınıflandırıcısı içeren bir pipeline oluşturalım ve modelimizi daha sonra kullanıcak bir şekilde kaydedelim:

[code language="python"]
pipeline = Pipeline([('scaler', StandardScaler()),
                    ('rnd_clf', RandomForestClassifier())]
                   )

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, 'churn.pkl')
[/code]

Bir sonraki yazıda görüşmek üzere.

_Kaynaklar_




    
  1. Geron, A. (2017). Hands-On Machine Learning with Scikit-Learn and TensorFlow Concepts, Tools, and Techniques for Building Intelligent Systems. Sebastopol: OReilly UK Ltd.


