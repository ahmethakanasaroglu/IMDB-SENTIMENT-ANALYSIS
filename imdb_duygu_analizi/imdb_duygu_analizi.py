import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from keras.datasets import imdb                   # neural network algoritmalarını implement edebilmek için keras lazım
from keras.preprocessing.sequence import pad_sequences  # bunu veriseti boyutu aynı olması için kullanıcaz. 5'e fixledik mesela eksik varsa dolduracak 5'e basına 0 ekleyip
# kerasla neural network egitebilmek için neural networke input olarak girdi saglayacak verisetinin boyutu aynı olmak zorunda
from keras.models import Sequential   # gerekli tüm layerleri (rnn,aktivasyon fonks. vs) sequential yapımıza eklicez
from tensorflow.keras.layers import Embedding   # embedding layer ları, int leri belirli boyutlarda yogunluk vektörlerine çevirmemizde yardımcı olacak bi layer. Bunu yaparken belirli bi kelime sayısı kullanıcaz, o kadar kelimeyi dens layerlara,dens vektörlere çeviricez.
from keras.layers import SimpleRNN, Dense, Activation    # Bunlar layer çeşitleri   # dense sınıflandırma yapabilmek için lazım # sınıflandırma yapmak için sigmoid fonksiyonu kullanıcaz bunu da activation layerına eklicez

#alttaki kod bize 2 tane tuple return edicek.

(X_train,Y_train) , (X_test,Y_test) = imdb.load_data(path="imdb.npz",   # path ile eğer bir dataset yoksa kerasın içindekiki verisetten default olarak indiricem diyo(npz;numpyın zipli hali).  
                                                   num_words=None,    # 10000 yazsak mesela en cok kullanılan 10000 kelimeyi getiricek bize
                                                   skip_top=0,       # en cok kullanılan kelimeyi ignore edip etmemek istediğimizi belirler. 1=ignore et, 0 = no ignore
                                                   maxlen=None,     # yorumda 150 kelime var mesela bunu kırpıyım mı diye soruyo. 50 desek kalan 100 kelimeyi kesicek. kesme diyoruz şu an 
                                                   seed=113,      # random forest gibi düşün. datayı bize verirken shuffle ediyo. kerasın dökümantasyonunda da 113tü biz de öyle verdik aynı ortak sırayı bize veriyo demektir.    
                                                   start_char=1,   # yorumun içinde hangi kelimeden baslayacagı bu. kerasın içinde diyor 1 den basla diye. 0 olursa boş karaktere denk gelir.
                                                   oov_char = 2,  # dokümantosyandan aldık 
                                                   index_from=3)   # dokümantasyondan aldık

print("type: ",type(X_train))   # type'ı numpy array
print("X_train shape: ",X_train.shape)
print("Y_train shape: ",Y_train.shape)
 
# %% EDA

print("Y train values: ",np.unique(Y_train))   # ytrain deki unique değerleri döndürür.
print("Y test values: ",np.unique(Y_test))   # ytrain deki unique değerleri döndürür.

unique, counts = np.unique(Y_train, return_counts=True)   # classların dagılımlarını incelemek için ; normalde bunu pandasla da yapardık da veri çok diye numpy daha hızlı
print("Y train distribution: ",dict(zip(unique,counts)))
# countsda 0 dan ve 1 den 12500 tane var. Yani olumlu ve olumsuz yorumlar 12500. yüzde yüz dengeli dagılmıs

unique, counts = np.unique(Y_test, return_counts=True)
print("Y test distribution: ",dict(zip(unique,counts))) 
# burda da 12500'er tane . düzgün dagılmıs eşit.

plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")   # kaç tane class oldugu
plt.ylabel("Freq")      # number of freq
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

d = X_train[0]
print(d)
print(len(d))

review_len_train = []
review_len_test = []
for i, ii in zip(X_train,X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))  # insanların yaptıkları yorumlardaki kelime sayılarını getirdik
    

# insanların yaptıgı yorumlardaki kelime sayılarının dagılımına bakıyoruz    
sns.distplot(review_len_train, hist_kws={"alpha":0.3})     # hist kws ile saydamlık-oposite ekliyoruz - plotun görünürlüğü demek
sns.distplot(review_len_test, hist_kws={"alpha":0.3})    # ikisini de 0.3 verdik üst üste cizsin diye
# üstteki histograma bakınca pozitif skewnesslik gördük. sağa dogru yatık kuyrugu.

print("Train mean: ",np.mean(review_len_train))
print("Train median: ",np.median(review_len_train))     # iki plotta aynı oldugu için birinin mean medyana baksak yeter.(traine baktık)
print("Train mode: ",stats.mode(review_len_train))

# keras kütüphanesinde kullanacağımız neural network parametreleri fix olmak zorunda. ama burada bir dagılım var fixlik yok. yorumlardaki kelime sayıları aynı değil. ortalama mod değerini kullanmak mantıklı olur. o yüzden bunlara bakıyoruz.


# number of words
word_index = imdb.get_word_index()   # her kelimenin bir int karşılığı var. o yüzden bunu kullanıyoruz.
print(type(word_index))      # mesela 1'de the var en çok kulanılan oymuş.
print(len(word_index))   # 88584 müş unique kelime sayısı

for keys,values in word_index.items():     # wordiindexin tipi dict oldugu için itemsle çagırıyoruz
    if values ==1:      # 1.harf 'the' ydi. onu yazdı.
        print(keys)

# d içinde depolanan reviewi gerçek bir texte çevirelim.
def whatItSay(index=24):
    reverse_index = dict([(value,key) for (key,value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i-3,"!") for i in X_train[index]])    # xtraindeki indexleri gezip !'leri get ediyorum
    print(decode_review)
    print(Y_train[index])   # görüşün olumlu olup olmadıgına bakıyoruz.
    return decode_review

decoded_review = whatItSay(5)

# %% Preprocess

num_words = 15000   # unique kelime sayısı 88k küsürdü 15k'ye aldık.
(X_train,Y_train) , (X_test,Y_test) = imdb.load_data(num_words=num_words)

# reviewlerimiz farklı boyutlardaydı. bunun içinde pathing işlemi yapıcaz.
maxlen = 130    # reviewdeki kelime sayısını sınırlıyoruz mean değeri bu oldugu için buna göre.
X_train = pad_sequences(X_train,maxlen=maxlen)
X_test = pad_sequences(X_test,maxlen=maxlen)

print(X_train[5])   # 130'a sabitlemiş mi diye bakıyoruz. kelimelerin indexleri yazıyor. 130 a tamamlamak içinde eksik kadar 0 ekliyor.
for i in X_train[0:10]:
    print(len(i))         # hepsi 130 olmus.

decoded_review = whatItSay(21)

# verisetimizi train edilebilir hale getirdik PREPROCESS ile.


# %% RNN - recurrent neural network yapısını inşa edicez

rnn = Sequential()    # bir sequential yapım olacak ve dizimin içine gerekli layer'ları tek tek eklicez
rnn.add(Embedding(num_words, 32, input_length=len(X_train[0])))    # embedding = int'leri belirli boyutlarda yoğunluk vektörlerine çevirmemize yarayan bir yapı.  # inputdimension # outputdimension# inputlength herhangi indexinin uzunlugu dedik hepsi 130 zaten
rnn.add(SimpleRNN(16,input_shape=(num_words,maxlen),return_sequences=False, activation="relu"))   # sequential yapıma simple rnn modelini ekliyorum # outputspace boyutu=16  # 15000 kelime olcak max uzunluk 130 olcak
rnn.add(Dense(1))     # rnn'e sınıflandırma işlemi gerçekleştirebilmek için bi tane flatten değeri eklememiz lazım,dense ile yaptık.
rnn.add(Activation("sigmoid"))    #rnn yapımıza bi tane de aktivasyon fonksiyonu ekliyoruz sınıflandırma yapabilmek için
# aktivasyon fonks. sigmoid kullanma sebebimiz BINARY CLASSIFICATION yapabilmek için.
print(rnn.summary())    # sumamry = layerların, parametrelerin sayısını print ettiricek yapı.
rnn.compile(loss="binary_crossentropy", optimizer= "rmsprop", metrics= ["accuracy"])

# rnn modelimizi kurduk. Birazdan fit edicez.

# %% RNN Fitting

history = rnn.fit(X_train,Y_train,validation_data=(X_test,Y_test), epochs=5, batch_size=128, verbose=1) # en başarılı oldugumuz sonuncu. yüzde 92'ye yüzde 85 acc. Hafif bi overfit var ama sonuclar iyi.

# %% Evaluate Result

score = rnn.evaluate(X_test,Y_test)
print("Accuracy: %",score[1]*100)

plt.figure()
plt.plot(history.history["accuracy"], label="Train")  # history.history kütüphanesindeki acc degerini alıyoruz
plt.plot(history.history["val_accuracy"], label="Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()     # en iyi değeri 1.5'a yakın bi epochda elde etmişiz. yüzde 84e yakın bi acc

plt.figure()
plt.plot(history.history["loss"], label="Train") 
plt.plot(history.history["val_loss"], label="Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()  































































