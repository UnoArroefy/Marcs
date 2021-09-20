# Marcs
Smart-door-lock-system

menjalankan file

1. install dependencies/packages\
    ^numpy\
    ^imutils\
    ^dlib\
    ^opencv-python\
    ^keras
2. cek keberadaan files & mengatur path\
    ^model masker (model8.h5) beserta file haarcascade\
    ^centroidtracker.py dan trackable object.py\
    ^video yang digunakan untuk people counter
3. set limit, dengan mengganti nilai variabel limit, default yang digunakan 10
4. pastikan cahaya ruangan mencukupi, untuk memudahkan mendeteksi wajah
5. jika model masker salah prediksi, maka itu diantara 1 cahaya yang menyorot ke wajah terlalu terang, atau cahaya terlalu gelap
6. file yang dijalankan merupakan file bernama door-lock-system dengan ekstensi .ipynb atau .py
7. beberapa error yang sering terjadi karena lalai, saat saya test terjadi karena
    1. file video corrupt
    2. file path yang salah
    3. webcam sedang digunakan app lain
    4. bermasalah dengan menginstall package dlib/ lupa install

Note : untuk datasets kami menggunakan datasets dari Prajna 
dan menambah beberapa lagi, untuk lebih lengkap dataset serta file
model masker ada di github.

Github : [model masker](https://github.com/UnoArroefy/Face-mask-detection)
