# Laporan Proyek Machine Learning - Amanda Riyas Utami

## Domain Proyek

Tiket pesawat sering kali berubah ubah, seseorang mengecek tiket pesawat dengan pesawat yang sama namun dengan berbeda waktu landing sering kali menyebabkan perbedaan harga tiket pesawat padahal tujuannya sama. Perbedaan harga tiket pesawat ini membuat banyak orang merasa bingung mengenai kapan sebaiknya membeli tiket pesawat. Sebab banyak orang menginginkan bisa mendapat harga lebih murah tapi dengan kualitas dan pelayanan yang sama. Masalah ini bisa diselesaikan dengan suatu pemodelan salah satunnya pemodelan dalam machine learning seperti menggunakan model KNN, Random Forest, ataupun Boosting Algorithm, namun permasalahannya kita juga tidak tahu dimana model terbaik dari ketiga model tersebut.  Pada sebuah penelitian sebelumnya, prediksi harga tiket pesawat dengan menggunakan Logistic Regression, Random Forest, dan Gradient  Boosting diperoleh hasil Random  Forest lebih  baik  dari  model Logistic  Regression dan Gradient  Boosting(Zebua et al., 2022). Dan pada anlisis prediksi ini nanti akan mencoba menggunakan KNN, Random Forest, dan Boosting Algorithm untuk mengetahui model terbaik mana 

## Business Understanding

Setiap orang selalu menginginkan suatu hal dengan harga yang murah namun dengan kualitas yang baik. Seperti ketika mereka membeli tiket pesawat, mereka akan mencoba menganalisis harga tiket peswat yang sesuai dengan keinginan mereka dengan mempertimbangkan berbagai aspek untuk mendapatkan pengalaman penerbangan yang baik. Hal ini juga menjadi suatu petimbangan maskapai dalam menentukan harga tiket pesawat agar konsumen yang diperoleh bisa semaksimal mungkin. Misalkan saja, harga tiket pesawat dari Yogyakarta ke Kalimantan pada hari Minggu sebesar Rp 1.000.000,00 namun pada hari Rabu tiket pesawat dengan tujuan yang sama dan pesawat yang sama harganya hanya Rp 950.000,00. Tentu saja ini akan menjadi suatu pertimbangan konsumen ketika membeli tiket pesawat.

Dalam bisnis tentu saja akan berusaha semaksimal mungkin untuk mendapat profit terbesar. Oleh karena itu, penting bagi perusahaan untuk mengetahui dan dapat memprediksi harga tiket pesawar di pasar. Prediksi akan digunakan untuk menentukan harga tiket pesawat yang terbaik agar perusahaan dapat memperoleh profit semaksimal mungkin.

### Problem Statements

Berapa harga pasar tiket pesawat dengan dipengaruhi beberapa variabel penentu?

### Goals

Mengetahui fitur yang paling berkorelasi dengan harga tiket pesawat dan membuat model machine learning yang dapat memprediksi harga tiket pesawat seakurat mungkin berdasarkan kriteria yang ada. Solusi dari permasalahan ini dapat diselesaikan menggunakan regresi dengan menguji beberapa pemodelan seperti KNN, Random Forest, dan Boosting Algorithm untuk mencari tahu model terbaik mana yang cocok digunakan untuk memprediksi harga tiket pesawat, nantinya juga akan dilakukan tuning untuk mendapatkan prediksi yang lebih baik. Metrik digunakan untuk mengevaluasi seberapa baik model dalam memprediksi harga. Metrik yang digunakan adalah Mean Squared Error (MSE) atau Root Mean Square Error (RMSE) untuk mengukur seberapa jauh hasil prediksi dengan nilai yang sebenarnya pada bagian model evaluasi.

## Data Understanding
Data pada proyek ini berasal dari kaggle yang berjudul Flight Price Prediction dengan link https://www.kaggle.com/datasets/viveksharmar/flight-price-data 

### Variabel-variabel pada Flight Price Prediction adalah sebagai berikut:
- Airlines: Nama maskapai penerbangan yang mengoperasikan penerbangan tersebut.
- Source: Kota tempat penerbangan berangkat.
- Destination: Kota tempat mendarat.
- Total Stops: Jumlah pemberhentian yang dilakukan oleh penerbangan.
- Price: Harga tiket untuk masing-masing penerbangan.
- Date, Month, and Year: Tanggal tertentu dimana penerbangan dijadwalkan.
- Departure and Arrival Times: Jam dan menit terperinci untuk keberangkatan dan kedatangan.
- Duration: Total durasi penerbangan dalam jam dan menit.

Untuk memahami seperti apa data yang ada dilakukan visualisasi menggunakan diagram batang.

## Data Preparation
Data preparation sangat penting dilakukan sebelum membuat pemodelan karena untuk mengetahui lebih mendalam mengenai seperti apa data yang digunakan untuk menganalisis suatu permasalahan dan memastikan tidak ada kesalahan sebelum pemodelan agar model yang dihasilkan nanti lebih akurat. Pada bagian Data Preparatin ada beberapa tahapan yang dilakukan yaitu:
1. Exploratory Data Analysis - Deskripsi Variabel
   Pada tahap ini dilakukan pengecekan jumlah data dan variabel. Terdapat 10683 data dengan 14 variabel yang memiliki tipe object dan integer. Kemudian cek apakah ada data kosong dan diperoleh tidak ada data yang kosong. Setelah itu dilakukan pengecekan data yang duplikat dan diperoleh ada 222 data duplikat. Terakhir cek missing value dan diperoleh terdapat missing value pada variabel Duration_min karena tidak mungkin suatu penerbangan berdurasi 0, kemudian pada variabel duration_hours juga terdapat missing value karena terdapat durasi penerbangan yang sangat lama hingga lebih dari 24 jam padahal ada aturan lama penerbangan hanya sampai 24 jam. Dan pada variabel Duration_Stop juga ada missing value karena ada pemberhentian yang lebih dari 2 kali padahal pemberhentian penerbangan hanya diperbolehkan maksimal 2 kali. 
2. Exploratory Data Analysis - Menangani Missing Value dan Outliers
   Karena terdapat beberapa permasalahan pada data maka yang pertama dilakukan penghapusan data duplikat, dan setelah di hapus terdapat sisa data 10461 data. Kemudian dilakukan penghapusan data pada duration_hours yang melebihi 24 jam dan penghapusan data pada Total_Stop yang melebihi 2 kali pemberhentian. Setelah itu karena variabel date, month, dan year terpisah, sebaiknya digabungkan menjadi variabel Date_of_Journey untuk mempermudah proses analisis dan dilakukan penghapusan variabel date, month, year karena sudah digabungkan. Setelah dilakukan penggabungan dilakukan pembuatan fitur turunan untuk mengetahui apakah tanggal tersebut merupakan hari apa dengan dikategorikan menjadi angka 0-6 yang diberi nama Day_of_Week dan kategori 0 atau 1 untuk mengetahui apakah hari tersebut weekend atau tidak dengan variabel Is_weekend. Langkah selanjutnya menghapus Duration_hours yang isinya 0 karena tidak mungkin waktu penerbangan 0. Setelah itu dilakukan pengecekan menggunakan boxplot agar terlihat jelas appakah terdapat outlier atau tidak, dari pengecekan terdapat outlier, untuk penanganannya menggunakan IQR kemudian dilakukan penghapusan dari data yang di luar IQR. Tipe data Total_Stops, Day_of Week, dan Is_Weekend diubah menjadi kategori karena nilai ini merepresentasikan suatu kata. 
3. Exploratory Data Analysis - Univariate Analysis
   Mengelompokkan categirical_features yang terdiri dari variabel Airline, Source, Destination, Total_Stops, Day_of_Week, Is_Weekend dan numerical_featues yang terdiri dari variabel Dep_hours, Dep_min, Arrival_hours, Arrival_min, Duration_hours, Duration_min untuk mempermudah dalam penganalisisan data nantinya. Kemudian dibuat visualisasi data berupa diagram batang dari categirical_features yang terdapat informasi jumlah sampel dan persentase. Pada numerical_featues juga dibuat visualisasi menggunakan diagram batang dan boxplot numerical_featues terhadap price.
4.  Exploratory Data Analysis - Multivariate Analysis
   Pada tahap ini dilakukan pembuatan visualisasi untuk mengetahui pengaruh variabel-variabel dalam categirical_features dengan price dan untuk variabel-variabel numerical_featues dibuat matriks korelasi untuk mengetahui korelasi dari setiap variabel.
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/correlation_matrix.png" width="500"/>
Dari matriks korelasi, diperoleh korelasi terkuat pada variabel Duration_hours dengan Price.
6. Encoding Fitur Kategori
   Karena terdapat beberapa variabel kategori maka diperlukan adanya encoding untuk merubah data kategorik ke data numerik.
7. Train-Test-Split
   Membagi data menjadi data train dan test. Dari output diperoleh:
   - total of sample in whole dataset: 9487
   - total of sample in train dataset: 8538
   - total of sample in test dataset: 949
8. Standarisasi
   Standarisasi sangat penting untuk menyamakan skala fitur yang ada. 

## Modeling
Ada 3 model yang dicoba untuk memprediksi harga tiket pesawat yaitu:
1. K-Nearest Neighbor
   K-Nearest Neighbor atau yang biasa disebut dengan KNN menggunakan algoritma‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai curse of dimensionality (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data.
2. Random Forest
   Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Namun model ini kurang baik untuk data yang sangat spars (jarang) atau high-dimensional, kurang interpretatif, lambat untuk prediksi, dan cenderung overfit pada data kecil
3. Boosting Algorithm
   Boosting Algorith bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest. Namun model ini memiliki kelemahan yaitu lambat dalam pelatihan, sangat sensitif terhadap outlier, butuh banyak tuning, sulit diinterpretasikan, dan membutuhkan memori yang cukup besar.

Pada pemodelan ini nantinya akan dilakukan tuning untuk mendapat akurasi yang lebih baik dan natinya juga akan dipilih salah satu model yang paling baik dalam memprediksi harga tiket pesawat.

## Evaluation
1. Model KNN
   Dari evaluasi model diperoleh hasil pembagian train dan test cukup stabil namun nilai MSE relatif tinggi. Dari prediksi diperoleh hasil bahwa model KNN meng-overestimate nilai aktual sebesar ~4.1 ribu, kesalahan masih besar, tetapi lebih kecil dibanding Random forest dan boosting algorithm.
2. Model Random Forest
   Nilai MSE train dan test paling rendah diantara model yang lain. Namun dari prediksi dengan kenyataan model ini sangat jauh dari nilai aktual dibandingkan model KNN dan boosting algorithm.
3. Boosting Algorithm
   Nilai MSE train dan test sangat tinggi, hal ini  mungkin dikarenakan model kurang belajar dengan baik (underfitting). Dan dari prediksi dengan nilai aktual model ini juga overestimate.

Berikut visualisasi metrik evaluasi dari ketiga model:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/plot_evaluasimetrik1.png" width="500"/>

Dari ketiga model tersebut masih buruk dalam memprediksi, oleh karena itu dilakukan tuning untuk mendapatkan model yang lebih baik.

Evaluasi setelah tuning:
1. K-Nearest Neighbor
   Model KNN cukup baik dalam mempelajari data training (R² > 0.7), performanya cukup stabil saat diuji di data test. Nilai MSE dan MAE menunjukkan kesalahan prediksi yang moderat. Tidak terjadi overfitting yang besar, meskipun ada penurunan performa dari train ke test. KNN juga memberikan prediksi paling dekat dengan nilai aktual pada sampel ini.
2. Random Forest
   Memiliki performa terbaik secara keseluruhan karena tidak overfit, nilai MSE dan MAE paling rendah di test set sehingga prediksi paling akurat dan stabil di antara ketiga model. Untuk nilai prediksinya masih cukup jauh dengan nilai aktual.
3. Boosting Algorithm
   Menunjukkan performa terburuk karena  model tidak cukup baik menjelaskan variasi target serta nilai MSE dan MAE tertinggi. Untuk nilai prediksinya masih cukup jauh dengan nilai aktual.

Berikut visualisasi evaluasi dari metrik evaluasi ketiga model
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/mse_plot.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/r2_plot.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/mae_plot.png" width="500"/>


Dari evaluasi ketiga metode tersebut jika hanya melihat 1 data point mungkin KNN lebih baik, namun jika mempertimbangkan performa keseluruhan model Random Forest lebih baik karena R² tertinggi (akurasi global terbaik), MAE dan MSE terendah pada test set, dan stabil di train-test atau tidak overfit. Oleh karen itu dapat disimpulkan bahwa random forest adalah model terbaik untuk memprediksi harga tiket pesawat.


_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

