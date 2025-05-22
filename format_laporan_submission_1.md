# Laporan Proyek Machine Learning - Amanda Riyas Utami

## Domain Proyek

Tiket pesawat sering kali berubah ubah, seseorang mengecek tiket pesawat dengan pesawat yang sama namun dengan berbeda waktu landing sering kali menyebabkan perbedaan harga tiket pesawat padahal tujuannya sama. Perbedaan harga tiket pesawat ini membuat banyak orang merasa bingung mengenai kapan sebaiknya membeli tiket pesawat. Sebab banyak orang menginginkan bisa mendapat harga lebih murah tapi dengan kualitas dan pelayanan yang sama. Masalah ini bisa diselesaikan dengan suatu pemodelan salah satunnya pemodelan dalam machine learning seperti menggunakan model KNN, Random Forest, ataupun Boosting Algorithm, namun permasalahannya kita juga tidak tahu dimana model terbaik dari ketiga model tersebut.  Pada sebuah penelitian sebelumnya, prediksi harga tiket pesawat dengan menggunakan Logistic Regression, Random Forest, dan Gradient  Boosting diperoleh hasil Random  Forest lebih  baik  dari  model Logistic  Regression dan Gradient  Boosting(Zebua et al., 2022). Dan pada anlisis prediksi ini nanti akan mencoba menggunakan KNN, Random Forest, dan Boosting Algorithm untuk mengetahui model terbaik mana 

## Business Understanding

Setiap orang selalu menginginkan suatu hal dengan harga yang murah namun dengan kualitas yang baik. Seperti ketika mereka membeli tiket pesawat, mereka akan mencoba menganalisis harga tiket peswat yang sesuai dengan keinginan mereka dengan mempertimbangkan berbagai aspek untuk mendapatkan pengalaman penerbangan yang baik. Hal ini juga menjadi suatu petimbangan maskapai dalam menentukan harga tiket pesawat agar konsumen yang diperoleh bisa semaksimal mungkin. Misalkan saja, harga tiket pesawat dari Yogyakarta ke Kalimantan pada hari Minggu sebesar Rp 1.000.000,00 namun pada hari Rabu tiket pesawat dengan tujuan yang sama dan pesawat yang sama harganya hanya Rp 950.000,00. Tentu saja ini akan menjadi suatu pertimbangan konsumen ketika membeli tiket pesawat.

Dalam bisnis tentu saja akan berusaha semaksimal mungkin untuk mendapat profit terbesar. Oleh karena itu, penting bagi perusahaan untuk mengetahui dan dapat memprediksi harga tiket pesawar di pasar. Prediksi akan digunakan untuk menentukan harga tiket pesawat yang terbaik agar perusahaan dapat memperoleh profit semaksimal mungkin.

### Problem Statements

Berapa harga pasar tiket pesawat dengan dipengaruhi beberapa variabel penentu?

### Goals
- Mengetahui variabel yang paling berkorelasi dengan harga tiket pesawat
- Memprediksi harga tiket pesawat

## Solution Statements
- Mengunakan regresi dalam pemecahan masalah
- Menerapkan beberapa pemodelan K-Nearest Neighbor, Random Forest, dan Boosting Algorithm untuk mencari tahu model terbaiknya
- Menggunakan metrik evaluasi Mean Squared Error (MSE) atau Root Mean Square Error (RMSE)

## Data Understanding
Data pada proyek ini berasal dari kaggle yang berjudul Flight Price Prediction dengan link https://www.kaggle.com/datasets/viveksharmar/flight-price-data 

### Variabel-variabel pada Flight Price Prediction adalah sebagai berikut:
- Airlines: Nama maskapai penerbangan yang mengoperasikan penerbangan tersebut.
- Source: Kota tempat penerbangan berangkat.
- Destination: Kota tempat mendarat.
- Total Stops: Jumlah pemberhentian yang dilakukan oleh penerbangan.
- Price: Harga tiket untuk masing-masing penerbangan.
- Date : Tanggal dimana penerbangan dijadwalkan.
- Month : Bulan dimanapenerbangan dijadwalkan
- Year: Tahun dimana penerbangan dijadwalkan.
- Dep_hours : Jam keberangkatan pesawat.
- Dep_min : Menit pada jam keberangkatan pesawat.
- Arrival_hours : Jam penerbangan tiba
- Arrival_min : Menit pada jam saat pesawat tiba
- Duration_hours : Durasi penerbangan dalam jam.
- Duration_min : Durasi penerbangan dalam menit

Untuk memahami seperti apa data yang digunakan maka dilakukan data understanding dengan beberapa tahapan yaitu:

1. Exploratory Data Analysis - Deskripsi Variabel
   Pada tahap ini dilakukan pengecekan jumlah data dan variabel. Terdapat 10683 data dengan 14 variabel yang memiliki tipe object dan integer. Variabel tersebut terdiri dari variabel Airline (object), Source (object), Destination (object), Total_Stops (int), Price (int), Date (int), Month (int), Year (int), Dep_hours(int), Dep_min (int), Arrival_hours (int), Arrival_min (int), Duration_hours (int), Duration_min (int). Kemudian cek apakah ada data kosong dan diperoleh tidak ada data yang kosong. Setelah itu dilakukan pengecekan data yang duplikat dan diperoleh ada 222 data duplikat. Terakhir cek missing value dan diperoleh terdapat missing value pada variabel duration_hours karena terdapat durasi penerbangan yang sangat lama hingga lebih dari 24 jam padahal ada aturan lama penerbangan hanya sampai 24 jam. Dan pada variabel Duration_Stop juga ada missing value karena ada pemberhentian yang lebih dari 2 kali padahal pemberhentian penerbangan hanya diperbolehkan maksimal 2 kali.

2. Exploratory Data Analysis - Menangani Missing Value dan Outliers
   Karena terdapat beberapa permasalahan pada data maka yang pertama dilakukan pengecaekan data pada duration_hours yang melebihi 24 jam dan dan menghapus penghapusan data yang isinya melebihi 24 jam. Kemudian dilakukan pengecekan pada variabel Total_Stop yang melebihi 2 kali pemberhentian dan diperoleh data yang terdapat 3 dan 4 kali pemberhentian padahal maksimal pemberhentian pesawat hanya 2 kali sehingga dilakukan penghapusan data pada data yang memiliki total pemberhentian lebih dari 2. Kemudian dilakukan pengecekan pada Duration_hours dan Duration_min untuk meyakinkan bahwa tidak ada data penerbangan pesawat yang isinya 0 semua. Setelah itu dilakukan pengecekan menggunakan boxplot agar terlihat jelas appakah terdapat outlier atau tidak pada price dan ditemukan ada beberapa outlier seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/boxplot%20dengan%20outlier.png" width="500"/>
Dan dilakukan penanganan menggunakan IQR sehingga diperoleh data yang lebih bersih dengan visualisasi seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/boxplot%20setelah%20pembersihan%20outlier.png" width="500"/>

3. Exploratory Data Analysis - Univariate Analysis
   Mengelompokkan categorical_features yang terdiri dari variabel (Airline, Source, Destination) dan numerical_featues yang terdiri dari variabel (Total_Stops, Date, Month, Year, Dep_hours, Dep_min, Arrival_hours, Arrival_min, Duration_hours, Duration_min) dengan target Price untuk mempermudah dalam penganalisisan data nantinya. Kemudian dibuat visualisasi data berupa diagram batang dari categirical_features seprti gambar di bawah ini untuk melihat seperti apa data yang ada:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Diagram%20categorical%20Airline.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/diagram%20categorical%20source.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/diagram%20categorical%20Destination.png" width="500"/>

Kemudian untuk numerical_featues dibuat diagram batang seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/diagram%20batang%20numerical%20features.png" width="500"/>

Dan dilihat juga distribusi harga tiket per maskapai dengan data tanpa outlier dan diperoleh visualisasi seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/boxplot%20price%20tanpa%20outlier.png" width="500"/>

4.  Exploratory Data Analysis - Multivariate Analysis
   Pada tahap ini dilakukan pembuatan visualisasi untuk mengetahui pengaruh variabel-variabel dalam categirical_features dengan price.
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/price%20terhadap%20airline.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/price%20terhadap%20source.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/price%20terhadap%20destination.png" width="500"/>
Untuk variabel-variabel numerical_featues dibuat matriks korelasi untuk mengetahui korelasi dari setiap variabel. Berikut matriks korelasinya:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/matriks%20korelasi%20terbaru.png" width="500"/>
Dari matriks korelasi, diperoleh korelasi terkuat pada variabel Duration_hours dengan Total_Stops sebesar 0,76

## Data Preparation
Data preparation sangat penting dilakukan sebelum membuat pemodelan karena untuk mengetahui lebih mendalam mengenai seperti apa data yang digunakan untuk menganalisis suatu permasalahan dan memastikan tidak ada kesalahan sebelum pemodelan agar model yang dihasilkan nanti lebih akurat. Pada bagian Data Preparation ada beberapa tahapan yang dilakukan yaitu:

1. Menghapus Duplikasi Data
   Setelah penghapusan data diperoleh sisa data sebanyak 9579 data.
2. Menggabungkan variabel date, month, dan year menjadi variabel Date_of_Journey dan membuat fitur turunan yaitu variabel Day_of_week dan Is_weekend. Setelah membuat fitur turunan variabel date, month, year, dan Date_of_Journey dihapus karena sudah tidak digunakan.
3. Menggabungkan variabel Duration_hours dan Duration_min kemudian membuat fitur turunan dengan nama Duration_total_min atau bentuk dari jam dan menit yang sudah dibuah ke menit semua untuk mepermudah dalam menganalisis total durasi penerbangan. Setelah itu hapus variabel Duration_hours dan Duration_min karena sudah tidak digunakan.
4. Menggabungkan variabel Arrival_hours dan Arrival_min kemudian membuat fitur turunan dengan nama Arrival_total_min atau bentuk dari jam dan menit yang sudah dibuah ke menit semua untuk mepermudah dalam menganalisis waktu pesawat tiba. Setelah itu hapus variabel Arrival_hours dan Arrival_min karena sudah tidak digunakan.
5. Menggabungkan variabel Dep_hours dan Dep_min kemudian membuat fitur turunan dengan nama Dep_total_min atau bentuk dari jam dan menit yang sudah dibuah ke menit semua untuk mepermudah dalam menganalisis waktu pesawat berangkat. Setelah itu hapus variabel Dep_hours dan Dep_min karena sudah tidak digunakan.
6. Hasil data setelah preparation terdiri dari 9579 data dengan 10 variabel yaitu variabel Airline (object), Souce (object), Destination (object), Total_Stops (int), Price (int), Day_of_week (int), Is_weekend (int), Duration_total_min (int), Arrival_total_min (int), Dep_total_min (int)
7. Encoding Fitur Kategori
Karena terdapat beberapa variabel kategori maka diperlukan adanya encoding untuk merubah data kategorik ke data numerik.
8. Train-Test-Split
   Membagi data menjadi data train dan test. Dari output diperoleh:
   - total of sample in whole dataset: 9487
   - total of sample in train dataset: 8538
   - total of sample in test dataset: 949
9. Standarisasi
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
   Dari evaluasi model diperoleh hasil train MSE 4265 yang artinya cukup besar sehingga KNN tidak terlalu overfit tapi juga tidak sangat presisi di training. Kemudian untuk nilai test MSE diperoleh nilai 4727, nilai ini cukup dekat dengan train MSE yang menunjukkan performa cukup konsisten.
2. Model Random Forest
   Nilai MSE train model ini sebesar 4176 dan nilai MSE test sebesar 4176, kedua nilai ini paling rendah diantara model yang lain. Nilai train MSE yang kecil ini menunjukkan model sangat baik dalam mempelajari data latih. Kemudian nilai MSE yang juga kecil pada model ini menunjukkan performa yang paling baik secara keseluruhan. Sehingga model ini paling stabil dan direkomendasikan.
3. Boosting Algorithm
   Nilai MSE train sebesar 6529, nilai ini menjadi yang terbesar diantara model lainnya sehingga model tidak terlalu presisi di data latih. Kemudian nilai test MSE sebesar 6123, nilai ini juga sangat besar yang menunjukkan model tidak belajar dengan baik secara umum. Data ini jika menggunakan boosting algorith konfigurasinya underfit atau kurang optimal. 

Berikut metrik evaluasi dari ketiga model:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Metrik%20Evaluasi.png" width="500"/>
Dan bentuk plot dari matrik evaluasi yaitu sebagai berikut:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/evaluasi%20model.png" width="500"/>
Pada perbandingan nilai prediksi antara ketiga model diperoleh nilai seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Aktual%20dan%20prediksi%20sebelum%20tuning.png" width="500"/>
Prediksi KNN paling dekat dengan nilai aktual, namun perbedaan nilai prediksi dan nilai aktual masih sangat jauh, oleh karena itu dilakukan tuning untuk mendapatkan model yang lebih baik.

Evaluasi setelah tuning:
1. K-Nearest Neighbor
   Nilai train MSE pada model K-Nearest Neighbor sebesar 4,13 dan nilai test MSE sebesar 4,57. Nilai MSE pada KNN lebih tinggi dari random forest. Perbedaan train dan test kecil yang menandakan model konsisten tapi kurang akurat. Sehingga performa KNN sedang, model ini mungkin cocok untuk model yang simpel tapi cukup stabil.
2. Random Forest
   Nilai train MSE sebesar 2,98 dan test MSE sebesar 3,91, kedua nilai ini menjadi nilai yang paling rendah dari model lainnya. Nilai train MSE yang rendah menunjukkan model belajar sangat baik dari data latih. Dan nilai test MSE yang paling rendah menunjukkan generalisasi yang baik ke data baru. Perbedaan nilai train dan test yang tidak terlalu besar memiliki arti stabil dan tidak overfit. Sehingga Random Forest menjadi model terbaik dari nilai MSE.
3. Boosting Algorithm
   Nilai train MSE sebesar 6,42, nilai ini menjadi yang paling besar dari model lainnya artinya model tidk berhasil mempelajari data dengan baik. 
Kemudian nilai test MSE sebesar 5,97 , nilai ini juga menjadi yang paling tinggi diantara model yang lain, nilai yang tinggi ini menunjukkan performa buruk pada data uji. Bisa jadi undefitting atau model tidak cocok dengan karakteristik data. Sehingga model boosting algorithm menjadi yang paling buruk dari model yang lain.
Berikut metrik evaluasi ketiga model
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Metrik%20Evaluasi%20Setelah%20Tuning.png" width="500"/>
Visualisasi grafik MSE setelah tuning:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/MSE%20model%20setelah%20tuning.png" width="500"/>
Pada perbandingan nilai prediksi antara ketiga model setelah tuning diperoleh nilai seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Aktual%20dan%20prediksi%20setelah%20tuning.png" width="500"/>

Prediksi Random Forest paling dekat dengan nilai aktual, dimana nilai prediksi random forest sebesar 6967,8 dan nilai aktualnya 7438

Dari evaluasi ketiga metode tersebut Random forest menjadi model terbaik yang dapat digunakan untuk memprediksi harga tiket pesawat. 
