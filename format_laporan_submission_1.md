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
   Karena terdapat beberapa permasalahan pada data maka yang pertama dilakukan penghapusan data duplikat, dan setelah di hapus terdapat sisa data 10461 data. Kemudian dilakukan penghapusan data pada duration_hours yang melebihi 24 jam dan penghapusan data pada Total_Stop yang melebihi 2 kali pemberhentian. Setelah itu karena variabel date, month, dan year terpisah, sebaiknya digabungkan menjadi variabel Date_of_Journey untuk mempermudah proses analisis dan dilakukan penghapusan variabel date, month, year karena sudah digabungkan. Setelah dilakukan penggabungan dilakukan pembuatan fitur turunan untuk mengetahui apakah tanggal tersebut merupakan hari apa dengan dikategorikan menjadi angka 0-6 yang diberi nama Day_of_Week dan kategori 0 atau 1 untuk mengetahui apakah hari tersebut weekend atau tidak dengan variabel Is_weekend. Tipe data Total_Stops, Day_of Week, dan Is_Weekend diubah menjadi kategori karena nilai ini merepresentasikan suatu kata.
3. Exploratory Data Analysis - Univariate Analysis
   Mengelompokkan categirical_features yang terdiri dari variabel Airline, Source, Destination, Total_Stops, Day_of_Week, Is_Weekend dan numerical_featues yang terdiri dari variabel Dep_hours, Dep_min, Arrival_hours, Arrival_min, Duration_hours, Duration_min untuk mempermudah dalam penganalisisan data nantinya. Kemudian dibuat visualisasi data berupa diagram batang dari categirical_features yang terdapat informasi jumlah sampel dan persentase. Pada numerical_featues juga dibuat visualisasi menggunakan diagram batang dan boxplot numerical_featues terhadap price.
4.  Exploratory Data Analysis - Multivariate Analysis
   Pada tahap ini dilakukan pembuatan visualisasi untuk mengetahui pengaruh variabel-variabel dalam categirical_features dengan price dan untuk variabel-variabel numerical_featues dibuat matriks korelasi untuk mengetahui korelasi dari setiap variabel. Dari matriks korelasi, diperoleh korelasi terkuat pada variabel Duration_hours dengan Price.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

