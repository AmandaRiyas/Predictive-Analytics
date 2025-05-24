# Laporan Proyek Machine Learning - Amanda Riyas Utami

## Domain Proyek

Proyek ini berfokus pada penerapan machine learning untuk memprediksi harga tiket pesawat berdasarkan data historis yang tersedia secara publik. Industri penerbangan merupakan salah satu sektor dengan dinamika harga yang sangat tinggi, sehingga menjadi tantangan menarik dalam pengembangan model prediktif yang andal.
Dataset yang digunakan mencakup berbagai fitur seperti maskapai penerbangan, asal dan tujuan, durasi penerbangan, waktu keberangkatan dan kedatangan, serta jumlah transit. Fitur-fitur ini dinilai relevan karena secara logis memiliki pengaruh terhadap fluktuasi harga tiket.
Pada penelitian sebelumnya, prediksi harga tiket pesawat menggunakan Logistic Regression, Random Forest, dan Gradient Boosting menunjukkan bahwa Random Forest menghasilkan performa lebih baik dibanding dua model lainnya (Zebua et al., 2022). Berdasarkan hal tersebut, proyek ini mengimplementasikan tiga algoritma regresi K-Nearest Neighbors (KNN), Random Forest, dan Boosting Algorithm karena masing-masing memiliki keunggulan dalam menangani data.
Evaluasi performa model dilakukan menggunakan metrik Mean Squared Error (MSE), dan proses tuning dilakukan dengan pendekatan GridSearchCV dan RandomizedSearchCV untuk mengoptimalkan hasil prediksi.
Masalah ini harus diselesaikan karena ketidakakuratan prediksi harga tiket berdampak pada strategi bisnis perusahaan dan keputusan pembelian konsumen. Dengan model prediktif yang baik, perusahaan dapat mengoptimalkan penentuan harga dinamis, sementara konsumen dapat membuat keputusan pembelian yang lebih tepat waktu dan ekonomis.

## Business Understanding
Harga tiket pesawat memiliki fluktuasi yang tinggi karena dipengaruhi oleh beberapa faktor seperti maskapai pesawat yang digunakan, lokasi penerbangan, lokasi tujuan, durasi penerbangan, waktu keberangkatan, waktu tiba, dan total transit. Ketidakmampuan perusahaan untuk memprediksi harga secara akurat dapat menghambat pengambilan keputusan strategis seperti penentuan harga. Oleh karena itu, diperlukan model prediktif yang mampu memperkirakan harga tiket pesawat secara tepat berdasarkan pola data historis.

### Problem Statements
- Fluktuasi tinggi pada harga tiket pesawat menyulitkan perusahaan dalam menetapkan harga tiket pesawat karena ketidaktepatan dalam memprediksi harga dapat menghambat strategi harga dinamis.
- Perusahaan membutuhkan model prediktif berbasis data historis yang dapat membantu memproyeksikan harga tiket pesawat secara akurat.

### Goals
- Mengembangkan model prediksi harga tiket pesawat berdasarkan data historis dan variabel variabel relevan seperti maskapai pesawat yang digunakan, lokasi penerbangan, lokasi tujuan, durasi penerbangan, waktu keberangkatan, waktu tiba, dan total transit.
- Membandingkan performa beberapa algoritma machine learning regresi seperti K-Neares Neighbors (KNN), Random Forest, dan Boosting Algorithm untuk mendapat model dengan akurasi terbaik.

## Solution Statements
- Menerapkan beberapa pemodelan K-Nearest Neighbor, Random Forest, dan Boosting Algorithm untuk mencari tahu model terbaiknya dengan menambahkan evaluasi performa menggunakan Mean Squared Error (MSE)
- Melakuakan hyperparameter tuning menggunakan GridSearchCV dan RandomizedSearchCV
- Membandingkan performa dari model K-Nearest Neighbor, Random Forest, dan Boosting Algorithm.

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
   Karena terdapat beberapa permasalahan pada data maka yang pertama dilakukan pengecekan data pada duration_hours yang melebihi 24 jam dan dan menghapus penghapusan data yang isinya melebihi 24 jam. Kemudian dilakukan pengecekan pada variabel Total_Stop yang melebihi 2 kali pemberhentian dan diperoleh data yang terdapat 3 dan 4 kali pemberhentian padahal maksimal pemberhentian pesawat hanya 2 kali sehingga dilakukan penghapusan data pada data yang memiliki total pemberhentian lebih dari 2. Kemudian dilakukan pengecekan pada Duration_hours dan Duration_min untuk meyakinkan bahwa tidak ada data penerbangan pesawat yang isinya 0 semua. Setelah itu dilakukan pengecekan menggunakan boxplot agar terlihat jelas appakah terdapat outlier atau tidak pada price dan ditemukan ada beberapa outlier seperti di bawah ini:
   
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/boxplot%20dengan%20outlier.png" width="500"/>

Banyaknya outlier pada boxplot ini kemungkinan dikarenakan harga tiket pesawat yang tidak wajar oleh karena itu dilakukan penanganan dengan menghapus outlier menggunakan IQR sehingga diperoleh data yang lebih bersih dengan visualisasi seperti di bawah ini:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/boxplot%20setelah%20pembersihan%20outlier.png" width="500"/>

Setelah dilakukan penghapusan outlier, terdapat masih ada 2 outlier yang sangat dekat dengan batas IQR, nilai ini sudah tidak perlu dihapus karena tidak terlalu berpengaruh untuk penganalisan selanjutnya sebab posisinya sudah sangat dekat dengan IQR.

4. Exploratory Data Analysis - Univariate Analysis
   Mengelompokkan categorical_features yang terdiri dari variabel (Airline, Source, Destination) dan numerical_featues yang terdiri dari variabel (Total_Stops, Date, Month, Year, Dep_hours, Dep_min, Arrival_hours, Arrival_min, Duration_hours, Duration_min) dengan target Price untuk mempermudah dalam penganalisisan data nantinya. Kemudian dibuat visualisasi data berupa diagram batang dari categirical_features seprti gambar di bawah ini untuk melihat seperti apa data yang ada:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Diagram%20categorical%20Airline.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/diagram%20categorical%20source.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/diagram%20categorical%20Destination.png" width="500"/>

- Pada diagram batang distribusi `Airline` terhadap jumlah, maskapai dengan jumlah tertinggi yaitu Jet Airways dengan jumlah sebanyak 3377 dan yang terendah adalah maskapai Trujet dengan jumlah 1.
- Pada diagram batang distribusi `Source`terhadap jumlah, tempat keberangkatan tertinggi yaitu di Delhi dengan jumlah 4059 dan yang terendah di Chennai dengan jumlah 381
- Pada diagram batang distribusi `Destination`terhadap jumlah, tujuan destinasi tertinggi yaitu Cochin dengan jumlah 4059 dan yang terendah di Kolkata dengan jumlah 381.

Kemudian untuk numerical_features dibuat diagram batang seperti di bawah ini:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/diagram%20batang%20numerical%20features.png" width="500"/>

- Dari diagram batang Distribusi `Total_Stops`, jumlah total stop tertinggi yaitu di angka 1 yang artinya pesawat hanya transit 1 kali dan total stop terendah di angka 2 yang artinya pesawat melakukan  2 kali transit
- Dari diagram batang distribusi `Date`, tanggal tertinggi berada di batang ke empat atau sekitar tanggal 8 dan yang terendah berada di batang ke 2 yaitu sekitar tanggal 3
- Dari diagram batang distribusi `month`, bulan dengan jumlah penerbangan tertinggi terjadi pada bulan 6 atau bulan Juni dan yang terendah yaitu pada bulan ke 4 atau bulan April.
- Dari diagram batang distribusi `Year` terlihat hanya ada satu batang yaitu di tahun 2019, yang artinya data hanya pada rentang tahin 2019.
- Dari diagram batang distribusi `Dep_hours` terlihat bahwa jam keberangkatan pesawat tertinggi terjadi pada jam 7 dan yang terendah pada jam 3.
- Dari diagram batang distribusi `Dep_min` terlihat bahwa menit keberangkatan pesawat tertinggi pada menit 0, artinya pesawat lebih sering berangkat di jam yang tepat todak lebih beberapa menit. Dan keberangkatan pesawat terendah di menit 40.
- Dari diagram batang distribusi `Arrival_hours` terlihat bahwa pesawat sering tiba di pukul 19 dan paling jarang tiba di pukul 6.
- Dari diagram batang distribusi `Arrival_min` terlihat bahwa pesawat sering tiba di menit 0 dan paling jarang tiba di menit 55.
- Dari diagram batang distribusi `Duration_hours` terlihat bahwa lama penerbangan paling sering yaitu selama 2 jam dan yang paling jarang yaitu selama 17 jam.
- Dari diagram batang distribusi `Duration_min` terlihat bahwa durasi menit dalam penerbangan paling sering di menit 30 dan yang paling jarang dengan durasi menit 10.

Dan dilihat juga distribusi harga tiket per maskapai dengan data yang tanpa outlier dan diperoleh visualisasi seperti di bawah:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/boxplot%20price%20tanpa%20outlier.png" width="500"/>

Dari boxplot diagram `Airline` terhadap `Price` terlihat bahwa range harga terpanjang pada maskapai Jet Airways dengan menyediakan tiket paling murah hingga paling mahal dan range harga terpendek pada maskapai Trujet.

4.  Exploratory Data Analysis - Multivariate Analysis
   Pada tahap ini dilakukan pembuatan visualisasi untuk mengetahui pengaruh variabel-variabel dalam categirical_features dengan price.
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/price%20terhadap%20airline.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/price%20terhadap%20source.png" width="500"/>
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/price%20terhadap%20destination.png" width="500"/>

- Dari diagram hubungan variabel `Airline` dengan `Price` terlihat bahwa maskapai dengan harga tertinggi yaitu Multiple carriers Premium economy dan yang terendah Trujet.
- Dari diagram hubungan variabel `Source` dengan `Price` terlihat bahwa harga lokasi keberangkatan termahal berada di Delhi dan yang paling murah di Mumbai
- Dari diagram hubungan variabel `Destination` dengan `Price` terlihat bahwa harga lokasi pendaratan termahal berada di New Delhi dan Cochin. Untuk yang termurah berada di Hyderabad

Untuk variabel-variabel numerical_featues dibuat matriks korelasi untuk mengetahui korelasi dari setiap variabel. Berikut matriks korelasinya:
<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/matriks%20korelasi%20terbaru.png" width="500"/>

- Korelasi terkuat pada variabel `Total_Stops` dan `Duration_hours`yaitu sebesar 0,67
- Korelasi terendah pada variabel -0 yaitu pada variabel `Date` dan `Arrival_hours`, variabel `Date` dan `Dep_min`, sertabvariabel `Date` dan `Total_Stops`. Nilai -0 ini artinya tidak ada korelasi sama sekali antar variabel
- Korelasi `Year` tidak terlihat karena isi tahun hanya sama

## Data Preparation
Data preparation sangat penting dilakukan sebelum membuat pemodelan karena untuk mengetahui lebih mendalam mengenai seperti apa data yang digunakan untuk menganalisis suatu permasalahan dan memastikan tidak ada kesalahan sebelum pemodelan agar model yang dihasilkan nanti lebih akurat. Pada bagian Data Preparation ada beberapa tahapan yang dilakukan yaitu:

A. Menghapus Duplikasi Data <br>
   Setelah penghapusan data diperoleh sisa data sebanyak 9579 data.

B. Filter Data<br>
1. Menggabungkan variabel date, month, dan year menjadi variabel Date_of_Journey dan membuat fitur turunan yaitu variabel Day_of_week dan Is_weekend. Setelah membuat fitur turunan variabel date, month, year, dan Date_of_Journey dihapus karena sudah tidak digunakan.
2. Menggabungkan variabel Duration_hours dan Duration_min kemudian membuat fitur turunan dengan nama Duration_total_min atau bentuk dari jam dan menit yang sudah dibuah ke menit semua untuk mepermudah dalam menganalisis total durasi penerbangan. Setelah itu hapus variabel Duration_hours dan Duration_min karena sudah tidak digunakan.
3. Menggabungkan variabel Arrival_hours dan Arrival_min kemudian membuat fitur turunan dengan nama Arrival_total_min atau bentuk dari jam dan menit yang sudah dibuah ke menit semua untuk mepermudah dalam menganalisis waktu pesawat tiba. Setelah itu hapus variabel Arrival_hours dan Arrival_min karena sudah tidak digunakan.
4. Menggabungkan variabel Dep_hours dan Dep_min kemudian membuat fitur turunan dengan nama Dep_total_min atau bentuk dari jam dan menit yang sudah dibuah ke menit semua untuk mepermudah dalam menganalisis waktu pesawat berangkat. Setelah itu hapus variabel Dep_hours dan Dep_min karena sudah tidak digunakan.

Hasil data setelah preparation terdiri dari 9579 data dengan 10 variabel yaitu variabel Airline (object), Souce (object), Destination (object), Total_Stops (int), Price (int), Day_of_week (int), Is_weekend (int), Duration_total_min (int), Arrival_total_min (int), Dep_total_min (int)

C. Encoding Fitur Kategori <br>
Sebelum melakukan encoding dilakuka penddefinisian ulang pada plane_no_outliers karena ada perubahan variabel pada data. Kemudian dilakukan pendefinisian ulang pada categorical_features dan numerical_features. Isi dari categorical_features yaitu Airline, Source, dan Destination. Dan isi pada numerical_features yaitu Total_Stops, Day_of_week, Is_weekend, Duration_total_min, Arrival_total_min, dan Dep_total_min. Setelah itu dilakukan encoding untuk merubah data kategorik ke data numerik.

D. Train-Test-Split <br>
   Membagi data menjadi data train dan test. Dari output diperoleh dari total 9487 data, sebanyak 8538 data untuk train, dan 949 data untuk test.
   
E. Standarisasi <br>
   Standarisasi sangat penting untuk menyamakan skala fitur yang ada. 

## Modeling
Pada tahap ini, dilakukan pemodelan prediksi harga tiket pesawat menggunakan tiga algoritma machine learning, yaitu K-Nearest Neighbors (KNN), Random Forest, dan Boosting. Ketiga model dilatih menggunakan data historis harga tiket pesawat yang telah diproses sebelumnya. Evaluasi awal dilakukan menggunakan metrik Mean Square Error (MSE).

1. K-Nearest Neighbors <br>

   K-Nearest Neighbors atau yang biasa disebut dengan KNN menggunakan algoritma‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar. Permasalahan ini sering disebut sebagai curse of dimensionality (kutukan dimensi). Pada dasarnya, permasalahan ini muncul ketika jumlah sampel meningkat secara eksponensial seiring dengan jumlah dimensi (fitur) pada data.
   Parameter yang digunakan dalam model K-Nearest Neighbors ini yaitu:

- `n_neighbors=10`: Menentukan jumlah tetangga terdekat yang digunakan untuk memprediksi nilai. Model akan melihat 10 titik data terdekat untuk menentukan prediksi harga tiket pesawat. Pemilihan nilai ini dapat mempengaruhi bias dan variansi model.
- `weights='uniform'` (default): Semua tetangga memiliki bobot yang sama dalam proses prediksi.
- `p=2` (default): Menggunakan Euclidean Distance untuk mengukur jarak antara titik data.
- `algorithm='auto'` (default): Scikit-learn akan memilih algoritma terbaik secara otomatis (ball tree, kd tree, atau brute-force) tergantung pada data.

Model ini dilatih pada data latih (`X_train`, `y_train`) dan performanya dievaluasi dengan metrik Mean Squared Error (MSE) untuk data latih dan data uji.

2. Random Forest <br>

   Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. Namun model ini kurang baik untuk data yang sangat spars (jarang) atau high-dimensional, kurang interpretatif, lambat untuk prediksi, dan cenderung overfit pada data kecil.
   Parameter yang di gunakan di model Random Forest ini yaitu:

- `n_estimators=50`: Jumlah pohon (tree) yang digunakan dalam hutan acak. Semakin banyak jumlah pohon, umumnya model semakin stabil, tetapi waktu komputasi juga meningkat. Dalam kasus ini digunakan 50 pohon.
- `max_depth=16`: Batas maksimum kedalaman setiap pohon. Parameter ini digunakan untuk mencegah pohon tumbuh terlalu dalam dan overfitting. Dengan kedalaman 16, model cukup kompleks untuk menangkap pola, tapi tetap terkontrol.
- `random_state=55`: Menjamin hasil yang konsisten setiap kali model dilatih dengan cara mengatur seed generator angka acak.
- `n_jobs=-1`: Menginstruksikan scikit-learn untuk menggunakan semua core CPU yang tersedia saat proses training dan prediksi, sehingga mempercepat komputasi.

Model ini dilatih pada data latih (`X_train`, `y_train`) dan hasil prediksi dievaluasi menggunakan Mean Squared Error (MSE) pada data latih dan data uji.

3. Boosting Algorithm <br>

   Boosting Algorithm bertujuan untuk meningkatkan performa atau akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma ini sangat powerful dalam meningkatkan akurasi prediksi. Algoritma boosting sering mengungguli model yang lebih sederhana seperti logistic regression dan random forest. Namun model ini memiliki kelemahan yaitu lambat dalam pelatihan, sangat sensitif terhadap outlier, butuh banyak tuning, sulit diinterpretasikan, dan membutuhkan memori yang cukup besar.
   Parameter yang digunakan dalam model Boosting Algorithm ini yaitu:

- `learning_rate=0.05`: Parameter ini mengontrol seberapa besar kontribusi setiap regressor lemah terhadap model akhir. Nilai yang lebih kecil membuat proses pelatihan lebih lambat namun bisa menghasilkan performa yang lebih stabil. Nilai 0.05 menunjukkan model akan belajar dengan kecepatan konservatif.
- `random_state=55`: Digunakan untuk menjamin hasil yang reprodusibel dengan menetapkan seed untuk generator angka acak.

Model dilatih pada data latih (`X_train`, `y_train`), dan performanya dievaluasi menggunakan Mean Squared Error (MSE) pada data latih dan data uji.

Pada pemodelan ini nantinya akan dilakukan tuning untuk mendapat akurasi yang lebih baik dan natinya juga akan dipilih salah satu model yang paling baik dalam memprediksi harga tiket pesawat.

## Evaluation
1. Model KNN
   Dari evaluasi model diperoleh hasil train MSE 4265,338616 yang artinya cukup besar sehingga KNN tidak terlalu overfit tapi juga tidak sangat presisi di training. Kemudian untuk nilai test MSE diperoleh nilai 4727,798823 nilai ini cukup dekat dengan train MSE yang menunjukkan performa cukup konsisten.
2. Model Random Forest
   Nilai MSE train model ini sebesar 2327,643793 dan nilai MSE test sebesar 4176,975962, kedua nilai ini paling rendah diantara model yang lain. Nilai train MSE yang kecil ini menunjukkan model sangat baik dalam mempelajari data latih. Kemudian nilai MSE yang juga kecil pada model ini menunjukkan performa yang paling baik secara keseluruhan. Sehingga model ini paling stabil dan direkomendasikan.
3. Boosting Algorithm
   Nilai MSE train sebesar 6529,342032 nilai ini menjadi yang terbesar diantara model lainnya sehingga model tidak terlalu presisi di data latih. Kemudian nilai test MSE sebesar 6123,370864 nilai ini juga sangat besar yang menunjukkan model tidak belajar dengan baik secara umum. Data ini jika menggunakan boosting algorith konfigurasinya underfit atau kurang optimal. 

Berikut metrik evaluasi dari ketiga model:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Metrik%20Evaluasi.png" width="500"/>

Dan bentuk plot dari matrik evaluasi yaitu sebagai berikut:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/evaluasi%20model.png" width="500"/>

Dari evaluasi yang telah dilakukan nilai mse yang terbaik yaitu pada model KNN dan yang terburuk pada Boosting

Pada perbandingan nilai prediksi antara ketiga model diperoleh nilai seperti di bawah ini:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Aktual%20dan%20prediksi%20sebelum%20tuning.png" width="500"/>

Prediksi KNN paling dekat dengan nilai aktual, yaitu nilai prediksinya 12504,6 dan nilai aktualnya 7438, namun perbedaan nilai prediksi dan nilai aktual masih sangat jauh, oleh karena itu dilakukan tuning untuk mendapatkan model yang lebih baik.

Tuning:
1. Pada model K-Nearest Neighbor menggunakan GriSearchCV untuk tuning model dengan parameter n_neighbors [3,5,10,15] sebagai jumlah tetangga terdekat yang digunakan untuk prediksi, weight yaitu uniform yang artinya sama rata dan distance yang artinya sama dekat, dan p:[1,2] atau parameter jarak, nilai 1 menunjukkan manhattan dan nilai 2 menunjukkan euclidean. Dari tuning diperoleh kombinasi parameter terbaik yaitu ketika n_neighbors=10 (model menggunakan 10 tetangga terdekat untuk menentukan prediksi), p=1 menggunakan Manhattan distance sebagai metrik kedekatan, dan weights='uniform' yang artinya semua tetangga memiliki bobot yang sama dalam menentukan nilai prediksi.
2. Pada model Random Forest menggunakan RandomizedSearchCV untuk tuning model dengan parameter n_estimators yaitu [50, 100, 150] yang menunjukkan Jumlah pohon dalam ensemble Random Forest (lebih sedikit = lebih cepat, tapi bisa lebih variatif), max_depth yaitu [10, 16, 20] yang menunjukkan maksimum kedalaman tiap pohon, min_samples_split yaitu [2, 5] yang menunjukkan minimum jumlah sampel yang dibutuhkan untuk membagi node, min_samples_leaf yaitu [1, 2] yang menunjukkan minimum jumlah sampel di leaf node, dan max_features yaitu ['sqrt', 'log2', None] yang menunjukkan fitur yang membagi node. Dan diperoleh best parameter yaitu n_estimators=50 yang artinya jumlah pohon yang optimal adalah 50, max_depth=10 yang menunjukkan pohon tidak terlalu dalam untuk mengurangi risiko overfitting, min_sampes leaf=2 dan min_samples_split=2 untuk menjaga pohon dari mempelajari terlalu banyak noise data, serta max_features=None yang menunjukkan penggunaan semua fitur saat menentukan split terbaik yang dapat membantu model jika semua fitur informatif.
3. Pada model boosting algorithm menggunakan GridSearchCV untuk tuning model dengan parameter n_estimators yaitu [50, 100, 200] yang menunjukkan jumlah total estimator, learning_rate yaitu [0.01, 0.05, 0.1, 0.2] yang menunjukkan pengontrol kontribusi tiap estimator, dan loss yaitu ['linear', 'square', 'exponential']. Linear menunjukkan kesalahan dihitung secara proporsional terhadap error, square menunjukkan kesalahan dikuadratkan, sehingga kesalahan besar dihukum lebih berat, dan exponential menunjukkan kesalahan tumbuh eksponensial seiring error meningkat. Dari tuning, diperoleh best parameternya yaitu learning_rate=0,01 yang menunjukkan bahwa model bekerja lebih baik dengan pembelajaran yang lambat, n_estimators=200 yang menunjukkan dibutuhkan banyak estimator agar boosting bisa belajar dari kesalahan sebelumnya karena learning rate kecil, dan loss=linear yang menunjukkan fungsi loss default.

Kemudian dibuat dictionary model dari ketiga model yaitu KNN, Random Forest, dan Boosting Algorithm.

Evaluasi setelah tuning:
1. K-Nearest Neighbor
   Nilai train MSE pada model K-Nearest Neighbor sebesar 4126519.224565 dan nilai test MSE sebesar 4573659.111391. Nilai MSE pada KNN lebih tinggi dari random forest. Perbedaan train dan test kecil yang menandakan model konsisten tapi kurang akurat. Sehingga performa KNN sedang, model ini mungkin cocok untuk model yang simpel tapi cukup stabil.
2. Random Forest
   Nilai train MSE sebesar 3397884.162461 dan test MSE sebesar 3682727.457989, kedua nilai ini menjadi nilai yang paling rendah dari model lainnya. Nilai train MSE yang rendah menunjukkan model belajar sangat baik dari data latih. Dan nilai test MSE yang paling rendah menunjukkan generalisasi yang baik ke data baru. Perbedaan nilai train dan test yang tidak terlalu besar memiliki arti stabil dan tidak overfit. Sehingga Random Forest menjadi model terbaik dari nilai MSE.
3. Boosting Algorithm
   Nilai train MSE sebesar 6422229.830446, nilai ini menjadi yang paling besar dari model lainnya artinya model tidk berhasil mempelajari data dengan baik. Kemudian nilai test MSE sebesar 5973400.421522 , nilai ini juga menjadi yang paling tinggi diantara model yang lain, nilai yang tinggi ini menunjukkan performa buruk pada data uji. Bisa jadi undefitting atau model tidak cocok dengan karakteristik data. Sehingga model boosting algorithm menjadi yang paling buruk dari model yang lain.
Berikut metrik evaluasi ketiga model

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Evaluasi%20Setelah%20Tuning.png" width="500"/>

Visualisasi grafik MSE setelah tuning:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/mse%20setelah%20tuning%20baru.png" width="500"/>

Dari evaluasi ketiga model setelah tuning Random Forest adalah model terbaik karena nilai MSE paling rendah diantara model yang lain

Pada perbandingan nilai prediksi antara ketiga model setelah tuning diperoleh nilai seperti di bawah ini:

<img src="https://raw.githubusercontent.com/AmandaRiyas/Predictive-Analytics/refs/heads/main/images/Aktual%20vs%20prediksi%20setelah%20tuning.png" width="500"/>

Prediksi Boosting Algorithm paling dekat dengan nilai aktual, dimana nilai prediksi Boosting Algorithm sebesar 4981.5 dan nilai aktualnya 7438


## Kesimpulan
Meskipun pada prediksi yang terbaik adalah Boosting Algorithm namun Random Forest adalah model yang terbaik karena nilai MSE terkecil. Sebab lebih baik mengutamakan kestabilan error individual dari pada perbandingan nilai prediksi dan nilai aktual. Sehingga model Random Forest adalah yang cocok untuk memprediksi harga tiket pesawat.
