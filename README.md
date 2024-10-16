# Laporan Proyek Machine Learning - Doli sawaluddin

## Domain Proyek
![9 Ikon Kota Bandung, Gedung Sate Hingga Masjid Terapung » Bandung Aktual](https://i0.wp.com/bandungaktual.com/wp-content/uploads/2013/10/kota-bandung-gedung-sate-1.jpg?fit=661%2C364&ssl=1)

sumber : bandungaktual.com


Kondisi banyak kota di Indonesia yang umumnya berkembang pesat dan berfungsi sebagai pusat kegiatan serta menyediakan layanan primer dan sekunder, telah mengundang penduduk dari daerah pedesaan untuk mendapatkan kehidupan yang lebih baik serta berbagai kemudahan lain termasuk lapangan kerja. Kondisi tersebut di atas mengakibatkan terjadinya pertambahan penduduk yang lebih pesat dibanding kemampuan pemerintah di dalam menyediakan hunian serta layanan primer lainnya.[\[1\]](https://jurnal.umj.ac.id/index.php/nalars/article/view/551/517)

**Kota Bandung** sebagai salah satu kota besar di Indonesia dengan daya tarik sebagai pusat pendidikan, bisnis, dan pariwisata, mengalami dinamika harga properti yang cukup signifikan. 
prediksi harga rumah di kota ini menjadi hal yang menarik untuk dipelajari karena beberapa faktor berikut:

**1. Permintaan yang Tinggi.**

**2. Keterbatasan Lahan.**

**3. Peningkatan Infrastruktur.**

**4. Faktor Ekonomi.**

**5. Faktor Psikologis.**

**6. Investasi Jangka Panjang.**


Bagi perusahan  maupun investor ini merupakan peluang sekaligus tantangan  dalam bisnis property rumah, dimana pemilihan property yang berkualitas bukan hanya dari harga yang murah namun harus memperhatikan faktor lain demi keuntungan yang lebih besar serta meminimalisir kerugian dimasa mendatang.

Selain itu pasar properti Bandung menjadi destinasi menarik bagi para investor yang mengincar pertumbuhan jangka panjang. Sejumlah faktor kunci, mulai dari peningkatan infrastruktur hingga pertumbuhan ekonomi yang stabil, membuka peluang besar bagi mereka yang ingin berinvestasi.[\[2\]](https://rumahbandungproperties.com/pasar-properti-bandung-kota-metropolitan-penuh-tren-dan-sejarah/)

Sebenarnya telah banyak penelitian yang melakukan prediksi harga rumah, salah satunya pada penelitian (Rahayuningtyas et al., 2021), dimana dengan memanfaatkan metode regresi linear, hasil penelitian ini menunjukkan nilai akurasi sebesar 86,41%. Harga rumah sangat berpengaruh kuat dengan luas bangunan dan harga rumah tidak dipengaruhi pada kriteria jenis sertifikat dikarenakan hasil yang didapat pada keterkaitan atau hubungan variabel luas bangunan sebesar 0,80 dan jenis sertifikat sebesar 0,00.[\[3\]](https://journal.utmmataram.ac.id/index.php/explore/article/view/123/109)

 Berdasarkan pemaparan diatas maka dalam proyek ini akan dibuat model machine learning untuk melakukan analisis prediksi terhadap harga rumah dengan memperhatikan lokasi property,luas tanah dan bangunan , serta fitur yang ada dalam property seperti jumlah kamar tidur, kamar mandi, dan garasi tempat parkir. Dengan adanya model machine learning yang akan dibangun ini diharapkan dapat membantu dalam pengambilan keputusan dalam pembelian maupun penjualan property.


## Business Understanding


### Problem Statements

Berdasarkan latar belakang diatas maka didapatkan rumusan permasalahan sebagai berikut:
- Bagaimana cara menganalisa fitur yang bermanfaat dalam predisiksi harga rumah?
- Bagaimana cara membangun model machine Learning untuk memprediksi harga rumah ?

### Goals

Berdasarkan pernyataan masalah diatas maka dapat kita tentukan tujuan sebagai berikut:
- Melakukan analisa menggunakan  Exploratory Data Analysis (EDA) untuk menemukan fitur yang bermanfaat untuk model.
- Membangun model machine learning yang dapat melakukan predikisi harga rumah dengan baik.

    ### Solution statements
Dari pemaparan sebelumnya, maka terdapat beberapa solusi yang bisa kita gunakan untuk mencapai tujuan dari proyek ini, yaitu:

**1.	 Tahap analisis menggunakan EDA.**
Exploratory Data Analysis (EDA) adalah bagian dari proses data science. EDA menjadi sangat penting sebelum melakukan feature engineering dan modeling karena dalam tahap ini kita harus memahami datanya terlebih dahulu.Exploratory Data Analysis memungkinkan analyst memahami isi data yang digunakan, mulai dari distribusi, frekuensi, korelasi dan lainnya [[4]](https://www.google.com/url?q=https%3A%2F%2Fmedium.com%2Fdata-folks-indonesia%2Fmemahami-data-dengan-exploratory-data-analysis-a53b230cce84).
![](https://media.geeksforgeeks.org/wp-content/uploads/20240509161456/Steps-for-Performing-Exploratory-Data-Analysis.png)
sumber :  media.geeksforgeeks.org

Meskipun proses EDA sangat fleksibel, secara umum, EDA dapat membantu kita menjawab berbagai pertanyaan fundamental tentang data, seperti karakteristik data, distribusi data, hubungan antar variabel, dan pola-pola yang tersembunyi di dalam data.
Pada proyek ini kita juga akan memanfaatkan EDA untuk menangani rentang harga yang terlalu tinggi, data yang meyimpang jauh dari kumpulan data (outliers) , menangani missing value, serta menemukan fitur yang bermanfaat untuk model kita.

**2.	  Dalam penelitian ini, kita akan membandingkan performa tiga algoritma pembelajaran mesin (KNN, Random Forest, dan AdaBoost)** untuk membangun model prediksi. Model dengan akurasi tertinggi akan dipilih sebagai model terbaik.


**-	Algoritma k-Nearest Neighbor** 
Algoritma k-Nearest Neighbor adalah algoritma supervised learning dimana hasil dari instance yang baru diklasifikasikan berdasarkan mayoritas dari kategori k-tetangga terdekat. Tujuan dari algoritma ini adalah untuk mengklasifikasikan obyek baru berdasarkan atribut dan sample-sample dari training data. Algoritma k-Nearest Neighbor menggunakan Neighborhood Classification sebagai nilai prediksi dari nilai instance yang baru.[\[5\]](https://medium.com/bee-solution-partners/cara-kerja-algoritma-k-nearest-neighbor-k-nn-389297de543e)

![](https://d2jdgazzki9vjm.cloudfront.net/tutorial/machine-learning/images/k-nearest-neighbor-algorithm-for-machine-learning2.png)

sumber : cloudfront.net

Secara umum, cara kerja algoritma KNN adalah sebagai berikut:
				- 	Tentukan jumlah tetangga (K) yang akan digunakan untuk pertimbangan penentuan kelas.
				-	Hitung jarak dari data baru ke masing-masing data point di dataset.
				-	Ambil sejumlah K data dengan jarak terdekat, kemudian tentukan kelas dari data baru tersebut. [\[6\]](https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi/)
				
Jarak yang Sering Digunakan:

a.	Jarak Euclidean: Rumus ini paling sering digunakan untuk menghitung jarak antara dua titik dalam ruang multidimensi.

$$d(x,y)=\sqrt{\sum_{i=1}^n (x_i-y_i)^2}$$
       
b.	Jarak Manhattan: Rumus ini menghitung jarak dengan menjumlahkan selisih absolut dari koordinat-koordinat yang sesuai.

$$d(x,y)=\sum_{i=1}^n |x_i-y_i|$$
  
c. Jarak Minkowski: Ini adalah generalisasi dari jarak Euclidean dan Manhattan.

$$d(x,y)=\left(\sum_{i=1}^n |x_i-y_i|^p\right)^\frac{1}{p}$$
  
d. Jarak Hamming digunakan untuk data biner (data yang hanya memiliki nilai 0 atau 1).

$$d(x,y)=\frac{1}{n}\sum_{n=1}^{n=n} |x_i-y_i|$$



**-	Algoritma Random Forest** 
Random Forest adalah kumpulan dari decision tree atau pohon keputusan. Algoritma ini merupakan kombinasi masing-masing tree dari decision tree yang kemudian digabungkan menjadi satu model. Biasanya, Random Forest dipakai untuk masalah regresi dan klasifikasi dengan kumpulan data yang berukuran besar. [\[7\]](https://algorit.ma/blog/cara-kerja-algoritma-random-forest-2022/)

![Diagram of Random Forest](https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/cdp/cf/ul/g/50/f9/ICLH_Diagram_Batch_03_27-RandomForest.png)
sumber: www.ibm.com

Jika ini merupakan masalah regresi maka prediksi untuk sampel uji xt dilakukan dengan mengambil rata-rata prediksi semua pohon.
$$\hat{f}=\frac{1}{B}\sum_{b=1}^{B} f_b(x^{'})$$



**-	Algoritma Adaptive Boosting** 

Algoritma AdaBoost (Adaptive Boosting), adalah sebuah teknik Boosting yang digunakan sebagai metode ensemble dalam  machine learning. Algoritma ini disebut Adaptive Boosting karena bobot diberikan ulang pada setiap instance, dengan bobot yang lebih tinggi diberikan pada instance yang salah diklasifikasikan. Boosting digunakan untuk mengurangi bias serta variasi dalam supervised learning. Algoritma Adaboost bekerja dengan cara secara iteratif melatih weak learners, seperti decision tree atau model linear, pada sebuah dataset dan memberikan bobot pada setiap instance training berdasarkan kesalahan klasifikasinya.[\[8\]](https://www.trivusi.web.id/2023/07/algoritma-adaboost.html)
![AdaBoost Algorithm](https://almablog-media.s3.ap-south-1.amazonaws.com/image_28_7cf514b000.png)
Sumber : www.almabetter.com

Secara runtun, cara kerja algoritma ini dapat dijabarkan sebagai berikut:
1.	Inisialisasi bobot sampel pelatihan
2.	Melatih weak classifier
3.	Evaluasi performa weak classifier
4.	Memperbarui bobot sampel pelatihan
5.	Ulangi langkah 2-4 sesuai jumlah iterasi yang ditentukan
6.	Menggabungkan weak classifier menjadi sebuah model yang kuat








## Data Understanding

Pada proyek ini kita akan menggunakan 3 dataset dengan 1 dataset utama ,1 dataset untuk menambah jumlah data dan 1 dataset pendukung untuk melengkapi alamat kecamatan. Dataset yang digunakan adalah sebagai berikut:

#### **1. Dataset Utama**  

Dataset Utama diambil dari Dataset Kaggele [Daftar Harga Rumah di Kota Bandung](https://www.kaggle.com/datasets/khaleeel347/harga-rumah-seluruh-kecamatan-di-kota-bandung). Didalam dataset tersebut terdapat file yang akan kita gunakan dengan nama *results_cleaned.csv*

##### Variabel-variabel pada *results_cleaned.csv* adalah sebagai berikut:
![variabel1](https://github.com/user-attachments/assets/31c43ddf-1594-4758-9e6a-f2be67e4b315)

-   **house_name:**  Nama atau judul properti residensial.
-   **location:**  Lokasi atau kecamatan di Bandung di mana properti tersebut berada.
-   **bedroom_count:**  Jumlah kamar tidur di properti tersebut.
-   **bathroom_count:**  Jumlah kamar mandi di properti tersebut.
-   **carport_count:**  Jumlah tempat parkir/garasi yang tersedia pada properti tersebut.
-   **price:**  Harga properti tersebut dalam Rupiah Indonesia (IDR).
-   **land_area:**  Total luas tanah properti tersebut dalam meter persegi.
-   **building_area (m2):**  Total luas bangunan properti tersebut dalam meter persegi.

Jumlah data yang terdapat pada data utama adalah 7609 baris dengan 8 kolom, berdasarkan informasi dari kaggle , data ini cukup bersih, namun kita akan merubah nama kolom land_area menjadi land_area(m2) ,dan mengubah nilai price yang mencapai milyaran menjadi juta untuk memudahkan dalam pembacaan informasi didalam dataset.
  
#### **2. Dataset Tambahan**  

Dataset yang digunakan untuk menambah jumlah data diambil dari Dataset Kaggele [Dataset Harga Rumah Bandung](https://www.kaggle.com/datasets/rafliaping/dataset-harga-rumah-bandung) (Data pendukung 1). file yang akan kita gunakan bernama *data_rumah.xlsx*

##### Variabel-variabel pada *data_rumah.xlsx* adalah sebagai berikut:
![image](https://github.com/user-attachments/assets/123384c8-8e24-4d1b-b435-1fc911a20d77)

 - **Unnamed: 0 ,** Berisi nomor urut
 - **Judul :** berisi text promosi ataupun nama rumah
 - **alamat :** lokasi rumah
 - **deskripsi :** berisi keterangan dari rumah
 - **kamar :** Jumlah kamar tidur di properti tersebut.
 - **bangunan :** Total luas bangunan properti tersebut dalam meter persegi.
 - **lahan :** Total luas tanah properti tersebut dalam meter persegi.
 - **harga :** Harga properti tersebut dalam Rupiah Indonesia (IDR).

Jumlah data yang terdapat pada dataset tambahan adalah 1470 baris dengan 8 kolom, kondisi data masih perlu dilakukan penyesuaian dengan data utama *results_cleaned* supaya nantinya kedua dataset dapat digabungkan.

#### **3. Dataset Pendukung**  
Dataset Pendukung [Datakelurahan](https://docs.google.com/spreadsheets/d/1Ub_VtM4_WMxCJeCSynKRtAhEonfzgRjI/export?format=xlsx&gid=371747489) , data ini diambil dari [stekom.ac.id](https://p2k.stekom.ac.id/ensiklopedia/Daftar_kecamatan_dan_kelurahan_di_Kota_Bandung) (Data pendukung 2). Tujuan dari data ini adalah untuk mengelompokkan alamat menjadi kecamatan.

#####  Variabel-variabel pada *kelurahan.xlsx* adalah sebagai berikut:
![image](https://github.com/user-attachments/assets/cb9235f9-5e8b-456e-b5b8-b1b667bc899c)

 - **Kecamatan :** Nama kecamatan pada kota bandung
 - **Lokasi :** lokasi jalan, kelurahan, nama tempat dibandung

Data set ini berisi alamat dan kecamatan dengan jumlah data 181 baris dan 2 kolom. Data ini nantinya akan digunakan untuk mengelompokkan alamat menjadi kecamatan pada *data_rumah.xlsx*


#### **4. Nilai yang hilang dan nilai duplikat**

- Nilai Duplikat
  
Nantinya setelah dilakukan penggabungan antara dataset utama dan dataset tambahan maka akan ditemukan duplikasi data sebagai berikut:

![image](https://github.com/user-attachments/assets/fec28654-648a-42d4-bcb0-7ffaf8c0c62c)

Dari gambar diatas bisa kita lihat terdapat duplikasi data sebanyak 1469 data. Kita akan menghilangkan data ini pada tahap data preparation untuk menjaga data agar tetap bersih.

- Nilai yang hilang

![image](https://github.com/user-attachments/assets/f6c87078-0b58-4d91-93e9-0dcf4f67b00e)

Bisa kita lihat, tidak ada nilai yang hilang pada data.


#### **5. Gabungan Dataset Utama dan Dataset Tambahan**

![image](https://github.com/user-attachments/assets/9eb36674-8734-4c8d-a322-d49b38dc86cd)

Setelah dilakukan pembersihan dan penggabungan data,maka kita mendapatkan data akhir dengan total data sebanyak 8243 baris dengan 8 kolom. Untuk deskripsi variabel atau keterangan kolom mengikuti Dataset utama.




### **Exploratory Data Analysis.**
Tahapan Selanjutnya setelah dataset sudah bersih adalah melakukan Exploratory Data Analysis.

**1.	Deskripsi Variabel**

![image](https://github.com/user-attachments/assets/6ca990c9-211f-4f48-a0c9-6a1eab7ba970)

Berdasarkan output diatas bisa kita lihat terdapat 6 atribut dengan tipe data int64 dan 2 atribut bertipe object yang berisi nama dan lokasi properti.dari data tersebut didapat klasifikasi sebagai berikut:
-   categorical features (fitur non-numerik): house name , location
-   numerical features (fitur numerik): bedroom_count, bathroom_count, carport_count, price, land_area(m2), building_area (m2)

**2.	Deskripsi Statistik**

![image](https://github.com/user-attachments/assets/3e55728f-19d2-40e2-ba49-afe674ca09a8)

Data di atas memberikan ringkasan statistik deskriptif dari fitur-fitur numerik dalam dataset. Ringkasan statistik ini memberikan gambaran awal tentang distribusi dan karakteristik data.
Keterangan:
-   **count:**  Jumlah data yang tidak kosong (non-missing) untuk setiap kolom.
-   **mean:**  Rata-rata dari nilai-nilai dalam kolom.
-   **std:**  Deviasi standar, menunjukkan sebaran data di sekitar rata-rata. Semakin tinggi deviasi standar, semakin tersebar data.
-   **min:**  Nilai minimum dalam kolom.
-   **25%:**  Persentil ke-25, menunjukkan nilai dimana 25% data berada di bawahnya.
-   **50%:**  Median, nilai di tengah dataset.
-   **75%:**  Persentil ke-75, menunjukkan nilai dimana 75% data berada di bawahnya.
-   **max:**  Nilai maksimum dalam kolom.

Jika kita perhatikan minimal nilai pada propery adalah 0. Di zaman sekarang ini sebuah rumah memiliki minimal 1 kamar tidur dan 1 kamar mandi, sehingga kita perlu curiga jika data tersebut merupakan data yang salah, serta terdapat anomali data pada land area dan building area,dimana tidak mungkin luas area memiliki luas dengan nilai minus atau 0.

**3.	Anomali dan Missing Value**

![image](https://github.com/user-attachments/assets/03f13492-e6e9-41a5-9d0b-cc0202ac106d)

Dari data diatas dapat kita lihat terdapat missing value  (nilai 0) pada bedroom dan bathroom serta anomali pada building area dan land_area dimana terdapat luas bangunan bernilai minus dan nol. Sedangkan Untuk carport_count tidak perlu dihiraukan karena beberapa rumah memang bisa tidak memiliki parkiran. Kita akan menghapus data ini pada tahap data preparation.

**4.	Outliers**

Outliers adalah data yang nilainya jauh menyimpang dari distribusi data mayoritas. Untuk mendeteksi outliers ini, kita akan memvisualisasikan datanya menggunakan boxplot dengan bantuan library Seaborn.
Untuk Menangani outlier kita akan menggunakan IQR (Inter Quartile Range) Method , berikut adalah bentuk persamaannya:

Batas bawah = Q1 - 1.5 * IQR

Batas atas = Q3 + 1.5 * IQR

Tampilan Outliers pada dataset:

![image](https://github.com/user-attachments/assets/e1ec5fb3-066f-4996-9364-7d79c8056d8a)

Bisa kita lihat terdapat data yang sangat jauh dari kelompok data pada fitur bedroom count, bathroom count, carport count, land area dan building area. Sehingga kita perlu menghilangkan outlier tersebut pada tahap data preparation.

**5.	EDA-Univariate Analysis**

![image](https://github.com/user-attachments/assets/94f734c4-18fb-4d95-a763-621552e551ac)

Terdapat 30 kecamatan pada kota bandung yang menjadi lokasi properti. Dari data tersebut bisa kita lihat penjualan properti paling banyak berada di kecamatan Arcamanik, buahbatu, dan daerah kota bandung, serta penjulan properti paling sedikit berada pada kecamatan cibeunying kaler dan cinambo.


![image](https://github.com/user-attachments/assets/db9c6953-f65f-425c-9744-40ed3198b652)

Berdasarkan histogram diatas kita bisa melihat distribusi data pada setiap fitur numerik, berikut penjelasan detailnya:
 -  **Bedroom Count:** Sebagian besar rumah memiliki jumlah kamar tidur antara 2 sampai 4, dengan sedikit rumah yang memiliki  jumlah kamar tidur lebih dari 4. Ini menunjukkan bahwa rumah dengan 2-4 kamar tidur adalah yang paling umum di pasaran.
- **Bathroom Count:** Distribusi data mirip dengan bedroom count, dengan sebagian besar rumah memiliki 2-3 kamar mandi. Hal ini mengindikasikan korelasi antara jumlah kamar tidur dan kamar mandi.
 -  **Carport Count:** Sebagian besar rumah memiliki carport dengan kapasitas 1-2 mobil.
 -  **Price:** Distribusi harga rumah cenderung miring ke kanan (right skewed), yang menunjukkan bahwa ada beberapa rumah dengan harga yang sangat tinggi dibandingkan dengan sebagian besar rumah. Ini adalah fenomena umum pada pasar properti, di mana beberapa properti mewah memiliki harga yang jauh lebih tinggi daripada rata-rata.
 -  **Land Area:** Data land area juga cenderung miring ke kanan. Artinya, sebagian besar rumah memiliki luas tanah yang relatif kecil, tetapi ada beberapa rumah yang memiliki luas tanah yang sangat besar.
 -  **Building Area:** Distribusi building area juga mirip dengan land area, cenderung miring ke kanan. Ini menunjukkan bahwa sebagian besar rumah memiliki luas bangunan yang relatif kecil, tetapi ada beberapa rumah dengan luas bangunan yang sangat besar.
    
    Kesimpulan: Dari histogram ini, kita bisa memahami distribusi data pada setiap fitur numerik. Selain itu, kita juga bisa melihat pola-pola tertentu, seperti korelasi antara jumlah kamar tidur dan kamar mandi, serta keberadaan outlier pada harga, luas tanah, dan luas bangunan. Informasi ini bisa membantu kita untuk lebih memahami karakteristik pasar properti di Bandung dan membantu dalam proses pembuatan model prediksi.

**6.	EDA - Multivariate Analysis**

![image](https://github.com/user-attachments/assets/b42a647f-79e6-4fb0-8064-8367dfce20ea)

Perhatikan rentang harga diatas, berdasarkan data diatas maka dapat disimpulkan:
- Rata-rata harga properti: 2226.20 juta
- Kecamatan sumurbandung dan bandung wetan memiliki property degan harga jual yang cukup tinggi
- Property dengan harga termurah berada di kecamatan cibiru dan ujungberung.
Dengan data ini dapat mempermudah dalam memilih lokasi property sesuai kebutuhan dan dana pembeli.

![image](https://github.com/user-attachments/assets/9250093e-96cc-42c1-9a81-4d03de049cbe)

Berdasarkan plot yang ditampilkan, kita dapat mengamati hubungan antara 'price' dengan fitur-fitur lainnya:
- Bedroom Count, Bathroom Count, Carport Count: Bisa kita lihat jumlah kamar tidur dan kamar mandi tidak terlalu mempengaruhi harga jual rumah, Jika kita baca data pada tahap awal salah satu penyebabnya adalah karena beberapa property mahal tidak menginputkan jumlah kamar dan garasi dengan benar dan beberapa property ternyata hanya menjual tanah saja.
-  Land Area: Terdapat korelasi positif yang kuat antara luas tanah dengan harga. Semakin luas tanah, cenderung semakin tinggi harga rumah. Ini menunjukkan bahwa luas tanah adalah faktor penting dalam menentukan harga properti.
-  Building Area: Sama seperti luas tanah, terdapat korelasi positif yang kuat antara luas bangunan dengan harga. Semakin luas bangunan, cenderung semakin tinggi harga rumah.

Kesimpulan:
Berdasarkan hasil analisis, fitur yang paling berpengaruh terhadap harga rumah adalah luas tanah dan luas bangunan. Kedua fitur ini menunjukkan korelasi yang kuat dan signifikan dengan harga. Jumlah kamar tidur dan kamar mandi juga memiliki pengaruh positif terhadap harga, tetapi tidak sekuat luas tanah dan bangunan. Jumlah carport memiliki pengaruh yang lebih kecil terhadap harga dibandingkan dengan fitur lainnya.

**7.	Correlation Matrix menggunakan Heatmap**

![image](https://github.com/user-attachments/assets/05ea2fac-886b-4d9a-9f16-fee611221999)

Berdasarkan heatmap korelasi yang ditampilkan, kita dapat mengamati hubungan antar fitur numerik:
1.  Korelasi Positif:
    -   Antara 'building_area (m2)' dan 'price (million)': Korelasi positif yang kuat (0.77), menunjukkan bahwa semakin luas bangunan, harga rumah cenderung semakin tinggi.
    -   Antara 'land_area(m2)' dan 'price (million)': Korelasi positif yang kuat (0.79), menunjukkan bahwa semakin luas tanah, harga rumah cenderung semakin tinggi.
    -   Jumlah kamar tidur dan kamar mandi masih memiliki kolerasi walau tidak terlalu kuat
2.  Korelasi Lemah:
    -   Antara 'carport_count' dan 'price (million)': Korelasi positif yang lemah (0.31), menunjukkan bahwa pengaruh jumlah carport terhadap harga rumah relatif kecil.


## Data Preparation
**Alasan mengapa diperlukan tahapan data preparation** adalah karena merupakan langkah krusial sebelum memulai pelatihan model machine learning. Tujuannya adalah untuk mengolah data mentah agar sesuai dan optimal untuk digunakan dalam proses pembelajaran mesin.
### **1. Dataset Preparation**
#### **1.1. Dataset Preparation pada Dataset Utama**
Pada tahapan ini hanya Mengganti nama kolom land_area menjadi land_area(m2) agar memudahkan pemaham terkait nilai didalamnya yang merupakan ukuran luas
![image](https://github.com/user-attachments/assets/418f62da-374b-48df-8efe-1f9003de80bc)


#### **1.2. Dataset Preparation pada Dataset tambahan**
1.	Pertama, kita akan menyesuaikan data pada dataset *data_rumah* berdasarkan dataset  *results_cleaned*, dengan tahapan berikut:
	-   Menyeragamkan alamat menjadi kecamatan, dengan memanfaatkan dataset *kelurahan.xlsx*
![image](https://github.com/user-attachments/assets/eec4363d-ed04-4a15-ac78-c4f9efaf2500)
	-   Menghapus kolom Unnamed: 0, alamat, dan deskripsi yang tidak dibutuhkan untuk analisa dan pembuatan model.
	-   Menyesuaikan nama kolom pada dataset *data_rumah*  berdasarkan nama kolom *results_cleaned* agar kedua dataset bisa digabungkan
	-   Menambahkan minimal kamar mandi untuk  dataset *data_rumah* , penambahan ini dilakukan untuk menyesuaikan nilai dengan dataset *results_cleaned*	
![image](https://github.com/user-attachments/assets/a662f40f-7eef-4524-b4b9-7e5a04a44581)

Berikut tampilan akhir rumah_2 (dataset *data_rumah*):

![image](https://github.com/user-attachments/assets/1fd1040e-3f92-4ca0-b728-a5e60e1ebb4a)

#### **1.3. Dataset Preparation pada Dataset Gabungan**

1.	 Menggabungkan kedua dataset menjadi dataframe *rumah_bersih*, dan menghapus duplikat data berdasarkan kolom house name.
	 
![image](https://github.com/user-attachments/assets/87f330c1-eaa7-4476-93dc-c665e4b0254d)

![image](https://github.com/user-attachments/assets/fbaa6396-79c5-4b8d-a29a-ac6a032603be)

2.	 Mengubah ukuran price dari milyar ke juta untuk memudahkan pembacaan data.
![image](https://github.com/user-attachments/assets/70f09242-a523-4315-8972-64eb65b1d0d1)

3.	Tampilan data setelah melalui tahapan 1 dan 2 adalah sebagai berikut:

![image](https://github.com/user-attachments/assets/b5d7eb9d-d456-4844-81c6-72094b46c869)

Bisa kita lihat Total data setelah digabungkan adalah sebanyak 8243 data.

### **2. Penyesuaian Data setelah Proses Exploratory Data Analysis (EDA)**

1.	Seperti yang dijelaskan Ketika melakukan analisa menggunakan EDA sebelumnya , ditemukan data yang tidak wajar (anomali) serta data dengan nilai nol, sehingga dilakukan penghapusan terhadap data tersebut:
![image](https://github.com/user-attachments/assets/5a92c8ef-4553-4e7b-a949-b57d8b4ccd89)

bisa kita lihat total data sekarang telah berkurang menjadi 8037 baris dan 8 kolom baru.

2.	Selanjutnya kita juga menghapus **data outliers** sehingga data berkurang lagi menjadi 6593 baris dengan 7 kolom baru,ini terjadi disebabkan kolom house_name telah dihapus karena tidak dibutuhkan untuk tahapan selanjutnya.
berikut adalah tampilan awal data yang memiliki outliers:
![image](https://github.com/user-attachments/assets/b771d15d-3ada-41ac-92f4-35ef27edb883)

- Sebelum membersihkan outliers kita akan menghapus kolom house_name karena sudah tidak diperlukan lagi.

![image](https://github.com/user-attachments/assets/060ced80-9642-410b-98d9-10861e552275)

- Lalu pada tahapan berikutnya kita akan menghapus outliers menggunakan  IQR sebagai berikut:

![image](https://github.com/user-attachments/assets/d47734a2-33aa-4850-afd1-73bb49c84e4c)

- Setelah dilakukan pembersihan terhadap outliers maka kita akan mendapatkan hasil sebagai berikut:

![image](https://github.com/user-attachments/assets/c44254d4-e54e-4865-8d01-21d19d062327)

walaupun masih ada outliers, namun tidak perlu khawatir karena beberapa rumah memang memiliki variasi harga yg lebih mahal dibanding rumah lainnya.

- Setelah dilakukan permbersihan, maka kita mendapatkan hasil akhir seperti yang diperlihatkan oleh gambar dibawah ini:
![image](https://github.com/user-attachments/assets/bf9a763a-c2ce-41d0-84b0-fbad9eefe139)

3.	Tahapan akhir kita mencoba mengurangi rentang harga yang terlalu tinggi menggunakan Winsorizing , Winsorizing akan menggantikan 5% nilai terendah dan 5% nilai tertinggi dalam kolom 'price (million)' dengan persentil ke-5 dan persentil ke-95, sehingga mengurangi pengaruh outlier pada distribusi harga.

![image](https://github.com/user-attachments/assets/bab77842-19cf-4f60-8b6d-915d9cc0b8bc)

Berikut adalah tampilan data setelah dilakukan metode winsorizing, terlihat rentang data sudah tidak terlalu tinggi.
![image](https://github.com/user-attachments/assets/e1d47158-0717-46e6-8b9f-4773262f908f)


### **3. Encoding Fitur Kategori**
Untuk mempersiapkan data kategorikal agar dapat diproses oleh algoritma machine learning, teknik one-hot encoding sering digunakan. Teknik ini mengubah setiap kategori menjadi sebuah fitur biner (0 atau 1), di mana nilai 1 menunjukkan keberadaan kategori tersebut. Scikit-learn menyediakan kelas OneHotEncoder yang memudahkan proses encoding ini.

![image](https://github.com/user-attachments/assets/30d35651-ca46-4ce0-aa51-6facb687b2e8)
Dapat kita lihat untuk setiap kategori dalam fitur location telah diubah kedalam bentuk biner.

### **4. Split Data**
Membagi data menjadi 90% untuk training dan 10% untuk test. Pembagian Data Latih (Train) 90% dan Data Uji (Test) 10% adalah praktik umum dalam machine learning. Tujuan utama dari pembagian ini adalah untuk mengevaluasi kinerja model secara objektif sebelum digunakan pada data yang benar-benar baru.

![image](https://github.com/user-attachments/assets/297a0fe3-e52a-463d-a2db-86ba405ec6be)

Setelah dilakukan pembagian data , kita mendapatkan data training sebanyak 5933 data dan data test sebanyak 660 data.



## Modeling
Setelah data disiapkan, langkah selanjutnya adalah mempersiapkan kerangka data (dataframe) untuk analisis model. DataFrame ini akan digunakan untuk menyimpan hasil evaluasi model yang akan dikembangkan menggunakan tiga algoritma berbeda, yaitu K-Nearest Neighbor (KNN), Random Forest, dan Adaptive Boosting (AdaBoost). Parameter `index` pada dataframe ini menunjukkan jenis metrik evaluasi yang akan digunakan, yaitu Mean Squared Error (MSE) pada data latih (train_mse) dan data uji (test_mse).

![image](https://github.com/user-attachments/assets/d0fe0d87-8fea-4198-bd2b-af880dfcae2e)

Langkah selanjutnya adalah menerapkan semua algoritma kedalam model diatas.

### **1.	 K-Nearest Neighbor (KNN) Algorithm**

![image](https://github.com/user-attachments/assets/2cae9956-64ca-4546-acdb-174347513dae)

Penjelasan Cara Kerja KNN pada Kode di Atas

1. Inisialisasi Model KNN:
   - `knn = KNeighborsRegressor(n_neighbors=10)`: 
     - Kode ini membuat objek model KNN dengan parameter `n_neighbors=10`.
     - `n_neighbors` menentukan jumlah tetangga terdekat yang akan dipertimbangkan saat melakukan prediksi.
     - Dalam kasus ini, model akan mencari 10 data point terdekat dengan data yang ingin diprediksi.

2. Pelatihan Model:
   - `knn.fit(X_train, y_train)`: 
     - Metode `fit` digunakan untuk melatih model KNN dengan data training (`X_train` dan `y_train`).
     - Pada tahap ini, model akan mempelajari pola dan hubungan antara fitur (X) dan target (y) dalam data training.
     - Model akan menyimpan data training dalam bentuk struktur data yang memungkinkan pencarian tetangga terdekat dengan efisien.

3. Evaluasi Model (Train MSE):
   - `models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)`:
     - Setelah model dilatih, kita perlu mengukur performanya. Kode ini menghitung Mean Squared Error (MSE) pada data training.
     - `knn.predict(X_train)`: Model digunakan untuk memprediksi harga (`y`) pada data training.
     - `mean_squared_error(...)`:  MSE dihitung dengan membandingkan prediksi model (`y_pred`) dengan nilai aktual harga (`y_true`) dari data training.
     - MSE merupakan ukuran kesalahan model dalam memprediksi harga pada data training. Nilai MSE yang rendah menunjukkan bahwa model memiliki performa baik pada data training.

Secara singkat, KNN bekerja dengan mencari data point terdekat dengan data yang ingin diprediksi.
Kemudian, berdasarkan harga dari tetangga terdekat tersebut, model akan membuat prediksi harga untuk data yang baru.
Pada kode di atas, model KNN dilatih dengan data training dan kemudian performanya diukur dengan MSE pada data training.

### **2.  Random Forest Algorithm**

![image](https://github.com/user-attachments/assets/b91f3905-5b36-4ff8-b477-1d978c07c84f)

Penjelasan Cara Kerja Random Forest pada Kode di Atas adalah sebagai berikut:
1. Inisialisasi Model Random Forest:
   - `RF = RandomForestRegressor(n_estimators=50, max_depth=32, random_state=55, n_jobs=-1)`:
     - Kode ini membuat objek model Random Forest dengan beberapa parameter:
       - `n_estimators=50`: Jumlah pohon keputusan yang akan dibuat dalam ensemble. Semakin banyak pohon, semakin baik performanya, tetapi juga membutuhkan waktu yang lebih lama untuk melatih.
       - `max_depth=32`: Kedalaman maksimum setiap pohon keputusan. Kedalaman pohon yang lebih besar memungkinkan model mempelajari pola yang lebih kompleks, tetapi juga berpotensi menyebabkan overfitting (model terlalu fokus pada data training dan performanya buruk pada data baru).
       - `random_state=55`: Menentukan random seed untuk reproduksibilitas hasil. Dengan nilai yang sama, model akan menghasilkan hasil yang sama setiap kali dijalankan.
       - `n_jobs=-1`: Menggunakan semua core CPU yang tersedia untuk mempercepat pelatihan model.

2. Pelatihan Model:
   - `RF.fit(X_train, y_train)`:
     - Metode `fit` digunakan untuk melatih model Random Forest dengan data training (`X_train` dan `y_train`).
     - Pada tahap ini, model akan membangun sejumlah pohon keputusan. Setiap pohon dibangun dengan menggunakan sampel data training yang berbeda (bootstrap aggregating atau bagging) dan subset fitur yang berbeda (random subspace).
     - Setiap pohon akan mempelajari pola dan hubungan antara fitur (X) dan target (y) dalam data training yang diberikan.

3. Evaluasi Model (Train MSE):
   - `models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)`:
     - Setelah model dilatih, kita perlu mengukur performanya. Kode ini menghitung Mean Squared Error (MSE) pada data training.
     - `RF.predict(X_train)`: Model digunakan untuk memprediksi harga (`y`) pada data training.
     - `mean_squared_error(...)`: MSE dihitung dengan membandingkan prediksi model (`y_pred`) dengan nilai aktual harga (`y_true`) dari data training.
     - MSE merupakan ukuran kesalahan model dalam memprediksi harga pada data training. Nilai MSE yang rendah menunjukkan bahwa model memiliki performa baik pada data training.

Secara singkat, Random Forest bekerja dengan membangun sejumlah pohon keputusan yang berbeda,
masing-masing dilatih dengan sampel data dan fitur yang berbeda. 
Kemudian, saat melakukan prediksi, setiap pohon akan memberikan hasil prediksinya sendiri.
Hasil prediksi akhir diperoleh dengan menggabungkan (aggregasi) hasil prediksi dari semua pohon,
misalnya dengan mengambil rata-rata prediksi untuk regresi.
Pada kode di atas, model Random Forest dilatih dengan data training dan kemudian performanya diukur dengan MSE pada data training.


### **3.  Adaptive Boosting (AdaBoost) Algorithm**

![image](https://github.com/user-attachments/assets/e086ec67-fe9a-44db-a985-9256006bc6d1)

Penjelasan Cara Kerja AdaBoost pada Kode di Atas:

1. Inisialisasi Model AdaBoost:
   - `boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)`:
     - Kode ini membuat objek model AdaBoost dengan beberapa parameter:
       - `learning_rate=0.05`: Parameter ini mengontrol seberapa besar pengaruh setiap model "weak learner" terhadap model akhir. Nilai yang lebih rendah berarti bahwa setiap model memiliki pengaruh yang lebih kecil, dan model akan belajar lebih lambat.
       - `random_state=55`: Menentukan random seed untuk reproduksibilitas hasil. Dengan nilai yang sama, model akan menghasilkan hasil yang sama setiap kali dijalankan.

2. Pelatihan Model:
   - `boosting.fit(X_train, y_train)`:
     - Metode `fit` digunakan untuk melatih model AdaBoost dengan data training (`X_train` dan `y_train`).
     - Pada tahap ini, model akan membangun sejumlah model "weak learner" (biasanya pohon keputusan yang sederhana) secara berurutan.
     - Setiap "weak learner" akan dilatih dengan memperhatikan kesalahan yang dibuat oleh model sebelumnya.
     - Data yang salah diprediksi oleh model sebelumnya akan diberi bobot yang lebih besar pada pelatihan model berikutnya, sehingga model berikutnya akan lebih fokus pada data yang sulit diprediksi.

3. Evaluasi Model (Train MSE):
   - `models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)`:
     - Setelah model dilatih, kita perlu mengukur performanya. Kode ini menghitung Mean Squared Error (MSE) pada data training.
     - `boosting.predict(X_train)`: Model digunakan untuk memprediksi harga (`y`) pada data training.
     - `mean_squared_error(...)`: MSE dihitung dengan membandingkan prediksi model (`y_pred`) dengan nilai aktual harga (`y_true`) dari data training.
     - MSE merupakan ukuran kesalahan model dalam memprediksi harga pada data training. Nilai MSE yang rendah menunjukkan bahwa model memiliki performa baik pada data training.

Secara singkat, AdaBoost bekerja dengan membangun serangkaian model "weak learner" secara berurutan.
Setiap model berikutnya akan fokus pada data yang sulit diprediksi oleh model sebelumnya.
Hasil prediksi akhir diperoleh dengan menggabungkan hasil prediksi dari semua "weak learner",
dengan memberikan bobot yang lebih besar pada model yang memiliki performa lebih baik.
Pada kode di atas, model AdaBoost dilatih dengan data training dan kemudian performanya diukur dengan MSE pada data training.


	

### **4. Pertimbangan untuk Dataset dengan Banyak Fitur:**

-   **KNN:** Kelebihan KNN adalah fleksibilitasnya ,namun tidak disarankan untuk dataset dengan banyak fitur karena perhitungan jarak yang intensif dan menyebabkan waktu komputasi bisa meningkat secara signifikan.
-   **Random Forest:** Kelebihannya adalah Sangat cocok untuk dataset dengan banyak fitur karena kemampuannya dalam menangani dimensi tinggi dan memilih fitur yang relevan. Namun memiliki kelemahan yakni bisa sulit diinterpretasi karena melibatkan banyak pohon keputusan dan membutuhkan waktu komputasi yang lebih lama
-   **AdaBoost:** Memiliki kelebihan dapat bekerja dengan baik pada dataset dengan banyak fitur, tetapi perlu diperhatikan sensitivitasnya terhadap noise. kekurangan lainnya adalah Model AdaBoost juga bisa sulit diinterpretasi

Berdasarkan data diatas, sementara dapat kita simpulkan bahwa random forest dan adaboost adalah model yang paling sesuai untuk dataset kita, dimana fitur bertambah banyak setelah dilakukan encoding pada fitur kategori.
Akan tetapi Ketiga model yang telah dikembangkan, yakni model K-Nearest Neighbor, Random Forest, dan Adaptive Boosting, akan dievaluasi lagi  terhadap performansinya. Evaluasi ini bertujuan untuk mengidentifikasi model terbaik yang menghasilkan prediksi paling akurat dengan tingkat kesalahan seminimal mungkin.	


## Evaluation

### **Metrik Evaluasi**
Setelah membangun ketiga model (KNN, Random Forest, dan AdaBoost), kita akan mengevaluasi performanya menggunakan metrik Mean Squared Error (MSE). MSE akan menghitung rata-rata kuadrat selisih antara nilai prediksi model dengan nilai sebenarnya, baik pada data latih maupun data uji. Model dengan nilai MSE terendah dianggap sebagai model yang memiliki kinerja terbaik. Meskipun MSE mudah dihitung, namun interpretasi nilai mutlaknya perlu dilakukan dengan hati-hati karena tidak memberikan skala referensi yang jelas.

$$MSE=\frac{1}{N}\sum_{i=1}^{N} (y_i-y\\_pred_i)^2$$

Rumus Mean Squared Error (MSE) diatas digunakan untuk mengukur rata-rata kuadrat selisih antara nilai prediksi suatu model dengan nilai sebenarnya. Semakin kecil nilai MSE, semakin baik kinerja model dalam melakukan prediksi.

-   **$N$:** Jumlah data yang digunakan untuk evaluasi.
-   **$y_i$:** Nilai sebenarnya dari data ke-i.
-   **$y\\_pred$:** Nilai prediksi dari model untuk data ke-i.

**Intinya:** Rumus ini menghitung rata-rata dari kuadrat selisih antara nilai sebenarnya dan nilai prediksi. Kuadrat digunakan untuk memberikan bobot yang lebih besar pada kesalahan yang lebih besar.

evaluasi model:

![image](https://github.com/user-attachments/assets/66735453-bb16-42a0-8dcd-d3589bee3e04)

Dari data tabel tersebut dapat divisualisasikan pada grafik batang berikut.
![image](https://github.com/user-attachments/assets/b1bcda8a-7715-44e6-922e-0ec1a127e717)

Berdasarkan hasil evaluasi model, algoritma RF memiliki nilai MSE terendah pada data testing. Hal ini mengindikasikan bahwa model RF memiliki performa yang lebih baik dalam memprediksi harga rumah dibandingkan dengan KNN dan Boosting. Model RF mampu menangkap pola yang kompleks dalam data dengan lebih baik, yang menyebabkan model ini memiliki kemampuan generalisasi yang lebih tinggi.

### **Eksplorasi fitur-fitur yang bermanfaat**

Dikarenakan kita tidak bisa menampilkan skor dengan heatmap yang disebabkan data kategori terlalu banyak maka kita akan menampilkan skornya dalam bentuk text sebagai berikut.

![image](https://github.com/user-attachments/assets/a60c956e-2820-4d6a-9970-b062d106a9ee)

Jika kita perhatikan fitur yang paling bermanfaat dalam menentukan harga rumah atau properti adalah luas tanah dan luas bangunan. jika kita lihat skornya yang hanya mencapai 79%, ini disebabkan karena harga propery mewah cenderung bervariasi dan lebih mahal dibandingkan harga rumah lainnya.

Selanjutnya kita akan melihat fitur-fitur yang tidak terlalu berpengaruh terhadap harga jual rumah:

![image](https://github.com/user-attachments/assets/2ab76d45-7928-4b83-8a77-6917ab18f243)

Dari skor kolerasi antara fitur lainnya dengan harga menunjukkan bahwa :
- jumlah kamar mandi dan kamar tidur tidak terlalu berpengaruh terhadap harga rumah, namun tetap bisa dijadikan sebagai tolak ukur dalam menentukan harga rumah walaupun tidak terlalu signifikan.
- Berdasarkan skor diatas, kita dapat menyimpulkan bahwa lokasi tidak mempengaruhi harga rumah dikota bandung

### **Menguji Model**

Tahapan terakhir adalah Menguji model dengan melakukan prediksi menggunakan beberapa data dari data test.

![image](https://github.com/user-attachments/assets/d020371d-bb34-4e48-881c-a8f49b756a94)

Dari 10 prediksi diatas bisa kita lihat pada kolom diff_RF, dimana selisih antara data sebenarnya  *(y_true )* dengan data prediksi sangat kecil dibandingkan selisih pada KNN dan boosting.


### **Kesimpulan**

Berdasarkan perbandingan prediksi dengan nilai sebenarnya, dapat disimpulkan bahwa ketiga model (KNN, RF, dan Boosting) yang telah kita bangun mampu memprediksi harga rumah dengan cukup baik. Namun, Random Forest memiliki keunggulan dalam hal akurasi dan kemampuan generalisasi, sehingga menjadi pilihan yang lebih baik untuk memprediksi harga rumah pada dataset ini. Serta Jika dilihat dari berbagai tahapan diatas ,Exploratory Data Analysis (EDA) ternyata dapat membantu dalam menemukan fitur-fitur yang bermanfaat untuk memprediksi harga rumah. Model yang telah dibangun dapat digunakan dalam membantu perusahaan dan investor dalam menganalisa harga property dipasaran dengan memperhatikan berbagai fitur-fitur yang ada seperti jumlah kamar ,luas area. serta membantu pengambilan keputusan sebelum melakukan pembelian atau penjualan property rumah.


## Referensi
[1] Kusumawardhani, V., Sutjahjo, S. H., & Dewi, I. K. (2016). PENYEDIAAN PERUMAHAN DAN INFRASTRUKTUR DASAR DI LINGKUNGAN PERMUKIMAN KUMUH PERKOTAAN (STUDI KASUS DI KOTA BANDUNG). _NALARs_, _15_(1), 13–24. https://doi.org/10.24853/nalars.15.1.13-24

[2] Liu. (2023). Pasar properti Bandung kota metropolitan penuh tren dan sejarah. _Rumah Bandung Properties_. Retrived from: https://rumahbandungproperties.com/pasar-properti-bandung-kota-metropolitan-penuh-tren-dan-sejarah/

[3] Nuris, N. (2024). Analisis Prediksi Harga Rumah Pada Machine Learning Menggunakan Metode Regresi Linear. _EXPLORE_, _14_(2), 108–112. https://doi.org/10.35200/ex.v14i2.123

[4] Tim Data Folks Indonesia. (2019, 10 Oktober). Memahami data dengan exploratory data analysis. _Medium_. Retrived from: https://medium.com/data-folks-indonesia/memahami-data-dengan-exploratory-data-analysis-a53b230cce84

[5] Ismail, A. M. (2018). Cara kerja algoritma k-Nearest Neighbor (k-NN). _Medium_. Retrived from: https://medium.com/bee-solution-partners/cara-kerja-algoritma-k-nearest-neighbor-k-nn-389297de543e

[6] Afifah, L. (n.d.). Algoritma K-Nearest Neighbor (KNN) untuk klasifikasi. _Ilmu Data Py_. Retrived from: https://ilmudatapy.com/algoritma-k-nearest-neighbor-knn-untuk-klasifikasi/

[7] Algorit. ma. (2022). Cara kerja algoritma random forest. _Algorit. ma_. Retrived from: https://algorit.ma/blog/cara-kerja-algoritma-random-forest-2022/




<!--stackedit_data:
eyJoaXN0b3J5IjpbMjA4Njg0NDcyN119
-->
