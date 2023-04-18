[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=9971783&assignment_repo_type=AssignmentRepo)
# Graded Challenge 1

_Graded Challenge ini dibuat guna mengevaluasi pembelajaran pada Hacktiv8 Data Science Fulltime Program khususnya pada konsep Basic SQL, Python, serta Data Preparation with Pandas._

---

## Dataset Description

* Pada graded challenge ini, data diakses menggunakan `bigquery-public-data` pada Google Cloud Big Query.
* Buka [Google Cloud Platform](https://console.cloud.google.com/), masuk ke BigQuery, lalu buka tab `bigquery-public-data` atau klik link [berikut](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=samples&page=dataset&_ga=2.245085957.1471931019.1642739417-486643658.1638156099) atau link [berikut](https://console.cloud.google.com/bigquery?p=bigquery-public-data&d=covid19_jhu_csse_eu&page=dataset&project=rock-wonder-317907) untuk langsung menuju ke dataset.

```{attention}
Perhatikan petunjuk penggunaan dataset!
```

1. Gunakan tabel `summary` pada database `covid19_jhu_csse_eu`.
2. Buatlah query sesuai dengan kebutuhan analisis/eksplorasi kamu (sesuaikan dengan problem statement).
3. Simpan dataset dalam bentuk csv, dengan nama `h8dsft_P0GC1_<nama-students>.csv`. Pastikan seluruh data tersimpan (*clue*: Simpan ke dalam Google Drive supaya data tersimpan seluruhnya, kemudian di download di local atau di-load ke Google Colab).
4. Salin query yang telah dibuat di Google Cloud Platform, tulislah pada bagian atas notebook!
5. Tampilkan `head` dan `tail` dari dataset pada notebook!

---

## Assignment Instructions

*Graded Challenge 1* dikerjakan dalam format ***notebook*** dengen beberapa **kriteria wajib** di bawah ini:

1. *Library* yang digunakan adalah **Pandas**.

2. *Project* dinyatakan selesai dan diterima untuk dinilai jika saat dilakukan `Run All` pada *notebook*, semua *cell* berhasil tereksekusi sampai akhir.

3. Isi *notebook* harus mengikuti *outline* di bawah ini:
   1. Perkenalan
      > Bab pengenalan harus diisi dengan identitas, latar belakang permasalahan, dan *problem statement* yang ingin dibahas.
   
   2. Import pustaka yang dibutuhkan
      > *Cell* pertama pada *notebook* **harus berisi dan hanya berisi** semua *library* yang digunakan dalam *project*.
   
   3. Data Loading
      > Bagian ini berisi query yang sudah dibuat dan proses *data loading* yang kemudian dilanjutkan dengan *explorasi data* secara sederhana.
   
   4. Data Cleaning
      > Bagian ini berisi proses penyiapan data berupa data cleaning sebelum dilakukan *explorasi data* lebih lanjut. Proses cleaning dapat berupa memberi nama baru untuk setiap kolom, mengisi missing values, menghapus kolom yang tidak dipakai, dan lain sebagainya.
   
   5. Explorasi Data
      > Bagian ini berisi explorasi data pada dataset diatas dengan menggunakan query, grouping, visualisasi sederhana, dan lain sebagainya.

   6. Pengambilan Kesimpulan
      > Pada bab terakhir ini, **harus berisi** kesimpulan yang mencerminkan hasil yang didapat dengan dibandingkan dengan *problem statement* yang sudah ditulis di bagian pengenalan.

4. *Notebook* harus diupload dalam repositori GitHub Clasroom masing-masing siswa untuk selanjutnya dinilai.

## Assignment Submission

- Simpan assignment pada sesi ini dengan nama `h8dsft_P0W1_<nama-student>.ipynb`, misal `h8dsft_P0W1_raka_ardhi.ipynb`.
- Push Assigment yang telah dibuat ke akun Github masing-masing student.

## Assignment Objectives

*Graded Challenge 1* ini dibuat guna mengevaluasi konsep Basic SQL, Python, serta Data Preparation with Pandas sebagai berikut:

- Mampu memuat data dengan Pandas
- Mampu menangani missing values pada dataset
- Mampu melakukan manipulasi kolom
- Mampu membuat query terhadap dataset
- Mampu melakukan grouping terhadap dataset
- Mampu melakukan visualisasi sederhana menggunakan Pandas

---

## Assignment Rubrics

|       KEY COMPONENT      |                                                 CRITERIA                                                 |                                                                                        NOTES                                                                                       | POINTS | TOTAL |
|:------------------------:|:--------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------:|:-----:|
|        SQL Queries       |                           Mampu menarik data dari BigQuery GCP menggunakan SQL                           | Siswa mencoba membuat query SQL BigQuery dengan dibuktikan pada notebook dan ditampilkan datanya                                                                                   |    1   |   3   |
|                          |                                                                                                          | Siswa mampu membuat query yang sesuai dengan logika, kondisi, dan hasil yang diinginkan oleh soal                                                                                  |    2   |       |
|       Data Loading       |                                      Mampu memuat data dengan Pandas                                     | Siswa mampu memuat data dengan pd.read_csv/pd.read_excel sesuai dengan format data                                                                                                 |    1   |   2   |
|                          |                                                                                                          | Siswa dapat memuat data dengan benar sampai tidak ada pesan error dan data termuat dengan baik (dibuktikan dengan menampilkan data)                                                |    1   |       |
|       Missing Value      |              Mampu menangani missing values sehingga tidak ada missing values dalam dataset              | Mampu mengecek keberadaan missing value baik standard maupun non-standard pada data                                                                                                |    2   |   3   |
|                          |                                                                                                          | Mampu menghandling missing value sampai tidak ada lagi missing value                                                                                                               |    1   |       |
|    Manipulating Column   |                  Mampu mengganti nama kolom. Mampu menghapus kolom yang tidak digunakan                  | Mampu melakukan penggantian nama kolom bila diperlukan                                                                                                                             |    1   |   3   |
|                          |                                                                                                          | Mampu menghapus kolom yang tidak diperlukan                                                                                                                                        |    1   |       |
|                          |                                                                                                          | Mampu memberikan penjelasan dan reasoning terhadap keputusan dalam melakukan manipulating column                                                                                   |    1   |       |
|        Data Query        |                                    Mampu membuat minimal 3 query data                                    | Siswa mencoba membuat pandas Query dengan df.query atau slicing biasa dengan [ ] sebanyak 3                                                                                        |    3   |   12  |
|                          |                                                                                                          | Siswa mampu membuat 3 pandas Query dengan logika dan kondisi yang benar sesuai dengan informasi apa yang ingin dilihat. Tiap Query harus dengan kondisi dan logika yang berbeda    |    3   |       |
|                          |                                                                                                          | Memberikan keterangan informasi apa yang ingin dilihat melalui Pandas Query                                                                                                        |    3   |       |
|                          |                                                                                                          | Mampu memberikan insight/kesimpulan di tiap hasil query dengan singkat, jelas, dan padat                                                                                           |    3   |       |
| Grouping and Aggregating |                           Mampu melakukan pengelompokan data minimal 3 kelompok                          | Siswa mencoba membuat 3 grouping data                                                                                                                                              |    3   |   12  |
|                          |                                                                                                          | Siswa mampu membuat 3 grouping data beserta agregatnya dengan benar sesuai dengan informasi apa yang ingin dilihat. Tiap grouping harus berdasarkan kolom dan agregat yang berbeda |    3   |       |
|                          |                                                                                                          | Memberikan keterangan informasi apa yang dingin dilihat melalui grouping dan aggregating                                                                                           |    3   |       |
|                          |                                                                                                          | Mampu memberikan insight/kesimpulan di tiap hasil grouping dan aggregating dengan singkat, jelas, dan padat                                                                        |    3   |       |
|   Pandas Visualization   |                                Mampu membuat minimal 3 plot dengan Pandas                                | Siswa mencoba membuat 3 data visualisasi dengan pandas plot                                                                                                                        |    3   |   12  |
|                          |                                                                                                          | Siswa mampu membuat data visualisasi untuk 3 kolom yang berbeda menggunakan pandas plot                                                                                            |    3   |       |
|                          |                                                                                                          | Memberikan keterangan informasi apa yang ingin dilihat dari visualisasi data                                                                                                       |    3   |       |
|                          |                                                                                                          | Mampu memberikan insight/kesimpulan di tiap hasil grouping dan aggregating dengan singkaat, jelas, dan padat                                                                       |    3   |       |
|      Runs Perfectly      |              Kode berjalan tanpa ada error. Seluruh kode berfungsi dan dibuat dengan benar.              | Pengerrjaan tugas sebelum dikumpulkan telah dilakukan Restart notebook dan Run All oleh student                                                                                    |    1   |   2   |
|                          |                                                                                                          | Jika tidak ada error sama sekali dari awal hingga akhir setelah dilakukan running ulang                                                                                            |    1   |       |
|     Overall Analysis     |               Menarik informasi/kesimpulan dari keseluruhan eksplorasi data yang dilakukan.              | Siswa mencoba menulis kesimpulan dari apa yang sudah dikerjakan meski tidak relevan                                                                                                |    1   |   6   |
|                          |                                                                                                          | Siswa mampu menarik benang merah dari apa yang telah dikerjakan dalam beberapa kalimat                                                                                             |    3   |       |
|                          |                                                                                                          | Siswa dapat menuliskan kesimpulan dengan singkat, jelas, dan padat                                                                                                                 |    2   |       |
|        Readability       | Semua cell di notebook terdokumentasikan dengan baik dengan markdown pada tiap cell ntuk penjelasan kode | Student menuliskan Nama dan Batch                                                                                                                                                  |    1   |   5   |
|                          |                                                                                                          | Student menuliskan problem statement dengan bahasa sendiri                                                                                                                         |    2   |       |
|                          |                                                                                                          | Semua markdown dan heading digunakan secara tepat sehingga notebook rapih dan dapat dengan mudah dicerna dan ditelusuri                                                            |    1   |       |
|                          |                                                                                                          | Student dapat memberikan penjelasan singkat pada cell code dengan komentar                                                                                                         |    1   |       |

---

```
Total Points : 60
```
---
