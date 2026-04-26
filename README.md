<h1>🦠 COVID-19 Indonesian Twitter Sentiment Analysis</h1>

  <p align="center">
    <strong>Analisis sentimen tweet berbahasa Indonesia terkait COVID-19 menggunakan IndoBERTweet, LDA Topic Modeling, dan SVM Classifier.</strong>
    <br /><br />
    <a href="https://www.kaggle.com/datasets/dionisiusdh/covid19-indonesian-twitter-sentiment">
      📦 Dataset Kaggle
    </a>
    &nbsp;&middot;&nbsp;
    <a href="https://colab.research.google.com/">
      🚀 Open in Colab
    </a>
  </p>

  <!-- Badges -->
  <p>
    <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" alt="Python"/>
    <img src="https://img.shields.io/badge/Platform-Google%20Colab-orange?style=flat-square&logo=googlecolab" alt="Colab"/>
    <img src="https://img.shields.io/badge/NLP-IndoBERTweet-green?style=flat-square" alt="IndoBERTweet"/>
    <img src="https://img.shields.io/badge/Model-SVM%20RBF-red?style=flat-square" alt="SVM"/>
    <img src="https://img.shields.io/badge/Accuracy-81.39%25-brightgreen?style=flat-square" alt="Accuracy"/>
    <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
  </p>

</div>

---

## 📋 Table of Contents

- [📖 About](#-about)
- [📊 Dataset](#-dataset)
- [🔬 Methodology](#-methodology)
- [🛠️ Tech Stack](#️-tech-stack)
- [📁 Project Structure](#-project-structure)
- [⚡ Quick Start](#-quick-start)
- [📈 Results](#-results)
- [👤 Author](#-author)

---

## 📖 About

Project ini adalah tugas **Big Data Scraping** (KDD01 - 034) yang bertujuan menganalisis sentimen publik Indonesia di Twitter terkait pandemi COVID-19. Pipeline lengkap mencakup preprocessing teks berbahasa Indonesia, pelabelan sentimen otomatis dengan model Transformer, topic modeling, hingga klasifikasi dengan machine learning.

> **Pertanyaan Penelitian:** Bagaimana sentimen masyarakat Indonesia di Twitter terhadap isu COVID-19? Topik apa yang paling banyak dibicarakan?

---

## 📊 Dataset

**Sumber:** [COVID-19 Indonesian Twitter Sentiment — Kaggle](https://www.kaggle.com/datasets/dionisiusdh/covid19-indonesian-twitter-sentiment)

| Atribut         | Detail                                  |
|-----------------|-----------------------------------------|
| **Platform**    | Twitter / X                             |
| **Bahasa**      | Indonesia                               |
| **Total Tweet** | ±45.949 tweet                           |
| **Kolom Utama** | `tweet` (teks mentah)                   |
| **Label Sentimen** | Positive, Neutral, Negative          |

### Distribusi Sentimen (Hasil Labeling IndoBERTweet)

| Label        | Jumlah  | Persentase |
|--------------|---------|------------|
| 🟡 Neutral   | 32.278  | ~70.2%     |
| 🔴 Negative  | 10.823  | ~23.6%     |
| 🟢 Positive  | 2.848   | ~6.2%      |

---

## 🔬 Methodology

Pipeline proyek ini mengikuti tahapan KDD (Knowledge Discovery in Databases):

```
Raw Tweets
    │
    ▼
1. TEXT PREPROCESSING
   ├── Case Folding (lowercase)
   ├── Hapus URL, mention, hashtag, angka, tanda baca
   ├── Tokenisasi
   ├── Stopword Removal (Bahasa Indonesia)
   └── Stemming (PySastrawi / Indonesian Stemmer)
    │
    ▼
2. SENTIMENT LABELING (IndoBERTweet)
   └── Model: Aardiiiy/indobertweet-base-Indonesian-sentiment-analysis
       ├── Positive
       ├── Neutral
       └── Negative
    │
    ▼
3. TOPIC MODELING (LDA)
   ├── Jumlah Topik: 5
   ├── Library: Gensim
   └── Visualisasi distribusi topik per sentimen
    │
    ▼
4. CLASSIFICATION (SVM)
   ├── Vectorizer: TF-IDF (max_features=5000)
   ├── Model: SVM Kernel RBF
   ├── Split: 80% train / 20% test
   └── Akurasi: 81.39%
    │
    ▼
5. VISUALIZATION
   ├── Pie Chart distribusi sentimen
   └── Word Cloud per kategori sentimen
```

---

## 🛠️ Tech Stack

| Kategori       | Library / Tool                                                  |
|----------------|-----------------------------------------------------------------|
| **Language**   | Python 3.12                                                     |
| **Platform**   | Google Colab                                                    |
| **NLP Model**  | `Aardiiiy/indobertweet-base-Indonesian-sentiment-analysis` (HuggingFace Transformers) |
| **Topic Model**| Gensim (LDA Multicore)                                          |
| **ML**         | Scikit-learn (SVM RBF, TF-IDF, train_test_split)               |
| **Data**       | Pandas, Swifter                                                 |
| **Visualisasi**| Matplotlib, Seaborn, WordCloud                                  |
| **Stemming**   | PySastrawi / Indonesian Stemmer                                 |

---

## 📁 Project Structure

```
project/
├── 📓 Scraping_Big_Data_-X-_Farrel_Farica_F_KDD01_034.ipynb   # Main notebook
├── 📄 README.md
│
├── 📂 data/ (generate saat runtime)
│   ├── covid19_tweets.csv                  # Dataset mentah dari Kaggle
│   ├── stemmedtweets_checkpoint.csv        # Checkpoint hasil stemming
│   └── content/sentimentanalysisresults.csv  # Hasil labeling sentimen
│
└── 📂 output/ (generate saat runtime)
    ├── sentiment_distribution_pie.png      # Visualisasi distribusi sentimen
    ├── wordcloud_positive.png              # Word Cloud sentimen positif
    ├── wordcloud_neutral.png              # Word Cloud sentimen netral
    └── wordcloud_negative.png             # Word Cloud sentimen negatif
```

---

## ⚡ Quick Start

### Prerequisites

- Akun Google untuk menjalankan di **Google Colab**
- Akun Kaggle + API Key (untuk download dataset)

### Langkah-langkah

**1. Buka notebook di Google Colab**

Klik badge **Open in Colab** di bagian atas, atau upload file `.ipynb` ke [colab.research.google.com](https://colab.research.google.com).

**2. Download dataset dari Kaggle**

```python
# Install Kaggle API
!pip install kaggle

# Upload kaggle.json (API key kamu)
from google.colab import files
files.upload()  # Upload kaggle.json

# Setup dan download dataset
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d dionisiusdh/covid19-indonesian-twitter-sentiment
!unzip covid19-indonesian-twitter-sentiment.zip
```

**3. Install dependencies**

```bash
pip install transformers gensim swifter wordcloud PySastrawi scikit-learn
```

**4. Jalankan sel notebook secara berurutan**

> ⚠️ **Perhatian:** Proses sentiment labeling dengan IndoBERTweet memerlukan waktu lama untuk ~45K tweet. Notebook sudah menyertakan **checkpoint CSV** agar tidak perlu mengulang proses dari awal jika sesi Colab terputus.

---

## 📈 Results

### Performa Model Klasifikasi

| Model       | Kernel | Vectorizer | Accuracy   |
|-------------|--------|------------|------------|
| **SVM**     | RBF    | TF-IDF     | **81.39%** |

### LDA Topic Modeling (5 Topik)

| Topic | Top Keywords                                                      | Tema                   |
|-------|-------------------------------------------------------------------|------------------------|
| 0     | pandemi, dampak, bantu, tangan, masyarakat, ekonomi              | Dampak Sosial-Ekonomi  |
| 1     | sebar, cegah, patuh, virus, protokol, sehat, himbauan            | Protokol Kesehatan     |
| 2     | normal, new, ya, pandemi, rakyat, indonesia, kalo                | New Normal             |
| 3     | putus, mudik, rantai, mata, ikut, imbau, polri                   | Larangan Mudik         |
| 4     | tangan, positif, corona, orang, pasien, virus, data              | Informasi Kasus        |

---

## 👤 Author

**Farrel Farica F.**
Universitas — Kelas KDD01 (NIM: 034)

- 🐙 GitHub: [@farrelfaricaf](https://github.com/farrelfaricaf)
- 💼 LinkedIn: [linkedin.com/in/farrelfaricaf](https://www.linkedin.com/in/farrelfaricaf/)

---

## 📜 License

Distributed under the **MIT License**.
