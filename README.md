# Hydraulic Anomaly Detection

[![CI - Anomaly Detection Pipeline](https://github.com/naufaldahafizh/hydraulic-anomaly-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/naufaldahafizh/hydraulic-anomaly-detection/actions)

Deteksi anomali pada sistem industri berbasis sensor menggunakan model unsupervised learning (Isolation Forest).  
Dibangun dengan struktur kode modular (OOP), unit testing, dan pipeline CI/CD production-ready.

---

## Struktur Proyek

- `src/`: pipeline modular (loader, preprocessor, model, evaluator)
- `tests/`: unit test untuk setiap komponen
- `data/`: raw dan processed dataset
- `notebooks/`: eksplorasi data dan analisis visual
- `pipeline.py`: eksekusi training pipeline end-to-end
- `.github/workflows/`: GitHub Actions CI
- `README.md`: dokumentasi proyek

---

## Dataset
- **UCI Hydraulic Condition Monitoring**
- Sensor tekanan, suhu, vibrasi
- 2205 sampel × 51 fitur
- Target: kondisi 4 komponen sistem (Cooler, Valve, Pump, Accumulator)

---

## Cara Menjalankan

1. **Clone repo & install dependensi**
```bash
git clone https://github.com/naufaldahafizh/hydraulic-anomaly-detection.git
cd hydraulic-anomaly-detection
pip install -r requirements.txt
```

2. **Jalankan pipeline**
```bash
python src/pipeline.py
```

3. **Jalankan unit tests**
```bash
pytest tests/
```

Citation
If using this dataset, please cite: Nikolai Helwig, Eliseo Pignanelli, Andreas Sch�tze, �Condition Monitoring of a Complex Hydraulic System Using Multivariate Statistics�, in Proc. I2MTC-2015 - 2015 IEEE International Instrumentation and Measurement Technology Conference, paper PPS1-39, Pisa, Italy, May 11-14, 2015, doi: 10.1109/I2MTC.2015.7151267.
