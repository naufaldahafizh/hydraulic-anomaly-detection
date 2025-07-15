# pipeline.py

import logging
from data_loader import DataLoader
from preprocessor import Preprocessor
from model_trainer import ModelTrainer
from evaluator import Evaluator

def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("[INFO] Memulai pipeline...")
    
    # Step 1: Load & bersihkan data
    data_path = "./data/raw/merged_data.csv"
    df = DataLoader(data_path).load_data()
    preprocessor = Preprocessor()
    df = preprocessor.clean_column_names(df)

    # Step 2: Pisahkan fitur sensor dan label kondisi
    X_raw, y_true = preprocessor.separate_features_targets(df)
    X_scaled = preprocessor.scale_features(X_raw)

    # Step 3: Latih model deteksi anomali
    trainer = ModelTrainer(contamination=0.05)
    trainer.train(X_scaled)
    predictions = trainer.predict(X_scaled)

    # Step 4: Evaluasi
    # Kita bisa pilih salah satu kondisi: stable sebagai acuan
    if 'stable' in y_true.columns:
        evaluator = Evaluator(y_true['stable'].values)
    else:
        evaluator = Evaluator()
    evaluator.evaluate(predictions)

    # Step 5: Simpan model
    trainer.save_model("./models/isolation_forest_model.pkl")

    logging.info("[INFO] Pipeline selesai dijalankan.")

if __name__ == "__main__":
    main()
