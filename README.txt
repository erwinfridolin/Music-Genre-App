Music Genre Classification with LSTM

## Workflow
1. Extract MFCC (Mel-Frequency Cepstral Coefficients) features from audio dataset
2. Train LSTM model 
3. Save trained model
4. Deploy inference using Streamlit





















Struktur program
music-genre-classification/
│
├── app/
│   ├── app.py
│   └── audio_utils.py
│
├── training/
│   ├── extract_features.py
│   ├── train_lstm.py
│   └── label_encoder.pkl   (hasil training)
│
├── model/
│   └── lstm_genre_model.h5 (hasil training)
│
├── data/ 
│   └── processed/
│       ├── X.npy
│       └── y.npy
|       raw/
│       ├── dataset
│      
|
|
│
├── requirements.txt
├── README.md
└── .gitignore