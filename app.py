import streamlit as st
import joblib
import pandas as pd

# 1. Load Model
# Pastikan nama file model di bawah ini SESUAI dengan nama file .pkl yang kamu unduh dari Colab
model = joblib.load('model_datamining.pkl') 

# 2. Judul & Deskripsi Aplikasi
st.title("Aplikasi Prediksi Harga Rumah")
st.write("Masukkan spesifikasi rumah untuk melihat estimasi harganya berdasarkan algoritma Data Mining.")

# 3. Form Input untuk User (Membagi layar menjadi 2 kolom agar rapi)
col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("Luas Bangunan (Square Footage)", min_value=500, value=2000)
    num_bedrooms = st.number_input("Jumlah Tempat Tidur", min_value=1, value=3)
    num_bathrooms = st.number_input("Jumlah Kamar Mandi", min_value=1, value=2)
    year_built = st.number_input("Tahun Dibangun", min_value=1950, max_value=2024, value=2000)

with col2:
    lot_size = st.number_input("Luas Tanah (Lot Size)", min_value=0.0, value=2.0)
    garage_size = st.number_input("Kapasitas Garasi (Mobil)", min_value=0, value=1)
    # Menggunakan slider karena kualitas lingkungan biasanya memiliki rentang pasti (misal 1-10)
    neighborhood_quality = st.slider("Kualitas Lingkungan", min_value=1, max_value=10, value=5)

# 4. Tombol Prediksi
if st.button("Prediksi Harga"):
    # 5. Menyusun input dari user ke dalam DataFrame Pandas
    # PENTING: Nama kolom di bawah ini harus SAMA PERSIS (huruf besar/kecil) dengan dataset saat training
    data_baru = pd.DataFrame({
        'Square_Footage': [square_footage],
        'Num_Bedrooms': [num_bedrooms],
        'Num_Bathrooms': [num_bathrooms],
        'Year_Built': [year_built],
        'Lot_Size': [lot_size],
        'Garage_Size': [garage_size],
        'Neighborhood_Quality': [neighborhood_quality]
    })
    
    # 6. Melakukan Prediksi
    try:
        hasil_prediksi = model.predict(data_baru)
        # Menampilkan output harga dengan format mata uang (Dolar)
        st.success(f"Estimasi Harga Rumah: ${hasil_prediksi[0]:,.2f}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memprediksi: {e}")
        st.warning("Pastikan jumlah kolom input ini sama dengan kolom (X_train) yang kamu gunakan saat melatih model di Colab.")