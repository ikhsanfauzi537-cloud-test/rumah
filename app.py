import streamlit as st
import joblib
import pandas as pd

# 1. Load Model
# Ganti 'model_datamining.pkl' dengan nama file modelmu yang sebenarnya jika berbeda
model = joblib.load('final_linear_regression_model.pkl') 

# 2. Judul & Deskripsi Aplikasi
st.title("Aplikasi Prediksi Harga Rumah")
st.write("Masukkan spesifikasi luas bangunan dan luas tanah untuk melihat estimasi harganya.")

# 3. Form Input untuk User 
# Kita gunakan 2 kolom agar tampilannya bersebelahan
col1, col2 = st.columns(2)

with col1:
    square_footage = st.number_input("Luas Bangunan (Square Footage)", min_value=500, value=2500)

with col2:
    lot_size = st.number_input("Luas Tanah (Lot Size)", min_value=0.0, value=2.5, step=0.1)

# 4. Tombol Prediksi
if st.button("Prediksi Harga"):
    # 5. Menyusun input dari user ke dalam DataFrame Pandas (Hanya 2 Fitur)
    data_baru = pd.DataFrame({
        'Square_Footage': [square_footage],
        'Lot_Size': [lot_size]
    })
    
    # 6. Melakukan Prediksi
    try:
        hasil_prediksi = model.predict(data_baru)
        
        # Menampilkan hasil dengan kotak hijau sukses, format sesuai dengan yang di Colab
        st.success(f"Estimasi Harga Rumah: ${hasil_prediksi[0]:,.2f}")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
