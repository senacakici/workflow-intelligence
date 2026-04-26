import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Simüle edilmiş artan hava akışı verileri
# Hesaplanmış yüzeysel hızlar (m/s), artan dosyasından hesaplanan u_artan değerleri
u_artan = np.array([0.000, 0.044, 0.065, 0.087, 0.096, 0.103, 0.106, 0.116, 0.131, 0.141])
# Simüle edilmiş basınç düşüşü değerleri (mm H2O)
# Baslangıçta lineer artan, sonra sabitlenmiş (örneğin 120 mmH2O)
dp_artan = np.array([0, 30, 60, 90, 110, 120, 120, 120, 120, 120])
# Simüle edilmiş yatak yüksekliği değerleri (mm) – genelde minimum akışkanlaşma sonrası sabitlenir
h_artan = np.array([35, 37, 38, 39, 40, 40, 40, 40, 40, 40])

# Belirleme: pre-akışkanlaşma bölgesi (lineer artış) için veri noktalarını seçelim
# Burada dp değerleri lineer artış gösterirken, sabitlenmeye yakın olanları hariç tutuyoruz.
# Örneğin, ilk 6 veri noktasını (indeks 0-5) kullanıyoruz.
u_linear = u_artan[:6]
dp_linear = dp_artan[:6]

# Lineer regresyon (teğet) hesaplaması
slope, intercept, r_value, p_value, std_err = linregress(u_linear, dp_linear)
# Regresyon doğrusunu çizmek için
u_fit = np.linspace(u_linear[0], u_linear[-1], 100)
dp_fit = intercept + slope*u_fit

# Minimum akışkanlaşma hızı (Umf) olarak, regresyon doğrusunun 120 mmH2O değerini verdiği u'yu hesaplayalım
umf = (120 - intercept) / slope

# Grafik oluşturma
plt.figure(figsize=(10, 6))

# Basınç düşüşü eğrisi (artan hava akışı)
plt.plot(u_artan, dp_artan, 's-', color='orangered', label='Yatak Boyunca Basınç Düşüşü (∆P)')


plt.plot(u_artan, h_artan, 'o-', color='blue', label='Yatak Yüksekliği (h)')


plt.plot(u_fit, dp_fit, '-', color='black', label='Teğet (Lineer Regresyon)')


plt.axvline(x=umf, color='green', linestyle='--', label=f'Minimum Akışkanlaşma Hızı (Umf = {umf:.3f} m/s)')


plt.plot(umf, 120, 'ko')  # nokta

plt.xlabel('Yüzeysel Hız (m/s)')
plt.ylabel('Basınç Düşüşü (mm H2O) / Yatak Yüksekliği (mm)')
plt.title('Artan Hava Akışı: Basınç Düşüşü, Yatak Yüksekliği ve Minimum Akışkanlaşma Hızı')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
