# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 23:12:11 2024

@author: 90545
"""
 ## fena değil
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

def capture_image(camera_id=0, output_filename='spectrum_image.jpg'):
    """Kamera ile fotoğraf çek ve kaydet"""
    camera = cv2.VideoCapture(camera_id)  # Kamerayı başlat
    ret, frame = camera.read()
    
    if ret: #Trueysa dondur
        cv2.imwrite(output_filename, frame)  # Fotoğrafı kaydet
        print(f"Image saved as {output_filename}")
    
    camera.release()  # Kamerayı serbest bırak
    return output_filename

def calibrate(pixel_positions, known_wavelengths):
    """Kalibrasyon katsayılarını hesapla"""
    coefficients = np.polyfit(pixel_positions, known_wavelengths, 1)  # Doğrusal fit
    print("Calibration Coefficients:", coefficients)
    return coefficients

def pixel_to_wavelength(pixel, coefficients):
    """Piksel pozisyonunu dalga boyuna çevir"""
    return coefficients[0] * pixel + coefficients[1]

#### !!!!bunu chatgptden aldım.
def wavelength_to_rgb(wavelength):
    """Dalga boyunu RGB renk değerine çevir"""
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R = G = B = 0

    if (wavelength >= 380) and (wavelength < 440):
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif (wavelength >= 440) and (wavelength < 490):
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif (wavelength >= 490) and (wavelength < 510):
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif (wavelength >= 510) and (wavelength < 580):
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif (wavelength >= 580) and (wavelength < 645):
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif (wavelength >= 645) and (wavelength <= 750):
        R = 1.0
        G = 0.0
        B = 0.0

    # Adjust intensity
    R = round(intensity_max * (R ** gamma))
    G = round(intensity_max * (G ** gamma))
    B = round(intensity_max * (B ** gamma))

    return (R, G, B)

def analyze_spectrum(image_filename, coefficients):
    """Görüntüyü analiz et ve dalga boyu yoğunluğunu hesapla"""
    image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)  # Gri tonlama yapıyorum ki intensity bulabileyim.
    if image is None:
        print(f"Failed to load image {image_filename}")
        return

    # Gürültü azaltmak için Gauss bulanıklığı uygulama
    image_blurred = cv2.GaussianBlur(image, (5, 5), 0) # kenar yumusatma icin kullanılıyor
    
    # Spektrum boyunca yoğunluk değerlerini hesapla (gri tonlamalı)
    intensity = np.mean(image_blurred, axis=0)  # Y ekseni boyunca ortalama
    pixel_positions = np.arange(len(intensity))  # Piksel pozisyonları
    
    # Dalga boylarını hesapla
    wavelengths = pixel_to_wavelength(pixel_positions, coefficients)

    # Yoğunluğu dalga boylarına karşı çiz
    plt.plot(wavelengths, intensity, label="Spectrum")
    plt.title('Spectral Analysis')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')


    ### yukarıdaki fonksiyonu /// chatgptden aldığım
    # Smooth renk geçişleriyle çizgi altını doldurmak için
    for i in range(1, len(wavelengths)):
        # Ardışık iki dalga boyu arasındaki renk geçişi
        color_start = wavelength_to_rgb(wavelengths[i-1])
        color_end = wavelength_to_rgb(wavelengths[i])
        
        # Linear interpolation (yumuşak geçiş) ile renkleri arada karıştır
        color = np.array(color_start) * (1 - (i / len(wavelengths))) + np.array(color_end) * (i / len(wavelengths))
        color = color.astype(int)
        
        # Renk ile alanı doldur
        plt.fill_between(wavelengths[i-1:i+1], 0, intensity[i-1:i+1], color=color/255, alpha=0.5)

    plt.legend()
    plt.show()

# 1. Görüntü yakala
image_filename = capture_image(camera_id=0, output_filename='spectrum_image.jpg')

if image_filename:
    # 2. Kalibrasyon
    # Kalibrasyon için bilinen LED'lerin dalga boyları ve görüntüdeki piksel pozisyonları
    known_wavelengths = [450, 525, 625]  # nm (mavi, yeşil, kırmızı)
    pixel_positions = [100, 250, 400]   # Görüntüdeki pikselimizin konumları, nasıl belirlerim?
    coefficients = calibrate(pixel_positions, known_wavelengths)

    # 3. Spektral Analiz
    analyze_spectrum(image_filename, coefficients)
