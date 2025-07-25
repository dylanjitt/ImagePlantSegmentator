# -*- coding: utf-8 -*-

"""
Script para visualizar imágenes hiperespectrales de reflectancia.
Usa spectral.imshow() para una visualización clara y especializada.
"""

# Librerías
from spectral import imshow, save_rgb
import spectral.io.envi as envi
import numpy as np
from os import getcwd, makedirs

# Directorio actual y carpeta de salida
wd = getcwd()
makedirs('Fig', exist_ok=True)

# Leer imagen hiperespectral, el envi.open() devuelve un objeto tipo spectral.image que representa el cubo 3d (alto x ancho x bandas espectrales)
reflectance_img = envi.open(
    r"C:\Users\ASUS TUF LAPTOP\Desktop\Autoencoders\codigoOficial\Datos\REFLECTANCE_2020-09-10_004.hdr",
    r"C:\Users\ASUS TUF LAPTOP\Desktop\Autoencoders\codigoOficial\Datos\REFLECTANCE_2020-09-10_004.dat"
)

# Mostrar metadatos disponibles
print("Metadatos encontrados:")
print(reflectance_img.metadata.keys())

# Obtener las bandas RGB por defecto desde el encabezado
rgb_bands = [int(b) for b in reflectance_img.metadata['default bands']]
#[100,80,40]
# Mostrar imagen RGB con bandas por defecto
imshow(reflectance_img, rgb_bands)
# Guardar RGB como imagen si lo deseas
save_rgb('Fig/imreflectance_rgb.png', reflectance_img, bands=rgb_bands)

# Zoom en una región de la imagen (hoja)
imZoom = reflectance_img[150:200, 150:200, :]
imshow(imZoom, rgb_bands)
# No se puede guardar directamente el zoom con save_rgb a menos que se cree una imagen falsa

# Obtener espectros de la subimagen para graficar
sp = np.reshape(imZoom, (50 * 50, reflectance_img.shape[2]))
lbds = [float(x) for x in reflectance_img.metadata['wavelength']]

# Mostrar los espectros de los primeros 71 píxeles
import matplotlib.pyplot as plt
plt.figure()
for i in range(71):
    plt.plot(lbds, sp[i, :])
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Spectra of the subimage')
plt.savefig('Fig/spectra.pdf')

# Análisis de componentes principales (SVD)
sp_centered = sp - sp.mean(axis=0)
U, s, V = np.linalg.svd(sp_centered)

# Obtener las primeras 4 componentes principales
pcs = [V.T[:, i] for i in range(4)]
titles = ['PC1', 'PC2', 'PC3', 'PC4']

# Graficar las cargas de cada componente
for i, pc in enumerate(pcs):
    plt.figure()
    plt.plot(lbds, pc)
    plt.axhline(color='gray', linestyle='--')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Loadings')
    plt.title(f'Loadings for {titles[i]}')
    plt.savefig(f'Fig/{titles[i].lower()}.pdf')

# Calcular scores para PC1-PC4
scores = [sp_centered.dot(pc) for pc in pcs]
xilX = np.tile(np.arange(0, 50, 1), 50)
xilY = np.sort(xilX)

# Graficar combinaciones de scores
plt.figure()
plt.scatter(scores[0], scores[1], s=xilX, c=xilY)
plt.axhline(color='gray', linestyle='--')
plt.axvline(color='gray', linestyle='--')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('Fig/sc12.pdf')

plt.figure()
plt.scatter(scores[2], scores[3], s=xilX, c=xilY)
plt.axhline(color='gray', linestyle='--')
plt.axvline(color='gray', linestyle='--')
plt.xlabel('PC3')
plt.ylabel('PC4')
plt.savefig('Fig/sc34.pdf')

input("Presiona ENTER para cerrar las ventanas...")
