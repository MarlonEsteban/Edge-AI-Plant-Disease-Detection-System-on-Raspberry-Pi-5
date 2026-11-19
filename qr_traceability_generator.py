import qrcode

# Reemplaza la IP si cambió hoy (usa ipconfig en tu CMD)
url_correcta = "http://192.168.100.254:5000/planta/plt-001"

qr = qrcode.make(url_correcta)
qr.save("qr_plt_001.png")
print("✅ Código QR para PLT-001 generado. Listo para pegar en tu documento.")