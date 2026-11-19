import os
import cv2
import sqlite3
import numpy as np
import qrcode
from datetime import datetime
from flask import Flask, render_template_string, g, request, redirect, url_for, send_file, Response, make_response
from fpdf import FPDF

# --- IMPORTACIÓN DE YOLO ---
try:
    from ultralytics import YOLO
    modelo_yolo = YOLO("best_ncnn_model", task="detect") 
    print("✅ Modelo YOLO NCNN cargado correctamente.")
except Exception as e:
    modelo_yolo = None
    print("========================================")
    print(f"⚠️ ERROR FATAL AL CARGAR EL MODELO IA:\n{str(e)}")
    print("========================================")
    print("⚠️ Funcionando en modo simulación.")

app = Flask(__name__)
DATABASE = 'cultivos.db'

# --- CREAR CARPETAS ---
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/referencias', exist_ok=True)
os.makedirs('static/qrs', exist_ok=True)

# --- CONFIGURACIÓN DE CÁMARA EN TIEMPO REAL ---
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

def generar_frames():
    """Captura video y aplica la IA en tiempo real"""
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # APLICAR IA EN TIEMPO REAL
            if modelo_yolo is not None:
                try:
                    resultados = modelo_yolo(frame, verbose=False)
                    cajas = resultados[0].boxes
                    
                    if len(cajas) == 0:
                        # Si no detecta enfermedades, escribe "Hoja Sana" en verde
                        cv2.putText(frame, "Hoja Sana", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    else:
                        # Si detecta algo, dibuja las cajas de detección normales
                        frame = resultados[0].plot()
                except Exception:
                    pass 

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ==========================================
# 1. FUNCIONES DE VISIÓN CLÁSICA
# ==========================================
def leer_qr_desde_imagen(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None: return None
    detector = cv2.QRCodeDetector()
    datos, bbox, _ = detector.detectAndDecode(img)
    if datos: return datos 
    return None

def extraer_area_hoja(ruta_original, ruta_recortada):
    img = cv2.imread(ruta_original)
    if img is None: return False
        
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    desenfoque = cv2.GaussianBlur(gris, (5, 5), 0)
    bordes = cv2.Canny(desenfoque, 50, 150)
    
    contornos, _ = cv2.findContours(bordes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    
    alto_img, ancho_img = img.shape[:2]
    area_total = alto_img * ancho_img
    
    recorte_exitoso = False
    for c in contornos:
        area = cv2.contourArea(c)
        if area < area_total * 0.05: continue
            
        perimetro = cv2.arcLength(c, True)
        aproximacion = cv2.approxPolyDP(c, 0.02 * perimetro, True)
        
        if 4 <= len(aproximacion) <= 8:
            x, y, w, h = cv2.boundingRect(aproximacion)
            if w > ancho_img * 0.95 and h > alto_img * 0.95: continue
                
            margen = 10
            y1, y2 = max(0, y+margen), min(img.shape[0], y+h-margen)
            x1, x2 = max(0, x+margen), min(img.shape[1], x+w-margen)
            
            recorte = img[y1:y2, x1:x2]
            if recorte.size > 0:
                cv2.imwrite(ruta_recortada, recorte)
                recorte_exitoso = True
                break
            
    if not recorte_exitoso:
        cv2.imwrite(ruta_recortada, img) 
    return True

# ==========================================
# 2. BASE DE DATOS, TENDENCIAS Y RECOMENDACIONES
# ==========================================
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row 
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None: db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS plantas (
                id_planta TEXT PRIMARY KEY,
                fecha_cultivo TEXT,
                fecha_fumigacion TEXT,
                ultima_enfermedad TEXT,
                ruta_foto_analizada TEXT,
                ruta_foto_referencia TEXT,
                proxima_fumigacion TEXT DEFAULT 'No programada'
            )
        ''')
        try:
            cursor.execute("ALTER TABLE plantas ADD COLUMN proxima_fumigacion TEXT DEFAULT 'No programada'")
        except sqlite3.OperationalError:
            pass

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS historial (
                id_historial INTEGER PRIMARY KEY AUTOINCREMENT,
                id_planta TEXT,
                fecha_escaneo TEXT,
                diagnostico TEXT
            )
        ''')
        db.commit()

def registrar_historial(id_planta, diagnostico):
    db = get_db()
    cursor = db.cursor()
    fecha_actual = datetime.now().strftime("%d/%m/%Y %H:%M")
    cursor.execute("INSERT INTO historial (id_planta, fecha_escaneo, diagnostico) VALUES (?, ?, ?)", (id_planta, fecha_actual, diagnostico))
    db.commit()

def calcular_tendencia(historial):
    if len(historial) < 2:
        return "Faltan datos", "⚪ Recolecte otro escaneo"
    
    pesos = {"sana": 0, "leve": 1, "moderada": 2, "medio": 2, "grave": 3}
    actual = historial[0]['diagnostico'].lower()
    anterior = historial[1]['diagnostico'].lower()
    
    p_actual, p_anterior = 0, 0
    for k, v in pesos.items():
        if k in actual: p_actual = v
        if k in anterior: p_anterior = v
        
    if p_actual < p_anterior: return "Mejorando", "🟢 La planta se está recuperando"
    elif p_actual > p_anterior: return "Empeorando", "🔴 La infección avanza. Fumigue pronto."
    else: return "Estable", "🟡 Sin cambios significativos respecto al último escaneo."

def obtener_recomendacion(diagnostico):
    """Devuelve el producto sugerido y el proceso según la gravedad de la enfermedad"""
    diag = diagnostico.lower()
    if "sana" in diag:
        producto = "Ninguno requerido."
        proceso = "Mantener el programa actual de riego y fertilizacion. Continuar monitoreo rutinario."
    elif "leve" in diag:
        producto = "Fungicida / Tratamiento preventivo (Ej. Base de Cobre o Azufre)."
        proceso = "Aplicar rociado foliar ligero. Aislar la planta preventivamente y revisar humedad del suelo."
    elif "medio" in diag or "moderada" in diag:
        producto = "Fungicida sistemico de amplio espectro."
        proceso = "Podar hojas severamente afectadas. Aplicar fungicida cada 7 dias. Mejorar la ventilacion y reducir la humedad ambiental."
    elif "grave" in diag:
        producto = "Tratamiento quimico de choque (Fungicida de rescate de accion rapida)."
        proceso = "Cuarentena estricta del lote. Poda agresiva. Si no hay mejora en 7 dias, proceder con la incineracion de la planta para evitar propagacion cruzada."
    else:
        producto = "Evaluacion manual requerida."
        proceso = "Contactar a un especialista agronomo para determinar tratamiento."
    
    return producto, proceso

# ==========================================
# 3. PLANTILLAS HTML
# ==========================================

PLANTILLA_CAMARA = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cámara en Vivo - IA</title>
    <style>
        body { font-family: Arial; padding: 20px; background-color: #f4f4f9; text-align: center; }
        .video-container { border: 4px solid #1abc9c; border-radius: 10px; display: inline-block; overflow: hidden; margin-bottom: 20px; background: black;}
        .btn { padding: 15px 25px; border-radius: 8px; color: white; text-decoration: none; font-weight: bold; display: inline-block; margin: 10px; cursor: pointer; border: none; font-size: 16px; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);}
        .btn-volver { background-color: #7f8c8d; }
        .btn-foto { background-color: #9b59b6; }
        .btn-galeria { background-color: #3498db; }
        .btn-instantaneo { background-color: #e67e22; display: block; max-width: 300px; margin: 20px auto; }
        .btn:hover { opacity: 0.9; transform: translateY(-2px); }
    </style>
</head>
<body>
    <h1 style="color: #2c3e50;">📸 Cámara IA en Tiempo Real</h1>
    <p style="color: #7f8c8d; font-size: 16px;">Apunta a la hoja para ver predicciones en vivo.</p>
    
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" width="640" style="max-width: 100%; height: auto; display: block;">
    </div>
    
    <div>
        <a href="/reporte_en_vivo" class="btn btn-instantaneo">📥 Descargar Reporte de esta Toma</a>
    </div>

    <div style="margin-top: 10px;">
        <form action="/escanear_global" method="POST" enctype="multipart/form-data" style="display: inline-block;">
            <label class="btn btn-foto">
                📸 Escanear Planta y QR <input type="file" name="foto_global" accept="image/*" capture="environment" style="display: none;" onchange="this.form.submit()">
            </label>
        </form>
        
        <form action="/escanear_global" method="POST" enctype="multipart/form-data" style="display: inline-block;">
            <label class="btn btn-galeria">
                🖼️ Subir desde Galería <input type="file" name="foto_global" accept="image/*" style="display: none;" onchange="this.form.submit()">
            </label>
        </form>
    </div>

    <div style="margin-top: 40px;">
        <a href="/panel" class="btn btn-volver">⬅ Volver al Dashboard</a>
    </div>
</body>
</html>
"""

PLANTILLA_PANEL = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panel Agrícola</title>
    <style>
        body { font-family: Arial; padding: 20px; background-color: #f4f4f9; }
        h1 { color: #2c3e50; text-align: center; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; background: white; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #2c3e50; color: white; }
        .estado-grave { background-color: #ffcccc; color: #cc0000; font-weight: bold; }
        .estado-moderado { background-color: #fff3cd; color: #856404; font-weight: bold; }
        .estado-leve { background-color: #fff9c4; color: #f39c12; font-weight: bold; }
        .estado-sano { background-color: #d4edda; color: #155724; font-weight: bold; }
        .btn-sm { padding: 8px 12px; border-radius: 4px; color: white; text-decoration: none; font-size: 14px; font-weight: bold; display: inline-block; margin: 2px;}
        .btn-ver { background-color: #3498db; }
        .btn-qr { background-color: #e67e22; }
        .btn-agregar { background-color: #2ecc71; font-size: 16px; }
        .btn-eliminar { background-color: #e74c3c; font-size: 14px; cursor: pointer; border: none;}
        .cabecera { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;}
        .escaner-global { background: white; border-left: 5px solid #1abc9c; padding: 20px; border-radius: 8px; margin-bottom: 25px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);}
        .btn-camara-grande { background-color: #1abc9c; color: white; padding: 15px 30px; text-decoration: none; font-size: 18px; font-weight: bold; border-radius: 8px; display: inline-block; margin-top: 10px;}
        .btn-camara-grande:hover { background-color: #16a085; }
    </style>
</head>
<body>
    <div class="cabecera">
        <h1>🎛️ Centro de Monitoreo - Tesis UPS</h1>
        <div style="margin-bottom: 10px;">
            <a href="/agregar" class="btn-sm btn-agregar">➕ Agregar Planta</a>
            <a href="/panel" class="btn-sm btn-ver" style="font-size: 16px;">🔄 Actualizar</a>
        </div>
    </div>
    
    <div class="escaner-global">
        <h3 style="margin-top: 0; color: #2c3e50;">🧠 Módulo de Inteligencia Artificial</h3>
        <p style="color: #7f8c8d; font-size: 15px; margin-bottom: 15px;">Accede a la cámara para ver predicciones en tiempo real, escanear códigos QR de tus plantas o generar un reporte rápido.</p>
        <a href="/camara_en_vivo" class="btn-camara-grande">🔴 Abrir Cámara en Vivo e IA</a>
    </div>

    <table>
        <tr>
            <th>ID Planta</th>
            <th>Diagnóstico Actual</th>
            <th>Acción Recomendada</th>
            <th>Próx. Fumigación</th>
            <th>Controles</th>
        </tr>
        {% for planta in plantas %}
            {% set diag = planta['ultima_enfermedad'] %}
            {% set clase_estado = 'estado-sano' if 'Sana' in diag else ('estado-grave' if 'Grave' in diag else ('estado-moderado' if 'Medio' in diag or 'Moderada' in diag else ('estado-leve' if 'Leve' in diag else ''))) %}
            <tr class="{{ clase_estado }}">
                <td>{{ planta['id_planta'] | upper }}</td>
                <td>{{ diag }}</td>
                <td>{% if 'Grave' in diag %} 🚨 Fumigar {% elif 'Medio' in diag or 'Moderada' in diag %} ⚠️ Revisar pronto {% elif 'Leve' in diag %} 🧐 Observar {% elif 'Sana' in diag %} ✅ Normal {% else %} ⏳ Esperando {% endif %}</td>
                <td style="font-weight:bold; color:#2980b9;">{{ planta['proxima_fumigacion'] }}</td>
                <td>
                    <a href="/descargar_qr/{{ planta['id_planta'] }}" class="btn-sm btn-qr">⬇️ QR</a>
                    <a href="/planta/{{ planta['id_planta'] }}" class="btn-sm btn-ver">📄 Reporte y Progreso</a>
                    <a href="/eliminar/{{ planta['id_planta'] }}" class="btn-sm btn-eliminar" onclick="return confirm('⚠️ Borrar planta y fotos?');">🗑️ Borrar</a>
                </td>
            </tr>
        {% endfor %}
    </table>
</body>
</html>
"""

PLANTILLA_INDIVIDUAL = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0"> 
    <title>Reporte de Progreso</title>
    <style>
        body { font-family: Arial; text-align: center; padding: 20px; background-color: #f4f4f9; }
        .tarjeta { background: white; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 10px rgba(0,0,0,0.1); text-align: left; max-width: 450px; margin: auto; margin-bottom: 20px;}
        .dato { font-size: 16px; margin: 8px 0; color: #34495e; }
        .alerta { color: #e74c3c; font-weight: bold; font-size: 18px;}
        .btn-panel { display: inline-block; margin-top: 15px; text-decoration: none; color: #7f8c8d; font-weight: bold;}
        .historial-item { background: #f8f9fa; border-left: 4px solid #3498db; padding: 10px; margin-bottom: 8px; border-radius: 4px; font-size:14px;}
        .input-fecha { padding: 8px; border: 1px solid #ccc; border-radius: 4px; width: 60%; }
        .btn-fecha { padding: 8px 12px; background: #e67e22; border:none; color: white; font-weight: bold; border-radius: 4px; cursor: pointer;}
        .btn-pdf { display: inline-block; padding: 10px 15px; background: #8e44ad; color: white; text-decoration: none; border-radius: 5px; font-weight: bold; margin-left: 10px; }
    </style>
</head>
<body>
    <h1>🌱 Reporte y Progreso: {{ id_lote | upper }}</h1>
    
    <div class="tarjeta">
        <h3 style="margin-top:0; border-bottom: 2px solid #ecf0f1; padding-bottom:10px;">📋 Datos Generales</h3>
        <p class="dato"><strong>Cultivo:</strong> {{ datos['fecha_cultivo'] }}</p>
        <p class="dato"><strong>Última Fumigación Realizada:</strong> {{ datos['fecha_fumigacion'] }}</p>
        <p class="dato alerta"><strong>Diagnóstico actual:</strong> {{ datos['ultima_enfermedad'] }}</p>
        
        <form action="/agendar_fumigacion/{{ id_lote }}" method="POST" style="margin-top: 15px; background: #fff3cd; padding: 10px; border-radius:5px;">
            <label style="font-weight:bold; color:#856404; display:block; margin-bottom:5px;">📅 Agendar Próxima Fumigación:</label>
            <input type="date" name="nueva_fecha" class="input-fecha" required>
            <button type="submit" class="btn-fecha">Guardar</button>
            <p style="margin:5px 0 0 0; font-size:14px; color:#2c3e50;">Agendada para: <b>{{ datos['proxima_fumigacion'] }}</b></p>
        </form>
    </div>

    <div class="tarjeta">
        <h3 style="margin-top:0; color: #2980b9;">📈 Análisis de Tendencia</h3>
        <p style="font-size:18px; font-weight:bold; margin-bottom:5px;">Estado: {{ tendencia }}</p>
        <p style="color: #7f8c8d; font-size: 14px; margin-top:0;">{{ msj_tendencia }}</p>
        
        <h4 style="margin-bottom:10px;">Últimos 5 Escaneos:</h4>
        {% if historial %}
            {% for h in historial %}
            <div class="historial-item">
                <b>{{ h['fecha_escaneo'] }}</b> - <i>{{ h['diagnostico'] }}</i>
            </div>
            {% endfor %}
        {% else %}
            <p style="color:gray; font-size:14px;">No hay escaneos previos.</p>
        {% endif %}
    </div>
    
    <div class="tarjeta">
        <h3 style="margin-top:0; color: #16a085;">📷 Nuevo Escaneo Directo</h3>
        <form action="/analizar/{{ id_lote }}" method="POST" enctype="multipart/form-data" style="margin-bottom: 10px;">
            <label style="display: block; cursor: pointer; padding: 12px; border-radius: 5px; background: #9b59b6; color: white; font-weight: bold; text-align: center;">
                📸 Tomar Foto Nueva <input type="file" name="foto_hoja" accept="image/*" capture="environment" style="display: none;" onchange="this.form.submit()">
            </label>
        </form>
        <form action="/analizar/{{ id_lote }}" method="POST" enctype="multipart/form-data">
            <label style="display: block; cursor: pointer; padding: 12px; border-radius: 5px; background: #3498db; color: white; font-weight: bold; text-align: center;">
                🖼️ Subir desde Galería <input type="file" name="foto_hoja" accept="image/*" style="display: none;" onchange="this.form.submit()">
            </label>
        </form>
    </div>
    
    <div style="margin-top: 20px;">
        <a href="/panel" class="btn-panel">⬅ Volver al Dashboard</a>
        <a href="/descargar_reporte/{{ id_lote }}" class="btn-pdf">📥 Descargar PDF</a>
    </div>
</body>
</html>
"""

PLANTILLA_AGREGAR = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Agregar</title><style>body { font-family: Arial; text-align: center; padding: 50px; background-color: #f4f4f9; } .formulario { background: white; padding: 30px; border-radius: 10px; max-width: 400px; margin: auto; text-align: left; } label { font-weight: bold; display: block; margin-top: 15px; } input { width: 100%; padding: 10px; margin-top: 5px; border: 1px solid #ccc; border-radius: 5px; box-sizing: border-box; } button { width: 100%; padding: 12px; background-color: #2ecc71; color: white; border: none; border-radius: 5px; font-weight: bold; margin-top: 20px; }</style></head><body><div class="formulario"><h1>🌱 Registrar Planta</h1><form action="/agregar" method="POST"><label>ID (Ej. plt-003):</label><input type="text" name="id_lote" required><label>Fecha Cultivo:</label><input type="text" name="fecha_cultivo" required><label>Última Fumigación:</label><input type="text" name="fecha_fumigacion" required><button type="submit">Guardar y Generar QR</button></form><a href="/panel" style="display:block; text-align:center; margin-top:15px;">Volver al Panel</a></div></body></html>"""

PLANTILLA_RESULTADO = """<!DOCTYPE html><html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Resultado</title><style>body { font-family: Arial; text-align: center; padding: 20px; background-color: #f4f4f9; } .contenedor-split { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-top: 20px;} .columna { background: white; padding: 15px; border-radius: 10px; max-width: 400px; flex: 1 1 300px; } img { width: 100%; border-radius: 5px; } .btn { padding: 12px 20px; border: none; border-radius: 5px; color: white; text-decoration: none; font-weight: bold; display: inline-block; margin-top: 20px; }</style></head><body><h1>✅ Análisis Completado: {{ id_lote | upper }}</h1><h2 style="color: #e74c3c;">Diagnóstico: {{ diagnostico }}</h2><div class="contenedor-split"><div class="columna"><h3>Tu Planta</h3><img src="{{ url_for('static', filename=ruta_analizada) }}"></div><div class="columna"><h3>Catálogo</h3><img src="{{ url_for('static', filename=ruta_ref) }}"></div></div><div><a href="/panel" class="btn" style="background:#3498db;">📊 Dashboard</a> <a href="/planta/{{ id_lote }}" class="btn" style="background:#2ecc71;">📄 Reporte</a></div></body></html>"""


# ==========================================
# 4. RUTAS DEL SERVIDOR
# ==========================================
@app.route('/')
def inicio():
    return redirect('/panel')

@app.route('/camara_en_vivo')
def camara_en_vivo():
    return render_template_string(PLANTILLA_CAMARA)

@app.route('/video_feed')
def video_feed():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reporte_en_vivo')
def reporte_en_vivo():
    """Captura el frame actual de la cámara, lo analiza y genera un PDF al instante con recomendaciones"""
    success, frame = camera.read()
    if not success:
        return "Error al acceder a la cámara", 500
    
    diagnostico = "Hoja Sana"
    img_procesada = frame.copy()
    
    if modelo_yolo is not None:
        try:
            resultados = modelo_yolo(frame, verbose=False)
            cajas = resultados[0].boxes
            nombres = resultados[0].names
            
            if len(cajas) > 0:
                enfermedades = [nombres[int(c.cls[0])] for c in cajas]
                if any("Grave" in e for e in enfermedades):
                    diagnostico = "Grave"
                elif any("Medio" in e for e in enfermedades) or any("Moderada" in e for e in enfermedades):
                    diagnostico = "Medio"
                elif any("Leve" in e for e in enfermedades):
                    diagnostico = "Leve"
                else:
                    diagnostico = enfermedades[0]
                
                img_procesada = resultados[0].plot()
        except Exception:
            pass

    # Extraer la recomendación del experto (producto y proceso)
    prod_recom, proc_recom = obtener_recomendacion(diagnostico)

    ruta_captura = "static/uploads/captura_vivo.jpg"
    cv2.imwrite(ruta_captura, img_procesada)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Reporte IA: Analisis en Tiempo Real", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    fecha_actual = datetime.now().strftime("%d de %B, %Y - %H:%M:%S")
    pdf.cell(200, 10, txt=f"Fecha de escaneo: {fecha_actual}", ln=True)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Diagnostico IA Detectado:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=diagnostico, ln=True)
    pdf.ln(5)

    # --- NUEVA SECCIÓN DE RECOMENDACIONES EN EL REPORTE EN VIVO ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="Plan de Accion Sugerido:", ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(20, 6, txt="Producto:", ln=False)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt=prod_recom)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(20, 6, txt="Proceso:", ln=False)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt=proc_recom)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Evidencia Visual:", ln=True, align='C')
    pdf.ln(5)
    
    if os.path.exists(ruta_captura):
        pdf.image(ruta_captura, x=55, w=100)
    
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=Reporte_Analisis_Vivo.pdf'
    return response

@app.route('/descargar_reporte/<id_lote>')
def descargar_reporte(id_lote):
    """Genera el PDF de una planta guardada e incluye las recomendaciones"""
    lote_limpio = id_lote.strip().lower()
    db = get_db()
    cursor = db.cursor()
    
    cursor.execute("SELECT * FROM plantas WHERE id_planta = ?", (lote_limpio,))
    datos = cursor.fetchone()
    if not datos: return "Planta no encontrada", 404

    cursor.execute("SELECT fecha_escaneo, diagnostico FROM historial WHERE id_planta = ? ORDER BY id_historial DESC LIMIT 5", (lote_limpio,))
    historial = cursor.fetchall()
    tendencia, msj_tendencia = calcular_tendencia(historial)

    # Extraer la recomendación del experto basado en la última enfermedad guardada
    prod_recom, proc_recom = obtener_recomendacion(datos['ultima_enfermedad'])

    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Reporte Tecnico de Cultivo - Lote: {lote_limpio.upper()}", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=f"Fecha de cultivo: {datos['fecha_cultivo']}", ln=True)
    pdf.cell(200, 8, txt=f"Ultima fumigacion realizada: {datos['fecha_fumigacion']}", ln=True)
    pdf.cell(200, 8, txt=f"Proxima fumigacion agendada: {datos['proxima_fumigacion']}", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="Diagnostico Actual (IA):", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=datos['ultima_enfermedad'], ln=True)
    
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="Analisis de Tendencia:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=f"Estado: {tendencia}", ln=True)
    pdf.ln(5)

    # --- NUEVA SECCIÓN DE RECOMENDACIONES EN EL REPORTE GUARDADO ---
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 8, txt="Plan de Accion Sugerido:", ln=True)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(20, 6, txt="Producto:", ln=False)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt=prod_recom)
    
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(20, 6, txt="Proceso:", ln=False)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 6, txt=proc_recom)
    pdf.ln(8)
    
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Evidencia Visual y Comparativa:", ln=True, align='C')
    pdf.ln(5)

    ruta_analizada = f"static/{datos['ruta_foto_analizada']}"
    ruta_referencia = f"static/{datos['ruta_foto_referencia']}"

    if datos['ruta_foto_analizada'] != 'ninguna' and os.path.exists(ruta_analizada) and os.path.exists(ruta_referencia):
        y_actual = pdf.get_y()
        pdf.image(ruta_analizada, x=15, y=y_actual, w=85)
        pdf.image(ruta_referencia, x=110, y=y_actual, w=85)
        pdf.set_y(y_actual + 85)
        pdf.set_font("Arial", 'I', 10)
        pdf.set_x(15)
        pdf.cell(85, 10, txt="Planta Analizada por IA", align='C')
        pdf.set_x(110)
        pdf.cell(85, 10, txt="Referencia del Catalogo", align='C')
    else:
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(200, 10, txt="(No hay fotos registradas para esta planta todavia. Realice un escaneo primero.)", ln=True, align='C')
    
    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = f'attachment; filename=Reporte_Tesis_{lote_limpio.upper()}.pdf'
    return response

@app.route('/panel')
def panel_control():
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM plantas ORDER BY id_planta")
    todas_las_plantas = cursor.fetchall()
    return render_template_string(PLANTILLA_PANEL, plantas=todas_las_plantas)

@app.route('/agregar', methods=['GET', 'POST'])
def agregar_planta():
    if request.method == 'POST':
        id_lote = request.form['id_lote'].strip().lower()
        fecha_c = request.form['fecha_cultivo']
        fecha_f = request.form['fecha_fumigacion']
        
        db = get_db()
        cursor = db.cursor()
        try:
            cursor.execute('''INSERT INTO plantas (id_planta, fecha_cultivo, fecha_fumigacion, ultima_enfermedad, ruta_foto_analizada, ruta_foto_referencia, proxima_fumigacion) VALUES (?, ?, ?, 'Pendiente', 'ninguna', 'ninguna', 'No programada')''', (id_lote, fecha_c, fecha_f))
            db.commit()
            qr_imagen = qrcode.make(f"{request.host_url}planta/{id_lote}")
            qr_imagen.save(f"static/qrs/{id_lote}.png")
            return redirect('/panel')
        except sqlite3.IntegrityError:
            return f"<h1 style='text-align:center; color:red;'>❌ El ID ya existe.</h1><a href='/agregar'>Volver</a>"
    return render_template_string(PLANTILLA_AGREGAR)

@app.route('/descargar_qr/<id_lote>')
def descargar_qr(id_lote):
    lote_limpio = id_lote.strip().lower()
    ruta_qr = f"static/qrs/{lote_limpio}.png"
    if not os.path.exists(ruta_qr):
        qrcode.make(f"{request.host_url}planta/{lote_limpio}").save(ruta_qr)
    return send_file(ruta_qr, as_attachment=True, download_name=f"QR_{lote_limpio.upper()}.png")

@app.route('/agendar_fumigacion/<id_lote>', methods=['POST'])
def agendar_fumigacion(id_lote):
    nueva_fecha = request.form['nueva_fecha']
    db = get_db()
    cursor = db.cursor()
    cursor.execute("UPDATE plantas SET proxima_fumigacion = ? WHERE id_planta = ?", (nueva_fecha, id_lote))
    db.commit()
    return redirect(f'/planta/{id_lote}')

@app.route('/planta/<id_lote>', strict_slashes=False)
def mostrar_datos(id_lote):
    lote_limpio = id_lote.strip().lower()
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM plantas WHERE id_planta = ?", (lote_limpio,))
    datos_planta = cursor.fetchone()
    
    if datos_planta:
        cursor.execute("SELECT fecha_escaneo, diagnostico FROM historial WHERE id_planta = ? ORDER BY id_historial DESC LIMIT 5", (lote_limpio,))
        historial = cursor.fetchall()
        tendencia, msj_tendencia = calcular_tendencia(historial)
        return render_template_string(PLANTILLA_INDIVIDUAL, id_lote=lote_limpio, datos=datos_planta, historial=historial, tendencia=tendencia, msj_tendencia=msj_tendencia)
    return "<h1 style='text-align:center;'>❌ Planta no encontrada</h1>", 404

def procesar_diagnostico(ruta_recorte):
    diagnostico = "Desconocido"
    if modelo_yolo is not None:
        try:
            resultados = modelo_yolo(ruta_recorte)
            img_con_cajas = resultados[0].plot() 
            cajas = resultados[0].boxes
            nombres = resultados[0].names
            
            if len(cajas) == 0: 
                diagnostico = "Hoja Sana"
            else: 
                enfermedades_detectadas = [nombres[int(c.cls[0])] for c in cajas]
                if any("Grave" in e for e in enfermedades_detectadas):
                    diagnostico = "Grave"
                elif any("Medio" in e for e in enfermedades_detectadas) or any("Moderada" in e for e in enfermedades_detectadas):
                    diagnostico = "Medio"
                elif any("Leve" in e for e in enfermedades_detectadas):
                    diagnostico = "Leve"
                else:
                    diagnostico = enfermedades_detectadas[0]
                    
            return diagnostico, img_con_cajas
        except Exception as e:
            return f"Error IA: {str(e)}", cv2.imread(ruta_recorte)
    return "Roya Grave (Simulada)", cv2.imread(ruta_recorte)

@app.route('/analizar/<id_lote>', methods=['POST'])
def analizar_foto_individual(id_lote):
    lote_limpio = id_lote.strip().lower()
    if 'foto_hoja' not in request.files: return "❌ No imagen.", 400
    archivo = request.files['foto_hoja']
    
    ruta_cruda = f"static/uploads/cruda_{lote_limpio}.jpg"
    ruta_recorte = f"static/uploads/recorte_{lote_limpio}.jpg"
    ruta_final = f"static/uploads/final_{lote_limpio}.jpg"
    
    archivo.save(ruta_cruda)
    extraer_area_hoja(ruta_cruda, ruta_recorte)
    
    diagnostico, img_procesada = procesar_diagnostico(ruta_recorte)
    cv2.imwrite(ruta_final, img_procesada)

    if 'Sana' in diagnostico: ref = 'referencias/hoja_sana.jpg'
    elif 'Leve' in diagnostico: ref = 'referencias/hongo_leve.jpg'
    elif 'Moderada' in diagnostico or 'Medio' in diagnostico: ref = 'referencias/hongo_medio.jpg'
    else: ref = 'referencias/hongo_grave.jpg'

    db = get_db()
    cursor = db.cursor()
    ruta_relativa_final = f"uploads/final_{lote_limpio}.jpg"
    cursor.execute("UPDATE plantas SET ultima_enfermedad = ?, ruta_foto_analizada = ?, ruta_foto_referencia = ? WHERE id_planta = ?", (diagnostico, ruta_relativa_final, ref, lote_limpio))
    db.commit()
    
    registrar_historial(lote_limpio, diagnostico)
    
    return render_template_string(PLANTILLA_RESULTADO, id_lote=lote_limpio, diagnostico=diagnostico, ruta_analizada=ruta_relativa_final, ruta_ref=ref)

@app.route('/escanear_global', methods=['POST'])
def escanear_global():
    if 'foto_global' not in request.files: return "❌ No imagen.", 400
    archivo = request.files['foto_global']
    ruta_temp = "static/uploads/temp_plantilla.jpg"
    archivo.save(ruta_temp)

    datos_qr = leer_qr_desde_imagen(ruta_temp)
    if not datos_qr: return "<h1 style='text-align:center; color:red;'>❌ QR no detectado</h1><a href='/panel'>Volver</a>", 400

    id_lote = datos_qr.strip().split('/')[-1].lower()

    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM plantas WHERE id_planta = ?", (id_lote,))
    if not cursor.fetchone(): return f"<h1 style='text-align:center; color:red;'>❌ Planta {id_lote} no registrada</h1>", 404

    ruta_cruda = f"static/uploads/cruda_{id_lote}.jpg"
    ruta_recorte = f"static/uploads/recorte_{id_lote}.jpg"
    ruta_final = f"static/uploads/final_{id_lote}.jpg"
    
    os.replace(ruta_temp, ruta_cruda)
    extraer_area_hoja(ruta_cruda, ruta_recorte)

    diagnostico, img_procesada = procesar_diagnostico(ruta_recorte)
    cv2.imwrite(ruta_final, img_procesada)

    if 'Sana' in diagnostico: ref = 'referencias/hoja_sana.jpg'
    elif 'Leve' in diagnostico: ref = 'referencias/hongo_leve.jpg'
    elif 'Moderada' in diagnostico or 'Medio' in diagnostico: ref = 'referencias/hongo_medio.jpg'
    else: ref = 'referencias/hongo_grave.jpg'

    cursor.execute("UPDATE plantas SET ultima_enfermedad = ?, ruta_foto_analizada = ?, ruta_foto_referencia = ? WHERE id_planta = ?", (diagnostico, f"uploads/final_{id_lote}.jpg", ref, id_lote))
    db.commit()
    
    registrar_historial(id_lote, diagnostico)
    
    return render_template_string(PLANTILLA_RESULTADO, id_lote=id_lote, diagnostico=diagnostico, ruta_analizada=f"uploads/final_{id_lote}.jpg", ruta_ref=ref)

@app.route('/eliminar/<id_lote>')
def eliminar_planta(id_lote):
    lote_limpio = id_lote.strip().lower()
    db = get_db()
    cursor = db.cursor()
    cursor.execute("DELETE FROM plantas WHERE id_planta = ?", (lote_limpio,))
    cursor.execute("DELETE FROM historial WHERE id_planta = ?", (lote_limpio,))
    db.commit()
    
    archivos_a_borrar = [f"static/qrs/{lote_limpio}.png", f"static/uploads/cruda_{lote_limpio}.jpg", f"static/uploads/recorte_{lote_limpio}.jpg", f"static/uploads/final_{lote_limpio}.jpg"]
    for archivo in archivos_a_borrar:
        if os.path.exists(archivo):
            try: os.remove(archivo)
            except: pass
    return redirect('/panel')

with app.app_context():
    init_db()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)