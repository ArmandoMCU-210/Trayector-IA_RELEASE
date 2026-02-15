from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# --- CARGAR EL CEREBRO ---
try:
    cerebro = joblib.load('modelo_vocacional.pkl')
    modelo = cerebro['modelo']
    le_q25 = cerebro['encoder_q25']
    le_target = cerebro['encoder_target']
    columnas_modelo = cerebro['columnas']
except FileNotFoundError:
    cerebro = None
    print("Advertencia: No se encontró 'modelo_vocacional.pkl'.")

# --- DATOS DEL TEST ---
escala_aptitudes = [
    "1 = Se me complica mucho", "2 = Me cuesta trabajo", 
    "3 = Ahí la llevo", "4 = Se me facilita", "5 = Es lo mío, me sale natural"
]

escala_intereses = [
    "1 = Me aburre/No me llama", "2 = Meh, no es lo mío", 
    "3 = Está bien, neutral", "4 = Me gusta bastante", "5 = Me encanta/Me apasiona"
]

escala_futuro = [
    "1 = Jamás, ni muerto", "2 = Lo haría si no hay de otra", 
    "3 = Puede ser, ocasionalmente", "4 = Sí, varias veces por semana", "5 = Todos los días, es mi sueño"
]

escala_q25 = [
    "a) Las de pura lógica y números: Cálculo, Álgebra, Programación",
    "b) Las de \"¿cómo funciona esto?\": Física, Mecánica, Electrónica, Circuitos",
    "c) Las de construir y diseñar: Resistencia de materiales, Estructuras, Topografía",
    "d) Las de dinero y empresas: Contabilidad, Finanzas, Costos, Economía",
    "e) Las de organización y gente: Administración, Recursos Humanos, Marketing",
    "f) Las de optimizar y mejorar: Estadística, Investigación de operaciones, Calidad"
]

preguntas = {
    1: {"texto": "1. Cuando tienes que resolver un problema con números o fórmulas, ¿cómo te va?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    2: {"texto": "2. ¿Qué tan cómodo te sientes trasteando con computadoras, apps o tecnología en general?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    3: {"texto": "3. Si tuvieras que hacer un plano, dibujo técnico o diseño con medidas exactas, ¿cómo te sentirías?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    4: {"texto": "4. ¿Qué tan bueno eres arreglando cosas, usando herramientas o armando/desarmando aparatos?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    5: {"texto": "5. Cuando ves números de dinero, presupuestos o estados de cuenta, ¿se te facilita entenderlos?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    6: {"texto": "6. ¿Qué tan natural se te hace organizar gente, delegar tareas o coordinar equipos?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    7: {"texto": "7. Cuando ves que algo se puede hacer más rápido o mejor, ¿se te ocurren formas de optimizarlo?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    8: {"texto": "8. ¿Qué tan fácil te es imaginar en tu mente cómo se vería un objeto en 3D o desde diferentes ángulos?", "opciones": escala_aptitudes, "seccion": "SECCIÓN II: ¿QUÉ SE TE DA BIEN?"},
    9: {"texto": "9. Crear programas, apps, juegos o cualquier cosa digital desde cero", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    10: {"texto": "10. Trabajar con robots, máquinas inteligentes o sistemas que se muevan solos", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    11: {"texto": "11. Diseñar edificios, puentes, carreteras o cualquier construcción grande", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    12: {"texto": "12. Manejar un negocio, tomar decisiones de empresa o dirigir proyectos", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    13: {"texto": "13. Trabajar con números financieros, impuestos, auditorías o cuentas de empresas", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    14: {"texto": "14. Hacer que las cosas funcionen más rápido, organizar procesos o eliminar desperdicios", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    15: {"texto": "15. Trabajar con motores, circuitos eléctricos o sistemas de energía", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    16: {"texto": "16. Entender qué quiere la gente, vender ideas o crear estrategias de mercado", "opciones": escala_intereses, "seccion": "SECCIÓN III: ¿QUÉ TE LLAMA LA ATENCIÓN?"},
    17: {"texto": "17. Estar frente a una computadora escribiendo código o resolviendo bugs", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    18: {"texto": "18. Ensuciarme las manos arreglando, armando o calibrando máquinas", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    19: {"texto": "19. Estar en una obra de construcción supervisando que todo salga bien", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    20: {"texto": "20. Sentado haciendo cuentas, revisando facturas o elaborando reportes financieros", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    21: {"texto": "21. En juntas tomando decisiones importantes sobre el rumbo de una empresa", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    22: {"texto": "22. Diseñando cómo se va a fabricar algo, optimizando líneas de producción o mejorando logística", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    23: {"texto": "23. Calculando si una estructura va a aguantar, revisando planos estructurales", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    24: {"texto": "24. Conectando lo mecánico con lo electrónico, haciendo que las máquinas \"piensen\"", "opciones": escala_futuro, "seccion": "SECCIÓN IV: ¿TE VES HACIENDO ESTO EN TU FUTURO?"},
    25: {"texto": "25. Pensando en toda tu carrera, ¿en qué materias te fue mejor (o sufriste menos)?", "opciones": escala_q25, "seccion": "SECCIÓN V: TUS EXPERIENCIAS REALES"}
}

mensajes_carreras = {
    'Ingeniería de Software': "¡Tienes lógica de programador! El código es lo tuyo.",
    'Ingeniería Civil': "¡Tienes visión espacial! El mundo necesita tus estructuras.",
    'Ingeniería Mecatrónica': "¡Te gusta la robótica y la integración! El futuro es tuyo.",
    'Administración': "¡Tienes madera de líder! Sabes gestionar recursos.",
    'Contaduría': "¡El orden y las finanzas son tu fuerte! Eres clave para cualquier empresa.",
    'Gestión y Dirección de Negocios': "¡Tienes visión estratégica e instinto para los proyectos!",
    'Ingeniería Mecánica Eléctrica': "¡Entiendes la energía y el movimiento como nadie!",
    'Ingeniería Industrial': "¡Tu mente busca optimizar todo! La eficiencia es tu bandera."
}

# --- RUTAS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    if cerebro is None:
        return "Error: Modelo no cargado. Verifica 'modelo_vocacional.pkl'.", 500
    
    # Pasamos el diccionario de preguntas completo
    return render_template('test.html', preguntas=preguntas)
    
    # Agrupar preguntas por sección para renderizarlas más bonito
    secciones = {}
    for q_id, datos in preguntas.items():
        sec = datos['seccion']
        if sec not in secciones:
            secciones[sec] = []
        secciones[sec].append({'id': q_id, 'texto': datos['texto'], 'opciones': datos['opciones']})
        
    return render_template('test.html', secciones=secciones)

@app.route('/analizar', methods=['POST'])
def analizar():
    if cerebro is None:
        return redirect(url_for('index'))

    respuestas = request.form
    vector_entrada = {}
    
    for col in columnas_modelo:
        raw_answer = respuestas.get(col)
        if not raw_answer:
             return "Por favor, responde todas las preguntas.", 400
             
        if col == 'Q25':
            letra = raw_answer[0] 
            val = le_q25.transform([letra])[0]
            vector_entrada[col] = [val]
        else:
            numero = int(raw_answer[0])
            vector_entrada[col] = [numero]
            
    df_entrada = pd.DataFrame(vector_entrada)
    
    # Predicción
    pred_idx = modelo.predict(df_entrada)[0]
    carrera_resultado = le_target.inverse_transform([pred_idx])[0]
    
    # Probabilidades
    resultados_probs = []
    confianza = 0
    if hasattr(modelo, "predict_proba"):
        probs = modelo.predict_proba(df_entrada)[0]
        confianza = np.max(probs) * 100
        
        for i, carrera in enumerate(le_target.classes_):
            resultados_probs.append({
                'carrera': carrera,
                'probabilidad': round(probs[i] * 100, 1)
            })
        resultados_probs.sort(key=lambda x: x['probabilidad'], reverse=True)

    mensaje = mensajes_carreras.get(carrera_resultado, "¡Excelente opción de carrera! Te deseamos mucho éxito.")
    
    return render_template('result.html', 
                           carrera=carrera_resultado, 
                           confianza=round(confianza, 1), 
                           probs=resultados_probs,
                           mensaje=mensaje)

if __name__ == '__main__':
    app.run(debug=True)