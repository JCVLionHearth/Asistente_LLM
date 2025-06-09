# app.py

import streamlit as st
import pdfplumber
from transformers import pipeline
import re

# ---------- 1. Cargar modelos y funciones ----------
@st.cache_resource
def cargar_modelos():
    resumen = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    ner = pipeline("ner", model="Davlan/bert-base-multilingual-cased-ner-hrl", grouped_entities=True)
    return resumen, ner

resumidor, reconocedor_entidades = cargar_modelos()


def extraer_texto_pdf(archivo_pdf):
    texto_total = ""
    with pdfplumber.open(archivo_pdf) as pdf:
        for pagina in pdf.pages:
            texto_total += pagina.extract_text() + "\n"
    return texto_total

def limpiar_texto(texto):
    texto = re.sub(r'\n+', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# ---------- 2. Interfaz Streamlit ----------
st.title("📚 Asistente Inteligente de Lectura de PDFs")
st.write("Carga un PDF para obtener un resumen y las entidades clave (personas, lugares, fechas, etc.).")

archivo = st.file_uploader("📎 Carga un archivo PDF", type=["pdf"])

if archivo:
    with st.spinner("🔍 Extrayendo texto del PDF..."):
        texto_extraido = extraer_texto_pdf(archivo)
        texto_limpio = limpiar_texto(texto_extraido)
        st.success("✅ Texto extraído correctamente.")
        st.subheader("📄 Fragmento del texto extraído")
        st.write(texto_limpio[:1000] + "...")  # Muestra primeros 1000 caracteres

    if st.button("📑 Generar resumen"):
        with st.spinner("⏳ Generando resumen..."):
            # El modelo de resumen tiene un límite, usamos solo una parte
            texto_corto = texto_limpio[:1024]
            resumen = resumidor(texto_corto, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
            st.subheader("🧠 Resumen del contenido:")
            st.write(resumen)

    if st.button("🔎 Extraer entidades"):
        with st.spinner("🧬 Analizando entidades..."):
            texto_corto = texto_limpio[:1000]
            entidades = reconocedor_entidades(texto_corto)

            st.subheader("🏷️ Entidades encontradas:")
            for entidad in entidades:
                st.markdown(f"• **{entidad['word']}** — {entidad['entity_group']} (confianza: {round(entidad['score']*100, 1)}%)")
