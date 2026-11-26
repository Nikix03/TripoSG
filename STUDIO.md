# Nikix 3D Studio — fase 01

Interfaz de escritorio en español para ejecutar el motor existente de TripoSG sin abrir Blender. No modifica el motor ni sustituye sus dependencias.

## Windows / entorno local

Primero instala y verifica TripoSG según el README principal. Usa el mismo entorno Python con PyTorch/CUDA compatible con tu GPU. Después, desde la raíz del repositorio:

```powershell
python -m pip install -r requirements-studio.txt
python -m studio.app
```

No se instala automáticamente PyTorch ni CUDA. La primera generación puede descargar los pesos oficiales. El motor original y sus dependencias mantienen sus respectivas licencias; revisa también los términos de los pesos de RMBG antes de distribuir o comercializar.

## Funciones

- Selección de una imagen, vista previa y parámetros reales del CLI.
- Generación en proceso separado, registro, indicador de actividad y cancelación.
- Cada ejecución crea una carpeta única en el destino elegido.
- Visor de geometría OpenGL con rotación y zoom, estadísticas y apertura de GLB.
- Exportación GLB original y OBJ de geometría, sin Blender.

No hay todavía rig, UV generadas, texturas, retopología anatómica, blendshapes ni exportación FBX. La reducción de caras es decimación, no retopología. OBJ puede importarse como objeto estático en Unity; GLB requiere un importador glTF compatible. Ninguno de estos resultados es aún un avatar de VRChat.

## Pruebas

```powershell
python -m unittest discover -s studio/tests
```

Prueba manual: abrir la interfaz, cargar una imagen individual, ejecutar el motor y comprobar el GLB; abrirlo en el visor, exportar OBJ e importarlo en Unity. También probar cancelación y rutas con espacios. El visor depende del soporte OpenGL del equipo. No se afirma compatibilidad probada con una RTX 5070.

## Próximas fases

1. Diagnóstico reproducible de CUDA, dependencias y memoria; manejo offline de pesos.
2. Validación de malla, limpieza, unidades y ejes; retopología apta para personajes.
3. Texturas/UV y segmentación; rigging y pesos mediante motores independientes.
4. Exportador FBX sin interfaz de Blender (evaluar SDK o biblioteca con soporte real de skin y morphs), validación Humanoid y asistente de importación para Unity/VRChat.

Mantener adaptadores desacoplados y no exponer opciones avanzadas como disponibles hasta comprobarlas.
