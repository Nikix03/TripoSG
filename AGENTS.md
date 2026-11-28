# AETHRYX Studio

- KIRIA es la asistente principal de desarrollo. Comunica los avances en español.
- Continúa el trabajo existente en `codex/development`. Revisa el estado de Git antes de editar y conserva los cambios del usuario.
- El objetivo es convertir imágenes en avatares anime completos, modulares, editables y animables para Unity y VRChat. El flujo local admite Blender como backend automatizado opcional y Unity para preparación y exportación. El usuario final puede completar el flujo sin manejar Blender. La edición profesional en Blender y Unity sigue disponible.
- Prioriza calidad anatómica, reconstrucción multivista y separación real de geometría. Conserva el flujo TripoSG monovista y los proyectos `.aeth` existentes.
- Hardware de referencia: Windows, RTX 5070 de 12 GB, PyTorch y CUDA. Conserva `.venv`, `models`, `pretrained_weights` y los recursos locales. Excluye del repositorio pesos, entornos, credenciales y referencias privadas.
- Preserva la malla original al procesar copias. Conserva escala, ejes y transformaciones de instancia. Distingue componentes topológicos de partes anatómicas confirmadas.
- Registra capacidades según la evidencia. Documenta entradas, parámetros, tiempos, VRAM y resultados cuando ejecutes inferencia. Verifica licencias de código, pesos y activos antes de distribuirlos.
- Ejecuta `.\.venv\Scripts\python.exe -m unittest discover -s tests`. Actualiza `AETHRYX_STATUS.md` con verificaciones, errores y trabajo pendiente. Publica cambios validados en el fork de Nikix03.
