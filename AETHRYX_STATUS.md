# Estado de AETHRYX Studio

## 26 de noviembre de 2026

- KIRIA (Codex) retomó el trabajo desde `a9eaeb77fa4bcde3e6ece54dff0f002fa199372f` en `codex/development`. El árbol de trabajo inicial estaba limpio. Se leyeron los 11 documentos AETHRYX existentes.
- La separación topológica recorre las instancias de la escena y conserva sus transformaciones. Una exportación crea GLB individuales y un `scene.glb` en una carpeta única. Studio abre la escena resultante a través de su importador habitual para mostrar sus piezas por separado.
- La prueba con `test_assets/anime/genesis-phase3/genesis-phase3-75-original.glb` produjo un componente de 284.697 vértices y 569.394 caras. El GLB reabierto conservó los límites del original y el archivo fuente mantuvo su hash SHA-256.
- La suite pasó de 37 pruebas de base a 38 aprobadas con `unittest`. La regresión verifica dos instancias desplazadas, cuatro piezas abiertas, los límites de los GLB individuales, los nodos de la escena conjunta y la conservación del original y de las salidas al repetir la operación. `pytest` no está instalado en este entorno.
- Se conservaron `.venv`, los pesos y modelos locales. La validación de esta entrega usa geometría estática. Quedan pendientes la prueba interactiva, una nueva inferencia CUDA y la validación en Unity y VRChat.

## Continuidad con KIRIA de ChatGPT

- El avatar anime evaluado tiene una sola superficie conectada. Para separar cabello, ojos y ropa hacen falta datos de segmentación semántica o edición explícita de las superficies. Las piezas topológicas se mantienen sin categoría anatómica.
- La reconstrucción activa de TripoSG sigue siendo monovista. Para una prueba multivista reproducible hacen falta referencias frontal, lateral y posterior del mismo avatar en la misma pose, con encuadre y cámaras conocidos, y métricas de silueta y proporciones por vista.
- Siguen pendientes la reconstrucción anatómica, el rig humanoide, el skinning y la validación del avatar en el SDK de VRChat. Los resultados históricos y parámetros de RTX 5070 están en los documentos Genesis, Vision y Adaptive Engine.
