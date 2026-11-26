# AETHRYX Studio · Genesis Extended

## Punto de partida

- Rama `codex/development`. Commit base validado `12bc16f`.
- RTX 5070 de 12 GB y Windows. Los cambios de esta operación reutilizan NumPy, SciPy, Trimesh, Pillow y VTK del entorno existente. Se conservaron los motores y los pesos. La plantilla derivada de MakeHuman conserva el aviso CC0-1.0 de `NOTICE`.
- La validación se realizó con la plantilla autorizada. La evaluación artística con las referencias originales de Nikix sigue pendiente.

## Primer bloque

| Fase | Estado y evidencia | Pendiente |
| --- | --- | --- |
| 9 · Anatomía | Avance parcial. Perfiles anime, control de labios, párpados con espesor, cavidad y dentición por piezas, cabello en cuatro zonas y degradado de raíz a punta. [Revisión frontal, lateral, posterior y facial](test_assets/anime/genesis-extended/phase9-review.png). | Anatomía fina de manos y pies, apertura y cierre de labios en movimiento, mechones individuales y revisión artística. See-through continúa como importación de máscaras externas. |
| 10 · Materiales | Avance parcial. Mapas PNG PBR por pieza, empaquetado roughness/metallic, alfa, ajuste y diagnóstico UV por muestreo. El visor conserva la previsualización de color base. | Atlas e islas, capas de edición y representación del resto de canales en el visor. Poiyomi pendiente de Unity. |
| 11 · Vestuario | Avance parcial. Top, falda y vestido cerrados derivados de cortes corporales, holgura, espesor y almacenamiento local de conjuntos GLB. El cuerpo completo se conserva. | Camisetas, pantalones, zapatos, guantes, corrección dinámica y ocultación regional reversible. |
| 12 · Rig humanoide | Avance parcial. GLB con 53 articulaciones aproximadas, pesos normalizados de 1–4 influencias, matrices inversas y un clip de prueba. La pose calculada desplaza los vértices del brazo y se comprobó que el pie opuesto se mantiene dentro de la tolerancia. | Edición y revisión visual de articulaciones, poses A/T, pesos de dedos y prendas en movimiento y evaluación dentro de Unity. |
| 13 · Animación facial | Avance parcial. Ocho expresiones y quince visemas iniciales exportados como morph targets de posición. Las pruebas comprueban índices y localización de sonrisa y parpadeo. | Párpados y boca coordinados, animador facial interactivo y validación fonética. |
| 14 · Unity y VRChat | Bloqueo de prueba externa. El lector local verifica pesos, articulaciones, matrices de enlace y morphs del GLB. Se encontró Unity 2022.3.22f1. | Unity cerró la prueba batch con código 1 y licencia inactiva. Pendientes importación, FBX, Poiyomi y SDK. [Versión oficial de Unity para VRChat](https://creators.vrchat.com/sdk/upgrade/current-unity-version/) · [Exportador FBX de Unity](https://docs.unity3d.com/Packages/com.unity.formats.fbx@4.2/manual/index.html). |

### Decisiones y errores corregidos

- La malla corporal conserva su topología. El cabello y los párpados tienen mallas independientes. Una orientación invertida del párpado inferior se detectó con la prueba de volumen y se corrigió. La revisión visual permitió bajar el interior bucal a la altura de los labios.
- El render usa PyVista con cierre explícito de cada cámara. La versión instalada de `Plotter` rechazó el gestor de contexto.
- Los conjuntos se guardan con nodos de prendas en GLB y la transferencia inicial de pesos usa el vértice corporal más cercano. El paso nativo conserva el original. Las animaciones y expresiones exportadas requieren revisión visual y una prueba en el editor de destino.
- Pruebas del bloque mediante `.\.venv\Scripts\python.exe -m unittest discover -s tests`. La prueba de Unity se realizó en un proyecto temporal y sin activar cuentas, importar SDK ni instalar paquetes.

## Continuidad

- Siguiente tarea concreta dentro de la operación. Conectar los diagnósticos KIRIA al proyecto activo y añadir recuperación local de proyectos.
