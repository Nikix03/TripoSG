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
- Primer bloque publicado y validado en `fb53de6` con 62 pruebas. La prueba de Unity se realizó en un proyecto temporal y sin activar cuentas, importar SDK ni instalar paquetes.

## Segundo bloque

| Fase | Avance y comprobación | Pendiente |
| --- | --- | --- |
| 11 · Vestuario | La prenda puede reintentar con hasta dos holguras mayores dentro del rango configurado. Studio oculta torso o cadera en el visor de forma reversible y mantiene todas las caras del GLB. El vestido medido dio 0 contactos y 0 vértices interiores en reposo y en la pose de prueba. [Revisión de la pose](test_assets/anime/genesis-extended/pose-review.png). | El vestido muestra un pliegue por transferencia de pesos entre las piernas. Los prototipos de zapato y guante ajustados por normales se descartaron por cruces y penetraciones. Los conteos de contacto y vértices internos fueron 1.754/223 y 328/40 respectivamente. |
| 12–13 · Pose y expresión | Vista giratoria con esqueleto, reposo, aproximación a T, rotación por articulación y control de intensidad facial. Se verificó movimiento unilateral de ojo y párpado. La validación GLB vuelve a leer desde disco el clip, pesos y matrices inversas y calcula la deformación. | Colocación anatómica de articulaciones, control de dedos y calidad de expresiones en una referencia de Nikix. |
| 15 · KIRIA | Avance parcial. Panel de contexto y herramientas locales, con diagnóstico de fondo y comandos estructurados para mostrar, ocultar o cambiar color. Previsualización textual, confirmación, registro y deshacer por la vía de escena existente. | Conversación con inferencia real, conexión de proveedores y extensión del registro a proporciones y vestuario. Ollama no apareció instalado ni en ejecución. [API local de Ollama](https://docs.ollama.com/api/tags). |
| 16 · Studio Pre-Alpha | Avance parcial. Recuperación lateral `.recovery.aeth` cada 45 s cuando cambia un proyecto guardado y al cerrar. Se ofrece al reabrir si es más reciente y se conserva el original. Lanzador con diagnóstico local y exportación de rig en segundo plano. | Instalación limpia, recuperación de escenas sin guardar, cancelación de trabajos nuevos y prueba Unity/VRChat. |

- La prueba integrada con Tk abrió un personaje con vestido, confirmó un comando KIRIA, guardó una recuperación, abrió la pose T aproximada y auditó un GLB de 53 articulaciones, 37.124 caras y 15 visemas. El fotograma leído del GLB desplazó un vértice corporal hasta 0,2147 unidades de escena. El diagnóstico del lanzador detectó Python 3.10.11, la RTX 5070 de 12.227 MiB y las dependencias esenciales. No se instaló software nuevo.
- Pasaron 66 pruebas mediante `.\.venv\Scripts\python.exe -m unittest discover -s tests`.

### Uso reproducible

1. Ejecutar `iniciar-aethryx-studio.bat --diagnostico` y luego `iniciar-aethryx-studio.bat`.
2. Abrir **Escena → Administrador de componentes**, crear el cuerpo y las piezas, y guardar la escena modular independiente.
3. Usar **Vista de pose** para comprobar articulaciones y expresiones. Exportar un nuevo GLB articulado y revisarlo con `.\.venv\Scripts\python.exe -m aethryx_bridge avatar.glb`.
4. Guardar el proyecto `.aeth` en **Archivo**. El proyecto enlaza sus recursos relativos cuando es posible.

## Continuidad

- Siguiente tarea concreta. Ajustar la transferencia de pesos en prendas acampanadas y validar la geometría de mangas y calzado con cortes seccionales, conservando el cuerpo completo. La licencia activa de Unity es una dependencia separada para el puente y el SDK.
