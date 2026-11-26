# AETHRYX Character Studio — especificación de producto prioritaria

Decisión del propietario (2026-11-26): el producto se especializa inicialmente en personajes humanoides originales para Unity y VRChat. La meta es un avatar editable, organizado, texturizado, riggeado y validado, comparable en estructura a un avatar distribuido en BOOTH; no copiar assets de terceros ni prometer equivalencia artística automática. El modo objeto genérico (p.ej. dinosaurios) es independiente y secundario: no aplicar rig humanoide o cara humana por defecto a cualquier malla. Este documento amplía AETHRYX_ROADMAP.md.

## Entrada: Character Reference Sheet
- Una hoja de diseño de personaje puede contener vistas de cuerpo completo (frontal, lateral, posterior, 3/4) y ampliaciones (ojos izquierdo/derecho, boca, accesorios, ropa, manos/pies). También admitir imágenes individuales y varias imágenes sueltas, PNG/JPG/WEBP.
- Ejemplo de trabajo (descripción sin publicar la imagen personal): personaje anime humanoide adulto, pelo rubio con puntas rosa/celeste, heterocromía rosa/celeste, collar, aros, ropa de dos piezas; cuatro vistas y recuadros de detalles. No subir la imagen privada al repositorio. Usar fixtures sintéticos o autorizados.
- Character Sheet Analyzer: detectar regiones, distinguir vistas completas de detalles, recortar sin contaminación de imágenes vecinas, identificar orientación, pose, fondo, oclusiones, accesorios y prendas; asociar las vistas al mismo personaje. Mantener referencia original, coordenadas de recortes, etiquetas, confianza, vistas faltantes, origen y confirmación manual. No confundir lado del personaje con lado del observador.
- Interfaz: arrastrar hoja, mostrar recuadros detectados/editables, reasignar tipo, ajustar recorte, excluir región, añadir imágenes, escoger vista primaria. Permitir entrada frontal única con advertencia de ambigüedad. No exigir exactamente cuatro vistas si no existen.
- Referencia no equivale a geometría: poses, cámaras, peinados y proporciones inconsistentes entre vistas requieren alineación y revisión. Una vista 3/4 no sustituye otra lateral ortográfica; reconstruir partes ocluidas es estimación, no información observada. Ofrecer sugerencias de mejores referencias (T/A pose consistente, pies apoyados, manos visibles, vistas sin perspectiva exagerada).
- Normalizar encuadre, escala, máscara y correspondencias; conservar color, proporciones y características de identidad. Las ampliaciones condicionan materiales/rasgos locales, no se envían como personajes completos.

## Motor de reconstrucción
- TripoSG actual recibe una sola imagen. No afirmar que fusiona cuatro vistas ni enviar una hoja de contacto completa a su CLI. Como fallback explícito, usar recorte frontal aprobado; registrar qué referencias realmente usó el motor.
- Diseñar interfaz de backend (capacidades single_view/multi_view, texturing, rigging, requisitos VRAM, licencia, versión); evaluar motor de reconstrucción multivista y método especializado humanoide. Comparar fidelidad, consistencia, tiempo, VRAM y licencia con imágenes autorizadas. No concatenar cuatro mallas independientes. Un avatar requiere geometría consistente y topología deformable.
- Reconstrucción del personaje completo y/o componentes a partir de una representación coherente; conservar arte original de alta resolución. Separación y reconstrucción de superficies ocultas, pelo, ropa y cara son desafíos independientes. No prometer anatomía invisible exacta.
- Ejecutar localmente, módulos separados, sin Blender obligatorio, sin créditos de generación como requisito. Evaluar compatibilidad Windows/RTX 5070 y licencias de código/pesos antes de integrar. No publicar datos personales, modelos descargados ni claves.

## Estructura editable del avatar
Scene graph con nodos persistentes, nombres únicos y relaciones:
- Avatar root / Armature (hips, spine/chest, neck/head, extremidades y dedos, ojos, mandíbula opcional, huesos secundarios de cabello/ropa).
- Body base: cabeza/cuerpo, manos, pies, anatomía coherente bajo ropa cuando sea posible; topología humana y pose T/A, simetría editable.
- Face: ojos L/R independientes, iris/materiales, párpados y pestañas; boca, dientes y lengua según diseño; expresiones mediante morph targets/blendshapes.
- Hair: base y mechones/grupos editables; materiales y huesos secundarios opcionales.
- Clothing: piezas de ropa independientes (no fusionadas con el cuerpo), mallas ajustadas y pesos transferidos; conjuntos intercambiables, materiales, control de clipping y ocultación corporal configurable. No confundir segmentar una superficie visible con recuperar una prenda o cuerpo completo.
- Accessories: collar, pendientes L/R, pulseras y otros elementos detectados, objetos con pivotes, anclajes/bone parenting y materiales propios.
- UVs/materials/textures: base color, normal, roughness/metallic, emission, opacity/masks donde corresponda; conservar rasgos como heterocromía y colores de pelo; atlas opcional sin destruir la editabilidad.
- AETHRYX project: assets, provenance, scene nodes, transforms, material refs, original/working meshes, skeleton, skin weights, morphs, poses, validation results; versionado, undo y checkpoints. El .aeth v1 de fase 2 es solo manifiesto externo inicial, no contiene todavía todo esto.

## Rig y animación
- Rig humanoide automático con posibilidad de corrección manual: jerarquía, orientación, bind pose, pesos normalizados, dedos, eye bones, jaw opcional, controles y pruebas de deformación; no basta con visualizar líneas de esqueleto.
- Facial Studio: ojos con pivotes y mirada, parpadeo L/R, mandíbula, boca, expresiones y visemas (15 si se usa blendshape lip sync de VRChat); comprobar morph targets con misma topología base. No simular movimientos con etiquetas o texturas.
- Preview: pose T/A, giro de cabeza, mirada L/R, parpadeo, visemas, brazos/rodillas/dedos, poses y animaciones de prueba; detección de clipping y deformaciones. Huesos de pelo/ropa listos para configuración secundaria cuando sean necesarios.
- Personaje humanoide como preset principal; otras anatomías requieren rig específico y quedan fuera del MVP.

## Salida para Unity/VRChat
- Entregar paquete de recursos: FBX riggeado con geometrías separadas, skin, bind pose, blendshapes y materiales; texturas PNG y manifiesto, opcional GLB riggeado intermedio. OBJ solo geometría estática. Evaluar exportador con licencia y soporte real sin Blender.
- Complemento Unity opcional: importar recursos, mapear Humanoid, materiales (shader compatible; Poiyomi si instalado/licenciado por usuario), Animator, VRChat Avatar Descriptor, eye look, lip sync, PhysBones, configuración de expresiones/menús/ropa y prefab. UnityPackage creado desde Unity. No redistribuir SDK o shaders propietarios sin autorización.
- Validación: escala/ejes, huesos requeridos Unity Humanoid, pesos, ojos/visemas, texturas, mallas separadas, deformaciones, performance rank/triángulos/materiales y restricciones de plataforma. No declarar listo para subir hasta pasar importación y SDK Build & Test. El usuario conserva control de publicación en VRChat.

## Prioridades de desarrollo (incrementales)
1. Character mode y analizador de hoja: asistente de importación, regiones editables, etiquetas, manifest de referencias, selección frontal fallback. Pruebas con hojas sintéticas; mantener modo objeto separado.
2. Visor GPU/editor de escena, proyectos con assets y nodos persistentes; evaluar backend multivista. Prueba de un personaje completo desde varias referencias, con métricas y artefactos reales.
3. Geometría humanoide editable, retopología y separación/reconstrucción de partes; validar cada componente, no fingir resultados.
4. UV, texturas, materiales y ropa/accesorios.
5. Auto-rig, skinning, ojos, blendshapes, expresiones y pruebas de animación.
6. Exportador y complemento Unity, avatar prefab y pruebas reales de Unity/VRChat.
Criterio de éxito final: importar una hoja del personaje y obtener, tras revisión y correcciones cuando sean necesarias, un avatar estructurado y animable que pueda importarse y validarse en Unity/VRChat. No prometer automatización perfecta de todos los diseños. Cada fase se entrega con pruebas, limitaciones y documentación; no mostrar botones falsos. Copilot y Kiria deben trabajar en ramas independientes y revisar antes de fusionar.
