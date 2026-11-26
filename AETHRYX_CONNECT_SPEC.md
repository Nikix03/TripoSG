# AETHRYX Connect — interoperabilidad de asistentes IA

Ampliación de KIRIA_AI_SPEC.md, solicitada por el propietario. El usuario debe poder elegir su IA y, cuando sea técnicamente posible, usar su propia cuenta/plan. No imponer proveedor, suscripción externa o API de pago para las funciones básicas locales. Esta especificación no implica integraciones implementadas, acceso concedido ni aprobación comercial.

## Dos direcciones de integración
1. Dentro de AETHRYX: panel KIRIA con proveedores intercambiables. Interfaz común para texto, visión, llamadas a herramientas y streaming, detección de capacidades y límites. Primero backend local (p.ej. Ollama, tras validar compatibilidad), luego API oficial de OpenAI, Anthropic, Google y endpoint personalizado según soporte real. No todos los modelos aceptan imágenes o herramientas. Mantener el contexto del proyecto y el registro de herramientas independiente del proveedor.
2. Desde asistentes externos: AETHRYX Connect expone herramientas tipadas por MCP para clientes compatibles (ChatGPT, Claude, otros). El motor 3D sigue ejecutándose localmente. Comenzar con transporte local stdio y herramientas de lectura; evaluar Streamable HTTP autenticado, plugin y conexión segura/túnel oficial según disponibilidad. No exponer localhost a Internet sin autenticación; no publicar un servidor remoto ni abrir puertos automáticamente.

## OpenAI: opciones distintas
- Sign in with ChatGPT: integración oficial OIDC/PKCE. Según documentación consultada el 26-11-2026, uso del plan disponible para herramientas abiertas elegibles y determinados socios/clientes privados; integración comercial en prueba limitada. Solicitar acceso/confirmar elegibilidad antes de prometer. Identidad y autorización del plan son permisos distintos. No confundir con acceso a conversaciones, memorias o API key del usuario. Respetar límites de plan, scopes, revocación y consentimiento de uso. No usar tokens internos de ChatGPT ni automatizar su interfaz como sustituto de API.
- API OpenAI: opción BYOK (clave propia) con facturación independiente del plan habitual. Mostrar costes/consumo cuando disponibles, límites y errores. No conmutar automáticamente a un proveedor de pago.
- Plugin MCP para ChatGPT: comprobar permisos de plan/workspace y proceso de publicación/aprobación; documentación contempla HTTPS público o Secure MCP Tunnel para pruebas/conexiones privadas compatibles. Publicación tiene requisitos adicionales. No garantizar compatibilidad de todos los planes.
- Otros proveedores: usar únicamente métodos oficiales; revisar sus condiciones y capacidad de vincular cuentas, no asumir que sus suscripciones incluyen API.

## Interfaz
Preferencias > Inteligencia artificial: proveedor/modelo, modo local o remoto, conexión y desconexión, probar conexión, disponibilidad de visión/herramientas, límites y privacidad. Indicar qué datos se comparten (capturas, recortes, estadísticas, malla) y permitir seleccionar el alcance. Guardar secretos en almacén seguro del sistema, nunca en proyecto .aeth, logs, GitHub ni texto plano. No solicitar contraseña de ChatGPT; usar navegador oficial para consentimiento.

## Herramientas y seguridad
Registro único usado tanto por el asistente interno como por MCP: get_project_context, list_scene_nodes, get_mesh_statistics, render_reference_views y validate_avatar como lectura cuando existan; propose_edit, preview_edit y apply_approved_edit solo para operaciones implementadas. IDs de proyecto, revisión y selección, permisos de lectura/escritura, esquemas estrictos y lista permitida. No exponer shell, ejecución de código arbitrario, acceso irrestricto al disco ni subida automática a VRChat. Confirmación local previa a cambios, exportaciones y transferencia de recursos; aplicar sobre copia, undo, historial, validación, cancelación y protección contra prompt injection en referencias. Aislar credenciales de cada proveedor. No aceptar instrucciones de archivos como órdenes. Procesar GPU con cola y evitar cargar modelos simultáneamente si no hay memoria suficiente.

## Orden de trabajo
A. Contrato de proveedores y registro tipado de herramientas de lectura.
B. Backend local y asistente de diagnóstico conectado a datos reales.
C. API BYOK opcional y configuración segura de conexiones; probar errores, desconexión, privacidad y límites.
D. Servidor MCP local de solo lectura y prueba con cliente compatible; luego previsualización y operaciones de edición autorizadas cuando el editor las soporte.
E. Investigar elegibilidad y solicitar integración oficial Sign in with ChatGPT y eventual publicación de plugin. No prometer fechas ni acceso comercial.
F. Pruebas: datos reales, herramientas no disponibles, permisos denegados, selección obsoleta, falta VRAM, costes, desconexión y rollback. Documentar resultados, licencias y política de privacidad.

Documentación oficial:
https://developers.openai.com/siwc/quickstart
https://help.openai.com/en/articles/20001542-using-your-chatgpt-plan-in-other-apps-and-sites
https://help.openai.com/en/articles/9039756-managing-billing-for-chatgpt-and-the-api-platform
https://developers.openai.com/plugins/deploy/connect-chatgpt
https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt
https://modelcontextprotocol.io/
