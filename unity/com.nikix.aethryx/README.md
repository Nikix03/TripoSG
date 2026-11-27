# AETHRYX Studio Bridge

En Unity 2022.3, añade el `package.json` desde el administrador de paquetes o copia esta carpeta a `Packages/com.nikix.aethryx`. Abre **Tools → AETHRYX → Importar avatar Genesis** y elige un GLB articulado exportado por Studio. El puente crea un prefab, mallas, materiales, clip y un Avatar Humanoid si Unity valida la jerarquía.

El importador inicial admite el GLB nativo de AETHRYX, con posiciones, UV, pesos, matrices de enlace, morphs, clips de rotación y materiales Standard con textura base. No importa todos los materiales glTF ni configura el SDK de VRChat.

Para reproducir la prueba local, copia `scripts/AethryxBridgeSmoke.cs` al `Assets/Editor` de un proyecto temporal con el paquete instalado. Define `AETHRYX_UNITY_FIXTURE` con la ruta a un GLB de prueba con cuerpo y ojo izquierdo. Ejecuta Unity con `-batchmode -nographics -executeMethod AethryxBridgeSmoke.Run`. El registro `AETHRYX_SMOKE` contiene las verificaciones.
