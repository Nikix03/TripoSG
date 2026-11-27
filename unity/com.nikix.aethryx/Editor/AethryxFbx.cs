using System;
using System.IO;
using System.Linq;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace Nikix.Aethryx
{
    public static class AethryxFbx
    {
        [MenuItem("Tools/AETHRYX/Exportar avatar seleccionado a FBX")]
        static void ExportSelection()
        {
            var root = Selection.activeGameObject;
            if (root == null) return;
            var path = EditorUtility.SaveFilePanel("Exportar avatar articulado", "", root.name, "fbx");
            if (string.IsNullOrEmpty(path)) return;
            try { Export(root, path); }
            catch (Exception exc) { Debug.LogError("AETHRYX: " + exc.Message); }
        }

        public static void Export(GameObject root, string path)
        {
            if (root == null || !root.GetComponentsInChildren<SkinnedMeshRenderer>(true).Any())
                throw new ArgumentException("Selecciona un avatar con mallas articuladas.");
            path = Path.GetFullPath(path);
            if (!path.EndsWith(".fbx", StringComparison.OrdinalIgnoreCase) || File.Exists(path))
                throw new ArgumentException("Selecciona un archivo FBX nuevo.");
            const string apiName = "UnityEditor.Formats.Fbx.Exporter.ModelExporter";
            var exporter = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetType(apiName)).FirstOrDefault(type => type != null);
            var method = exporter?.GetMethod("ExportObject", BindingFlags.Public | BindingFlags.Static,
                null, new[] { typeof(string), typeof(UnityEngine.Object) }, null);
            if (method == null)
                throw new NotSupportedException("Instala el paquete oficial com.unity.formats.fbx en este proyecto.");
            try
            {
                var result = method.Invoke(null, new object[] { path, root }) as string;
                if (string.IsNullOrEmpty(result) || !File.Exists(path) || new FileInfo(path).Length == 0)
                    throw new IOException("Unity FBX Exporter no produjo un archivo válido.");
            }
            catch (TargetInvocationException exc)
            {
                throw new InvalidOperationException("Unity FBX Exporter rechazó el avatar.", exc.InnerException);
            }
        }
    }
}
