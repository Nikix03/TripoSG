using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace Nikix.Aethryx
{
    public static class AethryxHairCombine
    {
        [MenuItem("Tools/AETHRYX/Crear copia con cabello consolidado")]
        static void CombineSelection()
        {
            var source = Selection.activeGameObject;
            if (source == null) return;
            GameObject copy = null;
            try
            {
                copy = UnityEngine.Object.Instantiate(source, source.transform.parent);
                copy.name = source.name + " · cabello consolidado";
                if (Consolidate(copy) == 0)
                    throw new InvalidOperationException("La selección no contiene mechones compatibles.");
                Undo.RegisterCreatedObjectUndo(copy, "Consolidar cabello Genesis");
                Selection.activeGameObject = copy;
            }
            catch (Exception exc)
            {
                if (copy != null) UnityEngine.Object.DestroyImmediate(copy);
                Debug.LogError("AETHRYX: " + exc.Message);
            }
        }

        public static int Consolidate(GameObject copy)
        {
            var groups = copy.GetComponentsInChildren<SkinnedMeshRenderer>()
                .Where(r => r.name.StartsWith("cabello-largo-", StringComparison.Ordinal) &&
                    r.sharedMesh != null && r.sharedMesh.blendShapeCount == 0 &&
                    r.sharedMesh.subMeshCount == 1 && r.sharedMaterial != null &&
                    r.transform.childCount == 0)
                .GroupBy(r => (r.sharedMaterial, r.rootBone)).Where(group => group.Count() > 1).ToArray();
            int removed = 0;
            foreach (var group in groups)
            {
                var renderers = group.ToArray();
                string folder = Path.GetDirectoryName(AssetDatabase.GetAssetPath(renderers[0].sharedMesh))
                    ?.Replace('\\', '/');
                if (string.IsNullOrEmpty(folder) || !folder.StartsWith("Assets/", StringComparison.Ordinal))
                    throw new InvalidOperationException("La copia necesita mallas Genesis guardadas en Assets.");
                var combined = new Mesh { name = "Cabello Genesis consolidado", indexFormat = IndexFormat.UInt32 };
                var bones = new List<Transform>();
                var binding = new List<Matrix4x4>();
                var weights = new List<BoneWeight>();
                var inputs = new List<CombineInstance>();
                foreach (var renderer in renderers)
                {
                    var mesh = renderer.sharedMesh;
                    var matrix = copy.transform.worldToLocalMatrix * renderer.transform.localToWorldMatrix;
                    var sourceBones = renderer.bones;
                    var inverse = mesh.bindposes;
                    var vertexWeights = mesh.boneWeights;
                    if (vertexWeights.Length != mesh.vertexCount ||
                        vertexWeights.Any(weight => Mathf.Abs(weight.weight0 - 1f) > .0001f))
                        throw new InvalidOperationException("La consolidación requiere mechones de unión rígida.");
                    foreach (var weight in vertexWeights)
                    {
                        int index = weight.boneIndex0;
                        var pose = inverse[index] * matrix.inverse;
                        var bone = sourceBones[index];
                        int target = bones.FindIndex(item => item == bone);
                        if (target < 0)
                        {
                            target = bones.Count;
                            bones.Add(bone);
                            binding.Add(pose);
                        }
                        else if (Enumerable.Range(0, 16).Any(i => Mathf.Abs(binding[target][i] - pose[i]) > .0001f))
                            throw new InvalidOperationException("El hueso comparte dos matrices de reposo incompatibles.");
                        weights.Add(new BoneWeight { boneIndex0 = target, weight0 = 1 });
                    }
                    inputs.Add(new CombineInstance { mesh = mesh, subMeshIndex = 0, transform = matrix });
                }
                combined.CombineMeshes(inputs.ToArray(), true, true);
                if (combined.vertexCount != weights.Count)
                    throw new InvalidOperationException("La combinación cambió el orden de los vértices.");
                combined.bindposes = binding.ToArray();
                combined.boneWeights = weights.ToArray();
                AssetDatabase.CreateAsset(combined, AssetDatabase.GenerateUniqueAssetPath(folder + "/cabello-consolidado.asset"));
                var node = new GameObject("cabello-consolidado");
                node.transform.SetParent(copy.transform, false);
                var targetRenderer = node.AddComponent<SkinnedMeshRenderer>();
                targetRenderer.sharedMesh = combined;
                targetRenderer.sharedMaterial = group.Key.Item1;
                targetRenderer.rootBone = group.Key.Item2;
                targetRenderer.bones = bones.ToArray();
                targetRenderer.localBounds = combined.bounds;
                foreach (var renderer in renderers)
                    UnityEngine.Object.DestroyImmediate(renderer.gameObject);
                removed += renderers.Length - 1;
            }
            AssetDatabase.SaveAssets();
            return removed;
        }
    }
}
