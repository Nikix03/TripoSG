using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Unity.Collections;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace Nikix.Aethryx
{
    public static class AethryxHairCombine
    {
        [MenuItem("Tools/AETHRYX/Crear copia con cabello consolidado")]
        static void CombineHairSelection() => CombineSelection(false);

        [MenuItem("Tools/AETHRYX/Crear copia con componentes consolidados")]
        static void CombineComponentSelection() => CombineSelection(true);

        static void CombineSelection(bool includeComponents)
        {
            var source = Selection.activeGameObject;
            if (source == null) return;
            GameObject copy = null;
            try
            {
                copy = UnityEngine.Object.Instantiate(source, source.transform.parent);
                copy.name = source.name + " · copia consolidada";
                if (Consolidate(copy, includeComponents) == 0)
                    throw new InvalidOperationException("La selección no contiene componentes compatibles.");
                Undo.RegisterCreatedObjectUndo(copy, "Consolidar componentes Genesis");
                Selection.activeGameObject = copy;
            }
            catch (Exception exc)
            {
                if (copy != null) UnityEngine.Object.DestroyImmediate(copy);
                Debug.LogError("AETHRYX: " + exc.Message);
            }
        }

        public static int Consolidate(GameObject copy, bool includeComponents = false)
        {
            var groups = copy.GetComponentsInChildren<SkinnedMeshRenderer>()
                .Where(r => (r.name.StartsWith("cabello-largo-", StringComparison.Ordinal) ||
                        (includeComponents && (r.name.StartsWith("cabello-", StringComparison.Ordinal) ||
                            r.name.StartsWith("oreja-", StringComparison.Ordinal) ||
                            r.name.StartsWith("prenda-zapato-", StringComparison.Ordinal)))) &&
                    r.sharedMesh != null && r.sharedMesh.blendShapeCount == 0 &&
                    r.sharedMesh.subMeshCount == 1 && r.sharedMaterial != null &&
                    r.transform.childCount == 0 && r.enabled)
                .GroupBy(r => (r.sharedMaterial, r.rootBone, r.shadowCastingMode, r.receiveShadows,
                               r.updateWhenOffscreen, r.quality, r.gameObject.layer))
                .Where(group => group.Count() > 1).ToArray();
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
                var weights = new List<BoneWeight1>();
                var influenceCounts = new List<byte>();
                var inputs = new List<CombineInstance>();
                foreach (var renderer in renderers)
                {
                    var mesh = renderer.sharedMesh;
                    var matrix = copy.transform.worldToLocalMatrix * renderer.transform.localToWorldMatrix;
                    var sourceBones = renderer.bones;
                    var inverse = mesh.bindposes;
                    var sourceCounts = mesh.GetBonesPerVertex();
                    var vertexWeights = mesh.GetAllBoneWeights();
                    if (sourceCounts.Length != mesh.vertexCount || sourceCounts.Any(count => count == 0))
                        throw new InvalidOperationException("El componente necesita pesos en todos sus vértices.");
                    influenceCounts.AddRange(sourceCounts);
                    var matrixInverse = matrix.inverse;
                    foreach (var weight in vertexWeights)
                    {
                        int index = weight.boneIndex;
                        var pose = inverse[index] * matrixInverse;
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
                        weights.Add(new BoneWeight1 { boneIndex = target, weight = weight.weight });
                    }
                    inputs.Add(new CombineInstance { mesh = mesh, subMeshIndex = 0, transform = matrix });
                }
                combined.CombineMeshes(inputs.ToArray(), true, true);
                if (combined.vertexCount != influenceCounts.Count)
                    throw new InvalidOperationException("La combinación cambió el orden de los vértices.");
                combined.bindposes = binding.ToArray();
                using (var counts = new NativeArray<byte>(influenceCounts.ToArray(), Allocator.Temp))
                using (var remapped = new NativeArray<BoneWeight1>(weights.ToArray(), Allocator.Temp))
                    combined.SetBoneWeights(counts, remapped);
                AssetDatabase.CreateAsset(combined, AssetDatabase.GenerateUniqueAssetPath(folder + "/cabello-consolidado.asset"));
                var node = new GameObject("cabello-consolidado");
                node.transform.SetParent(copy.transform, false);
                var targetRenderer = node.AddComponent<SkinnedMeshRenderer>();
                targetRenderer.sharedMesh = combined;
                targetRenderer.sharedMaterial = group.Key.sharedMaterial;
                targetRenderer.rootBone = group.Key.rootBone;
                targetRenderer.shadowCastingMode = group.Key.shadowCastingMode;
                targetRenderer.receiveShadows = group.Key.receiveShadows;
                targetRenderer.updateWhenOffscreen = group.Key.updateWhenOffscreen;
                targetRenderer.quality = group.Key.quality;
                node.layer = group.Key.layer;
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
