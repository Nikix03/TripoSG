using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Nikix.Aethryx
{
    public static class AethryxVrchat
    {
        static readonly string[] Visemes = {
            "sil", "pp", "ff", "th", "dd", "kk", "ch", "ss", "nn", "rr", "aa", "e", "ih", "oh", "ou"
        };

        [MenuItem("Tools/AETHRYX/Preparar Avatar Descriptor de VRChat")]
        static void ConfigureSelection()
        {
            if (Selection.activeGameObject == null) return;
            try
            {
                Configure(Selection.activeGameObject);
                AethryxImporter.DiagnoseSelection();
                ReportPerformance(Selection.activeGameObject);
            }
            catch (Exception exc) { Debug.LogError("AETHRYX: " + exc.Message); }
        }

        [MenuItem("Tools/AETHRYX/Informe de rendimiento VRChat")]
        static void DiagnoseSelection()
        {
            if (Selection.activeGameObject == null) return;
            try { ReportPerformance(Selection.activeGameObject); }
            catch (Exception exc) { Debug.LogError("AETHRYX: " + exc.Message); }
        }

        [MenuItem("Tools/AETHRYX/Preparar PhysBone de la pieza seleccionada")]
        static void PrepareSelectedPhysBone()
        {
            var renderer = Selection.activeGameObject?.GetComponent<SkinnedMeshRenderer>();
            try
            {
                if (renderer != null)
                    PreparePhysBone(renderer);
                else if (Selection.activeTransform != null)
                    PrepareHairChain(Selection.activeTransform);
            }
            catch (Exception exc) { Debug.LogError("AETHRYX: " + exc.Message); }
        }

        static Type PhysBoneType()
        {
            const string typeName = "VRC.SDK3.Dynamics.PhysBone.Components.VRCPhysBone";
            var type = AppDomain.CurrentDomain.GetAssemblies().Select(a => a.GetType(typeName))
                .FirstOrDefault(candidate => candidate != null);
            if (type == null) throw new NotSupportedException("VRChat SDK PhysBones no está disponible.");
            return type;
        }

        public static Component PrepareHairChain(Transform root)
        {
            const string suffix = "-junta-0";
            if (!root.name.StartsWith("cabello-largo-", StringComparison.Ordinal)
                || !root.name.EndsWith(suffix, StringComparison.Ordinal) || root.parent?.name != "Head")
                throw new ArgumentException("Selecciona la junta inicial de un mechón Genesis.");
            var prefix = root.name.Substring(0, root.name.Length - suffix.Length);
            var cursor = root;
            var chain = new Transform[4];
            chain[0] = root;
            for (int index = 1; index < 4; index++)
            {
                cursor = cursor.Find(prefix + "-junta-" + index);
                if (cursor == null) throw new InvalidOperationException("La cadena capilar está incompleta.");
                chain[index] = cursor;
            }
            var used = root.root.GetComponentsInChildren<SkinnedMeshRenderer>(true).Any(renderer =>
            {
                if (renderer.sharedMesh == null) return false;
                var bones = renderer.bones;
                var slots = new HashSet<int>(chain.Select(bone => Array.IndexOf(bones, bone)));
                return !slots.Contains(-1) && renderer.sharedMesh.GetAllBoneWeights().ToArray()
                    .Where(weight => weight.weight > 0 && slots.Contains(weight.boneIndex))
                    .Select(weight => weight.boneIndex).Distinct().Count() == 4;
            });
            if (!used) throw new InvalidOperationException("La cadena capilar carece de pesos activos.");
            var type = PhysBoneType();
            if (root.GetComponent(type) != null)
                throw new InvalidOperationException("La cadena ya tiene un PhysBone.");
            var component = Undo.AddComponent(root.gameObject, type);
            try
            {
                using (var data = new SerializedObject(component))
                {
                    Find(data, "rootTransform").objectReferenceValue = root;
                    data.ApplyModifiedProperties();
                }
                return component;
            }
            catch
            {
                Undo.DestroyObjectImmediate(component);
                throw;
            }
        }

        public static Component PreparePhysBone(SkinnedMeshRenderer renderer)
        {
            var type = PhysBoneType();
            if ((!renderer.name.StartsWith("cabello-largo-") && !renderer.name.StartsWith("accesorio-hamsa-"))
                || renderer.sharedMesh == null)
                throw new ArgumentException("Selecciona un mechón largo o un aro Genesis.");
            var bones = renderer.bones;
            int headIndex = Array.FindIndex(bones, bone => bone != null && bone.name == "Head");
            var mesh = renderer.sharedMesh;
            var chain = Enumerable.Range(0, 4)
                .Select(index => Array.FindIndex(bones, bone => bone != null && bone.name == renderer.name + "-junta-" + index))
                .ToArray();
            if (chain.Any(index => index >= 0))
            {
                if (headIndex < 0 || chain.Any(index => index < 0) || bones[chain[0]].parent != bones[headIndex]
                    || Enumerable.Range(1, 3).Any(i => bones[chain[i]].parent != bones[chain[i - 1]]))
                    throw new InvalidOperationException("La cadena capilar de Genesis está incompleta.");
                var activeWeights = mesh.GetAllBoneWeights().ToArray();
                if (activeWeights.Length == 0 || activeWeights.Any(weight => !chain.Contains(weight.boneIndex)))
                    throw new InvalidOperationException("La cadena capilar tiene pesos incompatibles.");
                return PrepareHairChain(bones[chain[0]]);
            }
            using (var weights = mesh.GetAllBoneWeights())
                if (headIndex < 0 || weights.Length == 0 || weights.ToArray().Any(
                    weight => weight.boneIndex != headIndex || weight.weight < .999f))
                    throw new InvalidOperationException("La pieza debe estar unida rígidamente a la cabeza.");
            string folder = Path.GetDirectoryName(AssetDatabase.GetAssetPath(mesh))?.Replace('\\', '/');
            if (string.IsNullOrEmpty(folder) || !folder.StartsWith("Assets/"))
                throw new InvalidOperationException("Importa primero la pieza en un avatar Genesis guardado en Assets.");

            var head = bones[headIndex];
            var bounds = mesh.bounds;
            var anchor = new GameObject(renderer.name + "-pivote");
            Undo.RegisterCreatedObjectUndo(anchor, "AETHRYX PhysBone");
            anchor.transform.SetParent(head, false);
            var top = new Vector3(bounds.center.x, bounds.max.y, bounds.center.z);
            var bottom = new Vector3(bounds.center.x, bounds.min.y, bounds.center.z);
            anchor.transform.position = renderer.transform.TransformPoint(top);
            var binding = UnityEngine.Object.Instantiate(mesh);
            try
            {
                var phys = Undo.AddComponent(anchor, type);
                using (var data = new SerializedObject(phys))
                {
                    Find(data, "rootTransform").objectReferenceValue = anchor.transform;
                    Find(data, "endpointPosition").vector3Value =
                        anchor.transform.InverseTransformPoint(renderer.transform.TransformPoint(bottom));
                    data.ApplyModifiedProperties();
                }
                var matrices = binding.bindposes;
                matrices[headIndex] = anchor.transform.worldToLocalMatrix * renderer.transform.localToWorldMatrix;
                binding.bindposes = matrices;
                AssetDatabase.CreateAsset(binding, AssetDatabase.GenerateUniqueAssetPath(folder + "/physbone-mesh.asset"));
                Undo.RecordObject(renderer, "AETHRYX PhysBone");
                bones[headIndex] = anchor.transform;
                renderer.bones = bones;
                renderer.sharedMesh = binding;
                return phys;
            }
            catch
            {
                Undo.DestroyObjectImmediate(anchor);
                if (!AssetDatabase.Contains(binding)) UnityEngine.Object.DestroyImmediate(binding);
                throw;
            }
        }

        public static string ReportPerformance(GameObject root)
        {
            const string name = "VRC.SDKBase.Validation.Performance.AvatarPerformance";
            var api = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetType(name)).FirstOrDefault(type => type != null);
            var calculate = api?.GetMethods().FirstOrDefault(method =>
                method.Name == "CalculatePerformanceStats" && method.GetParameters().Length == 4);
            if (calculate == null) throw new NotSupportedException("API de rendimiento de VRChat SDK ausente.");
            var statsType = calculate.GetParameters()[2].ParameterType;
            var rating = statsType.GetMethod("GetPerformanceRatingForCategory");
            if (rating == null) throw new NotSupportedException("Clasificación de VRChat SDK ausente.");
            var category = rating.GetParameters()[0].ParameterType;
            var overall = Enum.Parse(category, "Overall");
            string GetRating(bool mobile)
            {
                var stats = Activator.CreateInstance(statsType, new object[] { mobile });
                calculate.Invoke(null, new[] { (object)root.name, root, stats, mobile });
                var blockers = Enum.GetValues(category).Cast<object>()
                    .Where(value => !Equals(value, overall) && !value.ToString().EndsWith("CategoryCount") &&
                        string.Equals(rating.Invoke(stats, new[] { value }).ToString(), "VeryPoor",
                                      StringComparison.OrdinalIgnoreCase))
                    .Select(value => value.ToString());
                return rating.Invoke(stats, new[] { overall }) + " [" + string.Join(", ", blockers) + "]";
            }
            var result = "PC=" + GetRating(false) + " mobile=" + GetRating(true);
            Debug.Log("AETHRYX_VRCHAT_PERFORMANCE " + result);
            return result;
        }

        public static Component Configure(GameObject root)
        {
            const string descriptorName = "VRC.SDK3.Avatars.Components.VRCAvatarDescriptor";
            var type = AppDomain.CurrentDomain.GetAssemblies()
                .Select(assembly => assembly.GetType(descriptorName)).FirstOrDefault(candidate => candidate != null);
            if (type == null)
                throw new NotSupportedException("Instala VRChat SDK Avatars en el proyecto de Unity.");
            if (root.GetComponent(type) != null)
                throw new InvalidOperationException("El avatar ya tiene un descriptor configurado.");
            var animator = root.GetComponent<Animator>();
            if (animator == null || animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                throw new InvalidOperationException("El avatar necesita un Humanoid válido.");
            var renderers = root.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            var names = Visemes.Select(name => "vrc.v_" + name).ToArray();
            var body = renderers.FirstOrDefault(r => r.sharedMesh != null &&
                names.All(name => r.sharedMesh.GetBlendShapeIndex(name) >= 0));
            if (body == null) throw new InvalidOperationException("El cuerpo debe incluir los 15 visemas Genesis.");
            var left = body.bones.FirstOrDefault(b => b != null && b.name == "LeftEye");
            var right = body.bones.FirstOrDefault(b => b != null && b.name == "RightEye");
            if (left == null || right == null) throw new InvalidOperationException("Faltan los huesos de los ojos.");

            var descriptor = Undo.AddComponent(root, type);
            try
            {
                using (var data = new SerializedObject(descriptor))
                {
                    SetEnum(Find(data, "lipSync"), "VisemeBlendShape");
                    Find(data, "VisemeSkinnedMesh").objectReferenceValue = body;
                    var lips = Find(data, "VisemeBlendShapes");
                    lips.arraySize = names.Length;
                    for (int index = 0; index < names.Length; index++)
                        lips.GetArrayElementAtIndex(index).stringValue = names[index];
                    Find(data, "enableEyeLook").boolValue = true;
                    const string eye = "customEyeLookSettings.";
                    SetEnum(Find(data, eye + "eyeGazeMode"), "BoneRotation");
                    Find(data, eye + "leftEye").objectReferenceValue = left;
                    Find(data, eye + "rightEye").objectReferenceValue = right;
                    SetEyeRotation(data, "Straight", left, right, 0, 0);
                    SetEyeRotation(data, "Up", left, right, -12, 0);
                    SetEyeRotation(data, "Down", left, right, 12, 0);
                    SetEyeRotation(data, "Left", left, right, 0, -15);
                    SetEyeRotation(data, "Right", left, right, 0, 15);
                    Find(data, "ViewPosition").vector3Value =
                        root.transform.InverseTransformPoint((left.position + right.position) * .5f);
                    data.ApplyModifiedProperties();
                }
                return descriptor;
            }
            catch
            {
                Undo.DestroyObjectImmediate(descriptor);
                throw;
            }
        }

        static SerializedProperty Find(SerializedObject data, string path) =>
            data.FindProperty(path) ?? throw new NotSupportedException("Campo de VRChat SDK ausente: " + path);

        static void SetEnum(SerializedProperty property, string name)
        {
            var index = Array.IndexOf(property.enumNames, name);
            if (index < 0) throw new NotSupportedException("Opción de VRChat SDK ausente: " + name);
            property.enumValueIndex = index;
        }

        static void SetEyeRotation(SerializedObject data, string direction, Transform left, Transform right,
                                   float pitch, float yaw)
        {
            var path = "customEyeLookSettings.eyesLooking" + direction + ".";
            var delta = Quaternion.Euler(pitch, yaw, 0);
            Find(data, path + "left").quaternionValue = left.localRotation * delta;
            Find(data, path + "right").quaternionValue = right.localRotation * delta;
        }
    }
}
