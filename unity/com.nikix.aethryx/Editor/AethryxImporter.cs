using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;

namespace Nikix.Aethryx
{
    public static class AethryxImporter
    {
        [Serializable] class Root { public Node[] nodes; public MeshDef[] meshes; public Skin[] skins;
            public Accessor[] accessors; public View[] bufferViews; public MaterialDef[] materials;
            public TextureDef[] textures; public ImageDef[] images; public AnimationDef[] animations; }
        [Serializable] class Node { public string name; public int[] children; public float[] translation;
            public int mesh = -1; public int skin = -1; public float[] matrix; }
        [Serializable] class MeshDef { public Primitive[] primitives; public Extras extras; }
        [Serializable] class Primitive { public Attributes attributes; public int indices = -1;
            public int material = -1; public Morph[] targets; }
        [Serializable] class Attributes { public int POSITION = -1, NORMAL = -1, TEXCOORD_0 = -1, JOINTS_0 = -1, WEIGHTS_0 = -1; }
        [Serializable] class Extras { public string[] targetNames; }
        [Serializable] class Morph { public int POSITION; }
        [Serializable] class Skin { public int[] joints; public int inverseBindMatrices; public int skeleton; }
        [Serializable] class Accessor { public int bufferView, byteOffset, count, componentType; public string type; }
        [Serializable] class View { public int byteOffset, byteLength, byteStride; }
        [Serializable] class MaterialDef { public Pbr pbrMetallicRoughness; public string alphaMode;
            public float alphaCutoff = .5f; public float[] emissiveFactor; public TextureInfo emissiveTexture; }
        [Serializable] class Pbr { public float[] baseColorFactor; public TextureInfo baseColorTexture;
            public float roughnessFactor = 1f, metallicFactor = 1f; }
        [Serializable] class TextureInfo { public int index; }
        [Serializable] class TextureDef { public int source; }
        [Serializable] class ImageDef { public int bufferView; }
        [Serializable] class AnimationDef { public string name; public Channel[] channels; public Sampler[] samplers; }
        [Serializable] class Sampler { public int input, output; }
        [Serializable] class Channel { public int sampler; public Target target; }
        [Serializable] class Target { public int node; public string path; }

        sealed class Binary
        {
            public readonly Root Data;
            readonly byte[] bytes;
            readonly int origin;
            public Binary(string path)
            {
                bytes = File.ReadAllBytes(path);
                if (bytes.Length < 28 || BitConverter.ToUInt32(bytes, 0) != 0x46546c67 ||
                    BitConverter.ToUInt32(bytes, 4) != 2 || BitConverter.ToUInt32(bytes, 8) != bytes.Length)
                    throw new InvalidDataException("Se requiere un GLB 2 de AETHRYX.");
                var jsonLength = BitConverter.ToInt32(bytes, 12);
                if (jsonLength < 0 || jsonLength > bytes.Length - 28 ||
                    BitConverter.ToUInt32(bytes, 16) != 0x4e4f534a ||
                    BitConverter.ToUInt32(bytes, 24 + jsonLength) != 0x004e4942)
                    throw new InvalidDataException("Bloques GLB inválidos.");
                origin = 28 + jsonLength;
                Data = JsonUtility.FromJson<Root>(System.Text.Encoding.UTF8.GetString(bytes, 20, jsonLength));
            }
            public Accessor Get(int index) { return Data.accessors[index]; }
            int Sample(int index, int row, int col, int columns, int elementSize)
            {
                var a = Get(index);
                var v = Data.bufferViews[a.bufferView];
                int stride = v.byteStride == 0 ? columns * elementSize : v.byteStride;
                int offset = checked(a.byteOffset + row * stride + col * elementSize);
                if (offset < 0 || offset > v.byteLength - elementSize)
                    throw new InvalidDataException("Accesor GLB fuera del búfer.");
                return checked(origin + v.byteOffset + offset);
            }
            public float Float(int index, int row, int col, int columns)
            {
                if (Get(index).componentType != 5126) throw new InvalidDataException("Accesor sin float32.");
                return BitConverter.ToSingle(bytes, Sample(index, row, col, columns, 4));
            }
            public int Index(int index, int row, int col = 0, int columns = 1)
            {
                switch (Get(index).componentType)
                {
                    case 5121: return bytes[Sample(index, row, col, columns, 1)];
                    case 5123: return BitConverter.ToUInt16(bytes, Sample(index, row, col, columns, 2));
                    case 5125: return checked((int)BitConverter.ToUInt32(bytes, Sample(index, row, col, columns, 4)));
                    default: throw new InvalidDataException("Tipo de índice GLB no compatible.");
                }
            }
            public Vector3 Vector(int index, int row)
            {
                return new Vector3(Float(index, row, 0, 3), Float(index, row, 1, 3), -Float(index, row, 2, 3));
            }
            public byte[] Image(int index)
            {
                var view = Data.bufferViews[Data.images[index].bufferView];
                var result = new byte[view.byteLength];
                Buffer.BlockCopy(bytes, origin + view.byteOffset, result, 0, result.Length);
                return result;
            }
        }

        [MenuItem("Tools/AETHRYX/Importar avatar Genesis")]
        public static void ImportFromMenu()
        {
            string path = EditorUtility.OpenFilePanel("Importar GLB articulado de AETHRYX", "", "glb");
            if (string.IsNullOrEmpty(path)) return;
            try { Import(path); }
            catch (Exception exc) { Debug.LogError("AETHRYX: " + exc.Message); }
        }

        [MenuItem("Tools/AETHRYX/Diagnosticar avatar seleccionado")]
        public static void DiagnoseSelection()
        {
            var root = Selection.activeGameObject;
            if (root == null) return;
            var renderers = root.GetComponentsInChildren<SkinnedMeshRenderer>(true).Where(r => r.sharedMesh != null).ToArray();
            var meshes = renderers.Select(r => r.sharedMesh).ToArray();
            var animator = root.GetComponent<Animator>();
            Debug.Log($"AETHRYX: mallas {meshes.Length}, vértices {meshes.Sum(m => m.vertexCount)}, " +
                $"triángulos {meshes.Sum(m => m.triangles.Length / 3)}, morphs {meshes.Sum(m => m.blendShapeCount)}, " +
                $"materiales {renderers.SelectMany(r => r.sharedMaterials).Distinct().Count()}, " +
                $"Humanoid {(animator != null && animator.avatar != null && animator.avatar.isHuman)}.");
        }

        [MenuItem("Tools/AETHRYX/Aplicar materiales Toon")]
        public static void ApplyToonToSelection()
        {
            var root = Selection.activeGameObject;
            if (root == null) return;
            var shader = Shader.Find("AETHRYX/Toon");
            if (shader == null || ShaderUtil.ShaderHasError(shader))
                throw new InvalidOperationException("No se pudo preparar el shader AETHRYX/Toon.");
            if (!AssetDatabase.IsValidFolder("Assets/AETHRYX")) AssetDatabase.CreateFolder("Assets", "AETHRYX");
            var folder = AssetDatabase.GenerateUniqueAssetPath("Assets/AETHRYX/Toon");
            AssetDatabase.CreateFolder("Assets/AETHRYX", Path.GetFileName(folder));
            var converted = new Dictionary<Material, Material>();
            foreach (var renderer in root.GetComponentsInChildren<Renderer>(true))
            {
                Undo.RecordObject(renderer, "AETHRYX Toon");
                renderer.sharedMaterials = renderer.sharedMaterials.Select(previous => {
                    if (previous == null) return null;
                    if (converted.TryGetValue(previous, out var result)) return result;
                    result = new Material(shader);
                    if (previous.HasProperty("_Color")) result.color = previous.color;
                    if (previous.HasProperty("_MainTex")) result.mainTexture = previous.mainTexture;
                    if (previous.HasProperty("_EmissionColor"))
                        result.SetColor("_EmissionColor", previous.GetColor("_EmissionColor"));
                    if (previous.HasProperty("_EmissionMap"))
                        result.SetTexture("_EmissionMap", previous.GetTexture("_EmissionMap"));
                    if (previous.IsKeywordEnabled("_ALPHATEST_ON") && previous.HasProperty("_Cutoff"))
                        result.SetFloat("_Cutoff", previous.GetFloat("_Cutoff"));
                    AssetDatabase.CreateAsset(result, folder + "/toon-" + converted.Count + ".mat");
                    converted.Add(previous, result);
                    return result;
                }).ToArray();
            }
            AssetDatabase.SaveAssets();
        }

        public static GameObject Import(string path)
        {
            var file = new Binary(path);
            var doc = file.Data;
            if (doc.skins == null || doc.skins.Length != 1)
                throw new InvalidDataException("Exporta el avatar con un skin nativo de Genesis.");
            if (!AssetDatabase.IsValidFolder("Assets/AETHRYX")) AssetDatabase.CreateFolder("Assets", "AETHRYX");
            var folder = AssetDatabase.GenerateUniqueAssetPath("Assets/AETHRYX/Genesis");
            AssetDatabase.CreateFolder("Assets/AETHRYX", Path.GetFileName(folder));
            var root = new GameObject("GenesisAvatar");
            try
            {
                var transforms = doc.nodes.Select(node => new GameObject(node.name).transform).ToArray();
                foreach (var transform in transforms) transform.SetParent(root.transform, false);
                for (int i = 0; i < doc.nodes.Length; i++)
                {
                    var node = doc.nodes[i];
                    if (node.translation != null)
                        transforms[i].localPosition = new Vector3(node.translation[0], node.translation[1], -node.translation[2]);
                    if (node.children != null)
                        foreach (var child in node.children) transforms[child].SetParent(transforms[i], false);
                }
                var skin = doc.skins[0];
                var bind = new Matrix4x4[skin.joints.Length];
                for (int i = 0; i < bind.Length; i++)
                    for (int row = 0; row < 4; row++)
                        for (int col = 0; col < 4; col++)
                            bind[i][row, col] = file.Float(skin.inverseBindMatrices, i, col * 4 + row, 16)
                                * ((row == 2) != (col == 2) ? -1 : 1);
                var materials = (doc.materials ?? new MaterialDef[0]).Select((m, i) => BuildMaterial(file, m, folder, i)).ToArray();
                Material fallbackMaterial = null;
                for (int i = 0; i < doc.nodes.Length; i++)
                {
                    var node = doc.nodes[i];
                    if (node.mesh < 0 || node.skin != 0) continue;
                    var definition = doc.meshes[node.mesh];
                    for (int j = 0; j < definition.primitives.Length; j++)
                    {
                        var primitive = definition.primitives[j];
                        var owner = transforms[i].gameObject;
                        if (j > 0)
                        {
                            owner = new GameObject("Primitive_" + j);
                            owner.transform.SetParent(transforms[i], false);
                        }
                        var mesh = BuildMesh(file, primitive, definition.extras, bind);
                        AssetDatabase.CreateAsset(mesh, folder + "/mesh-" + i + "-" + j + ".asset");
                        var renderer = owner.AddComponent<SkinnedMeshRenderer>();
                        renderer.sharedMesh = mesh;
                        renderer.bones = skin.joints.Select(joint => transforms[joint]).ToArray();
                        renderer.rootBone = transforms[skin.skeleton];
                        if (primitive.material >= 0)
                            renderer.sharedMaterial = materials[primitive.material];
                        else
                        {
                            if (fallbackMaterial == null)
                                fallbackMaterial = BuildMaterial(file, new MaterialDef(), folder, materials.Length);
                            renderer.sharedMaterial = fallbackMaterial;
                        }
                    }
                }
                var human = BuildHumanoid(root, transforms, skin);
                if (human.isValid && human.isHuman)
                {
                    AssetDatabase.CreateAsset(human, folder + "/Humanoid.asset");
                    root.AddComponent<Animator>().avatar = human;
                }
                else
                {
                    UnityEngine.Object.DestroyImmediate(human);
                    Debug.LogWarning("AETHRYX: el skin se importó, pero Unity no validó la configuración Humanoid.");
                }
                if (doc.animations != null)
                    for (int i = 0; i < doc.animations.Length; i++)
                        AssetDatabase.CreateAsset(BuildClip(file, doc.animations[i], root, transforms), folder + "/clip-" + i + ".anim");
                PrefabUtility.SaveAsPrefabAsset(root, folder + "/GenesisAvatar.prefab");
                AssetDatabase.SaveAssets();
                Selection.activeGameObject = root;
                Debug.Log("AETHRYX: " + folder);
                return root;
            }
            catch
            {
                UnityEngine.Object.DestroyImmediate(root);
                AssetDatabase.DeleteAsset(folder);
                throw;
            }
        }

        static Mesh BuildMesh(Binary file, Primitive block, Extras extras, Matrix4x4[] bind)
        {
            var a = block.attributes;
            int count = file.Get(a.POSITION).count;
            var mesh = new Mesh { indexFormat = count > 65535 ? IndexFormat.UInt32 : IndexFormat.UInt16 };
            mesh.vertices = Enumerable.Range(0, count).Select(i => file.Vector(a.POSITION, i)).ToArray();
            if (a.NORMAL >= 0) mesh.normals = Enumerable.Range(0, count).Select(i => file.Vector(a.NORMAL, i)).ToArray();
            if (a.TEXCOORD_0 >= 0)
                mesh.uv = Enumerable.Range(0, count).Select(i => new Vector2(
                    file.Float(a.TEXCOORD_0, i, 0, 2), 1f - file.Float(a.TEXCOORD_0, i, 1, 2))).ToArray();
            var indices = Enumerable.Range(0, file.Get(block.indices).count).Select(i => file.Index(block.indices, i)).ToArray();
            for (int i = 0; i < indices.Length; i += 3)
            {
                int temp = indices[i + 1]; indices[i + 1] = indices[i + 2]; indices[i + 2] = temp;
            }
            mesh.triangles = indices;
            if (a.NORMAL < 0) mesh.RecalculateNormals();
            mesh.bindposes = bind;
            mesh.boneWeights = Enumerable.Range(0, count).Select(i => new BoneWeight {
                boneIndex0 = file.Index(a.JOINTS_0, i, 0, 4), boneIndex1 = file.Index(a.JOINTS_0, i, 1, 4),
                boneIndex2 = file.Index(a.JOINTS_0, i, 2, 4), boneIndex3 = file.Index(a.JOINTS_0, i, 3, 4),
                weight0 = file.Float(a.WEIGHTS_0, i, 0, 4), weight1 = file.Float(a.WEIGHTS_0, i, 1, 4),
                weight2 = file.Float(a.WEIGHTS_0, i, 2, 4), weight3 = file.Float(a.WEIGHTS_0, i, 3, 4),
            }).ToArray();
            if (block.targets != null)
                for (int i = 0; i < block.targets.Length; i++)
                {
                    var delta = Enumerable.Range(0, count).Select(j => file.Vector(block.targets[i].POSITION, j)).ToArray();
                    mesh.AddBlendShapeFrame(extras.targetNames[i], 100, delta, null, null);
                }
            mesh.RecalculateBounds();
            return mesh;
        }

        static Material BuildMaterial(Binary file, MaterialDef def, string folder, int index)
        {
            var shader = Shader.Find("Standard");
            if (shader == null) throw new InvalidOperationException("El puente inicial requiere el shader Standard.");
            var material = new Material(shader);
            var pbr = def.pbrMetallicRoughness;
            if (pbr != null)
            {
                if (pbr.baseColorFactor != null)
                    material.color = new Color(pbr.baseColorFactor[0], pbr.baseColorFactor[1],
                        pbr.baseColorFactor[2], pbr.baseColorFactor[3]);
                material.SetFloat("_Metallic", pbr.metallicFactor);
                material.SetFloat("_Glossiness", 1 - pbr.roughnessFactor);
                if (pbr.baseColorTexture != null)
                {
                    material.mainTexture = BuildTexture(file, pbr.baseColorTexture, folder + "/color-" + index);
                }
            }
            if (def.emissiveFactor != null && def.emissiveFactor.Length == 3)
            {
                material.SetColor("_EmissionColor",
                    new Color(def.emissiveFactor[0], def.emissiveFactor[1], def.emissiveFactor[2]));
                if (def.emissiveTexture != null)
                    material.SetTexture("_EmissionMap", BuildTexture(file, def.emissiveTexture, folder + "/emission-" + index));
                material.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;
                material.EnableKeyword("_EMISSION");
            }
            if (def.alphaMode == "MASK")
            {
                material.EnableKeyword("_ALPHATEST_ON");
                material.SetFloat("_Cutoff", def.alphaCutoff);
                material.renderQueue = (int)RenderQueue.AlphaTest;
            }
            AssetDatabase.CreateAsset(material, folder + "/pbr-" + index + ".mat");
            return material;
        }

        static Texture2D BuildTexture(Binary file, TextureInfo info, string path)
        {
            var texture = new Texture2D(2, 2, TextureFormat.RGBA32, true);
            if (!ImageConversion.LoadImage(texture, file.Image(file.Data.textures[info.index].source)))
                throw new InvalidDataException("No se pudo decodificar la textura GLB.");
            AssetDatabase.CreateAsset(texture, path + ".asset");
            return texture;
        }

        static Avatar BuildHumanoid(GameObject root, Transform[] transforms, Skin skin)
        {
            var available = skin.joints.Select(i => transforms[i].name).ToDictionary(name => name);
            var human = HumanTrait.BoneName.Select(name => new { label = name, bone = name.Replace(" ", "") })
                .Where(pair => available.ContainsKey(pair.bone))
                .Select(pair => new HumanBone {
                    humanName = pair.label, boneName = pair.bone, limit = new HumanLimit { useDefaultValues = true }
                }).ToArray();
            var skeleton = new[] { root.transform }.Concat(transforms).Select(t => new SkeletonBone {
                name = t.name, position = t.localPosition, rotation = t.localRotation, scale = t.localScale
            }).ToArray();
            var description = new HumanDescription {
                human = human, skeleton = skeleton, armStretch = .05f, legStretch = .05f,
                upperArmTwist = .5f, lowerArmTwist = .5f, upperLegTwist = .5f, lowerLegTwist = .5f,
            };
            return AvatarBuilder.BuildHumanAvatar(root, description);
        }

        static AnimationClip BuildClip(Binary file, AnimationDef def, GameObject root, Transform[] transforms)
        {
            var clip = new AnimationClip { name = def.name };
            foreach (var channel in def.channels)
            {
                if (channel.target.path != "rotation") continue;
                var sample = def.samplers[channel.sampler];
                var path = AnimationUtility.CalculateTransformPath(transforms[channel.target.node], root.transform);
                for (int axis = 0; axis < 4; axis++)
                {
                    var keys = new Keyframe[file.Get(sample.input).count];
                    for (int i = 0; i < keys.Length; i++)
                        keys[i] = new Keyframe(file.Float(sample.input, i, 0, 1),
                            file.Float(sample.output, i, axis, 4) * (axis < 2 ? -1 : 1));
                    var binding = EditorCurveBinding.FloatCurve(path, typeof(Transform), "m_LocalRotation." + "xyzw"[axis]);
                    AnimationUtility.SetEditorCurve(clip, binding, new AnimationCurve(keys));
                }
            }
            clip.EnsureQuaternionContinuity();
            return clip;
        }
    }
}
