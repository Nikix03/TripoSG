using System;
using System.Linq;
using UnityEditor;
using UnityEngine;
using Nikix.Aethryx;
public static class AethryxBridgeSmoke {
    public static void Run() {
        try {
            var root = AethryxImporter.Import(Environment.GetEnvironmentVariable("AETHRYX_UNITY_FIXTURE"));
            var renderers = root.GetComponentsInChildren<SkinnedMeshRenderer>();
            var body = renderers.First(r => r.name == "cuerpo-base");
            bool full = Environment.GetEnvironmentVariable("AETHRYX_UNITY_FULL") == "1";
            if ((!full && renderers.Length != 2) || body.bones.Length != 53 || body.sharedMesh.blendShapeCount != 23)
                throw new InvalidOperationException("Counts unexpected.");
            if (Enumerable.Range(0, body.sharedMesh.blendShapeCount)
                    .Count(i => body.sharedMesh.GetBlendShapeName(i).StartsWith("vrc.v_")) != 15)
                throw new InvalidOperationException("Missing visemes.");
            if (full) {
                var required = new[]{"ojo-derecho", "cabello-largo-01", "ropa-vestido",
                                     "accesorio-collar-arbol", "accesorio-hamsa-izquierdo",
                                     "accesorio-hamsa-derecho", "prenda-zapato-derecho"};
                if (required.Any(name => !renderers.Any(r => r.name == name)) ||
                    renderers.Any(r => r.bones.Length != 53 || r.sharedMaterial.shader.name != "Standard"))
                    throw new InvalidOperationException("Incomplete modular PBR avatar.");
            }
            var arm = body.bones.First(t => t.name == "LeftUpperArm");
            var before = new Mesh(); body.BakeMesh(before);
            arm.localRotation = Quaternion.Euler(0, 0, 30);
            var after = new Mesh(); body.BakeMesh(after);
            var a = before.vertices; var b = after.vertices;
            float displacement = Enumerable.Range(0, a.Length).Max(i => Vector3.Distance(a[i], b[i]));
            if (displacement < .03f) throw new InvalidOperationException("Skin did not deform.");
            var animator = root.GetComponent<Animator>();
            if (animator == null || animator.avatar == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                throw new InvalidOperationException("Invalid Humanoid.");
            var leftEye = renderers.First(r => r.name == "ojo-izquierdo");
            var texture = leftEye.sharedMaterial.mainTexture;
            if (texture == null) throw new InvalidOperationException("Missing eye texture.");
            if (full) {
                var rightEye = renderers.First(r => r.name == "ojo-derecho");
                if (rightEye.sharedMaterial.mainTexture == null || rightEye.sharedMaterial.mainTexture == texture)
                    throw new InvalidOperationException("Missing independent iris texture.");
                var eyeBone = body.bones.First(t => t.name == "LeftEye");
                leftEye.BakeMesh(before); rightEye.BakeMesh(after);
                var leftRest = before.vertices; var rightRest = after.vertices;
                eyeBone.localRotation = Quaternion.Euler(0, 18, 0);
                leftEye.BakeMesh(before); rightEye.BakeMesh(after);
                if (Enumerable.Range(0, leftRest.Length).Max(i => Vector3.Distance(leftRest[i], before.vertices[i])) < .001f ||
                    Enumerable.Range(0, rightRest.Length).Max(i => Vector3.Distance(rightRest[i], after.vertices[i])) > .0001f)
                    throw new InvalidOperationException("Eye rotation is not independent.");
                body.BakeMesh(before);
                body.SetBlendShapeWeight(body.sharedMesh.GetBlendShapeIndex("vrc.v_aa"), 100);
                body.BakeMesh(after);
                a = before.vertices; b = after.vertices;
                if (Enumerable.Range(0, a.Length).Max(i => Vector3.Distance(a[i], b[i])) < .001f)
                    throw new InvalidOperationException("Viseme did not deform.");
                eyeBone.localRotation = Quaternion.identity;
                body.SetBlendShapeWeight(body.sharedMesh.GetBlendShapeIndex("vrc.v_aa"), 0);
            }
            arm.localRotation = Quaternion.identity;
            var clips = AssetDatabase.FindAssets("t:AnimationClip", new[]{"Assets/AETHRYX"}).Length;
            if (clips < 1) throw new InvalidOperationException("Missing animation clip.");
            var trim = renderers.FirstOrDefault(r => r.name == "ropa-ribete-emisivo");
            if (trim != null && (!trim.sharedMaterial.IsKeywordEnabled("_EMISSION") ||
                                 trim.sharedMaterial.GetColor("_EmissionColor").maxColorComponent < .1f))
                throw new InvalidOperationException("Missing PBR emission: " + trim.sharedMaterial.GetColor("_EmissionColor")
                    + " keyword=" + trim.sharedMaterial.IsKeywordEnabled("_EMISSION"));
            Selection.activeGameObject = root;
            AethryxImporter.ApplyToonToSelection();
            if (renderers.Any(r => r.sharedMaterial.shader.name != "AETHRYX/Toon"))
                throw new InvalidOperationException("Missing Toon.");
            if (trim != null && trim.sharedMaterial.GetColor("_EmissionColor").maxColorComponent < .1f)
                throw new InvalidOperationException("Missing toon emission.");
            if (Environment.GetEnvironmentVariable("AETHRYX_UNITY_FBX") == "1")
                TestFbx(root);
            if (Environment.GetEnvironmentVariable("AETHRYX_UNITY_VRCHAT") == "1")
            {
                var descriptor = AethryxVrchat.Configure(root);
                using (var data = new SerializedObject(descriptor))
                {
                    var lips = data.FindProperty("VisemeBlendShapes");
                    if (lips.arraySize != 15 || lips.GetArrayElementAtIndex(11).stringValue != "vrc.v_e" ||
                        data.FindProperty("VisemeSkinnedMesh").objectReferenceValue != body ||
                        !data.FindProperty("enableEyeLook").boolValue ||
                        data.FindProperty("customEyeLookSettings.leftEye").objectReferenceValue == null ||
                        data.FindProperty("customEyeLookSettings.rightEye").objectReferenceValue == null)
                        throw new InvalidOperationException("Avatar Descriptor mapping incomplete.");
                }
                Debug.Log("AETHRYX_VRCHAT descriptor=" + descriptor.GetType().FullName + " visemes=15 eyes=2");
                AethryxImporter.DiagnoseSelection();
                AethryxVrchat.ReportPerformance(root);
                var earring = renderers.Single(renderer => renderer.name == "accesorio-hamsa-izquierdo");
                earring.BakeMesh(before);
                var rest = before.vertices;
                var phys = AethryxVrchat.PreparePhysBone(earring);
                earring.BakeMesh(after);
                if (Enumerable.Range(0, rest.Length).Max(i => Vector3.Distance(rest[i], after.vertices[i])) > .0001f)
                    throw new InvalidOperationException("PhysBone shifted the bind pose.");
                phys.transform.localRotation = Quaternion.Euler(0, 0, 20);
                earring.BakeMesh(after);
                if (Enumerable.Range(0, rest.Length).Max(i => Vector3.Distance(rest[i], after.vertices[i])) < .002f)
                    throw new InvalidOperationException("The accessory did not follow the PhysBone pivot.");
                Debug.Log("AETHRYX_VRCHAT_PHYSBONE " + phys.GetType().FullName + " bind and deformation passed");
            }
            Debug.Log("AETHRYX_SMOKE renderer=" + renderers.Length + " bones=" + body.bones.Length +
                " morphs=" + body.sharedMesh.blendShapeCount + " delta=" + displacement +
                " human=" + (animator != null && animator.avatar.isHuman) + " clips=" + clips + " toon=True");
            EditorApplication.Exit(0);
        } catch (Exception exc) {
            Debug.LogException(exc);
            EditorApplication.Exit(1);
        }
    }

    public static void TestProfiles()
    {
        try
        {
            var folder = Environment.GetEnvironmentVariable("AETHRYX_UNITY_PROFILES");
            var files = System.IO.Directory.GetFiles(folder, "ascension-profile-*.glb");
            if (files.Length != 4) throw new InvalidOperationException("Expected four profile fixtures.");
            foreach (var path in files.OrderBy(file => file))
            {
                var root = AethryxImporter.Import(path);
                var body = root.GetComponentsInChildren<SkinnedMeshRenderer>().First(r => r.name == "cuerpo-base");
                var animator = root.GetComponent<Animator>();
                if (body.sharedMesh.blendShapeCount != 23 || body.bones.Length != 53 ||
                    animator == null || !animator.avatar.isValid || !animator.avatar.isHuman)
                    throw new InvalidOperationException("The profile lost the rig or facial expressions.");
                AethryxVrchat.Configure(root);
                Debug.Log("AETHRYX_PROFILE " + System.IO.Path.GetFileName(path) + " " +
                          AethryxVrchat.ReportPerformance(root));
            }
            EditorApplication.Exit(0);
        }
        catch (Exception exc)
        {
            Debug.LogException(exc);
            EditorApplication.Exit(1);
        }
    }

    public static void TestHairConsolidation()
    {
        try
        {
            var source = AethryxImporter.Import(Environment.GetEnvironmentVariable("AETHRYX_UNITY_FIXTURE"));
            var copy = UnityEngine.Object.Instantiate(source);
            var hair = copy.GetComponentsInChildren<SkinnedMeshRenderer>()
                .First(renderer => renderer.name == "cabello-largo-01");
            var phys = AethryxVrchat.PreparePhysBone(hair);
            var original = new Mesh();
            hair.BakeMesh(original);
            int before = source.GetComponentsInChildren<SkinnedMeshRenderer>().Length;
            int removed = AethryxHairCombine.Consolidate(copy);
            var merged = copy.GetComponentsInChildren<SkinnedMeshRenderer>()
                .Single(renderer => renderer.bones.Contains(phys.transform));
            var rest = new Mesh();
            merged.BakeMesh(rest);
            var restVertices = rest.vertices;
            var spatial = new System.Collections.Generic.HashSet<Vector3>(restVertices);
            if (removed < 1 || !original.vertices.All(vertex => spatial.Contains(vertex) ||
                    restVertices.Any(other => (vertex - other).sqrMagnitude < 1e-8f)))
                throw new InvalidOperationException("Consolidation shifted the source silhouette.");
            phys.transform.localRotation = Quaternion.Euler(0, 0, 18);
            var moved = new Mesh();
            merged.BakeMesh(moved);
            var posed = moved.vertices;
            float delta = restVertices.Select((vertex, index) => (vertex - posed[index]).magnitude).Max();
            if (delta < .002f || before != source.GetComponentsInChildren<SkinnedMeshRenderer>().Length)
                throw new InvalidOperationException("Consolidation lost the PhysBone pivot or changed the original.");
            int after = copy.GetComponentsInChildren<SkinnedMeshRenderer>().Length;
            Debug.Log("AETHRYX_ORIGIN_HAIR renderers=" + before + "->" + after + " delta=" + delta);
            AethryxVrchat.ReportPerformance(copy);
            EditorApplication.Exit(0);
        }
        catch (Exception exc)
        {
            Debug.LogException(exc);
            EditorApplication.Exit(1);
        }
    }

    static void TestFbx(GameObject root)
    {
        var path = AssetDatabase.GenerateUniqueAssetPath("Assets/AETHRYX/nikix-roundtrip.fbx");
        AethryxFbx.Export(root, System.IO.Path.GetFullPath(path));
        AssetDatabase.ImportAsset(path, ImportAssetOptions.ForceSynchronousImport);
        var importer = (ModelImporter)AssetImporter.GetAtPath(path);
        importer.importBlendShapes = true;
        importer.optimizeGameObjects = false;
        importer.SaveAndReimport();
        var imported = AssetDatabase.LoadAssetAtPath<GameObject>(path);
        var mesh = imported.GetComponentsInChildren<SkinnedMeshRenderer>()
            .Single(r => r.name.Replace('_', '-') == "cuerpo-base");
        var source = root.GetComponentsInChildren<SkinnedMeshRenderer>().Single(r => r.name == "cuerpo-base");
        var joints = imported.GetComponentsInChildren<Transform>().Select(t => t.name).ToHashSet();
        var body = mesh.sharedMesh;
        using (var weights = body.GetAllBoneWeights())
        using (var original = source.sharedMesh.GetAllBoneWeights()) {
            int visemes = Enumerable.Range(0, body.blendShapeCount)
                .Count(i => body.GetBlendShapeName(i).Contains("vrc.v_"));
            var originalWeighted = original.ToArray().Where(w => w.weight > .001f)
                .Select(w => source.bones[w.boneIndex].name).ToHashSet();
            var reimportedWeighted = weights.ToArray().Where(w => w.weight > .001f)
                .Select(w => mesh.bones[w.boneIndex].name).ToHashSet();
            int skeletonCount = source.bones.Count(bone => joints.Contains(bone.name));
            Debug.Log("AETHRYX_FBX skeleton=" + skeletonCount + " weighted=" + reimportedWeighted.Count
                + " weights=" + weights.Length + " morphs=" + body.blendShapeCount + " visemes=" + visemes);
            if (skeletonCount != 53 || !originalWeighted.SetEquals(reimportedWeighted) ||
                body.blendShapeCount != 23 || visemes != 15)
                throw new InvalidOperationException("FBX round-trip lost a binding or blendshape.");
        }
    }
}
