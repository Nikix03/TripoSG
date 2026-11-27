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
            }
            var clips = AssetDatabase.FindAssets("t:AnimationClip", new[]{"Assets/AETHRYX"}).Length;
            if (clips < 1) throw new InvalidOperationException("Missing animation clip.");
            Selection.activeGameObject = root;
            AethryxImporter.ApplyToonToSelection();
            if (renderers.Any(r => r.sharedMaterial.shader.name != "AETHRYX/Toon"))
                throw new InvalidOperationException("Missing Toon.");
            Debug.Log("AETHRYX_SMOKE renderer=" + renderers.Length + " bones=" + body.bones.Length +
                " morphs=" + body.sharedMesh.blendShapeCount + " delta=" + displacement +
                " human=" + (animator != null && animator.avatar.isHuman) + " clips=" + clips + " toon=True");
            EditorApplication.Exit(0);
        } catch (Exception exc) {
            Debug.LogException(exc);
            EditorApplication.Exit(1);
        }
    }
}
