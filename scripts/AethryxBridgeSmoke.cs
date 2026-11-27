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
            if (renderers.Length != 2 || body.bones.Length != 53 || body.sharedMesh.blendShapeCount != 23)
                throw new InvalidOperationException("Counts unexpected.");
            var arm = body.bones.First(t => t.name == "LeftUpperArm");
            var before = new Mesh(); body.BakeMesh(before);
            arm.localRotation = Quaternion.Euler(0, 0, 30);
            var after = new Mesh(); body.BakeMesh(after);
            var a = before.vertices; var b = after.vertices;
            float displacement = Enumerable.Range(0, a.Length).Max(i => Vector3.Distance(a[i], b[i]));
            if (displacement < .03f) throw new InvalidOperationException("Skin did not deform.");
            var animator = root.GetComponent<Animator>();
            var texture = renderers.First(r => r.name == "ojo-izquierdo").sharedMaterial.mainTexture;
            if (texture == null) throw new InvalidOperationException("Missing eye texture.");
            var clips = AssetDatabase.FindAssets("t:AnimationClip", new[]{"Assets/AETHRYX"}).Length;
            if (clips < 1) throw new InvalidOperationException("Missing animation clip.");
            Selection.activeGameObject = root;
            AethryxImporter.ApplyToonToSelection();
            if (body.sharedMaterial.shader.name != "AETHRYX/Toon") throw new InvalidOperationException("Missing Toon.");
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
