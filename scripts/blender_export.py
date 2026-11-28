"""Blender-side worker for importing an articulated GLB and exporting a professional copy."""

import json
from pathlib import Path
import sys

import bpy
import bmesh


def inspect_geometry(objects, repair):
    evidence = []
    for obj in objects:
        mesh = obj.data
        topology = bmesh.new()
        try:
            topology.from_mesh(mesh)
            def counts():
                return {"vertices": len(topology.verts), "faces": len(topology.faces),
                        "open_edges": sum(edge.is_boundary for edge in topology.edges),
                        "overloaded_edges": sum(len(edge.link_faces) > 2 for edge in topology.edges)}
            before = counts()
            editable = not mesh.shape_keys and not obj.vertex_groups and not obj.find_armature()
            if repair and editable:
                bmesh.ops.dissolve_degenerate(topology, dist=1e-8, edges=list(topology.edges))
                bmesh.ops.recalc_face_normals(topology, faces=list(topology.faces))
                topology.to_mesh(mesh)
                mesh.update()
            evidence.append({"mesh": obj.name, "before": before, "after": counts(),
                             "repair_applied": bool(repair and editable)})
        finally:
            topology.free()
    return evidence


def main():
    manifest = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text(encoding="utf-8"))
    source, output = Path(manifest["input"]), Path(manifest["output"])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if "FINISHED" not in bpy.ops.import_scene.gltf(filepath=str(source)):
        raise RuntimeError("glTF import did not complete")
    objects = list(bpy.context.scene.objects)
    meshes = [obj for obj in objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("The source scene has no meshes")
    geometry = inspect_geometry(meshes, manifest.get("repair") is True)
    if output.suffix == ".blend":
        bpy.ops.file.pack_all()
        result = bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=False)
    elif output.suffix == ".fbx":
        result = bpy.ops.export_scene.fbx(filepath=str(output), add_leaf_bones=False,
                                          use_mesh_modifiers=False, path_mode="COPY", embed_textures=True)
    elif output.suffix == ".glb":
        result = bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB")
    else:
        raise ValueError("Unsupported output format")
    if "FINISHED" not in result:
        raise RuntimeError("Export did not complete")
    report = {
        "status": "complete", "format": output.suffix,
        "meshes": len(meshes), "bones": sum(len(obj.data.bones) for obj in objects if obj.type == "ARMATURE"),
        "shape_keys": sum(len(obj.data.shape_keys.key_blocks) - 1 for obj in meshes if obj.data.shape_keys),
        "materials": len(bpy.data.materials),
        "geometry": geometry,
    }
    Path(manifest["response"]).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
