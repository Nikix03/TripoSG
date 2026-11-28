"""Native delivery from a saved, reviewed Genesis reconstruction."""

import json
from hashlib import sha256
from copy import deepcopy
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp
from time import perf_counter
from zipfile import ZIP_DEFLATED, ZipFile

import trimesh

from aethryx_anatomy import export_studio_reconstruction, reconstruct_data
from aethryx_bridge import inspect_avatar_glb
from aethryx_modular import BODY, ModularAvatar
from aethryx_rig import export_rigged_avatar
from studio_project import load_project, save_project


def package_for_unity(avatar, output, *, hair_chains=None):
    audit = inspect_avatar_glb(avatar)
    if not audit["valid_native_glb"]:
        raise ValueError("El paquete requiere un GLB articulado válido.")
    root = Path(__file__).resolve().parent
    package = root / "unity" / "com.nikix.aethryx"
    manifest = {
        "schema": 1, "avatar": "Assets/AETHRYX/avatar.glb",
        "dependencies": {"unity_reference": "2022.3.22f1", "vrchat_sdk_avatars_reference": "3.10.6",
                         "optional_unity_fbx_exporter_reference": "4.2.1"},
        "included": ["GLB with embedded materials", "AETHRYX importer", "installation instructions"],
        "sdk_validated": False,
        "audit": {key: audit[key] for key in ("joints", "visemes", "skinned_nodes", "triangles")},
    }
    if hair_chains:
        chains_audit = inspect_avatar_glb(hair_chains)
        if not chains_audit["valid_native_glb"]:
            raise ValueError("La variante de cadenas capilares no pasó la auditoría GLB.")
        manifest["hair_chain_variant"] = "Assets/AETHRYX/avatar-cadenas.glb"
    instructions = (
        "Abre un proyecto autorizado de Avatars en Unity mediante VRChat Creator Companion.\n"
        "Copia las carpetas Assets y Packages desde este ZIP al proyecto de prueba.\n"
        "En Tools / AETHRYX / Importar avatar Genesis, selecciona Assets/AETHRYX/avatar.glb.\n"
        "Prepara el Avatar Descriptor con Tools / AETHRYX y revisa el resultado con el SDK.\n"
        "Usa Build & Test para la primera prueba. El SDK requiere revisión antes de Build & Publish.\n"
        "La importación crea los activos y el prefab. Si necesitas FBX instala Unity FBX Exporter.\n"
        "Con la variante de cadenas, selecciona la junta inicial de cada mechón y usa Preparar PhysBone.\n"
    )
    with ZipFile(output, "x", compression=ZIP_DEFLATED) as archive:
        archive.write(avatar, manifest["avatar"])
        if hair_chains:
            archive.write(hair_chains, manifest["hair_chain_variant"])
        for path in sorted(package.rglob("*")):
            if path.is_file():
                archive.write(path, (Path("Packages") / package.name / path.relative_to(package)).as_posix())
        archive.write(root / "LICENSE", "LICENSE")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2))
        archive.writestr("IMPORTAR.txt", instructions)


def _portable_references(source, anatomy, output):
    """Localize known project resources. The Unity ZIP contains only the avatar and importer."""
    anatomy = deepcopy(anatomy)
    references = deepcopy(source["references"])
    directory = output / "referencias"
    copies = {}

    def copy_resource(value):
        if not value:
            return value
        original = Path(value).resolve(strict=True)
        if not original.is_file():
            raise ValueError("La referencia guardada debe ser un archivo.")
        if original not in copies:
            directory.mkdir(exist_ok=True)
            target = directory / f"recurso-{len(copies) + 1:03d}{original.suffix}"
            copyfile(original, target)
            copies[original] = str(target)
        return copies[original]

    for state in anatomy["views"].values():
        for key in ("image", "mask", "body_mask"):
            if key in state:
                state[key] = copy_resource(state[key])
    if not anatomy.get("bundled_template") and anatomy.get("template"):
        anatomy["template"] = copy_resource(anatomy["template"])
    for region in references:
        region["path"] = copy_resource(region["path"])
    image = copy_resource(source.get("image")) or next(
        (view["image"] for view in anatomy["views"].values() if view.get("image")), None,
    )
    return image, references, anatomy


class DeliveryCancelled(Exception):
    """Cancellation requested between export stages."""


DELIVERY_STAGES = (
    "Comprobar proyecto", "Preparar geometría", "Articular avatar",
    "Optimizar variantes", "Empaquetar para Unity", "Guardar proyecto editable",
)


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def delivery_inputs(project, source):
    anatomy = source.get("anatomy_setup") or {}
    paths = [project, source.get("model"), source.get("image")]
    if not anatomy.get("bundled_template"):
        paths.append(anatomy.get("template"))
    paths.extend(region["path"] for region in source["references"])
    paths.extend(view.get(key) for view in anatomy.get("views", {}).values()
                 for key in ("image", "mask", "body_mask"))
    return {str(path): _hash_file(path)
            for path in dict.fromkeys(Path(value).resolve() for value in paths if value)}


def verify_delivery(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "integridad.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    required = {"Avatar.aeth", "avatar-modular.glb", "avatar-articulado.glb", "avatar-pc.glb",
                "avatar-cadenas-capilares.glb", "Unity-AETHRYX.zip", "procedencia.json"}
    actual = {path.relative_to(folder).as_posix(): _hash_file(path)
              for path in folder.rglob("*") if path.is_file() and path != folder / "integridad.json"}
    if not required.issubset(expected) or actual != expected:
        raise ValueError("La entrega tiene archivos faltantes o modificados.")
    return manifest


def build_reviewed_delivery(project, folder, *, on_stage=None, cancelled=None, expected_inputs=None, on_inputs=None, on_output=None):
    started = perf_counter()

    def check_cancelled():
        if cancelled is not None and cancelled.is_set():
            raise DeliveryCancelled()

    def stage(index):
        check_cancelled()
        if on_stage is not None:
            on_stage(index, DELIVERY_STAGES[index])

    stage(0)
    source = load_project(project)
    inputs = delivery_inputs(project, source)
    if expected_inputs is not None and inputs != expected_inputs:
        raise ValueError("Las entradas cambiaron desde el primer intento. Registra una entrega nueva si deseas usar los cambios.")
    if on_inputs is not None:
        on_inputs(inputs)
    data = source.get("anatomy_setup")
    if not data or not data.get("views"):
        raise ValueError("El proyecto necesita un ajuste de Genesis guardado y revisado.")
    if any(view.get("suggested") for view in data["views"].values()):
        # A saved setup can contain reviewed suggestions. The caller confirms reuse explicitly.
        data = {**data, "reviewed_suggestions_reused": True}
    data = {**data, "export_hybrid": True, "export_rig": True, "repair_hair_in_poses": True}
    stage(1)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if source.get("model"):
        if Path(source["model"]).suffix.lower() != ".glb":
            raise ValueError("El modelo editable guardado debe estar en GLB para preparar la entrega.")
        output = Path(mkdtemp(prefix="evolution-", dir=folder))
        if on_output is not None:
            on_output(output)
        modular = output / "avatar-modular.glb"
        copyfile(source["model"], modular)
    else:
        report = {}
        body, patches, errors = reconstruct_data(data, deformation_report=report)
        base, _ = export_studio_reconstruction(folder, data, body, patches, report, errors)
        output = base.parent
        if on_output is not None:
            on_output(output)
        modular = output / "avatar-modular.glb"
    scene = trimesh.load(modular, force="scene", process=False)
    articulated = output / "avatar-articulado.glb"
    hidden = source.get("hidden_components", []) if source.get("model") else []
    if hidden:
        avatar = ModularAvatar(scene, hidden)
        visible = trimesh.Scene({name: avatar.get(name, world=True) for name in avatar.nodes
                                 if name not in avatar.hidden or name == BODY})
        visible.metadata.update(scene.metadata)
        scene = visible
    stage(2)
    if not articulated.is_file():
        export_rigged_avatar(scene, articulated)
    stage(3)
    pc = output / "avatar-pc.glb"
    export_rigged_avatar(scene, pc, profile="VRChat PC Preserve Shape")
    chain_variant = output / "avatar-cadenas-capilares.glb"
    export_rigged_avatar(scene, chain_variant, profile="VRChat PC Hair Chains")
    stage(4)
    package_for_unity(pc, output / "Unity-AETHRYX.zip", hair_chains=chain_variant)
    stage(5)
    image, references, data = _portable_references(source, data, output)
    save_project(output / "Avatar.aeth", image=image, model=modular, output=articulated,
                 **source["settings"], references=references, anatomy_setup=data,
                 hidden_components=hidden)
    (output / "procedencia.json").write_text(json.dumps({
        "geometry": "saved_modular_project" if source.get("model") else "reviewed_genesis_reconstruction",
        "anatomy_observations": list(data["views"]),
        "rigging": "native_genesis",
        "variants": ["editable", "articulated", "pc", "hair_chains"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    check_cancelled()
    if delivery_inputs(project, source) != inputs:
        raise ValueError("Las entradas cambiaron durante la preparación.")
    audits = {path.name: inspect_avatar_glb(path) for path in (articulated, pc, chain_variant)}
    if any(not audit["valid_native_glb"] for audit in audits.values()):
        raise ValueError("Una variante articulada no superó la auditoría GLB.")
    with ZipFile(output / "Unity-AETHRYX.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("Falló la verificación del ZIP de Unity.")
    files = {path.relative_to(output).as_posix(): _hash_file(path)
             for path in output.rglob("*") if path.is_file()}
    (output / "integridad.json").write_text(json.dumps({
        "inputs": inputs, "files": files, "audits": audits, "seconds": perf_counter() - started,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    verify_delivery(output)
    check_cancelled()
    return output
