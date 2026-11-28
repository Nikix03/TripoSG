"""Fixed correspondences for the bundled, normalized MakeHuman body."""

from pathlib import Path


BUNDLED_BODY = Path(__file__).resolve().parent / "assets" / "genesis" / "humanoid-base.ply"

# MediaPipe Pose landmark index → verified index in the bundled PLY.
POSE_VERTICES = {
    0: 201,
    11: 8120, 12: 1432,
    13: 10116, 14: 3448,
    15: 9890, 16: 3222,
    23: 10967, 24: 4342,
    25: 11213, 26: 4595,
    27: 11458, 28: 4840,
}

BODY_PROFILES = {
    "Anime neutro": {"head": 1.08},
    "Anime femenino": {"head": 1.10, "hips": 1.07, "waist": .94, "shoulders": .98, "legs": 1.04},
    "Anime masculino": {"head": 1.06, "shoulders": 1.10, "neck": 1.06, "hips": .97, "legs": 1.02},
}


def resolve_pose_vertices(mesh):
    """Map canonical MediaPipe pose anchors onto either the bundled or sculpted body."""
    import trimesh
    from scipy.spatial import cKDTree

    if len(mesh.vertices) == 13380:
        return dict(POSE_VERTICES)
    reference = trimesh.load(BUNDLED_BODY, force="mesh", process=False)
    tree = cKDTree(mesh.vertices)
    return {
        joint: int(tree.query(reference.vertices[vertex])[1])
        for joint, vertex in POSE_VERTICES.items()
    }
