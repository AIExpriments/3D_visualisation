############################################## chatgpt ############################################
import json
import math
from pathlib import Path

import cv2
import numpy as np
from skimage.measure import marching_cubes


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

NUM_SLICES = 50

# Distance between radiography slices
SLICE_THICKNESS_MM = 3.0

# Detector pixel size
PIXEL_SIZE_MM = 0.20

OUTPUT_HTML = "radome_3d_reconstruction.html"


# ============================================================
# 1. SYNTHETIC DETECTION DATA
# ============================================================

def generate_synthetic_data(
    num_slices,
    width,
    height
):
    """
    Generate realistic-looking synthetic detector output.

    Returns:

        circle_params[z]

        crack_polygons[z]

    Every crack is generated INSIDE the annular
    region between inner_radius and outer_radius.
    """

    rng = np.random.default_rng(10)

    circle_params = []
    crack_polygons = []

    base_cx = width / 2
    base_cy = height / 2

    base_outer_radius = 190
    shell_thickness = 35

    for z in range(num_slices):

        # ----------------------------------------------------
        # Radome circle
        # ----------------------------------------------------

        cx = (
            base_cx
            + 2.0 * np.sin(z * 0.12)
        )

        cy = (
            base_cy
            + 2.0 * np.cos(z * 0.10)
        )

        outer_radius = (
            base_outer_radius
            + 3.0 * np.sin(z * 0.08)
        )

        inner_radius = (
            outer_radius
            - shell_thickness
        )

        circle = {
            "center_x": cx,
            "center_y": cy,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius
        }

        circle_params.append(circle)

        # ----------------------------------------------------
        # Cracks
        # ----------------------------------------------------

        cracks_this_slice = []

        # ====================================================
        # CRACK 1
        #
        # A long crack existing across many slices
        # ====================================================

        if 5 <= z <= 38:

            # IMPORTANT:
            # Select radial position BETWEEN the circles.

            crack_radius = (
                inner_radius
                + 0.55 *
                (outer_radius - inner_radius)
            )

            # Crack moves slightly with Z
            theta = (
                np.deg2rad(30)
                + 0.02 * z
            )

            crack_cx = (
                cx
                + crack_radius * np.cos(theta)
            )

            crack_cy = (
                cy
                + crack_radius * np.sin(theta)
            )

            # Crack orientation
            crack_angle = np.deg2rad(65)

            length = 30
            crack_width = 5

            dx = np.cos(crack_angle)
            dy = np.sin(crack_angle)

            px = -dy
            py = dx

            polygon = [

                (
                    crack_cx
                    - dx * length / 2
                    + px * crack_width,

                    crack_cy
                    - dy * length / 2
                    + py * crack_width
                ),

                (
                    crack_cx
                    + dx * length / 2
                    + px * crack_width,

                    crack_cy
                    + dy * length / 2
                    + py * crack_width
                ),

                (
                    crack_cx
                    + dx * length / 2
                    - px * crack_width,

                    crack_cy
                    + dy * length / 2
                    - py * crack_width
                ),

                (
                    crack_cx
                    - dx * length / 2
                    - px * crack_width,

                    crack_cy
                    - dy * length / 2
                    - py * crack_width
                )
            ]

            cracks_this_slice.append(polygon)

        # ====================================================
        # CRACK 2
        #
        # Smaller defect
        # ====================================================

        if 15 <= z <= 45 and z % 2 == 0:

            crack_radius = (
                inner_radius
                + 0.70 *
                (outer_radius - inner_radius)
            )

            theta = np.deg2rad(140)

            crack_cx = (
                cx
                + crack_radius * np.cos(theta)
                + rng.normal(0, 1)
            )

            crack_cy = (
                cy
                + crack_radius * np.sin(theta)
                + rng.normal(0, 1)
            )

            length = 18
            crack_width = 4

            angle = np.deg2rad(-20)

            dx = np.cos(angle)
            dy = np.sin(angle)

            px = -dy
            py = dx

            polygon = [

                (
                    crack_cx - dx * length / 2 + px * crack_width,
                    crack_cy - dy * length / 2 + py * crack_width
                ),

                (
                    crack_cx + dx * length / 2 + px * crack_width,
                    crack_cy + dy * length / 2 + py * crack_width
                ),

                (
                    crack_cx + dx * length / 2 - px * crack_width,
                    crack_cy + dy * length / 2 - py * crack_width
                ),

                (
                    crack_cx - dx * length / 2 - px * crack_width,
                    crack_cy - dy * length / 2 - py * crack_width
                )
            ]

            cracks_this_slice.append(polygon)

        crack_polygons.append(
            cracks_this_slice
        )

    return circle_params, crack_polygons


# ============================================================
# 2. CREATE RING MASK
# ============================================================

def create_ring_mask(
    circle,
    width,
    height
):
    """
    Creates:

        1 = radome material
        0 = everything else

    Ring = outside inner circle AND
           inside outer circle.
    """

    cx = circle["center_x"]
    cy = circle["center_y"]

    inner_radius = circle["inner_radius"]
    outer_radius = circle["outer_radius"]

    y, x = np.ogrid[
        0:height,
        0:width
    ]

    distance_squared = (
        (x - cx) ** 2
        +
        (y - cy) ** 2
    )

    ring = (
        (distance_squared >= inner_radius ** 2)
        &
        (distance_squared <= outer_radius ** 2)
    )

    return ring.astype(np.uint8)


# ============================================================
# 3. CREATE CRACK MASK
# ============================================================

def create_crack_mask(
    polygons,
    width,
    height
):
    """
    Fill crack polygons.

    Does NOT apply ring constraint here.
    """

    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    for polygon in polygons:

        if len(polygon) < 3:
            continue

        points = np.asarray(
            polygon,
            dtype=np.int32
        )

        cv2.fillPoly(
            mask,
            [points],
            1
        )

    return mask


# ============================================================
# 4. BUILD 3D VOLUMES
# ============================================================

def build_volumes(
    circle_params,
    crack_polygons,
    width,
    height
):
    """
    Build:

        ring_volume
        crack_volume

    Shape:

        (Z, Y, X)
    """

    if len(circle_params) != len(crack_polygons):
        raise ValueError(
            "circle_params and crack_polygons "
            "must have the same number of slices."
        )

    num_slices = len(circle_params)

    ring_volume = np.zeros(
        (
            num_slices,
            height,
            width
        ),
        dtype=np.uint8
    )

    crack_volume = np.zeros(
        (
            num_slices,
            height,
            width
        ),
        dtype=np.uint8
    )

    for z in range(num_slices):

        # ----------------------------------------------------
        # Ring
        # ----------------------------------------------------

        ring_mask = create_ring_mask(
            circle_params[z],
            width,
            height
        )

        # ----------------------------------------------------
        # Crack polygons
        # ----------------------------------------------------

        detected_crack_mask = create_crack_mask(
            crack_polygons[z],
            width,
            height
        )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Keep only crack pixels that are actually
        # inside the radome material.
        # ----------------------------------------------------

        valid_crack_mask = (
            detected_crack_mask
            &
            ring_mask
        ).astype(np.uint8)

        ring_volume[z] = ring_mask

        crack_volume[z] = valid_crack_mask

    return ring_volume, crack_volume


# ============================================================
# 5. MARCHING CUBES
# ============================================================

def volume_to_mesh(
    volume
):
    """
    Convert a binary volume into a mesh.

    Physical spacing:

        Z = 3 mm
        Y = pixel size
        X = pixel size
    """

    if not np.any(volume):

        print(
            "WARNING: Volume is empty!"
        )

        return None, None

    vertices, faces, normals, values = (
        marching_cubes(
            volume.astype(np.float32),
            level=0.5,
            spacing=(
                SLICE_THICKNESS_MM,
                PIXEL_SIZE_MM,
                PIXEL_SIZE_MM
            )
        )
    )

    return vertices, faces


# ============================================================
# 6. CENTER MESHES TOGETHER
# ============================================================

def center_meshes(
    radome_vertices,
    crack_vertices
):
    """
    Use the SAME center for both meshes.

    This is important.

    Do NOT independently center the crack mesh.
    """

    if radome_vertices is None:
        return radome_vertices, crack_vertices

    # Bounding box of radome
    min_xyz = radome_vertices.min(axis=0)
    max_xyz = radome_vertices.max(axis=0)

    center = (
        min_xyz + max_xyz
    ) / 2.0

    radome_vertices = (
        radome_vertices - center
    )

    if crack_vertices is not None:

        crack_vertices = (
            crack_vertices - center
        )

    return (
        radome_vertices,
        crack_vertices
    )


# ============================================================
# 7. MESH → JSON
# ============================================================

def mesh_to_json(
    vertices,
    faces
):

    if vertices is None:

        return {
            "vertices": [],
            "faces": []
        }

    return {
        "vertices": vertices.tolist(),
        "faces": faces.tolist()
    }


# ============================================================
# 8. THREE.JS HTML
# ============================================================

def generate_html(
    radome_mesh,
    crack_mesh,
    output_file
):

    radome_json = json.dumps(
        radome_mesh,
        separators=(",", ":")
    )

    crack_json = json.dumps(
        crack_mesh,
        separators=(",", ":")
    )

    html = f"""<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<title>Radome 3D Reconstruction</title>

<style>

html,
body {{
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    background: #111;
}}

#info {{
    position: absolute;
    top: 15px;
    left: 15px;

    color: white;

    font-family: Arial, sans-serif;

    background: rgba(0,0,0,0.65);

    padding: 12px 16px;

    border-radius: 8px;

    z-index: 10;

    line-height: 1.5;
}}

</style>

<script type="importmap">

{{
    "imports": {{

        "three":
        "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",

        "three/addons/":
        "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
    }}
}}

</script>

</head>


<body>


<div id="info">

<b>Radome 3D Reconstruction</b>

<br><br>

<span style="color:#cccccc">
Gray = Radome shell
</span>

<br>

<span style="color:#ff3333">
Red = Crack defects
</span>

<br><br>

Drag → Rotate

<br>

Scroll → Zoom

</div>


<script type="module">

import * as THREE from 'three';

import {{
    OrbitControls
}} from 'three/addons/controls/OrbitControls.js';


// ==========================================================
// EMBEDDED DATA
// ==========================================================

const radomeData =
{radome_json};

const crackData =
{crack_json};


// ==========================================================
// SCENE
// ==========================================================

const scene =
    new THREE.Scene();

scene.background =
    new THREE.Color(0x111111);


// ==========================================================
// CAMERA
// ==========================================================

const camera =
    new THREE.PerspectiveCamera(
        45,
        window.innerWidth /
        window.innerHeight,
        0.1,
        100000
    );


// ==========================================================
// RENDERER
// ==========================================================

const renderer =
    new THREE.WebGLRenderer({{
        antialias: true
    }});

renderer.setPixelRatio(
    window.devicePixelRatio
);

renderer.setSize(
    window.innerWidth,
    window.innerHeight
);

document.body.appendChild(
    renderer.domElement
);


// ==========================================================
// LIGHT
// ==========================================================

const ambient =
    new THREE.AmbientLight(
        0xffffff,
        1.5
    );

scene.add(ambient);


const light =
    new THREE.DirectionalLight(
        0xffffff,
        2.0
    );

light.position.set(
    500,
    500,
    500
);

scene.add(light);


// ==========================================================
// CREATE MESH
// ==========================================================

function createMesh(
    data,
    material
) {{

    const geometry =
        new THREE.BufferGeometry();


    // ------------------------------------------------------
    // Vertices
    // ------------------------------------------------------

    const positions =
        new Float32Array(
            data.vertices.length * 3
        );


    for (
        let i = 0;
        i < data.vertices.length;
        i++
    ) {{

        /*
            Python coordinates:

                [Z, Y, X]

            Three.js:

                X
                Y
                Z
        */

        positions[
            i * 3 + 0
        ] =
            data.vertices[i][2];


        positions[
            i * 3 + 1
        ] =
            -data.vertices[i][1];


        positions[
            i * 3 + 2
        ] =
            data.vertices[i][0];

    }}


    geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(
            positions,
            3
        )
    );


    // ------------------------------------------------------
    // Faces
    // ------------------------------------------------------

    const indices = [];


    for (
        let i = 0;
        i < data.faces.length;
        i++
    ) {{

        indices.push(
            data.faces[i][0],
            data.faces[i][1],
            data.faces[i][2]
        );

    }}


    geometry.setIndex(
        indices
    );


    geometry.computeVertexNormals();


    return new THREE.Mesh(
        geometry,
        material
    );
}}


// ==========================================================
// RADOME MATERIAL
// ==========================================================

const radomeMaterial =
    new THREE.MeshStandardMaterial({{

        color: 0xaaaaaa,

        transparent: true,

        opacity: 0.35,

        side: THREE.DoubleSide,

        depthWrite: false
    }});


const radome =
    createMesh(
        radomeData,
        radomeMaterial
    );

scene.add(radome);


// ==========================================================
// CRACK MATERIAL
// ==========================================================

const crackMaterial =
    new THREE.MeshStandardMaterial({{

        color: 0xff0000,

        emissive: 0x550000,

        side: THREE.DoubleSide,

        transparent: false
    }});


const cracks =
    createMesh(
        crackData,
        crackMaterial
    );

scene.add(cracks);


// ==========================================================
// CONTROLS
// ==========================================================

const controls =
    new OrbitControls(
        camera,
        renderer.domElement
    );

controls.enableDamping =
    true;

controls.dampingFactor =
    0.05;


// ==========================================================
// CAMERA FIT
// ==========================================================

const box =
    new THREE.Box3()
        .setFromObject(radome);


const sphere =
    box.getBoundingSphere(
        new THREE.Sphere()
    );


controls.target.set(
    0,
    0,
    0
);


camera.position.set(
    sphere.radius * 1.8,
    sphere.radius * 1.3,
    sphere.radius * 2.0
);


camera.near =
    Math.max(
        0.01,
        sphere.radius / 1000
    );


camera.far =
    sphere.radius * 20;


camera.lookAt(
    0,
    0,
    0
);


camera.updateProjectionMatrix();


// ==========================================================
// RESIZE
// ==========================================================

window.addEventListener(
    "resize",
    () => {{

        camera.aspect =
            window.innerWidth /
            window.innerHeight;

        camera.updateProjectionMatrix();

        renderer.setSize(
            window.innerWidth,
            window.innerHeight
        );

    }}
);


// ==========================================================
// RENDER
// ==========================================================

function animate() {{

    requestAnimationFrame(
        animate
    );

    controls.update();

    renderer.render(
        scene,
        camera
    );

}}

animate();

</script>

</body>

</html>
"""

    Path(
        output_file
    ).write_text(
        html,
        encoding="utf-8"
    )


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("RADOME 3D RECONSTRUCTION")
    print("=" * 60)


    # --------------------------------------------------------
    # Generate fake detector output
    # --------------------------------------------------------

    print(
        "\n[1] Generating synthetic detections..."
    )

    circle_params, crack_polygons = \
        generate_synthetic_data(
            NUM_SLICES,
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )


    # Print some information
    total_cracks = sum(
        len(x)
        for x in crack_polygons
    )

    print(
        f"    Slices       : {NUM_SLICES}"
    )

    print(
        f"    Crack regions: {total_cracks}"
    )


    # --------------------------------------------------------
    # Build 3D volumes
    # --------------------------------------------------------

    print(
        "\n[2] Building 3D masks..."
    )

    ring_volume, crack_volume = \
        build_volumes(
            circle_params,
            crack_polygons,
            IMAGE_WIDTH,
            IMAGE_HEIGHT
        )


    print(
        f"    Ring volume  : {ring_volume.shape}"
    )

    print(
        f"    Crack volume : {crack_volume.shape}"
    )


    # --------------------------------------------------------
    # Check crack voxels
    # --------------------------------------------------------

    crack_voxels = np.count_nonzero(
        crack_volume
    )

    ring_voxels = np.count_nonzero(
        ring_volume
    )

    print(
        f"    Ring voxels  : {ring_voxels}"
    )

    print(
        f"    Crack voxels : {crack_voxels}"
    )


    if crack_voxels == 0:

        raise RuntimeError(
            "CRACK VOLUME IS EMPTY. "
            "Check crack polygon generation."
        )


    # --------------------------------------------------------
    # Marching cubes
    # --------------------------------------------------------

    print(
        "\n[3] Running marching cubes..."
    )


    radome_vertices, radome_faces = \
        volume_to_mesh(
            ring_volume
        )


    crack_vertices, crack_faces = \
        volume_to_mesh(
            crack_volume
        )


    print(
        f"    Radome vertices: "
        f"{len(radome_vertices)}"
    )

    print(
        f"    Radome triangles: "
        f"{len(radome_faces)}"
    )

    print(
        f"    Crack vertices: "
        f"{len(crack_vertices)}"
    )

    print(
        f"    Crack triangles: "
        f"{len(crack_faces)}"
    )


    # --------------------------------------------------------
    # SAME coordinate system
    # --------------------------------------------------------

    print(
        "\n[4] Aligning meshes..."
    )

    radome_vertices, crack_vertices = \
        center_meshes(
            radome_vertices,
            crack_vertices
        )


    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    print(
        "\n[5] Preparing mesh data..."
    )

    radome_mesh = mesh_to_json(
        radome_vertices,
        radome_faces
    )

    crack_mesh = mesh_to_json(
        crack_vertices,
        crack_faces
    )


    # --------------------------------------------------------
    # HTML
    # --------------------------------------------------------

    print(
        "\n[6] Creating interactive HTML..."
    )

    generate_html(
        radome_mesh,
        crack_mesh,
        OUTPUT_HTML
    )


    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"\nOpen this file in your browser:"
    )

    print(
        Path(OUTPUT_HTML).resolve()
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()

##################################################### Claudai ###########################################################
"""
Radome 3D Reconstruction Pipeline
==================================

Reconstructs a 3D model of a ring-shaped radome from a stack of 2D
radiography slices, using per-slice detection outputs:

    - Radome boundary: circle params (center_x, center_y, inner_radius, outer_radius)
    - Crack defects: polygon contours (list of (x, y) points), zero or more per slice

Pipeline:
    1. Build ring_mask (3D) from circle params.
    2. Build crack_mask (3D) from filled polygons, ANDed with ring_mask so
       cracks can never appear outside the material band.
    3. Run marching cubes on each volume -> two meshes.
    4. Center both meshes on the same origin.
    5. Export a self-contained interactive Three.js HTML file.

To use with real data: replace the call to `generate_synthetic_data()` in
`main()` with a function that returns the same structure:
    circle_params:   List[Dict] of length Z, each {"center_x", "center_y",
                      "inner_radius", "outer_radius"}  (pixel units)
    crack_polygons:  List[List[np.ndarray]] of length Z, each element is a
                      list of polygons for that slice, each polygon an
                      (N, 2) array of (x, y) pixel coordinates.
Everything downstream (build_masks, meshing, export) is unchanged.
"""

import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import cv2
from skimage.measure import marching_cubes


# --------------------------------------------------------------------------
# 1. Synthetic data generator (swap out for real detections later)
# --------------------------------------------------------------------------

def generate_synthetic_data(
    num_slices: int = 60,
    image_shape: Tuple[int, int] = (256, 256),
    inner_radius: float = 60.0,
    outer_radius: float = 90.0,
    num_cracks: int = 3,
    seed: int = 42,
) -> Tuple[List[Dict], List[List[np.ndarray]]]:
    """
    Generates fake per-slice circle_params and crack_polygons.

    Circle center stays fixed (a real radome is roughly coaxial across
    slices); radii wobble slightly slice-to-slice to mimic real detector
    noise. Cracks are generated as thin radial wedge-shaped polygons whose
    points are always kept strictly within [inner_radius, outer_radius] so
    they land inside the material band.

    Returns
    -------
    circle_params : list of dict, length num_slices
        Each: {"center_x", "center_y", "inner_radius", "outer_radius"}
    crack_polygons : list of list of (N,2) arrays, length num_slices
        crack_polygons[z] is a list of polygons (possibly empty) for slice z.
    """
    rng = random.Random(seed)
    cy, cx = image_shape[0] / 2.0, image_shape[1] / 2.0

    circle_params = []
    for z in range(num_slices):
        wobble = rng.uniform(-1.5, 1.5)
        circle_params.append({
            "center_x": cx,
            "center_y": cy,
            "inner_radius": inner_radius + wobble,
            "outer_radius": outer_radius + wobble,
        })

    # Pick a handful of crack "tracks": each spans a contiguous range of
    # slices and a fixed angular position, so it looks like a real crack
    # running partway through the ring thickness and along several slices.
    crack_polygons: List[List[np.ndarray]] = [[] for _ in range(num_slices)]

    for _ in range(num_cracks):
        angle = rng.uniform(0, 2 * math.pi)
        z_start = rng.randint(0, num_slices - 10)
        z_len = rng.randint(5, 15)
        z_end = min(num_slices - 1, z_start + z_len)

        # Radial extent of the crack within the material band (a crack
        # rarely spans the *entire* wall thickness, so keep a small margin).
        margin = 0.15 * (outer_radius - inner_radius)
        r0 = inner_radius + margin + rng.uniform(0, 5)
        r1 = outer_radius - margin - rng.uniform(0, 5)
        if r1 <= r0:
            r0, r1 = inner_radius + 2, outer_radius - 2

        half_width_angle = math.radians(rng.uniform(1.0, 2.5))

        for z in range(z_start, z_end + 1):
            params = circle_params[z]
            cx_z, cy_z = params["center_x"], params["center_y"]
            # Thin quadrilateral wedge: two points near r0, two near r1,
            # slightly perturbed so it isn't a perfect rectangle.
            jitter = rng.uniform(-0.3, 0.3)
            a0 = angle - half_width_angle + jitter
            a1 = angle + half_width_angle + jitter
            pts = np.array([
                [cx_z + r0 * math.cos(a0), cy_z + r0 * math.sin(a0)],
                [cx_z + r0 * math.cos(a1), cy_z + r0 * math.sin(a1)],
                [cx_z + r1 * math.cos(a1), cy_z + r1 * math.sin(a1)],
                [cx_z + r1 * math.cos(a0), cy_z + r1 * math.sin(a0)],
            ], dtype=np.float64)
            crack_polygons[z].append(pts)

    return circle_params, crack_polygons


# --------------------------------------------------------------------------
# 2. Mask construction
# --------------------------------------------------------------------------

def build_ring_mask_2d(params: Dict, shape: Tuple[int, int]) -> np.ndarray:
    """True where inner_radius <= distance_from_center <= outer_radius."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - params["center_x"]) ** 2 + (yy - params["center_y"]) ** 2)
    return (dist >= params["inner_radius"]) & (dist <= params["outer_radius"])


def build_crack_mask_2d(polygons: List[np.ndarray], shape: Tuple[int, int]) -> np.ndarray:
    """Fills each polygon into a boolean mask via cv2.fillPoly."""
    mask = np.zeros(shape, dtype=np.uint8)
    if polygons:
        int_polys = [np.round(p).astype(np.int32) for p in polygons]
        cv2.fillPoly(mask, int_polys, color=1)
    return mask.astype(bool)


def build_masks(
    circle_params: List[Dict],
    crack_polygons: List[List[np.ndarray]],
    image_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the two 3D volumes (Z, Y, X).

    crack_mask is always ANDed with ring_mask so a crack polygon that
    overshoots the ring boundary can never register outside the material.
    """
    num_slices = len(circle_params)
    assert len(crack_polygons) == num_slices, "circle_params and crack_polygons must have same length"

    ring_mask = np.zeros((num_slices, *image_shape), dtype=bool)
    crack_mask = np.zeros((num_slices, *image_shape), dtype=bool)

    for z in range(num_slices):
        ring_slice = build_ring_mask_2d(circle_params[z], image_shape)
        crack_slice_raw = build_crack_mask_2d(crack_polygons[z], image_shape)
        ring_mask[z] = ring_slice
        crack_mask[z] = crack_slice_raw & ring_slice  # constrain to material band

    return ring_mask, crack_mask


# --------------------------------------------------------------------------
# 3. Meshing
# --------------------------------------------------------------------------

@dataclass
class Mesh:
    vertices: np.ndarray  # (N, 3) float, in mm, XYZ order
    faces: np.ndarray     # (M, 3) int


def volume_to_mesh(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    level: float = 0.5,
) -> Mesh:
    """
    Runs marching cubes on a boolean/float volume.

    spacing is (z_mm, y_mm, x_mm) matching the volume's (Z, Y, X) axes,
    i.e. (slice_thickness_mm, pixel_size_mm, pixel_size_mm).

    Returns a Mesh with vertices reordered to (X, Y, Z) mm for direct use
    in Three.js (right-handed, X/Y/Z convention).
    """
    if not volume.any():
        return Mesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))

    vol_float = volume.astype(np.float32)
    verts_zyx, faces, _normals, _values = marching_cubes(
        vol_float, level=level, spacing=spacing
    )
    # marching_cubes returns verts as (z, y, x) in real units; reorder to (x, y, z)
    verts_xyz = verts_zyx[:, [2, 1, 0]]
    return Mesh(vertices=verts_xyz, faces=faces)


def center_meshes(meshes: List[Mesh]) -> List[Mesh]:
    """Centers all meshes on a single shared origin (combined bounding-box center)."""
    all_verts = [m.vertices for m in meshes if m.vertices.shape[0] > 0]
    if not all_verts:
        return meshes
    combined = np.vstack(all_verts)
    center = (combined.min(axis=0) + combined.max(axis=0)) / 2.0
    centered = []
    for m in meshes:
        if m.vertices.shape[0] > 0:
            centered.append(Mesh(vertices=m.vertices - center, faces=m.faces))
        else:
            centered.append(m)
    return centered


# --------------------------------------------------------------------------
# 4. Three.js HTML export
# --------------------------------------------------------------------------

def mesh_to_json_dict(mesh: Mesh) -> Dict:
    return {
        "vertices": mesh.vertices.astype(np.float32).flatten().tolist(),
        "faces": mesh.faces.astype(np.int32).flatten().tolist(),
    }


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Radome 3D Reconstruction</title>
<style>
  html, body { margin: 0; height: 100%; background: #111417; overflow: hidden; font-family: system-ui, sans-serif; }
  #info {
    position: absolute; top: 12px; left: 12px; color: #ddd; font-size: 13px;
    background: rgba(0,0,0,0.4); padding: 8px 12px; border-radius: 6px; line-height: 1.5;
  }
  #legend span { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }
</style>
</head>
<body>
<div id="info">
  <div id="legend"><span style="background:#c9c9c9;"></span>Radome shell (translucent)</div>
  <div id="legend"><span style="background:#ff3b30;"></span>Crack defects (solid red)</div>
  <div>Drag to rotate &middot; Scroll to zoom</div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const RING_MESH = __RING_MESH_JSON__;
const CRACK_MESH = __CRACK_MESH_JSON__;

function buildGeometry(meshData) {
  const geometry = new THREE.BufferGeometry();
  const vertices = new Float32Array(meshData.vertices);
  geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.setIndex(meshData.faces);
  geometry.computeVertexNormals();
  return geometry;
}

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111417);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 10000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(1, 1, 1);
scene.add(dirLight);
const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
dirLight2.position.set(-1, -0.5, -1);
scene.add(dirLight2);

const ringGeom = buildGeometry(RING_MESH);
const ringMaterial = new THREE.MeshPhongMaterial({
  color: 0xc9c9c9,
  transparent: true,
  opacity: 0.35,
  side: THREE.DoubleSide,
  depthWrite: false,
});
const ringMesh = new THREE.Mesh(ringGeom, ringMaterial);
scene.add(ringMesh);

let crackMesh = null;
if (CRACK_MESH.vertices.length > 0) {
  const crackGeom = buildGeometry(CRACK_MESH);
  const crackMaterial = new THREE.MeshPhongMaterial({
    color: 0xff3b30,
    side: THREE.DoubleSide,
  });
  crackMesh = new THREE.Mesh(crackGeom, crackMaterial);
  scene.add(crackMesh);
}

// Fit camera to the ring's bounding sphere
ringGeom.computeBoundingSphere();
const bs = ringGeom.boundingSphere;
const dist = (bs ? bs.radius : 100) * 3.0;
camera.position.set(dist * 0.6, dist * 0.4, dist * 0.8);
camera.lookAt(0, 0, 0);

// Simple orbit controls (drag to rotate, scroll to zoom) without extra deps
let isDragging = false;
let prevX = 0, prevY = 0;
let theta = Math.atan2(camera.position.x, camera.position.z);
let phi = Math.acos(camera.position.y / camera.position.length());
let radius = camera.position.length();

function updateCamera() {
  phi = Math.max(0.05, Math.min(Math.PI - 0.05, phi));
  camera.position.x = radius * Math.sin(phi) * Math.sin(theta);
  camera.position.y = radius * Math.cos(phi);
  camera.position.z = radius * Math.sin(phi) * Math.cos(theta);
  camera.lookAt(0, 0, 0);
}
updateCamera();

renderer.domElement.addEventListener('mousedown', (e) => {
  isDragging = true; prevX = e.clientX; prevY = e.clientY;
});
window.addEventListener('mouseup', () => { isDragging = false; });
window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  const dx = e.clientX - prevX, dy = e.clientY - prevY;
  prevX = e.clientX; prevY = e.clientY;
  theta -= dx * 0.007;
  phi -= dy * 0.007;
  updateCamera();
});
renderer.domElement.addEventListener('wheel', (e) => {
  e.preventDefault();
  radius *= (1 + e.deltaY * 0.001);
  radius = Math.max((bs ? bs.radius : 10) * 0.5, Math.min((bs ? bs.radius : 10) * 20, radius));
  updateCamera();
}, { passive: false });

// Touch support
let lastTouchDist = null;
renderer.domElement.addEventListener('touchstart', (e) => {
  if (e.touches.length === 1) { isDragging = true; prevX = e.touches[0].clientX; prevY = e.touches[0].clientY; }
});
renderer.domElement.addEventListener('touchend', () => { isDragging = false; lastTouchDist = null; });
renderer.domElement.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (e.touches.length === 1 && isDragging) {
    const dx = e.touches[0].clientX - prevX, dy = e.touches[0].clientY - prevY;
    prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
    theta -= dx * 0.007; phi -= dy * 0.007;
    updateCamera();
  } else if (e.touches.length === 2) {
    const d = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
    if (lastTouchDist !== null) {
      radius *= (1 + (lastTouchDist - d) * 0.005);
      updateCamera();
    }
    lastTouchDist = d;
  }
}, { passive: false });

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>
"""


def export_html(ring_mesh: Mesh, crack_mesh: Mesh, output_path: str) -> None:
    ring_json = json.dumps(mesh_to_json_dict(ring_mesh))
    crack_json = json.dumps(mesh_to_json_dict(crack_mesh))
    html = HTML_TEMPLATE.replace("__RING_MESH_JSON__", ring_json).replace(
        "__CRACK_MESH_JSON__", crack_json
    )
    with open(output_path, "w") as f:
        f.write(html)


# --------------------------------------------------------------------------
# 5. Main pipeline
# --------------------------------------------------------------------------

def main(
    output_html_path: str = "radome_3d.html",
    pixel_size_mm: float = 0.3,
    slice_thickness_mm: float = 3.0,
    image_shape: Tuple[int, int] = (256, 256),
):
    # --- Step 0: get per-slice detections ---------------------------------
    # Swap this call for your real detection outputs. Real data must return
    # the same two structures (see docstring at top of file).
    circle_params, crack_polygons = generate_synthetic_data(
        num_slices=60, image_shape=image_shape
    )

    print(f"Loaded {len(circle_params)} slices.")

    # --- Step 1-2: build masks ---------------------------------------------
    ring_mask, crack_mask = build_masks(circle_params, crack_polygons, image_shape)
    print(f"ring_mask voxels: {ring_mask.sum()}, crack_mask voxels: {crack_mask.sum()}")
    assert not (crack_mask & ~ring_mask).any(), "crack_mask leaked outside ring_mask!"

    # --- Step 3: marching cubes ---------------------------------------------
    spacing = (slice_thickness_mm, pixel_size_mm, pixel_size_mm)  # (Z, Y, X)
    ring_mesh = volume_to_mesh(ring_mask, spacing=spacing)
    crack_mesh = volume_to_mesh(crack_mask, spacing=spacing)
    print(f"ring mesh: {ring_mesh.vertices.shape[0]} verts, {ring_mesh.faces.shape[0]} faces")
    print(f"crack mesh: {crack_mesh.vertices.shape[0]} verts, {crack_mesh.faces.shape[0]} faces")

    # --- Step 4: center on shared origin -------------------------------------
    ring_mesh, crack_mesh = center_meshes([ring_mesh, crack_mesh])

    # --- Step 5: export interactive HTML -------------------------------------
    export_html(ring_mesh, crack_mesh, output_html_path)
    print(f"Wrote {output_html_path}")


if __name__ == "__main__":
    main()

############################################ thresholding based ##########################################################
"""
Radome 3D Reconstruction Pipeline
==================================
Stacks 2D radiography slices into a 3D volume, segments the radome material
and crack defects separately, generates two meshes (semi-transparent shell +
solid red cracks), and exports an interactive HTML viewer.

CURRENT MODE: synthetic data (auto-generated ring-with-cracks test images)
TO USE REAL IMAGES: see the `load_real_slices()` function at the bottom and
the toggle in `main()`.

Requirements:
    pip install numpy scipy scikit-image pillow

Output:
    radome_viewer.html  -> open in any browser, drag to rotate, scroll to zoom
"""

import numpy as np
import json
import os
from scipy.ndimage import binary_dilation, binary_fill_holes, zoom
from skimage import measure


# ----------------------------------------------------------------------
# 1. SYNTHETIC DATA GENERATION
#    Replace this block with load_real_slices() once you have real images.
# ----------------------------------------------------------------------

def generate_synthetic_slice(size=256, crack_seed=0):
    """Creates one synthetic radiograph: a gray ring (radome cross-section)
    on a dark background, with a few black branching cracks along the ring."""
    rng = np.random.default_rng(crack_seed)
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size / 2, size / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    outer_r = size * 0.38
    inner_r = size * 0.24

    img = np.full((size, size), 17.0)  # background
    ring = (r <= outer_r) & (r >= inner_r)
    img[ring] = 100.0  # ring material intensity

    # add a few random crack-like scratches along the ring using short
    # random-walk lines set to near-black
    n_cracks = rng.integers(4, 8)
    for _ in range(n_cracks):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(inner_r + 3, outer_r - 3)
        py = cy + radius * np.sin(angle)
        px = cx + radius * np.cos(angle)
        length = rng.integers(15, 45)
        for _ in range(length):
            py += rng.normal(0, 1.2)
            px += rng.normal(0, 1.2)
            iy, ix = int(round(py)), int(round(px))
            if 0 <= iy < size and 0 <= ix < size:
                yy2, xx2 = np.ogrid[-1:2, -1:2]
                mask = (yy2 ** 2 + xx2 ** 2) <= 1
                y0, y1 = max(0, iy - 1), min(size, iy + 2)
                x0, x1 = max(0, ix - 1), min(size, ix + 2)
                img[y0:y1, x0:x1] = 2.0

    img += rng.normal(0, 2.0, img.shape)  # sensor noise
    return img


def generate_synthetic_volume(n_slices=25, size=256):
    """Stacks synthetic slices with slight geometric variation per slice
    (mimics a slightly conical radome and per-slice crack differences)."""
    volume = np.zeros((n_slices, size, size), dtype=np.float32)
    for i in range(n_slices):
        base = generate_synthetic_slice(size=size, crack_seed=i)
        scale = 1.0 + 0.04 * np.sin(i / n_slices * np.pi)  # subtle taper
        scaled = zoom(base, scale, order=1)
        h, w = scaled.shape
        if h >= size:
            y0 = (h - size) // 2
            cropped = scaled[y0:y0 + size, y0:y0 + size]
        else:
            pad = (size - h) // 2
            cropped = np.pad(scaled, ((pad, size - h - pad), (pad, size - h - pad)))
        volume[i] = cropped
    return volume


# ----------------------------------------------------------------------
# 2. LOAD REAL IMAGES (use this once you have your actual slice files)
# ----------------------------------------------------------------------

def load_real_slices(folder_path, extension="png"):
    """Loads a folder of grayscale slice images, sorted by filename, into
    a (Z, Y, X) numpy volume. Filenames should sort in acquisition order,
    e.g. slice_001.png, slice_002.png, ...
    """
    from PIL import Image
    files = sorted(
        f for f in os.listdir(folder_path) if f.lower().endswith(extension)
    )
    if not files:
        raise FileNotFoundError(f"No .{extension} files found in {folder_path}")

    slices = []
    for f in files:
        img = np.array(Image.open(os.path.join(folder_path, f)).convert("L"), dtype=np.float32)
        slices.append(img)

    # sanity check: all slices must be same shape
    shapes = {s.shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"Slice images have inconsistent shapes: {shapes}")

    return np.stack(slices, axis=0)


# ----------------------------------------------------------------------
# 3. SEGMENTATION
#    Two options below. Use build_masks_from_detections() since you
#    already have circle params + crack polygons per slice. The old
#    threshold-based segment_volume() is kept for reference/fallback.
# ----------------------------------------------------------------------

def segment_volume(volume, ring_thresh=55, crack_thresh=12, dilation_iters=2):
    """[FALLBACK] Threshold-based segmentation from raw pixel intensity.
    Not needed if you already have circle params + crack polygons."""
    ring_mask = volume > ring_thresh
    footprint = np.zeros_like(ring_mask)
    for z in range(ring_mask.shape[0]):
        footprint[z] = binary_fill_holes(
            binary_dilation(ring_mask[z], iterations=dilation_iters)
        )
    crack_mask = (volume < crack_thresh) & footprint
    return ring_mask, crack_mask


def ring_mask_from_circles(shape, center, inner_r, outer_r):
    """Builds a single-slice ring mask from circle parameters.
    shape: (H, W) of the slice image
    center: (cx, cy)
    inner_r, outer_r: inner and outer radius in pixels
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = center
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    return (r >= inner_r) & (r <= outer_r)


def crack_mask_from_polygons(shape, polygons):
    """Builds a single-slice crack mask by filling in each polygon.
    shape: (H, W) of the slice image
    polygons: list of polygons, each a list/array of (x, y) points,
              e.g. [[(120,80),(125,90),(130,95),...], [...]]
    """
    import cv2
    mask = np.zeros(shape, dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], color=1)
    return mask.astype(bool)


def build_masks_from_detections(shape, n_slices, circle_params, crack_polygons):
    """Builds the full (Z, Y, X) ring_mask and crack_mask volumes from your
    existing detection outputs.

    circle_params: list of length n_slices, each entry a dict
        {"center": (cx, cy), "inner_r": float, "outer_r": float}
    crack_polygons: list of length n_slices, each entry a list of polygons
        (list of (x, y) point lists) for that slice - can be an empty list
        if a slice has no cracks
    """
    ring_mask = np.zeros((n_slices, *shape), dtype=bool)
    crack_mask = np.zeros((n_slices, *shape), dtype=bool)

    for z in range(n_slices):
        cp = circle_params[z]
        ring_mask[z] = ring_mask_from_circles(shape, cp["center"], cp["inner_r"], cp["outer_r"])
        crack_mask[z] = crack_mask_from_polygons(shape, crack_polygons[z])

    return ring_mask, crack_mask


# ----------------------------------------------------------------------
# 4. MESH GENERATION (marching cubes)
# ----------------------------------------------------------------------

def mask_to_mesh(mask, spacing=(3.0, 1.0, 1.0), decimals=2):
    """Runs marching cubes on a binary volume mask and returns vertex /
    face / normal arrays rounded for compact JSON export.

    spacing = (z, y, x) physical size per voxel, e.g. (slice_thickness_mm,
    pixel_size_mm, pixel_size_mm). Set this to your real acquisition spacing.
    """
    verts, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.float32), level=0.5, spacing=spacing
    )
    return verts, faces, normals


def flatten_mesh(verts, faces, normals, decimals=2):
    return {
        "vertices": [round(float(v), decimals) for v in verts.flatten()],
        "faces": faces.astype(np.int32).flatten().tolist(),
        "normals": [round(float(n), 3) for n in normals.flatten()],
    }


# ----------------------------------------------------------------------
# 5. HTML VIEWER EXPORT
# ----------------------------------------------------------------------

VIEWER_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin:0; height:100%; background:#1a1a1e; overflow:hidden; font-family: -apple-system, system-ui, sans-serif; }
  #info {
    position:absolute; top:12px; left:12px; color:#ddd; font-size:13px;
    background:rgba(0,0,0,0.55); padding:8px 12px; border-radius:6px; line-height:1.6;
    z-index: 10;
  }
  #info b { color:#fff; }
  .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
  #status { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); color:#888; font-size:14px; }
</style>
</head>
<body>
<div id="info">
  <div><span class="dot" style="background:#b4b4be;"></span><b>Radome shell</b> - semi-transparent</div>
  <div><span class="dot" style="background:#dc1e1e;"></span><b>Cracks</b> - highlighted red</div>
  <div style="margin-top:6px; color:#999;">Drag to rotate &middot; scroll to zoom</div>
</div>
<div id="status">Loading Three.js...</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
window.addEventListener('load', function() {
  const statusEl = document.getElementById('status');
  if (typeof THREE === 'undefined') {
    statusEl.textContent = 'Failed to load Three.js (check internet connection).';
    return;
  }
  statusEl.textContent = 'Building scene...';

  const MESH_DATA = __MESH_DATA__;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1a1e);

  const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.1, 5000);
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  let isDragging = false, prevX = 0, prevY = 0;
  let theta = Math.PI/4, phi = Math.PI/3.2, radius = 300;

  function updateCamera() {
    camera.position.x = radius * Math.sin(phi) * Math.sin(theta);
    camera.position.y = radius * Math.cos(phi);
    camera.position.z = radius * Math.sin(phi) * Math.cos(theta);
    camera.lookAt(0,0,0);
  }

  renderer.domElement.addEventListener('mousedown', e => { isDragging = true; prevX = e.clientX; prevY = e.clientY; });
  window.addEventListener('mouseup', () => isDragging = false);
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    theta -= (e.clientX - prevX) * 0.008;
    phi -= (e.clientY - prevY) * 0.008;
    phi = Math.max(0.15, Math.min(Math.PI-0.15, phi));
    prevX = e.clientX; prevY = e.clientY;
    updateCamera();
  });
  renderer.domElement.addEventListener('wheel', e => {
    e.preventDefault();
    radius *= (1 + e.deltaY*0.001);
    radius = Math.max(20, Math.min(2000, radius));
    updateCamera();
  }, { passive:false });

  renderer.domElement.addEventListener('touchstart', e => {
    if (e.touches.length===1) { isDragging=true; prevX=e.touches[0].clientX; prevY=e.touches[0].clientY; }
  });
  renderer.domElement.addEventListener('touchmove', e => {
    e.preventDefault();
    if (e.touches.length===1 && isDragging) {
      theta -= (e.touches[0].clientX - prevX)*0.008;
      phi -= (e.touches[0].clientY - prevY)*0.008;
      phi = Math.max(0.15, Math.min(Math.PI-0.15, phi));
      prevX = e.touches[0].clientX; prevY = e.touches[0].clientY;
      updateCamera();
    }
  }, { passive:false });
  renderer.domElement.addEventListener('touchend', () => { isDragging=false; });

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(5, 10, 7);
  scene.add(dirLight);
  const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
  dirLight2.position.set(-5, -5, -7);
  scene.add(dirLight2);

  function buildMesh(meshData, color, opacity) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(meshData.vertices), 3));
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(meshData.normals), 3));
    geometry.setIndex(meshData.faces);
    const material = new THREE.MeshPhongMaterial({
      color: color, transparent: opacity < 1, opacity: opacity,
      side: THREE.DoubleSide, depthWrite: opacity > 0.9, shininess: 30
    });
    return new THREE.Mesh(geometry, material);
  }

  try {
    scene.add(buildMesh(MESH_DATA.ring, 0xb4b4be, 0.35));
    scene.add(buildMesh(MESH_DATA.crack, 0xdc1e1e, 1.0));
    statusEl.style.display = 'none';
    updateCamera();
  } catch (err) {
    statusEl.textContent = 'Error building mesh: ' + err.message;
    console.error(err);
  }

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });

  function animate() {
    requestAnimationFrame(animate);
    renderer.render(scene, camera);
  }
  animate();
});
</script>
</body>
</html>
"""


def export_viewer(ring_mesh_data, crack_mesh_data, output_path="radome_viewer.html"):
    mesh_json = json.dumps({"ring": ring_mesh_data, "crack": crack_mesh_data})
    html = VIEWER_TEMPLATE.replace("__MESH_DATA__", mesh_json)
    with open(output_path, "w") as f:
        f.write(html)
    print(f"Viewer exported: {output_path} ({os.path.getsize(output_path)/1e6:.2f} MB)")


# ----------------------------------------------------------------------
# 6. MAIN
# ----------------------------------------------------------------------

def generate_synthetic_detections(n_slices=25, size=256, seed=0):
    """Fakes what your circle-detector + crack object-detector would output,
    so you can test build_masks_from_detections() before wiring in your
    real detection results. Replace this with your actual detection outputs.
    """
    rng = np.random.default_rng(seed)
    circle_params = []
    crack_polygons = []

    for z in range(n_slices):
        cx, cy = size / 2, size / 2
        taper = 1.0 + 0.04 * np.sin(z / n_slices * np.pi)
        outer_r = size * 0.38 * taper
        inner_r = size * 0.24 * taper
        circle_params.append({"center": (cx, cy), "inner_r": inner_r, "outer_r": outer_r})

        polys = []
        n_cracks = rng.integers(3, 6)
        for _ in range(n_cracks):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(inner_r + 3, outer_r - 3)
            px, py = cx + radius * np.cos(angle), cy + radius * np.sin(angle)
            pts = []
            for _ in range(6):
                px += rng.normal(0, 4)
                py += rng.normal(0, 4)
                pts.append((px, py))
            polys.append(pts)
        crack_polygons.append(polys)

    return circle_params, crack_polygons


def main():
    USE_SYNTHETIC = True   # <-- flip to False once you have real detections
    REAL_SLICE_FOLDER = "slices"  # <-- folder with real images (for shape reference)
    SLICE_THICKNESS_MM = 3.0      # <-- your actual slice spacing
    PIXEL_SIZE_MM = 1.0           # <-- your actual pixel spacing (set from your scanner)
    N_SLICES = 25
    IMG_SIZE = 256  # (height, width) of each slice - match your real image size

    if USE_SYNTHETIC:
        print("Generating synthetic circle params + crack polygons...")
        circle_params, crack_polygons = generate_synthetic_detections(
            n_slices=N_SLICES, size=IMG_SIZE
        )
        shape = (IMG_SIZE, IMG_SIZE)
    else:
        # --- Replace this block with your real detection pipeline output ---
        # circle_params: list per slice -> {"center": (cx,cy), "inner_r":.., "outer_r":..}
        # crack_polygons: list per slice -> list of polygons [[(x,y),...], ...]
        raise NotImplementedError(
            "Plug in your real circle_params and crack_polygons lists here, "
            "one entry per slice, then set `shape` to your image (H, W)."
        )

    print("Building masks from detections...")
    ring_mask, crack_mask = build_masks_from_detections(
        shape, len(circle_params), circle_params, crack_polygons
    )
    print("Ring voxels:", ring_mask.sum(), "| Crack voxels:", crack_mask.sum())

    if crack_mask.sum() == 0:
        print("WARNING: no crack voxels found - check your crack_polygons.")

    spacing = (SLICE_THICKNESS_MM, PIXEL_SIZE_MM, PIXEL_SIZE_MM)

    print("Generating ring mesh (marching cubes)...")
    verts_r, faces_r, normals_r = mask_to_mesh(ring_mask, spacing=spacing)

    print("Generating crack mesh (marching cubes)...")
    verts_c, faces_c, normals_c = mask_to_mesh(crack_mask, spacing=spacing)

    # center both meshes on the ring's centroid so they align visually
    center = verts_r.mean(axis=0)
    verts_r -= center
    verts_c -= center

    ring_data = flatten_mesh(verts_r, faces_r, normals_r)
    crack_data = flatten_mesh(verts_c, faces_c, normals_c)

    print(f"Ring mesh: {len(verts_r)} verts, {len(faces_r)} faces")
    print(f"Crack mesh: {len(verts_c)} verts, {len(faces_c)} faces")

    export_viewer(ring_data, crack_data, output_path="radome_viewer.html")
    print("Done. Open radome_viewer.html in a browser.")


if __name__ == "__main__":
    main()
########################## Prompts ################################################################
**Prompt 1 — Threshold-based segmentation (intensity-only detection)**

> Write a Python script that reconstructs a 3D model from a stack of 2D radiography slice images of a radome (ring-shaped object), taken 3mm apart like CT slices. Each grayscale image shows: a dark background, a bright/gray ring (the radome material), and crack defects that appear as near-black pixels running through the ring material.
>
> Requirements:
>
> 1. Load a folder of grayscale slice images (sorted by filename, e.g. slice_001.png, slice_002.png...) into a 3D numpy volume of shape (Z, Y, X), where Z is slice index.
> 2. Segment the ring material using an intensity threshold: pixels brighter than a threshold value = radome material → this gives a `ring_mask` volume.
> 3. Segment cracks using a second, darker intensity threshold (near-black pixels). Critically, cracks must ONLY be detected inside the ring material region — build a filled footprint of the ring (using binary dilation + fill holes so the hollow center and any crack gaps count as "inside"), and mask the crack detection to only that footprint. This prevents the dark background outside the ring from being misidentified as a crack. The resulting `crack_mask` volume must contain only crack pixels that fall between the ring's inner and outer boundary — never outside the outer edge or outside the ring material area.
> 4. Run marching cubes (skimage.measure.marching_cubes) separately on `ring_mask` and `crack_mask` to generate two 3D meshes, using correct real-world voxel spacing: slice thickness in mm (Z axis) and pixel size in mm (Y, X axes).
> 5. Center both meshes on the same coordinate origin so they align correctly when rendered together.
> 6. Export a single self-contained interactive HTML file using Three.js (no external server needed, mesh data embedded as JSON) that renders the radome shell semi-transparent (gray, ~30–40% opacity) and the cracks in solid red, visible through the shell. Include mouse drag-to-rotate and scroll-to-zoom controls.
> 7. Include a synthetic data generator that creates fake ring+crack slice images (with cracks placed only within the ring band, not in the background) so the full pipeline can be tested before real images are available. Make it easy to swap in real image loading later.

---

**Prompt 2 — Detection-based masks (circle + polygon outputs, constrained to ring)**

> I have a stack of 2D radiography slice images of a radome (ring-shaped object), taken 3mm apart like CT slices. For each slice, I already have two detection outputs from my own pipeline:
>
> 1. **Radome boundary**: inner and outer circle parameters — center (x, y), inner_radius, outer_radius — defining exactly where the ring material is (the material exists only in the band between inner_radius and outer_radius from the center; everything inside inner_radius is the hollow interior, everything outside outer_radius is background).
> 2. **Crack defects**: polygon contour points (list of (x, y) points) outlining each crack region, detected separately by an object detection model. A slice can have zero, one, or multiple crack polygons.
>
> Requirements:
>
> 1. For each slice, build a `ring_mask` (2D boolean array) from the circle parameters: a pixel is True only if its distance from center is between inner_radius and outer_radius — i.e., strictly within the material band.
> 2. For each slice, build a `crack_mask` (2D boolean array) by filling the crack polygons (using cv2.fillPoly). Then constrain this crack mask to only the ring material band: `crack_mask = crack_mask AND ring_mask` for that slice. This guarantees cracks can never appear outside the radome material — not in the hollow center, and not outside the outer edge — even if a detected polygon's points slightly overshoot the ring boundary.
> 3. Stack the per-slice 2D masks across all slices into two 3D volumes: `ring_mask` and `crack_mask`, each of shape (Z, Y, X).
> 4. Run marching cubes (skimage.measure.marching_cubes) separately on each 3D volume to generate two meshes — one for the radome shell, one for the cracks — using correct real-world voxel spacing (slice thickness in mm, pixel size in mm) so the model has accurate proportions.
> 5. Center both meshes on the same coordinate origin so they align when rendered together.
> 6. Export a single self-contained interactive HTML file using Three.js (no external server, mesh data embedded as JSON) rendering the radome shell semi-transparent (gray, ~30–40% opacity) with cracks shown in solid red, visible through the shell. Include mouse drag-to-rotate and scroll-to-zoom controls.
> 7. Include a function that generates synthetic/fake circle_params and crack_polygons (one entry per slice), with crack points always placed within the ring's material band (between inner and outer radius), so I can test the entire pipeline today. Structure the code so swapping in my real detection outputs later only requires replacing the synthetic data function with real per-slice lists.
