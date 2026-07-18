import os
import math
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CAD / CAM CORE
# ============================================================
import cadquery as cq

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection
)

from shapely.affinity import (
    translate,
    rotate
)

from shapely.geometry import JOIN_STYLE

try:
    from shapely import make_valid
except ImportError:
    try:
        from shapely.validation import make_valid
    except ImportError:
        make_valid = None


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CNC CAM ENGINE PRO v7.0",
    layout="wide"
)

st.markdown(
    """
    # 🏭 CNC CAM ENGINE PRO v7.0

    ### AUTOMATIC 3D ASSEMBLY → NESTING → MULTI-TOOL CAM

    **STEP Assembly → Tách chi tiết → Hạ phẳng → Nesting → Gom Toolpath theo Dao → Xuất G-Code**
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2. MATERIAL / SHEET CONFIGURATION
# ============================================================

st.sidebar.header("📐 THÔNG SỐ VẬT LIỆU PHÔI")

sheet_W = st.sidebar.number_input(
    "Chiều rộng khổ ván X (mm)",
    min_value=100.0,
    value=2440.0,
    step=10.0
)

sheet_H = st.sidebar.number_input(
    "Chiều cao khổ ván Y (mm)",
    min_value=100.0,
    value=1220.0,
    step=10.0
)

sheet_thickness = st.sidebar.number_input(
    "Độ dày ván tiêu chuẩn Z (mm)",
    min_value=0.1,
    value=17.0,
    step=0.1
)

margin = st.sidebar.number_input(
    "Khoảng cách biên tấm ván (mm)",
    min_value=0.0,
    value=15.0,
    step=1.0
)

safety_spacing = st.sidebar.number_input(
    "Khoảng cách an toàn giữa chi tiết (mm)",
    min_value=0.0,
    value=6.0,
    step=0.5
)


# ============================================================
# 3. TOOL CONFIGURATION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH MULTI-TOOL CNC")


# ------------------------------------------------------------
# TOOL T1 - INNER CUT
# ------------------------------------------------------------

st.sidebar.markdown("### 🔵 T1 - INNER CUT / LỖ")

t1_dia = st.sidebar.number_input(
    "T1 - Đường kính dao (mm)",
    min_value=0.1,
    value=3.0,
    step=0.1
)

t1_feed = st.sidebar.number_input(
    "T1 - Feed (mm/min)",
    min_value=100,
    value=2500,
    step=100
)

t1_plunge = st.sidebar.number_input(
    "T1 - Plunge (mm/min)",
    min_value=50,
    value=800,
    step=50
)

t1_spindle = st.sidebar.number_input(
    "T1 - Spindle (RPM)",
    min_value=1000,
    value=18000,
    step=500
)

t1_max_stepdown = st.sidebar.number_input(
    "T1 - Max Stepdown (mm)",
    min_value=0.5,
    value=4.0,
    step=0.5
)


# ------------------------------------------------------------
# TOOL T2 - POCKET
# ------------------------------------------------------------

st.sidebar.markdown("### 🟣 T2 - POCKET / HẠ NỀN")

t2_dia = st.sidebar.number_input(
    "T2 - Đường kính dao (mm)",
    min_value=0.1,
    value=6.0,
    step=0.1
)

t2_feed = st.sidebar.number_input(
    "T2 - Feed (mm/min)",
    min_value=100,
    value=3500,
    step=100
)

t2_plunge = st.sidebar.number_input(
    "T2 - Plunge (mm/min)",
    min_value=50,
    value=1200,
    step=50
)

t2_spindle = st.sidebar.number_input(
    "T2 - Spindle (RPM)",
    min_value=1000,
    value=18000,
    step=500
)

t2_max_stepdown = st.sidebar.number_input(
    "T2 - Max Stepdown (mm)",
    min_value=0.5,
    value=6.0,
    step=0.5
)


# ------------------------------------------------------------
# TOOL T3 - OUTER CUT
# ------------------------------------------------------------

st.sidebar.markdown("### 🟢 T3 - OUTER CUT / CẮT BIÊN")

t3_dia = st.sidebar.number_input(
    "T3 - Đường kính dao (mm)",
    min_value=0.1,
    value=6.0,
    step=0.1
)

t3_feed = st.sidebar.number_input(
    "T3 - Feed (mm/min)",
    min_value=100,
    value=3500,
    step=100
)

t3_plunge = st.sidebar.number_input(
    "T3 - Plunge (mm/min)",
    min_value=50,
    value=1200,
    step=50
)

t3_spindle = st.sidebar.number_input(
    "T3 - Spindle (RPM)",
    min_value=1000,
    value=18000,
    step=500
)

t3_max_stepdown = st.sidebar.number_input(
    "T3 - Max Stepdown (mm)",
    min_value=0.5,
    value=6.0,
    step=0.5
)


# ============================================================
# 4. GENERAL CNC CONFIGURATION
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙ CẤU HÌNH CNC")

chord_tolerance = st.sidebar.number_input(
    "Dung sai spline (mm)",
    min_value=0.005,
    max_value=0.5,
    value=0.02,
    step=0.005,
    format="%.3f"
)

enable_leadin = st.sidebar.checkbox(
    "Kích hoạt Lead-in",
    value=True
)

leadin_length = st.sidebar.number_input(
    "Chiều dài Lead-in (mm)",
    min_value=2.0,
    value=5.0,
    step=0.5
)

enable_ramping = st.sidebar.checkbox(
    "Kích hoạt Continuous Spiral Ramp",
    value=True
)

enable_tabs = st.sidebar.checkbox(
    "Kích hoạt Structural Tabs",
    value=True
)

tab_width = st.sidebar.number_input(
    "Chiều dài Tab (mm)",
    min_value=5.0,
    value=20.0,
    step=1.0
)

tab_thickness = st.sidebar.number_input(
    "Độ dày vật liệu còn lại tại Tab (mm)",
    min_value=0.5,
    value=4.0,
    step=0.5
)

tab_count_default = st.sidebar.slider(
    "Số lượng Tab / chi tiết",
    min_value=2,
    max_value=8,
    value=4
)

cnc_dialect = st.sidebar.selectbox(
    "Post Processor",
    [
        "Fanuc / Syntec",
        "Mach3 / Grbl",
        "UGS",
        "Weihong"
    ]
)

safe_Z = st.sidebar.number_input(
    "Safe Z (mm)",
    min_value=1.0,
    value=25.0,
    step=1.0
)

thru_overlap = st.sidebar.number_input(
    "Độ sâu cắt xuyên thêm (mm)",
    min_value=0.0,
    value=0.5,
    step=0.1
)


# ============================================================
# 5. TOOL DATABASE
# ============================================================

TOOLS = {

    "T1": {
        "name": "INNER_CUT",
        "operation_types": [
            "CNC_INNER_CUT"
        ],
        "diameter": t1_dia,
        "radius": t1_dia / 2.0,
        "feed": t1_feed,
        "plunge": t1_plunge,
        "spindle": t1_spindle,
        "max_stepdown": t1_max_stepdown,
        "order": 1
    },

    "T2": {
        "name": "POCKET",
        "operation_types": [
            "CNC_POCKET"
        ],
        "diameter": t2_dia,
        "radius": t2_dia / 2.0,
        "feed": t2_feed,
        "plunge": t2_plunge,
        "spindle": t2_spindle,
        "max_stepdown": t2_max_stepdown,
        "order": 2
    },

    "T3": {
        "name": "OUTER_CUT",
        "operation_types": [
            "CNC_OUTER_CUT"
        ],
        "diameter": t3_dia,
        "radius": t3_dia / 2.0,
        "feed": t3_feed,
        "plunge": t3_plunge,
        "spindle": t3_spindle,
        "max_stepdown": t3_max_stepdown,
        "order": 3
    }
}


# ============================================================
# 6. GEOMETRY REPAIR
# ============================================================

def repair_geometry(geom):

    if geom is None:
        return geom

    if geom.is_empty:
        return geom

    if geom.is_valid:
        return geom

    if make_valid is not None:

        try:

            fixed = make_valid(geom)

            if not fixed.is_empty:
                return fixed

        except Exception:
            pass

    try:

        fixed = geom.buffer(0)

        if not fixed.is_empty:
            return fixed

    except Exception:
        pass

    return geom


def extract_largest_polygon(geom):

    if geom is None:
        return None

    if geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):

        return max(
            geom.geoms,
            key=lambda p: p.area
        )

    if isinstance(geom, GeometryCollection):

        polygons = [
            g for g in geom.geoms
            if isinstance(g, Polygon)
        ]

        if polygons:
            return max(
                polygons,
                key=lambda p: p.area
            )

    return None


# ============================================================
# 7. CAD EDGE EXTRACTION
# ============================================================

def get_local_coordinates(
    cq_edge,
    plane_obj,
    tolerance=0.02
):

    p_start = plane_obj.toLocalCoords(
        cq_edge.startPoint()
    )

    p_end = plane_obj.toLocalCoords(
        cq_edge.endPoint()
    )

    g_type = cq_edge.geomType()

    if g_type == "LINE":

        return {
            "type": "LINE",
            "start": (
                p_start.x,
                p_start.y
            ),
            "end": (
                p_end.x,
                p_end.y
            )
        }

    elif g_type == "CIRCLE":

        p_center = plane_obj.toLocalCoords(
            cq_edge.Center()
        )

        return {
            "type": "CIRCLE",
            "center": (
                p_center.x,
                p_center.y
            ),
            "radius": cq_edge.radius()
        }

    else:

        try:

            occ_curve = cq_edge.ToAdaptor3d()

            first_param = occ_curve.FirstParameter()
            last_param = occ_curve.LastParameter()

            length_est = cq_edge.Length()

            segments = max(
                32,
                min(
                    512,
                    int(
                        length_est /
                        math.sqrt(
                            tolerance
                            if tolerance > 0
                            else 0.02
                        )
                    )
                )
            )

            pts = []

            for i in range(segments + 1):

                t = (
                    first_param
                    +
                    (last_param - first_param)
                    *
                    i
                    /
                    segments
                )

                p_loc = plane_obj.toLocalCoords(
                    cq_edge.valueAt(t)
                )

                pts.append(
                    (
                        p_loc.x,
                        p_loc.y
                    )
                )

            return {
                "type": "DISCRETE_CURVE",
                "points": pts
            }

        except Exception:

            return {
                "type": "LINE",
                "start": (
                    p_start.x,
                    p_start.y
                ),
                "end": (
                    p_end.x,
                    p_end.y
                )
            }


def discrete_edges(edges):

    pts = []

    for edge in edges:

        if edge["type"] == "LINE":

            pts.append(edge["start"])
            pts.append(edge["end"])

        elif edge["type"] == "CIRCLE":

            cx, cy = edge["center"]
            radius = edge["radius"]

            for angle in np.linspace(
                0,
                360,
                128,
                endpoint=False
            ):

                rad = math.radians(angle)

                pts.append(
                    (
                        cx
                        +
                        radius
                        *
                        math.cos(rad),

                        cy
                        +
                        radius
                        *
                        math.sin(rad)
                    )
                )

        elif edge["type"] == "DISCRETE_CURVE":

            pts.extend(
                edge["points"]
            )

    return pts


def clean_polygon_points(
    points,
    tolerance=0.01
):

    if not points:
        return []

    cleaned = []

    for p in points:

        p = (
            float(p[0]),
            float(p[1])
        )

        if not cleaned:

            cleaned.append(p)

        else:

            if not np.allclose(
                cleaned[-1],
                p,
                atol=tolerance
            ):

                cleaned.append(p)

    if len(cleaned) > 2:

        if not np.allclose(
            cleaned[0],
            cleaned[-1],
            atol=tolerance
        ):

            cleaned.append(
                cleaned[0]
            )

    return cleaned


# ============================================================
# 8. STEP ASSEMBLY EXPLODER
# ============================================================

def process_full_assembly_step(
    file_bytes,
    filename,
    std_thickness,
    tol_val
):

    temp_path = None
    parsed_parts = []

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(filename)[1]
        ) as temp_file:

            temp_file.write(
                file_bytes
            )

            temp_path = temp_file.name

        imported_shape = cq.importers.importStep(
            temp_path
        )

        solids = imported_shape.solids().vals()

        if not solids:

            raise ValueError(
                "Không tìm thấy Solid hợp lệ trong STEP."
            )

        st.info(
            f"🔎 Đã phát hiện "
            f"**{len(solids)}** "
            f"Solid trong assembly."
        )

        for idx, solid in enumerate(solids):

            # Loại bỏ chi tiết quá nhỏ
            if solid.Area() < 500:
                continue

            faces = solid.faces().vals()

            plane_faces = [
                f
                for f in faces
                if f.geomType() == "PLANE"
            ]

            if not plane_faces:
                continue

            # Mặt lớn nhất
            target_face = max(
                plane_faces,
                key=lambda f: f.Area()
            )

            face_center = target_face.Center()

            face_normal = target_face.normalAt(
                face_center
            )

            ref_plane = cq.Plane(
                origin=face_center,
                normal=face_normal
            )

            # OUTER CONTOUR
            outer_wire = target_face.outerWire()

            outer_edges = [
                get_local_coordinates(
                    edge,
                    ref_plane,
                    tol_val
                )
                for edge in outer_wire.Edges()
            ]

            # FEATURES
            features = []

            # INNER CUT / LỖ
            for inner_wire in target_face.innerWires():

                wire_edges = [
                    get_local_coordinates(
                        edge,
                        ref_plane,
                        tol_val
                    )
                    for edge in inner_wire.Edges()
                ]

                features.append(
                    {
                        "type": "CNC_INNER_CUT",
                        "edges": wire_edges,
                        "depth": std_thickness
                    }
                )

            # POCKET
            pocket_signatures = set()

            for face in faces:

                if face is target_face:
                    continue

                if face.geomType() != "PLANE":
                    continue

                local_center = ref_plane.toLocalCoords(
                    face.Center()
                )

                depth = abs(
                    local_center.z
                )

                if (
                    0.5
                    <= depth
                    <
                    std_thickness + 2.0
                ):

                    p_edges = [
                        get_local_coordinates(
                            edge,
                            ref_plane,
                            tol_val
                        )
                        for edge in face.outerWire().Edges()
                    ]

                    if not p_edges:
                        continue

                    raw_p = clean_polygon_points(
                        discrete_edges(
                            p_edges
                        )
                    )

                    if len(raw_p) < 4:
                        continue

                    try:

                        poly = repair_geometry(
                            Polygon(raw_p)
                        )

                        if poly is None:
                            continue

                        centroid = poly.centroid

                        signature = (
                            round(
                                centroid.x,
                                2
                            ),
                            round(
                                centroid.y,
                                2
                            ),
                            round(
                                depth,
                                1
                            )
                        )

                        if signature in pocket_signatures:
                            continue

                        pocket_signatures.add(
                            signature
                        )

                        features.append(
                            {
                                "type": "CNC_POCKET",
                                "edges": p_edges,
                                "depth": depth
                            }
                        )

                    except Exception:
                        continue

            # BOUNDING BOX
            raw_outer_pts = clean_polygon_points(
                discrete_edges(
                    outer_edges
                )
            )

            if len(raw_outer_pts) < 4:
                continue

            poly_outer = repair_geometry(
                Polygon(raw_outer_pts)
            )

            if (
                poly_outer is None
                or poly_outer.is_empty
            ):
                continue

            min_x, min_y, max_x, max_y = (
                poly_outer.bounds
            )

            width_local = max_x - min_x
            height_local = max_y - min_y

            parsed_parts.append(
                {
                    "name": f"Tam_Van_{idx + 1}",
                    "width": width_local,
                    "height": height_local,
                    "outer_edges": outer_edges,
                    "features": features,
                    "origin_x": face_center.x,
                    "origin_y": face_center.y
                }
            )

    finally:

        if (
            temp_path
            and
            os.path.exists(temp_path)
        ):

            os.remove(
                temp_path
            )

    return parsed_parts


# ============================================================
# 9. TRANSFORMATION
# ============================================================

def transform_point_production(
    x,
    y,
    dx,
    dy,
    angle,
    ox,
    oy
):

    x_local = x - ox
    y_local = y - oy

    rad = math.radians(
        angle
    )

    cos_a = math.cos(
        rad
    )

    sin_a = math.sin(
        rad
    )

    return (
        x_local * cos_a
        -
        y_local * sin_a
        +
        dx,

        x_local * sin_a
        +
        y_local * cos_a
        +
        dy
    )


def transform_edge_production(
    edge,
    dx,
    dy,
    angle,
    ox,
    oy
):

    if edge["type"] == "LINE":

        return {
            "type": "LINE",
            "start": transform_point_production(
                edge["start"][0],
                edge["start"][1],
                dx,
                dy,
                angle,
                ox,
                oy
            ),
            "end": transform_point_production(
                edge["end"][0],
                edge["end"][1],
                dx,
                dy,
                angle,
                ox,
                oy
            )
        }

    elif edge["type"] == "CIRCLE":

        return {
            "type": "CIRCLE",
            "center": transform_point_production(
                edge["center"][0],
                edge["center"][1],
                dx,
                dy,
                angle,
                ox,
                oy
            ),
            "radius": edge["radius"]
        }

    elif edge["type"] == "DISCRETE_CURVE":

        return {
            "type": "DISCRETE_CURVE",
            "points": [
                transform_point_production(
                    x,
                    y,
                    dx,
                    dy,
                    angle,
                    ox,
                    oy
                )
                for x, y in edge["points"]
            ]
        }

    return edge


def transform_edges_production(
    edges,
    dx,
    dy,
    angle,
    ox,
    oy
):

    return [
        transform_edge_production(
            edge,
            dx,
            dy,
            angle,
            ox,
            oy
        )
        for edge in edges
    ]


# ============================================================
# 10. T-BONE RELIEF
# ============================================================

def apply_t_bone_relief(
    polygon_points,
    tool_radius
):

    if len(polygon_points) < 4:
        return polygon_points

    pts = list(
        polygon_points
    )

    if np.allclose(
        pts[0],
        pts[-1]
    ):

        pts.pop()

    poly = repair_geometry(
        Polygon(pts)
    )

    if (
        not isinstance(
            poly,
            Polygon
        )
        or
        poly.is_empty
    ):

        return polygon_points

    if not poly.exterior.is_ccw:

        pts.reverse()

    result = []

    n = len(pts)

    for i in range(n):

        p_prev = np.array(
            pts[i - 1],
            dtype=float
        )

        p_curr = np.array(
            pts[i],
            dtype=float
        )

        p_next = np.array(
            pts[
                (i + 1) % n
            ],
            dtype=float
        )

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        len_v1 = np.linalg.norm(
            v1
        )

        len_v2 = np.linalg.norm(
            v2
        )

        if (
            len_v1 < 1e-5
            or
            len_v2 < 1e-5
        ):

            result.append(
                tuple(p_curr)
            )

            continue

        v1_u = v1 / len_v1
        v2_u = v2 / len_v2

        dot = np.clip(
            np.dot(
                v1_u,
                v2_u
            ),
            -1.0,
            1.0
        )

        angle = math.acos(
            dot
        )

        cross_z = (
            v1_u[0]
            *
            v2_u[1]
            -
            v1_u[1]
            *
            v2_u[0]
        )

        is_concave = (
            cross_z > 0.001
        )

        is_right_angle = (
            abs(
                angle
                -
                math.pi / 2
            )
            <
            math.radians(10)
        )

        result.append(
            tuple(p_curr)
        )

        if (
            is_concave
            and
            is_right_angle
        ):

            bisector = (
                v1_u
                +
                v2_u
            )

            norm_b = np.linalg.norm(
                bisector
            )

            if norm_b > 1e-5:

                bisector_u = (
                    bisector
                    /
                    norm_b
                )

                relief_point = (
                    p_curr
                    +
                    bisector_u
                    *
                    tool_radius
                )

                result.append(
                    tuple(
                        relief_point
                    )
                )

                result.append(
                    tuple(
                        p_curr
                    )
                )

    result.append(
        result[0]
    )

    return result


# ============================================================
# 11. NESTING
# ============================================================

def execute_production_nesting(
    parts_list,
    sheet_w,
    sheet_h,
    offset_val,
    margin_val
):

    sheet_bound = Polygon(
        [
            (
                margin_val,
                margin_val
            ),

            (
                sheet_w - margin_val,
                margin_val
            ),

            (
                sheet_w - margin_val,
                sheet_h - margin_val
            ),

            (
                margin_val,
                sheet_h - margin_val
            )
        ]
    )

    sorted_parts = sorted(
        parts_list,
        key=lambda x:
        x["width"]
        *
        x["height"],
        reverse=True
    )

    sheets = []

    for part in sorted_parts:

        raw_points = clean_polygon_points(
            discrete_edges(
                part["outer_edges"]
            )
        )

        if len(raw_points) < 4:
            continue

        poly_geom = extract_largest_polygon(
            repair_geometry(
                Polygon(raw_points)
            )
        )

        if poly_geom is None:
            continue

        buffered_poly = extract_largest_polygon(
            repair_geometry(
                poly_geom.buffer(
                    offset_val,
                    resolution=16,
                    join_style=JOIN_STYLE.round
                )
            )
        )

        if buffered_poly is None:
            continue

        min_x, min_y, _, _ = (
            buffered_poly.bounds
        )

        normalized_poly = translate(
            buffered_poly,
            xoff=-min_x,
            yoff=-min_y
        )

        raw_normalized = translate(
            poly_geom,
            xoff=-min_x,
            yoff=-min_y
        )

        best_pos = None
        target_sheet_idx = -1
        best_score = float("inf")

        for sheet_idx, sheet_data in enumerate(sheets):

            placed_union = sheet_data[
                "placed_union_geom"
            ]

            anchors = [
                (
                    margin_val,
                    margin_val
                )
            ]

            for pb in sheet_data[
                "placed_buffered_polygons"
            ]:

                b = pb.bounds

                anchors.extend(
                    [
                        (
                            b[2],
                            b[1]
                        ),

                        (
                            b[0],
                            b[3]
                        ),

                        (
                            b[2],
                            b[3]
                        )
                    ]
                )

            for angle in [
                0,
                90,
                180,
                270
            ]:

                rotated_poly = rotate(
                    normalized_poly,
                    angle,
                    origin=(0, 0)
                )

                r_min_x, r_min_y, _, _ = (
                    rotated_poly.bounds
                )

                for anchor_x, anchor_y in anchors:

                    dx = (
                        anchor_x
                        -
                        r_min_x
                    )

                    dy = (
                        anchor_y
                        -
                        r_min_y
                    )

                    candidate = translate(
                        rotated_poly,
                        xoff=dx,
                        yoff=dy
                    )

                    if not sheet_bound.covers(
                        candidate
                    ):

                        continue

                    if (
                        placed_union is not None
                        and
                        candidate.intersects(
                            placed_union
                        )
                    ):

                        continue

                    bounds = candidate.bounds

                    score = (
                        bounds[0]
                        +
                        bounds[1]
                        *
                        2.5
                    )

                    if score < best_score:

                        best_score = score

                        target_sheet_idx = sheet_idx

                        best_pos = {
                            "dx": dx,
                            "dy": dy,
                            "angle": angle,
                            "cand_poly": candidate,
                            "raw_trans": translate(
                                rotate(
                                    raw_normalized,
                                    angle,
                                    origin=(0, 0)
                                ),
                                xoff=dx,
                                yoff=dy
                            )
                        }

        if (
            best_pos is not None
            and
            target_sheet_idx >= 0
        ):

            sheets[
                target_sheet_idx
            ][
                "parts"
            ].append(
                {
                    "part_ref": part,
                    "original_offset": (
                        min_x,
                        min_y
                    ),
                    "placed_polygon":
                    best_pos[
                        "raw_trans"
                    ],
                    "dx":
                    best_pos[
                        "dx"
                    ],
                    "dy":
                    best_pos[
                        "dy"
                    ],
                    "angle":
                    best_pos[
                        "angle"
                    ]
                }
            )

            sheets[
                target_sheet_idx
            ][
                "placed_buffered_polygons"
            ].append(
                best_pos[
                    "cand_poly"
                ]
            )

            sheets[
                target_sheet_idx
            ][
                "placed_union_geom"
            ] = (
                sheets[
                    target_sheet_idx
                ][
                    "placed_union_geom"
                ]
                .union(
                    best_pos[
                        "cand_poly"
                    ]
                )
            )

        else:

            new_sheet_id = (
                len(sheets)
                +
                1
            )

            dx = (
                margin_val
                -
                min_x
            )

            dy = (
                margin_val
                -
                min_y
            )

            init_poly = translate(
                normalized_poly,
                xoff=dx,
                yoff=dy
            )

            sheets.append(
                {
                    "sheet_id":
                    new_sheet_id,

                    "parts":
                    [
                        {
                            "part_ref": part,
                            "original_offset": (
                                min_x,
                                min_y
                            ),
                            "placed_polygon": translate(
                                raw_normalized,
                                xoff=dx,
                                yoff=dy
                            ),
                            "dx": dx,
                            "dy": dy,
                            "angle": 0
                        }
                    ],

                    "placed_buffered_polygons": [
                        init_poly
                    ],

                    "placed_union_geom": init_poly
                }
            )

    return sheets
