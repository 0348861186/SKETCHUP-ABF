import io
import os
import math
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CAD / CAM INDUSTRIAL CORE
# ============================================================

import cadquery as cq

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection
)

from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE

try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None

import ezdxf


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Production-Ready CNC CAM Engine Pro v4.2",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#0F172A;'>
    🏭 PRODUCTION-READY CNC CAM ENGINE PRO V4.2
    </h1>

    <p style='text-align:center; color:#475569;'>
    STEP → CAD Analysis → Nesting → Toolpath → DXF Layer → ATC G-Code
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2. SIDEBAR CONFIGURATION
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
    "Độ dày ván thực tế Z (mm)",
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
    "Khoảng cách giữa các chi tiết (mm)",
    min_value=0.0,
    value=6.0,
    step=0.5
)


st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH DAO & CẮT G-CODE")

t1_dia = st.sidebar.number_input(
    "Đường kính dao T1 (mm)",
    min_value=0.1,
    value=6.0,
    step=0.1
)

t1_feed = st.sidebar.number_input(
    "Tốc độ cắt F (mm/min)",
    min_value=100,
    value=3500,
    step=100
)

t1_spindle = st.sidebar.number_input(
    "Tốc độ trục chính S (RPM)",
    min_value=1000,
    value=18000,
    step=500
)

max_stepdown = st.sidebar.number_input(
    "Chiều sâu mỗi lát cắt Stepdown (mm)",
    min_value=0.5,
    value=6.0,
    step=0.5
)


st.sidebar.markdown("### 🔩 LEAD-IN / RAMP / TABS")

ramp_angle = st.sidebar.slider(
    "Góc xuống dao Ramping (độ)",
    min_value=5,
    max_value=25,
    value=10
)

tab_width = st.sidebar.number_input(
    "Chiều dài Tab (mm)",
    min_value=5.0,
    value=15.0,
    step=1.0
)

tab_thickness = st.sidebar.number_input(
    "Độ dày Tab (mm)",
    min_value=0.5,
    value=3.5,
    step=0.5
)

tab_count_default = st.sidebar.slider(
    "Số lượng Tab / chi tiết",
    min_value=2,
    max_value=6,
    value=3
)


st.sidebar.markdown("### ⚙ POST-PROCESSOR")

cnc_dialect = st.sidebar.selectbox(
    "Hệ điều hành máy CNC",
    [
        "Fanuc / Syntec",
        "Mach3 / Grbl",
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


tool_radius = t1_dia / 2.0

total_offset = tool_radius + safety_spacing


# ============================================================
# 3. GEOMETRY REPAIR
# ============================================================

def repair_geometry(poly):

    if poly is None:
        return poly

    if poly.is_empty:
        return poly

    if poly.is_valid:
        return poly

    if make_valid is not None:

        try:
            fixed = make_valid(poly)

            if not fixed.is_empty:
                return fixed

        except Exception:
            pass

    try:

        fixed = poly.buffer(0)

        if not fixed.is_empty:
            return fixed

    except Exception:
        pass

    return poly


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
# 4. CAD EDGE → LOCAL 2D COORDINATES
# ============================================================

def get_local_coordinates(cq_edge, plane_obj):

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

            segments = 64

            pts = []

            for i in range(segments + 1):

                t = first_param + (
                    last_param - first_param
                ) * i / segments

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


# ============================================================
# 5. EDGE → DISCRETE POINTS
# ============================================================

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
                96,
                endpoint=False
            ):

                rad = math.radians(angle)

                pts.append(
                    (
                        cx + radius * math.cos(rad),
                        cy + radius * math.sin(rad)
                    )
                )

        elif edge["type"] == "DISCRETE_CURVE":

            pts.extend(
                edge["points"]
            )

    return pts


# ============================================================
# 6. CLEAN POLYGON POINTS
# ============================================================

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
# 7. T-BONE RELIEF
# ============================================================

def apply_t_bone_relief(
    polygon_points,
    tool_radius
):

    if len(polygon_points) < 4:

        return polygon_points

    pts = list(polygon_points)

    if np.allclose(
        pts[0],
        pts[-1]
    ):

        pts.pop()

    poly = Polygon(pts)

    poly = repair_geometry(poly)

    if not isinstance(poly, Polygon):

        return polygon_points

    # Đảm bảo CCW
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
            pts[(i + 1) % n],
            dtype=float
        )

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)

        if (
            len_v1 < 1e-6
            or len_v2 < 1e-6
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

        angle = math.acos(dot)

        cross_z = (
            v1_u[0] * v2_u[1]
            -
            v1_u[1] * v2_u[0]
        )

        # Góc lõm của polygon CCW
        is_concave = (
            cross_z > 0
        )

        is_right_angle = (
            abs(
                angle - math.pi / 2
            ) < math.radians(8)
        )

        if (
            is_concave
            and is_right_angle
        ):

            bisector = (
                v1_u
                +
                v2_u
            )

            norm_b = np.linalg.norm(
                bisector
            )

            if norm_b > 1e-6:

                bisector_u = (
                    bisector
                    /
                    norm_b
                )

                relief_distance = (
                    tool_radius
                    *
                    math.sqrt(2)
                )

                relief_point = (
                    p_curr
                    +
                    bisector_u
                    *
                    relief_distance
                )

                result.append(
                    tuple(
                        relief_point
                    )
                )

        else:

            result.append(
                tuple(p_curr)
            )

    result.append(
        result[0]
    )

    return result


# ============================================================
# 8. STEP FILE ANALYSIS
# ============================================================

def process_cad_file_production(
    file_bytes,
    filename,
    sheet_thick
):

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(
                filename
            )[1]
        ) as temp_file:

            temp_file.write(
                file_bytes
            )

            temp_path = temp_file.name

        part = cq.importers.importStep(
            temp_path
        )

        all_faces = part.faces().vals()

        if not all_faces:

            raise ValueError(
                "STEP không chứa mặt hình học hợp lệ."
            )

        top_faces = [

            f for f in all_faces

            if (
                f.geomType() == "PLANE"
                and f.normalAt().z > 0.8
            )
        ]

        if not top_faces:

            top_faces = all_faces

        target_face = max(
            top_faces,
            key=lambda f: f.Area()
        )

        ref_plane = cq.Plane(
            target_face
        )

        face_z_level = (
            target_face.Center().z
        )

        outer_wire = (
            target_face.outerWire()
        )

        outer_edges = [

            get_local_coordinates(
                edge,
                ref_plane
            )

            for edge in outer_wire.Edges()
        ]

        features = []

        # --------------------------------
        # INNER WIRES
        # --------------------------------

        for inner_wire in target_face.innerWires():

            wire_edges = [

                get_local_coordinates(
                    edge,
                    ref_plane
                )

                for edge in inner_wire.Edges()
            ]

            features.append(
                {
                    "type": "CNC_INNER_CUT",
                    "edges": wire_edges,
                    "depth": sheet_thick
                }
            )

        # --------------------------------
        # POCKET DETECTION
        # --------------------------------

        for face in all_faces:

            if face is target_face:

                continue

            if (
                face.geomType() == "PLANE"
                and face.Area()
                <
                target_face.Area() * 0.9
            ):

                f_z = face.Center().z

                if (
                    f_z < face_z_level
                    and
                    f_z >= (
                        face_z_level
                        -
                        sheet_thick
                    )
                ):

                    depth = abs(
                        face_z_level
                        -
                        f_z
                    )

                    if depth > 0.2:

                        p_edges = [

                            get_local_coordinates(
                                edge,
                                ref_plane
                            )

                            for edge in face.outerWire().Edges()
                        ]

                        features.append(
                            {
                                "type": "CNC_POCKET",
                                "edges": p_edges,
                                "depth": depth
                            }
                        )

        bbox = target_face.BoundingBox()

        return {

            "name": os.path.splitext(
                filename
            )[0],

            "width": bbox.xlen,

            "height": bbox.ylen,

            "outer_edges": outer_edges,

            "features": features

        }

    finally:

        if (
            temp_path
            and
            os.path.exists(
                temp_path
            )
        ):

            os.remove(
                temp_path
            )


# ============================================================
# 9. TRANSFORM
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

    cos_a = math.cos(rad)

    sin_a = math.sin(rad)

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


# ============================================================
# 10. NESTING
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
        x["width"] * x["height"],

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

        poly_geom = Polygon(
            raw_points
        )

        poly_geom = repair_geometry(
            poly_geom
        )

        poly_geom = extract_largest_polygon(
            poly_geom
        )

        if poly_geom is None:

            continue

        buffered_poly = poly_geom.buffer(
            offset_val,
            resolution=8,
            join_style=JOIN_STYLE.round
        )

        buffered_poly = extract_largest_polygon(
            buffered_poly
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

        best_score = float(
            "inf"
        )

        for sheet_idx, sheet_data in enumerate(
            sheets
        ):

            placed_polys = (
                sheet_data[
                    "placed_buffered_polygons"
                ]
            )

            anchors = [

                (
                    margin_val,
                    margin_val
                )
            ]

            for pb in placed_polys:

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

                    collision = any(

                        candidate.intersects(
                            placed
                        )

                        for placed
                        in placed_polys

                    )

                    if collision:

                        continue

                    bounds = candidate.bounds

                    score = (

                        bounds[0]
                        +
                        bounds[1] * 2.5

                    )

                    if score < best_score:

                        best_score = score

                        target_sheet_idx = (
                            sheet_idx
                        )

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

                    "placed_polygon": (

                        best_pos[
                            "raw_trans"
                        ]

                    ),

                    "dx": best_pos["dx"],

                    "dy": best_pos["dy"],

                    "angle": best_pos["angle"]

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

            sheets.append(

                {

                    "sheet_id":
                    new_sheet_id,

                    "parts": [

                        {

                            "part_ref":
                            part,

                            "original_offset": (

                                min_x,

                                min_y

                            ),

                            "placed_polygon":
                            translate(

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

                        translate(

                            normalized_poly,

                            xoff=dx,

                            yoff=dy

                        )

                    ]

                }

            )

    return sheets


# ============================================================
# 11. TRUE OFFSET TOOLPATH
# ============================================================

def get_true_offset_toolpath(
    edges,
    op_type,
    tool_radius
):

    raw_pts = discrete_edges(
        edges
    )

    cleaned = clean_polygon_points(
        raw_pts
    )

    if len(cleaned) < 4:

        return cleaned

    if op_type == "CNC_INNER_CUT":

        cleaned = apply_t_bone_relief(

            cleaned,

            tool_radius

        )

    poly = Polygon(
        cleaned
    )

    poly = repair_geometry(
        poly
    )

    poly = extract_largest_polygon(
        poly
    )

    if poly is None:

        return cleaned

    if op_type == "CNC_OUTER_CUT":

        offset_geom = poly.buffer(

            tool_radius,

            resolution=8,

            join_style=JOIN_STYLE.round

        )

    else:

        offset_geom = poly.buffer(

            -tool_radius,

            resolution=8,

            join_style=JOIN_STYLE.round

        )

    offset_geom = repair_geometry(
        offset_geom
    )

    offset_poly = extract_largest_polygon(
        offset_geom
    )

    if offset_poly is None:

        return cleaned

    return list(
        offset_poly.exterior.coords
    )


# ============================================================
# 12. ARC FITTING
# ============================================================

def fit_arc_from_three_points(
    p1,
    p2,
    p3
):

    p1 = np.array(
        p1,
        dtype=float
    )

    p2 = np.array(
        p2,
        dtype=float
    )

    p3 = np.array(
        p3,
        dtype=float
    )

    D = 2 * (

        p1[0] * (
            p2[1] - p3[1]
        )

        +

        p2[0] * (
            p3[1] - p1[1]
        )

        +

        p3[0] * (
            p1[1] - p2[1]
        )

    )

    if abs(D) < 1e-7:

        return None

    cx = (

        (
            p1[0] ** 2
            +
            p1[1] ** 2
        )
        *
        (
            p2[1]
            -
            p3[1]
        )

        +

        (
            p2[0] ** 2
            +
            p2[1] ** 2
        )
        *
        (
            p3[1]
            -
            p1[1]
        )

        +

        (
            p3[0] ** 2
            +
            p3[1] ** 2
        )
        *
        (
            p1[1]
            -
            p2[1]
        )

    ) / D

    cy = (

        (
            p1[0] ** 2
            +
            p1[1] ** 2
        )
        *
        (
            p3[0]
            -
            p2[0]
        )

        +

        (
            p2[0] ** 2
            +
            p2[1] ** 2
        )
        *
        (
            p1[0]
            -
            p3[0]
        )

        +

        (
            p3[0] ** 2
            +
            p3[1] ** 2
        )
        *
        (
            p2[0]
            -
            p1[0]
        )

    ) / D

    center = np.array(
        [
            cx,
            cy
        ]
    )

    r1 = np.linalg.norm(
        p1 - center
    )

    r2 = np.linalg.norm(
        p2 - center
    )

    r3 = np.linalg.norm(
        p3 - center
    )

    if r1 <= 1e-7:

        return None

    radius_error = max(

        abs(r1 - r2),

        abs(r2 - r3),

        abs(r1 - r3)

    )

    if radius_error > max(
        0.05,
        r1 * 0.01
    ):

        return None

    return {

        "center": center,

        "radius": r1

    }


def fit_arcs_and_emit_gcode(
    pts,
    feed_rate
):

    lines = []

    i = 0

    n = len(pts)

    while i < n - 1:

        if i + 2 < n:

            p1 = pts[i]

            p2 = pts[i + 1]

            p3 = pts[i + 2]

            arc = fit_arc_from_three_points(

                p1,

                p2,

                p3

            )

            if arc is not None:

                p1_np = np.array(
                    p1
                )

                p2_np = np.array(
                    p2
                )

                p3_np = np.array(
                    p3
                )

                cross = (

                    (
                        p2_np[0]
                        -
                        p1_np[0]
                    )
                    *
                    (
                        p3_np[1]
                        -
                        p2_np[1]
                    )

                    -

                    (
                        p2_np[1]
                        -
                        p1_np[1]
                    )
                    *
                    (
                        p3_np[0]
                        -
                        p2_np[0]
                    )

                )

                g_cmd = (
                    "G3"
                    if cross > 0
                    else "G2"
                )

                center = arc[
                    "center"
                ]

                I = (
                    center[0]
                    -
                    p1_np[0]
                )

                J = (
                    center[1]
                    -
                    p1_np[1]
                )

                lines.append(

                    f"{g_cmd} "
                    f"X{p3[0]:.3f} "
                    f"Y{p3[1]:.3f} "
                    f"I{I:.3f} "
                    f"J{J:.3f} "
                    f"F{feed_rate}"

                )

                i += 2

                continue

        next_pt = pts[i + 1]

        lines.append(

            f"G1 "
            f"X{next_pt[0]:.3f} "
            f"Y{next_pt[1]:.3f} "
            f"F{feed_rate}"

        )

        i += 1

    return lines


# ============================================================
# 13. DXF EXPORT
# ============================================================

def transform_edge_point(
    point,
    dx,
    dy,
    angle,
    ox,
    oy
):

    return transform_point_production(

        point[0],

        point[1],

        dx,

        dy,

        angle,

        ox,

        oy

    )


def write_edge_to_dxf(
    msp,
    edge,
    layer,
    dx,
    dy,
    angle,
    ox,
    oy
):

    attribs = {

        "layer":
        layer

    }

    if edge["type"] == "LINE":

        p1 = transform_edge_point(

            edge["start"],

            dx,

            dy,

            angle,

            ox,

            oy

        )

        p2 = transform_edge_point(

            edge["end"],

            dx,

            dy,

            angle,

            ox,

            oy

        )

        msp.add_line(

            p1,

            p2,

            dxfattribs=attribs

        )

    elif edge["type"] == "CIRCLE":

        center = transform_edge_point(

            edge["center"],

            dx,

            dy,

            angle,

            ox,

            oy

        )

        msp.add_circle(

            center=center,

            radius=edge["radius"],

            dxfattribs=attribs

        )

    elif edge["type"] == "DISCRETE_CURVE":

        points = [

            transform_edge_point(

                p,

                dx,

                dy,

                angle,

                ox,

                oy

            )

            for p in edge["points"]

        ]

        for p1, p2 in zip(

            points[:-1],

            points[1:]

        ):

            msp.add_line(

                p1,

                p2,

                dxfattribs=attribs

            )


def generate_dxf_industrial_layered(
    sheet_data,
    sheet_w,
    sheet_h
):

    doc = ezdxf.new(
        dxfversion="R2010"
    )

    msp = doc.modelspace()

    layers = {

        "CNC_SHEET_BORDER": 8,

        "CNC_OUTER_CUT": 1,

        "CNC_INNER_CUT": 4,

        "CNC_POCKET": 6

    }

    for name, color in layers.items():

        if name not in doc.layers:

            doc.layers.new(

                name=name,

                dxfattribs={

                    "color":
                    color

                }

            )

    border = [

        (
            0,
            0
        ),

        (
            sheet_w,
            0
        ),

        (
            sheet_w,
            sheet_h
        ),

        (
            0,
            sheet_h
        ),

        (
            0,
            0
        )

    ]

    for p1, p2 in zip(

        border[:-1],

        border[1:]

    ):

        msp.add_line(

            p1,

            p2,

            dxfattribs={

                "layer":
                "CNC_SHEET_BORDER"

            }

        )

    for placed in sheet_data["parts"]:

        part = placed[
            "part_ref"
        ]

        dx = placed[
            "dx"
        ]

        dy = placed[
            "dy"
        ]

        angle = placed[
            "angle"
        ]

        ox, oy = placed[
            "original_offset"
        ]

        for edge in part[
            "outer_edges"
        ]:

            write_edge_to_dxf(

                msp,

                edge,

                "CNC_OUTER_CUT",

                dx,

                dy,

                angle,

                ox,

                oy

            )

        for feat in part[
            "features"
        ]:

            for edge in feat[
                "edges"
            ]:

                write_edge_to_dxf(

                    msp,

                    edge,

                    feat[
                        "type"
                    ],

                    dx,

                    dy,

                    angle,

                    ox,

                    oy

                )

    out = io.StringIO()

    doc.write(
        out
    )

    return out.getvalue()


# ============================================================
# 14. TOOLPATH HELPERS
# ============================================================

def distance_between(
    p1,
    p2
):

    return math.hypot(

        p2[0] - p1[0],

        p2[1] - p1[1]

    )


def interpolate_point(
    p1,
    p2,
    distance
):

    segment_length = distance_between(
        p1,
        p2
    )

    if segment_length <= 1e-6:

        return p1

    ratio = (
        distance
        /
        segment_length
    )

    ratio = max(
        0.0,
        min(
            1.0,
            ratio
        )
    )

    return (

        p1[0]
        +
        (
            p2[0]
            -
            p1[0]
        )
        *
        ratio,

        p1[1]
        +
        (
            p2[1]
            -
            p1[1]
        )
        *
        ratio

    )


def create_tab_positions(
    points,
    tab_count
):

    if len(points) < 4:

        return []

    segments = []

    total_length = 0.0

    for i in range(
        len(points) - 1
    ):

        p1 = points[i]

        p2 = points[i + 1]

        length = distance_between(
            p1,
            p2
        )

        segments.append(

            {

                "p1": p1,

                "p2": p2,

                "length": length,

                "start":
                total_length,

                "end":
                total_length
                +
                length

            }

        )

        total_length += length

    if total_length <= 1e-6:

        return []

    tab_positions = []

    for i in range(
        tab_count
    ):

        target = (
            total_length
            *
            (
                i + 1
            )
            /
            (
                tab_count
                +
                1
            )
        )

        for seg in segments:

            if (

                seg["start"]
                <= target
                <= seg["end"]

            ):

                local_dist = (

                    target
                    -
                    seg["start"]

                )

                tab_positions.append(

                    interpolate_point(

                        seg["p1"],

                        seg["p2"],

                        local_dist

                    )

                )

                break

    return tab_positions


def point_is_near(
    p1,
    p2,
    tolerance
):

    return (

        distance_between(
            p1,
            p2
        )
        <=
        tolerance

    )


# ============================================================
# 15. INDUSTRIAL TOOLPATH
# ============================================================

def write_industrial_toolpath(
    gcode_list,
    pts,
    total_depth,
    stepdown,
    safe_z,
    feed_rate,
    has_tabs,
    ramp_angle
):

    cleaned = clean_polygon_points(
        pts
    )

    if len(cleaned) < 4:

        return

    p1 = cleaned[0]

    p2 = cleaned[1]

    direction = np.array(

        p2

    ) - np.array(

        p1

    )

    direction_length = np.linalg.norm(
        direction
    )

    if direction_length <= 1e-6:

        return

    unit_direction = (
        direction
        /
        direction_length
    )

    ramp_rad = math.radians(
        ramp_angle
    )

    if math.tan(
        ramp_rad
    ) > 0:

        lead_in_length = (

            stepdown
            /
            math.tan(
                ramp_rad
            )

        )

    else:

        lead_in_length = 15.0

    lead_in_length = min(

        lead_in_length,

        30.0

    )

    lead_in = (

        np.array(
            p1
        )
        -
        unit_direction
        *
        lead_in_length

    )

    lead_in = (

        float(
            lead_in[0]
        ),

        float(
            lead_in[1]
        )

    )

    tab_positions = []

    if has_tabs:

        tab_positions = create_tab_positions(

            cleaned,

            tab_count_default

        )

    current_z = 0.0

    while current_z > -total_depth:

        previous_z = current_z

        current_z -= stepdown

        if current_z < -total_depth:

            current_z = -total_depth

        # --------------------------------
        # RAPID TO LEAD-IN
        # --------------------------------

        gcode_list.append(

            f"G0 Z{safe_z:.3f}"

        )

        gcode_list.append(

            f"G0 X{lead_in[0]:.3f} "
            f"Y{lead_in[1]:.3f}"

        )

        # --------------------------------
        # RAMPING
        # --------------------------------

        gcode_list.append(

            f"G1 X{p1[0]:.3f} "
            f"Y{p1[1]:.3f} "
            f"Z{current_z:.3f} "
            f"F{feed_rate * 0.5:.0f} "
            f"; RAMP {ramp_angle} DEG"

        )

        # --------------------------------
        # TOOLPATH
        # --------------------------------

        idx = 1

        while idx < len(cleaned):

            pt = cleaned[idx]

            is_tab = False

            if (

                has_tabs
                and
                math.isclose(

                    current_z,

                    -total_depth,

                    abs_tol=0.05

                )

            ):

                for tab_point in tab_positions:

                    if point_is_near(

                        pt,

                        tab_point,

                        5.0

                    ):

                        is_tab = True

                        break

            if is_tab:

                tab_z = (

                    current_z
                    +
                    tab_thickness

                )

                if tab_z > 0:

                    tab_z = 0

                gcode_list.append(

                    f"G1 Z{tab_z:.3f} "
                    f"F{feed_rate * 0.4:.0f} "
                    f"; TAB"

                )

                gcode_list.append(

                    f"G1 X{pt[0]:.3f} "
                    f"Y{pt[1]:.3f} "
                    f"F{feed_rate}"

                )

                gcode_list.append(

                    f"G1 Z{current_z:.3f} "
                    f"F{feed_rate * 0.4:.0f}"

                )

                idx += 1

                continue

            # --------------------------------
            # ARC FITTING
            # --------------------------------

            if idx + 1 < len(cleaned):

                arc_lines = fit_arcs_and_emit_gcode(

                    [

                        cleaned[idx - 1],

                        cleaned[idx],

                        cleaned[idx + 1]

                    ],

                    feed_rate

                )

                if any(

                    line.startswith(
                        "G2"
                    )
                    or
                    line.startswith(
                        "G3"
                    )

                    for line in arc_lines

                ):

                    gcode_list.extend(
                        arc_lines
                    )

                    idx += 2

                    continue

            # --------------------------------
            # LINEAR MOVE
            # --------------------------------

            gcode_list.append(

                f"G1 X{pt[0]:.3f} "
                f"Y{pt[1]:.3f} "
                f"F{feed_rate}"

            )

            idx += 1

        gcode_list.append(

            f"G0 Z{safe_z:.3f}"

        )


# ============================================================
# 16. PRODUCTION G-CODE
# ============================================================

def generate_production_atc_gcode(

    sheet_data,

    sheet_th,

    stepdown,

    safe_z,

    overlap,

    dialect,

    ramp_angle

):

    gcode = []

    gcode.append(

        f"; CNC CAM ENGINE PRO V4.2"

    )

    gcode.append(

        f"; MACHINE: {dialect.upper()}"

    )

    gcode.append(

        f"; GENERATED: "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    )

    gcode.append(

        "G21 ; MM"

    )

    gcode.append(

        "G90 ; ABSOLUTE"

    )

    gcode.append(

        "G17 ; XY PLANE"

    )

    gcode.append(

        f"G0 Z{safe_z:.3f}"

    )

    queue_pockets = []

    queue_inners = []

    queue_outers = []

    for placed in sheet_data["parts"]:

        part = placed[
            "part_ref"
        ]

        for feat in part[
            "features"
        ]:

            if feat[
                "type"
            ] == "CNC_POCKET":

                queue_pockets.append(

                    (

                        part,

                        placed,

                        feat

                    )

                )

            elif feat[
                "type"
            ] == "CNC_INNER_CUT":

                queue_inners.append(

                    (

                        part,

                        placed,

                        feat

                    )

                )

        queue_outers.append(

            (

                part,

                placed,

                {

                    "type":
                    "CNC_OUTER_CUT",

                    "edges":
                    part[
                        "outer_edges"
                    ]

                }

            )

        )

    # ========================================================
    # TOOL CHANGE
    # ========================================================

    gcode.append(

        "\n; ===================================="

    )

    gcode.append(

        "; TOOL 1 - CUTTER"

    )

    gcode.append(

        "; ===================================="

    )

    gcode.append(

        "M6 T1"

    )

    gcode.append(

        f"M3 S{int(t1_spindle)}"

    )

    gcode.append(

        f"G0 Z{safe_z:.3f}"

    )

    # ========================================================
    # POCKET
    # ========================================================

    if queue_pockets:

        gcode.append(

            "\n; ===================================="

        )

        gcode.append(

            "; OPERATION 1 - POCKET"

        )

        gcode.append(

            "; ===================================="

        )

        for part, placed, feat in queue_pockets:

            toolpath = get_true_offset_toolpath(

                feat["edges"],

                "CNC_POCKET",

                tool_radius

            )

            transformed = [

                transform_point_production(

                    p[0],

                    p[1],

                    placed["dx"],

                    placed["dy"],

                    placed["angle"],

                    placed[
                        "original_offset"
                    ][0],

                    placed[
                        "original_offset"
                    ][1]

                )

                for p in toolpath

            ]

            write_industrial_toolpath(

                gcode,

                transformed,

                feat["depth"],

                stepdown,

                safe_z,

                t1_feed,

                False,

                ramp_angle

            )

    # ========================================================
    # INNER CUT
    # ========================================================

    if queue_inners:

        gcode.append(

            "\n; ===================================="

        )

        gcode.append(

            "; OPERATION 2 - INNER CUT"

        )

        gcode.append(

            "; ===================================="

        )

        for part, placed, feat in queue_inners:

            toolpath = get_true_offset_toolpath(

                feat["edges"],

                "CNC_INNER_CUT",

                tool_radius

            )

            transformed = [

                transform_point_production(

                    p[0],

                    p[1],

                    placed["dx"],

                    placed["dy"],

                    placed["angle"],

                    placed[
                        "original_offset"
                    ][0],

                    placed[
                        "original_offset"
                    ][1]

                )

                for p in toolpath

            ]

            write_industrial_toolpath(

                gcode,

                transformed,

                sheet_th + overlap,

                stepdown,

                safe_z,

                t1_feed,

                False,

                ramp_angle

            )

    # ========================================================
    # OUTER CUT
    # ========================================================

    if queue_outers:

        gcode.append(

            "\n; ===================================="

        )

        gcode.append(

            "; OPERATION 3 - OUTER CUT + TABS"

        )

        gcode.append(

            "; ===================================="

        )

        for part, placed, feat in queue_outers:

            toolpath = get_true_offset_toolpath(

                feat["edges"],

                "CNC_OUTER_CUT",

                tool_radius

            )

            transformed = [

                transform_point_production(

                    p[0],

                    p[1],

                    placed["dx"],

                    placed["dy"],

                    placed["angle"],

                    placed[
                        "original_offset"
                    ][0],

                    placed[
                        "original_offset"
                    ][1]

                )

                for p in toolpath

            ]

            write_industrial_toolpath(

                gcode,

                transformed,

                sheet_th + overlap,

                stepdown,

                safe_z,

                t1_feed,

                True,

                ramp_angle

            )

    gcode.append(

        "\nM5"

    )

    gcode.append(

        f"G0 Z{safe_z:.3f}"

    )

    gcode.append(

        "M30"

    )

    return "\n".join(
        gcode
    )


# ============================================================
# 17. STREAMLIT USER INTERFACE
# ============================================================

st.markdown("---")

st.subheader(

    "📥 HỆ THỐNG KIỂM ĐỊNH FILE STEP"

)

uploaded_files = st.file_uploader(

    "Nạp file cấu kiện STEP / STP",

    type=[

        "step",

        "stp"

    ],

    accept_multiple_files=True

)


if uploaded_files:

    production_db = []

    progress = st.progress(
        0
    )

    for idx, uploaded_file in enumerate(
        uploaded_files
    ):

        with st.spinner(

            f"Đang phân tích "
            f"{uploaded_file.name}..."

        ):

            try:

                parsed = process_cad_file_production(

                    uploaded_file.read(),

                    uploaded_file.name,

                    sheet_thickness

                )

                production_db.append(
                    parsed
                )

            except Exception as e:

                st.error(

                    f"Lỗi file "
                    f"{uploaded_file.name}: "
                    f"{str(e)}"

                )

        progress.progress(

            (
                idx + 1
            )
            /
            len(
                uploaded_files
            )

        )

    if production_db:

        st.success(

            f"Đã phân tích thành công "
            f"{len(production_db)} chi tiết."

        )

        summary_data = []

        for part in production_db:

            types = [

                feat[
                    "type"
                ]

                for feat
                in part[
                    "features"
                ]

            ]

            summary_data.append(

                {

                    "Mã cấu kiện":
                    part[
                        "name"
                    ],

                    "Chiều rộng X":
                    round(
                        part[
                            "width"
                        ],

                        2

                    ),

                    "Chiều cao Y":
                    round(
                        part[
                            "height"
                        ],

                        2

                    ),

                    "Pocket":
                    types.count(
                        "CNC_POCKET"
                    ),

                    "Inner Cut":
                    types.count(
                        "CNC_INNER_CUT"
                    ),

                    "Độ dày":
                    sheet_thickness

                }

            )

        st.dataframe(

            pd.DataFrame(
                summary_data
            ),

            use_container_width=True

        )

        st.markdown("---")

        st.subheader(

            "🧩 NESTING VÀ TOOLPATH"

        )

        with st.spinner(

            "Đang tối ưu hóa nesting..."

        ):

            sheets_result = execute_production_nesting(

                production_db,

                sheet_W,

                sheet_H,

                total_offset,

                margin

            )

        st.metric(

            "Tổng số tấm ván",

            len(
                sheets_result
            )

        )

        if sheets_result:

            tabs = st.tabs(

                [

                    f"TẤM #{sheet['sheet_id']}"

                    for sheet
                    in sheets_result

                ]

            )

            for idx, sheet in enumerate(
                sheets_result
            ):

                with tabs[idx]:

                    col_graph, col_output = st.columns(

                        [

                            3,

                            2

                        ]

                    )

                    # ====================================================
                    # VISUALIZATION
                    # ====================================================

                    with col_graph:

                        st.markdown(

                            "##### Sơ đồ Nesting"

                        )

                        fig, ax = plt.subplots(

                            figsize=(

                                12,

                                6

                            )

                        )

                        ax.set_xlim(

                            -50,

                            sheet_W + 50

                        )

                        ax.set_ylim(

                            -50,

                            sheet_H + 50

                        )

                        ax.set_aspect(
                            "equal"
                        )

                        ax.add_patch(

                            mpatches.Rectangle(

                                (

                                    0,

                                    0

                                ),

                                sheet_W,

                                sheet_H,

                                fill=False,

                                linewidth=2,

                                linestyle="--"

                            )

                        )

                        for placed in sheet[
                            "parts"
                        ]:

                            poly = placed[
                                "placed_polygon"
                            ]

                            part = placed[
                                "part_ref"
                            ]

                            if isinstance(

                                poly,

                                Polygon

                            ):

                                x, y = poly.exterior.xy

                                ax.fill(

                                    x,

                                    y,

                                    alpha=0.5

                                )

                                ax.plot(

                                    x,

                                    y

                                )

                                ax.text(

                                    poly.centroid.x,

                                    poly.centroid.y,

                                    part[
                                        "name"
                                    ],

                                    fontsize=8,

                                    ha="center",

                                    va="center"

                                )

                                for feat in part[
                                    "features"
                                ]:

                                    feature_points = [

                                        transform_point_production(

                                            p[0],

                                            p[1],

                                            placed[
                                                "dx"
                                            ],

                                            placed[
                                                "dy"
                                            ],

                                            placed[
                                                "angle"
                                            ],

                                            placed[
                                                "original_offset"
                                            ][0],

                                            placed[
                                                "original_offset"
                                            ][1]

                                        )

                                        for p

                                        in discrete_edges(

                                            feat[
                                                "edges"
                                            ]

                                        )

                                    ]

                                    feature_points = clean_polygon_points(

                                        feature_points

                                    )

                                    if len(

                                        feature_points

                                    ) >= 2:

                                        fx, fy = zip(

                                            *feature_points

                                        )

                                        ax.plot(

                                            fx,

                                            fy,

                                            linewidth=1

                                        )

                        st.pyplot(
                            fig
                        )

                        plt.close(
                            fig
                        )

                    # ====================================================
                    # OUTPUT
                    # ====================================================

                    with col_output:

                        st.markdown(

                            "##### 💾 XUẤT FILE CNC"

                        )

                        dxf_data = generate_dxf_industrial_layered(

                            sheet,

                            sheet_W,

                            sheet_H

                        )

                        st.download_button(

                            label=(

                                f"📥 Tải DXF "
                                f"Tấm #{sheet['sheet_id']}"

                            ),

                            data=dxf_data,

                            file_name=(

                                f"Factory_Sheet_"
                                f"{sheet['sheet_id']}.dxf"

                            ),

                            mime="image/vnd.dxf",

                            key=(

                                f"dxf_"
                                f"{sheet['sheet_id']}"

                            )

                        )

                        gcode_data = generate_production_atc_gcode(

                            sheet,

                            sheet_thickness,

                            max_stepdown,

                            safe_Z,

                            thru_overlap,

                            cnc_dialect,

                            ramp_angle

                        )

                        st.download_button(

                            label=(

                                f"📟 Tải G-code "
                                f"Tấm #{sheet['sheet_id']}"

                            ),

                            data=gcode_data,

                            file_name=(

                                f"ATC_Sheet_"
                                f"{sheet['sheet_id']}.nc"

                            ),

                            mime="text/plain",

                            key=(

                                f"gcode_"
                                f"{sheet['sheet_id']}"

                            )

                        )

                        with st.expander(

                            "Xem trước G-code"

                        ):

                            st.code(

                                gcode_data[
                                    :3000
                                ],

                                language="gcode"

                            )

else:

    st.info(

        "💡 Vui lòng tải file STEP/STP để bắt đầu."

    )
