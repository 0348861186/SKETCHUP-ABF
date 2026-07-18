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

        return 0

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

            "features": features,
            
            "origin_x": target_face.Center().x,
            
            "origin_y": target_face.Center().y

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
        (0.0, 0.0),
        (sheet_w, 0.0),
        (sheet_w, sheet_h),
        (0.0, sheet_h),
        (0.0, 0.0)
    ]
    
    for p1, p2 in zip(border[:-1], border[1:]):
        msp.add_line(
            p1, 
            p2, 
            dxfattribs={"layer": "CNC_SHEET_BORDER"}
        )
        
    for p_data in sheet_data["parts"]:
        part = p_data["part_ref"]
        dx, dy, angle = p_data["dx"], p_data["dy"], p_data["angle"]
        ox = part.get("origin_x", 0.0)
        oy = part.get("origin_y", 0.0)
        
        for edge in part["outer_edges"]:
            write_edge_to_dxf(
                msp, edge, "CNC_OUTER_CUT", 
                dx, dy, angle, ox, oy
            )
            
        for feat in part["features"]:
            for edge in feat["edges"]:
                write_edge_to_dxf(
                    msp, edge, feat["type"], 
                    dx, dy, angle, ox, oy
                )
                
    out_buf = io.StringIO()
    doc.write(out_buf)
    return out_buf.getvalue()


# ============================================================
# 14. G-CODE GENERATION ENGINE
# ============================================================

def generate_sheet_gcode(
    sheet_data, 
    sheet_thick, 
    safe_z, 
    thru_z, 
    stepdown, 
    feed, 
    spindle, 
    dialect
):
    lines = [
        f"( G-CODE GENERATED BY CAM ENGINE PRO V4.2 )",
        f"( DATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} )",
        f"G21 G90 G17 G40 G49 M03 S{spindle}"
    ]
    
    total_depth = sheet_thick + thru_z
    
    # Thứ tự công đoạn gia công an toàn: POCKET -> INNER_CUT -> OUTER_CUT
    for target_op in ["CNC_POCKET", "CNC_INNER_CUT", "CNC_OUTER_CUT"]:
        lines.append(f"({target_op} OPERATIONS ROUTINE)")
        
        for p_data in sheet_data["parts"]:
            part = p_data["part_ref"]
            dx, dy, angle = p_data["dx"], p_data["dy"], p_data["angle"]
            ox = part.get("origin_x", 0.0)
            oy = part.get("origin_y", 0.0)
            
            profiles = []
            if target_op == "CNC_OUTER_CUT":
                profiles.append({
                    "edges": part["outer_edges"], 
                    "depth": total_depth
                })
            else:
                for feat in part["features"]:
                    if feat["type"] == target_op:
                        d = total_depth if target_op == "CNC_INNER_CUT" else feat["depth"]
                        profiles.append({
                            "edges": feat["edges"], 
                            "depth": d
                        })
                        
            for prof in profiles:
                raw_offset_pts = get_true_offset_toolpath(
                    prof["edges"], target_op, tool_radius
                )
                if not raw_offset_pts:
                    continue
                
                global_pts = [
                    transform_point_production(
                        pt[0], pt[1], dx, dy, angle, ox, oy
                    ) 
                    for pt in raw_offset_pts
                ]
                
                z_limit = prof["depth"]
                z_curr = 0.0
                start_pt = global_pts[0]
                
                lines.append(
                    f"G0 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{safe_z:.3f}"
                )
                
                while z_curr < z_limit:
                    z_curr = min(z_curr + stepdown, z_limit)
                    lines.append(f"G1 Z{-z_curr:.3f} F{feed/2:.0f}")
                    
                    contour_lines = fit_arcs_and_emit_gcode(global_pts, feed)
                    lines.extend(contour_lines)
                    
                lines.append(f"G0 Z{safe_z:.3f}")
                
    lines.append("M05 M30")
    return "\n".join(lines)


# ============================================================
# 15. CONTROLLER & USER INTERFACE LOOP
# ============================================================

uploaded_files = st.file_uploader(
    "Tải lên các tệp thiết kế mẫu STEP (.step / .stp)", 
    type=["step", "stp"], 
    accept_multiple_files=True
)

if uploaded_files:
    parsed_parts = []
    progress = st.progress(0.0)
    
    for idx, f in enumerate(uploaded_files):
        try:
            bytes_data = f.read()
            part_info = process_cad_file_production(
                bytes_data, f.name, sheet_thickness
            )
            parsed_parts.append(part_info)
        except Exception as e:
            st.error(f"Lỗi hệ thống khi đọc cấu trúc tệp {f.name}: {str(e)}")
        progress.progress((idx + 1) / len(uploaded_files))
        
    if parsed_parts:
        st.success(f"Đã trích xuất hình học thành công {len(parsed_parts)} chi tiết CAD.")
        
        df_summary = pd.DataFrame([{
            "Tên chi tiết": p["name"],
            "Rộng X (mm)": round(p["width"], 2),
            "Cao Y (mm)": round(p["height"], 2),
            "Tính năng phụ": len(p["features"])
        } for p in parsed_parts])
        st.dataframe(df_summary, use_container_width=True)
        
        with st.spinner("Đang chạy thuật toán sắp xếp ván hình học tối ưu..."):
            nesting_sheets = execute_production_nesting(
                parsed_parts, sheet_W, sheet_H, total_offset, margin
            )
            
        st.subheader(f"📊 KẾT QUẢ TỐI ƯU XẾP VÁN ({len(nesting_sheets)} TẤM PHÔI)")
        
        for s_idx, sheet in enumerate(nesting_sheets):
            st.markdown(f"#### 📄 Tấm phôi thứ {sheet['sheet_id']}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlim(-50, sheet_W + 50)
            ax.set_ylim(-50, sheet_H + 50)
            ax.set_aspect('equal')
            
            ax.add_patch(
                mpatches.Rectangle(
                    (0, 0), sheet_W, sheet_H, 
                    fill=False, edgecolor='black', linewidth=2
                )
            )
            ax.add_patch(
                mpatches.Rectangle(
                    (margin, margin), sheet_W - 2*margin, sheet_H - 2*margin, 
                    fill=False, edgecolor='gray', linestyle='--'
                )
            )
            
            for p_data in sheet["parts"]:
                poly = p_data["placed_polygon"]
                if isinstance(poly, Polygon) and not poly.is_empty:
                    x, y = poly.exterior.xy
                    ax.fill(
                        x, y, alpha=0.4, 
                        facecolor='#0EA5E9', edgecolor='#0284C7', linewidth=1.5
                    )
                    ax.text(
                        poly.centroid.x, poly.centroid.y, 
                        p_data["part_ref"]["name"], 
                        fontsize=8, ha='center', va='center'
                    )
                    
            st.pyplot(fig)
            
            col1, col2 = st.columns(2)
            
            gcode_str = generate_sheet_gcode(
                sheet, sheet_thickness, safe_Z, thru_overlap, 
                max_stepdown, t1_feed, t1_spindle, cnc_dialect
            )
            col1.download_button(
                label=f"💾 Tải ATC G-Code Tấm {sheet['sheet_id']}",
                data=gcode_str,
                file_name=f"CAM_Engine_Sheet_{sheet['sheet_id']}.nc",
                mime="text/plain"
            )
            
            dxf_str = generate_dxf_industrial_layered(
                sheet, sheet_W, sheet_H
            )
            col2.download_button(
                label=f"📐 Tải Bản Vẽ DXF Layer Tấm {sheet['sheet_id']}",
                data=dxf_str,
                file_name=f"CAM_Engine_Sheet_{sheet['sheet_id']}.dxf",
                mime="image/vnd.dxf"
            )
