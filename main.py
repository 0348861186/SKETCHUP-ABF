import io
import os
import math
import tempfile
import json
import zipfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CAD / 3D
# ============================================================

import cadquery as cq
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE

try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None

import ezdxf

# ============================================================
# 3D VIEW
# ============================================================

import pyvista as pv
from stpyvista import stpyvista


# ============================================================
# 1. CẤU HÌNH STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Auto CNC Industrial Nesting Pro",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#1E3A8A;'>
    🏭 AUTO CNC INDUSTRIAL NESTING PRO
    </h1>

    <p style='text-align:center;'>
    STEP 3D → Phân tích hình học → Nesting → Phân loại CNC → DXF theo Layer → Aspire → G-code
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 2. SIDEBAR - THÔNG SỐ VẬT LIỆU
# ============================================================

st.sidebar.header("📐 KHỔ VÁN")

sheet_W = st.sidebar.number_input(
    "Chiều rộng khổ ván X (mm)",
    min_value=100.0,
    value=2440.0,
    step=1.0
)

sheet_H = st.sidebar.number_input(
    "Chiều cao khổ ván Y (mm)",
    min_value=100.0,
    value=1220.0,
    step=1.0
)

sheet_thickness = st.sidebar.number_input(
    "Độ dày ván (mm)",
    min_value=0.1,
    value=17.0,
    step=0.1
)

margin = st.sidebar.number_input(
    "Chừa biên tấm ván (mm)",
    min_value=0.0,
    value=15.0,
    step=1.0
)

safety_spacing = st.sidebar.number_input(
    "Khoảng cách an toàn giữa chi tiết (mm)",
    min_value=0.0,
    value=4.0,
    step=0.5
)


# ============================================================
# 3. SIDEBAR - DAO CẮT
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("🔧 DAO CẮT")

tool_diameter = st.sidebar.number_input(
    "Đường kính dao cắt (mm)",
    min_value=0.1,
    value=6.0,
    step=0.1
)

tool_radius = tool_diameter / 2.0

# Khoảng cách dùng để tránh va chạm giữa 2 chi tiết
total_offset = tool_radius + safety_spacing


# ============================================================
# 4. SIDEBAR - PHÂN LOẠI LỖ
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("🕳 PHÂN LOẠI LỖ")

hole_drill_diameter = st.sidebar.number_input(
    "Đường kính tối đa xem là lỗ khoan (mm)",
    min_value=0.1,
    value=20.0,
    step=0.5
)


# ============================================================
# 5. SIDEBAR - CHẾ ĐỘ BÙ DAO
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("⚙️ BÙ DAO")

compensation_mode = st.sidebar.selectbox(
    "Phương thức bù bán kính dao",
    [
        "Bù dao trong CAM Aspire / VCarve",
        "Xuất đường tâm dao trực tiếp"
    ]
)


# ============================================================
# 6. SIDEBAR - QUY TẮC GIA CÔNG
# ============================================================

st.sidebar.markdown("---")
st.sidebar.header("🧠 QUY TẮC CNC")

st.sidebar.info(
    """
    Hệ thống tự động phân loại:

    🔴 CNC_OUTER_CUT
    → Cắt biên ngoài

    🟢 CNC_INNER_CUT
    → Cắt biên trong

    🔵 CNC_INNER_DRILL
    → Khoan lỗ tròn

    🟣 CNC_ENGRAVE
    → Khắc / đường tâm

    🟠 CNC_POCKET
    → Phay lòng / pocket
    """
)


# ============================================================
# 7. HÀM SỬA HÌNH HỌC
# ============================================================

def repair_geometry(geometry):

    if geometry is None:
        return geometry

    if geometry.is_empty:
        return geometry

    if geometry.is_valid:
        return geometry

    if make_valid is not None:

        try:
            return make_valid(geometry)

        except Exception:
            pass

    try:
        return geometry.buffer(0)

    except Exception:
        return geometry


# ============================================================
# 8. ĐỌC FILE STEP
# ============================================================

def process_cad_file_with_occ(
    file_bytes,
    filename,
    hole_drill_diameter
):

    temp_path = None
    temp_stl_path = None

    try:

        # ----------------------------------------------------
        # Ghi file STEP tạm
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(filename)[1]
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name


        # ----------------------------------------------------
        # Đọc STEP bằng CadQuery
        # ----------------------------------------------------

        part = cq.importers.importStep(temp_path)

        all_faces = part.faces().vals()

        if not all_faces:

            raise ValueError(
                "Không tìm thấy bề mặt trong file STEP."
            )


        # ----------------------------------------------------
        # Xuất STL phục vụ xem 3D
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".stl"
        ) as temp_stl:

            temp_stl_path = temp_stl.name


        cq.exporters.export(
            part,
            temp_stl_path,
            cq.exporters.ExportTypes.STL
        )


        # ----------------------------------------------------
        # Tìm mặt phẳng lớn nhất
        # ----------------------------------------------------

        planar_faces = [

            face

            for face in all_faces

            if face.geomType() == "PLANE"

        ]

        if not planar_faces:

            planar_faces = all_faces


        target_face = max(
            planar_faces,
            key=lambda face: face.Area()
        )


        # ----------------------------------------------------
        # Phân tích biên dạng ngoài
        # ----------------------------------------------------

        outer_wire = target_face.outerWire()

        outer_edges = parse_wire_edges_high_precision(
            outer_wire
        )


        # ----------------------------------------------------
        # Phân tích các lỗ bên trong
        # ----------------------------------------------------

        holes = []

        for inner_wire in target_face.innerWires():

            hole_edges = parse_wire_edges_high_precision(
                inner_wire
            )

            if not hole_edges:

                continue


            is_pure_circle = (

                len(hole_edges) == 1

                and hole_edges[0]["type"] == "CIRCLE"

            )


            radius = (

                hole_edges[0]["radius"]

                if is_pure_circle

                else 0

            )


            holes.append(

                {

                    "edges": hole_edges,

                    "is_drill": (

                        is_pure_circle

                        and radius * 2

                        <= hole_drill_diameter

                    ),

                    "radius": radius

                }

            )


        # ----------------------------------------------------
        # Kích thước mặt phẳng
        # ----------------------------------------------------

        bbox = target_face.BoundingBox()


        # ----------------------------------------------------
        # Kích thước toàn bộ khối 3D
        # ----------------------------------------------------

        total_bbox = part.val().BoundingBox()


        thickness = (

            total_bbox.zlen

            if total_bbox.zlen > 0

            else 17.0

        )


        return {

            "name": os.path.splitext(filename)[0],

            "width": bbox.xlen,

            "height": bbox.ylen,

            "thickness": thickness,

            "outer_edges": outer_edges,

            "holes": holes,

            "stl_path": temp_stl_path

        }


    finally:

        if temp_path and os.path.exists(temp_path):

            os.remove(temp_path)


# ============================================================
# 9. ĐỌC EDGE CAD
# ============================================================

def parse_wire_edges_high_precision(
    wire,
    tolerance=0.05
):

    edges_data = []


    for edge in wire.Edges():

        g_type = edge.geomType()


        start = edge.startPoint()

        end = edge.endPoint()


        # ----------------------------------------------------
        # LINE
        # ----------------------------------------------------

        if g_type == "LINE":

            edges_data.append(

                {

                    "type": "LINE",

                    "start": (

                        start.x,

                        start.y

                    ),

                    "end": (

                        end.x,

                        end.y

                    )

                }

            )


        # ----------------------------------------------------
        # CIRCLE / ARC
        # ----------------------------------------------------

        elif g_type == "CIRCLE":

            radius = edge.radius()

            center = edge.Center()


            if (

                start - end

            ).Length < 1e-4:

                edges_data.append(

                    {

                        "type": "CIRCLE",

                        "center": (

                            center.x,

                            center.y

                        ),

                        "radius": radius

                    }

                )


            else:

                edges_data.append(

                    {

                        "type": "ARC",

                        "center": (

                            center.x,

                            center.y

                        ),

                        "radius": radius,

                        "start_angle": math.degrees(

                            math.atan2(

                                start.y - center.y,

                                start.x - center.x

                            )

                        ),

                        "end_angle": math.degrees(

                            math.atan2(

                                end.y - center.y,

                                end.x - center.x

                            )

                        )

                    }

                )


        # ----------------------------------------------------
        # BSPLINE / BEZIER
        # ----------------------------------------------------

        elif g_type in [

            "BSPLINE",

            "BEZIER",

            "OFFSET"

        ]:

            try:

                occ_curve = edge.ToAdaptor3d()


                first_p = occ_curve.FirstParameter()

                last_p = occ_curve.LastParameter()


                segments = max(

                    16,

                    min(

                        200,

                        int(

                            edge.Length()

                            / tolerance

                        )

                    )

                )


                pts = []


                for i in range(

                    segments + 1

                ):

                    t = (

                        first_p

                        + (

                            last_p - first_p

                        )

                        * i

                        / segments

                    )


                    p = edge.valueAt(t)


                    pts.append(

                        (

                            p.x,

                            p.y

                        )

                    )


                for i in range(

                    len(pts) - 1

                ):

                    edges_data.append(

                        {

                            "type": "LINE",

                            "start": pts[i],

                            "end": pts[i + 1]

                        }

                    )


            except Exception:

                pass


    return edges_data


# ============================================================
# 10. ARC → POINTS
# ============================================================

def sample_arc(

    center_x,

    center_y,

    radius,

    start_angle,

    end_angle,

    segments=32

):

    if end_angle < start_angle:

        end_angle += 360


    return [

        (

            center_x

            + radius

            * math.cos(

                math.radians(t)

            ),

            center_y

            + radius

            * math.sin(

                math.radians(t)

            )

        )

        for t in np.linspace(

            start_angle,

            end_angle,

            segments

        )

    ]


# ============================================================
# 11. EDGE → POINTS
# ============================================================

def edges_to_points(

    edges,

    circle_segments=64,

    arc_segments=32

):

    points = []


    for edge in edges:


        if edge["type"] == "LINE":

            points.extend(

                [

                    edge["start"],

                    edge["end"]

                ]

            )


        elif edge["type"] == "ARC":

            points.extend(

                sample_arc(

                    edge["center"][0],

                    edge["center"][1],

                    edge["radius"],

                    edge["start_angle"],

                    edge["end_angle"],

                    arc_segments

                )

            )


        elif edge["type"] == "CIRCLE":

            cx, cy = edge["center"]

            r = edge["radius"]


            points.extend(

                [

                    (

                        cx

                        + r

                        * math.cos(

                            math.radians(t)

                        ),

                        cy

                        + r

                        * math.sin(

                            math.radians(t)

                        )

                    )

                    for t in np.linspace(

                        0,

                        360,

                        circle_segments,

                        endpoint=False

                    )

                ]

            )


    return points


# ============================================================
# 12. LÀM SẠCH ĐIỂM
# ============================================================

def clean_points(

    points,

    snap_distance=0.01

):

    cleaned = []


    for p in points:

        if (

            not cleaned

            or not np.allclose(

                cleaned[-1],

                p,

                atol=snap_distance

            )

        ):

            cleaned.append(p)


    if (

        len(cleaned) > 2

        and not np.allclose(

            cleaned[0],

            cleaned[-1],

            atol=snap_distance

        )

    ):

        cleaned.append(

            cleaned[0]

        )


    return cleaned


# ============================================================
# 13. TẠO POLYGON
# ============================================================

def build_shapely_polygon_fixed(

    part,

    snap_distance=0.01

):

    outer_points = clean_points(

        edges_to_points(

            part["outer_edges"]

        ),

        snap_distance

    )


    if len(outer_points) < 3:

        return Polygon(

            [

                (0, 0),

                (part["width"], 0),

                (

                    part["width"],

                    part["height"]

                ),

                (0, part["height"])

            ]

        )


    interiors = []


    for hole in part["holes"]:

        pts = clean_points(

            edges_to_points(

                hole["edges"]

            ),

            snap_distance

        )


        if len(pts) >= 3:

            interiors.append(pts)


    polygon = Polygon(

        outer_points,

        interiors

    )


    return repair_geometry(

        polygon

    )


# ============================================================
# 14. XÁC ĐỊNH LOẠI GIA CÔNG
# ============================================================

def classify_hole_operation(hole):

    if hole["is_drill"]:

        return "CNC_INNER_DRILL"

    return "CNC_INNER_CUT"


# ============================================================
# 15. TẠO DANH SÁCH TÁC VỤ
# ============================================================

def analyze_part_operations(part):

    operations = []


    # --------------------------------------------------------
    # Biên ngoài
    # --------------------------------------------------------

    operations.append(

        {

            "operation": "OUTER_CUT",

            "layer": "CNC_OUTER_CUT",

            "description": "Cắt biên ngoài",

            "tool_type": "End Mill"

        }

    )


    # --------------------------------------------------------
    # Các lỗ
    # --------------------------------------------------------

    for index, hole in enumerate(

        part["holes"],

        start=1

    ):

        operation = classify_hole_operation(

            hole

        )


        if operation == "CNC_INNER_DRILL":

            operations.append(

                {

                    "operation": "DRILL",

                    "layer": "CNC_INNER_DRILL",

                    "description": (

                        f"Lỗ khoan {index}"

                    ),

                    "tool_type": "Drill"

                }

            )

        else:

            operations.append(

                {

                    "operation": "INNER_CUT",

                    "layer": "CNC_INNER_CUT",

                    "description": (

                        f"Cắt biên dạng trong {index}"

                    ),

                    "tool_type": "End Mill"

                }

            )


    return operations


# ============================================================
# 16. NESTING
# ============================================================

def perform_advanced_best_fit_nesting(

    parts_list,

    sheet_w,

    sheet_h,

    offset_val,

    margin_val

):

    sheet_boundary = Polygon(

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


    # Sắp xếp chi tiết lớn trước

    sorted_parts = sorted(

        parts_list,

        key=lambda x:

        x["width"]

        * x["height"],

        reverse=True

    )


    nested_sheets = []


    angles_to_try = [

        0,

        90,

        180,

        270

    ]


    for part in sorted_parts:


        poly_geom = build_shapely_polygon_fixed(

            part

        )


        buffered_poly = repair_geometry(

            poly_geom.buffer(

                offset_val,

                resolution=16,

                join_style=JOIN_STYLE.round

            )

        )


        min_x, min_y, _, _ = buffered_poly.bounds


        normalized_poly = translate(

            buffered_poly,

            xoff=-min_x,

            yoff=-min_y

        )


        raw_normalized_poly = translate(

            poly_geom,

            xoff=-min_x,

            yoff=-min_y

        )


        best_position = None

        best_sheet_idx = -1

        best_score = float("inf")


        # ====================================================
        # Tìm vị trí tốt nhất trong các tấm hiện có
        # ====================================================

        for s_idx, sheet_info in enumerate(

            nested_sheets

        ):


            placed_polys = (

                sheet_info[

                    "placed_buffered_polygons"

                ]

            )


            anchor_points = [

                (

                    margin_val,

                    margin_val

                )

            ]


            for p_poly in placed_polys:


                p_minx, p_miny, p_maxx, p_maxy = (

                    p_poly.bounds

                )


                anchor_points.extend(

                    [

                        (

                            p_maxx,

                            p_miny

                        ),

                        (

                            p_minx,

                            p_maxy

                        ),

                        (

                            p_maxx,

                            p_maxy

                        )

                    ]

                )


            )


            anchor_points = list(

                set(anchor_points)

            )


            for angle in angles_to_try:


                rot_poly = rotate(

                    normalized_poly,

                    angle,

                    origin=(0, 0)

                )


                r_minx, r_miny, _, _ = (

                    rot_poly.bounds

                )


                for ax, ay in anchor_points:


                    candidate_poly = translate(

                        rot_poly,

                        xoff=ax - r_minx,

                        yoff=ay - r_miny

                    )


                    # Không vượt khổ ván

                    if not sheet_boundary.covers(

                        candidate_poly

                    ):

                        continue


                    # Không giao nhau

                    if any(

                        candidate_poly.intersects(

                            p_poly

                        )

                        for p_poly in placed_polys

                    ):

                        continue


                    # Tính diện tích bao

                    all_x = []

                    all_y = []


                    for p_poly in placed_polys:

                        b = p_poly.bounds

                        all_x.extend(

                            [

                                b[0],

                                b[2]

                            ]

                        )

                        all_y.extend(

                            [

                                b[1],

                                b[3]

                            ]

                        )


                    b = candidate_poly.bounds


                    all_x.extend(

                        [

                            b[0],

                            b[2]

                        ]

                    )


                    all_y.extend(

                        [

                            b[1],

                            b[3]

                        ]

                    )


                    envelope_area = (

                        max(all_x)

                        - min(all_x)

                    ) * (

                        max(all_y)

                        - min(all_y)

                    )


                    if envelope_area < best_score:


                        best_score = envelope_area


                        best_sheet_idx = s_idx


                        best_position = {

                            "dx": ax - r_minx,

                            "dy": ay - r_miny,

                            "angle": angle,

                            "candidate_poly": candidate_poly,

                            "raw_poly_transformed":

                            translate(

                                rotate(

                                    raw_normalized_poly,

                                    angle,

                                    origin=(0, 0)

                                ),

                                xoff=ax - r_minx,

                                yoff=ay - r_miny

                            )

                        }


        # ====================================================
        # Đưa vào tấm đã tồn tại
        # ====================================================

        if (

            best_position

            and best_sheet_idx != -1

        ):


            sheet_info = nested_sheets[

                best_sheet_idx

            ]


            sheet_info["parts"].append(

                {

                    "part_ref": part,

                    "original_offset": (

                        min_x,

                        min_y

                    ),

                    "placed_polygon": best_position[

                        "raw_poly_transformed"

                    ],

                    "dx": best_position["dx"],

                    "dy": best_position["dy"],

                    "angle": best_position["angle"]

                }

            )


            sheet_info[

                "placed_buffered_polygons"

            ].append(

                best_position[

                    "candidate_poly"

                ]

            )


        # ====================================================
        # Tạo tấm mới
        # ====================================================

        else:


            new_idx = len(

                nested_sheets

            ) + 1


            rotated_poly = rotate(

                normalized_poly,

                0,

                origin=(0, 0)

            )


            r_minx, r_miny, _, _ = (

                rotated_poly.bounds

            )


            dx = margin_val - r_minx

            dy = margin_val - r_miny


            nested_sheets.append(

                {

                    "sheet_id": new_idx,

                    "parts": [

                        {

                            "part_ref": part,

                            "original_offset": (

                                min_x,

                                min_y

                            ),

                            "placed_polygon":

                            translate(

                                rotate(

                                    raw_normalized_poly,

                                    0,

                                    origin=(0, 0)

                                ),

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

                            rotated_poly,

                            xoff=dx,

                            yoff=dy

                        )

                    ]

                }

            )


    return nested_sheets


# ============================================================
# 17. BIẾN ĐỔI TỌA ĐỘ
# ============================================================

def transform_point(

    x,

    y,

    dx,

    dy,

    angle,

    orig_x,

    orig_y

):

    tx = x - orig_x

    ty = y - orig_y


    rad = math.radians(

        angle

    )


    return (

        tx * math.cos(rad)

        - ty * math.sin(rad)

        + dx,

        tx * math.sin(rad)

        + ty * math.cos(rad)

        + dy

    )


# ============================================================
# 18. TẠO LAYER DXF
# ============================================================

def create_cnc_layers(doc):

    layers = {

        "CNC_SHEET_BORDER": 8,

        "CNC_OUTER_CUT": 1,

        "CNC_INNER_CUT": 4,

        "CNC_INNER_DRILL": 2,

        "CNC_ENGRAVE": 5,

        "CNC_POCKET": 6

    }


    for layer_name, color in layers.items():

        if layer_name not in doc.layers:

            doc.layers.new(

                name=layer_name,

                dxfattribs={

                    "color": color

                }

            )


# ============================================================
# 19. TẠO BIÊN TẤM VÁN
# ============================================================

def add_sheet_border(

    msp,

    ox,

    sheet_w,

    sheet_h

):

    points = [

        (

            ox,

            0

        ),

        (

            ox + sheet_w,

            0

        ),

        (

            ox + sheet_w,

            sheet_h

        ),

        (

            ox,

            sheet_h

        ),

        (

            ox,

            0

        )

    ]


    for p1, p2 in zip(

        points[:-1],

        points[1:]

    ):

        msp.add_line(

            p1,

            p2,

            dxfattribs={

                "layer": "CNC_SHEET_BORDER"

            }

        )


# ============================================================
# 20. GHI EDGE RA DXF
# ============================================================

def write_edge_to_dxf(

    msp,

    edge,

    layer,

    dx,

    dy,

    angle,

    orig_x,

    orig_y

):


    if edge["type"] == "LINE":


        p1 = transform_point(

            edge["start"][0],

            edge["start"][1],

            dx,

            dy,

            angle,

            orig_x,

            orig_y

        )


        p2 = transform_point(

            edge["end"][0],

            edge["end"][1],

            dx,

            dy,

            angle,

            orig_x,

            orig_y

        )


        msp.add_line(

            p1,

            p2,

            dxfattribs={

                "layer": layer

            }

        )


    elif edge["type"] == "CIRCLE":


        center = transform_point(

            edge["center"][0],

            edge["center"][1],

            dx,

            dy,

            angle,

            orig_x,

            orig_y

        )


        msp.add_circle(

            center=center,

            radius=edge["radius"],

            dxfattribs={

                "layer": layer

            }

        )


    elif edge["type"] == "ARC":


        center = transform_point(

            edge["center"][0],

            edge["center"][1],

            dx,

            dy,

            angle,

            orig_x,

            orig_y

        )


        start_angle = (

            edge["start_angle"]

            + angle

        )


        end_angle = (

            edge["end_angle"]

            + angle

        )


        msp.add_arc(

            center=center,

            radius=edge["radius"],

            start_angle=start_angle,

            end_angle=end_angle,

            dxfattribs={

                "layer": layer

            }

        )


# ============================================================
# 21. XUẤT GEOMETRY THEO LAYER
# ============================================================

def export_original_geometry_to_dxf(

    msp,

    nested_sheets,

    sheet_w,

    sheet_h

):

    sheet_offset_x = 0


    for sheet_data in nested_sheets:


        ox = sheet_offset_x


        # ----------------------------------------------------
        # Biên tấm
        # ----------------------------------------------------

        add_sheet_border(

            msp,

            ox,

            sheet_w,

            sheet_h

        )


        # ----------------------------------------------------
        # Từng chi tiết
        # ----------------------------------------------------

        for p_node in sheet_data["parts"]:


            ref = p_node["part_ref"]


            dx = (

                p_node["dx"]

                + ox

            )


            dy = p_node["dy"]


            angle = p_node["angle"]


            orig_x, orig_y = (

                p_node["original_offset"]

            )


            # =================================================
            # BIÊN NGOÀI
            # =================================================

            for edge in ref["outer_edges"]:


                write_edge_to_dxf(

                    msp,

                    edge,

                    "CNC_OUTER_CUT",

                    dx,

                    dy,

                    angle,

                    orig_x,

                    orig_y

                )


            # =================================================
            # LỖ BÊN TRONG
            # =================================================

            for hole in ref["holes"]:


                if hole["is_drill"]:

                    layer = (

                        "CNC_INNER_DRILL"

                    )

                else:

                    layer = (

                        "CNC_INNER_CUT"

                    )


                for edge in hole["edges"]:


                    write_edge_to_dxf(

                        msp,

                        edge,

                        layer,

                        dx,

                        dy,

                        angle,

                        orig_x,

                        orig_y

                    )


        # ----------------------------------------------------
        # Khoảng cách giữa các tấm
        # ----------------------------------------------------

        sheet_offset_x += (

            sheet_w

            + 300

        )


# ============================================================
# 22. TẠO FILE CSV MAPPING TOOLPATH
# ============================================================

def create_toolpath_mapping(

    nested_sheets

):

    rows = []


    for sheet in nested_sheets:


        for p_node in sheet["parts"]:


            part = p_node["part_ref"]


            part_name = part["name"]


            # ----------------------------------------------
            # Biên ngoài
            # ----------------------------------------------

            rows.append(

                {

                    "Sheet":

                    sheet["sheet_id"],

                    "Part":

                    part_name,

                    "Operation":

                    "OUTER_CUT",

                    "DXF_Layer":

                    "CNC_OUTER_CUT",

                    "Suggested_Tool":

                    "End Mill",

                    "Toolpath_Type":

                    "Profile Outside",

                    "Order":

                    30

                }

            )


            # ----------------------------------------------
            # Lỗ
            # ----------------------------------------------

            for index, hole in enumerate(

                part["holes"],

                start=1

            ):


                if hole["is_drill"]:


                    rows.append(

                        {

                            "Sheet":

                            sheet["sheet_id"],

                            "Part":

                            part_name,

                            "Operation":

                            f"DRILL_{index}",

                            "DXF_Layer":

                            "CNC_INNER_DRILL",

                            "Suggested_Tool":

                            "Drill",

                            "Toolpath_Type":

                            "Drill",

                            "Order":

                            10

                        }

                    )


                else:


                    rows.append(

                        {

                            "Sheet":

                            sheet["sheet_id"],

                            "Part":

                            part_name,

                            "Operation":

                            f"INNER_CUT_{index}",

                            "DXF_Layer":

                            "CNC_INNER_CUT",

                            "Suggested_Tool":

                            "End Mill",

                            "Toolpath_Type":

                            "Profile Inside",

                            "Order":

                            20

                        }

                    )


    df = pd.DataFrame(rows)


    if not df.empty:

        df = df.sort_values(

            by=[

                "Sheet",

                "Order"

            ]

        )


    return df


# ============================================================
# 23. XUẤT JSON MAPPING
# ============================================================

def create_toolpath_json(

    nested_sheets,

    sheet_w,

    sheet_h,

    sheet_thickness

):

    data = {

        "project": "Auto CNC Industrial Nesting Pro",

        "created_at": datetime.now().isoformat(),

        "sheet": {

            "width": sheet_w,

            "height": sheet_h,

            "thickness": sheet_thickness

        },

        "toolpath_order": [

            {

                "order": 10,

                "layer": "CNC_INNER_DRILL",

                "operation": "DRILL"

            },

            {

                "order": 20,

                "layer": "CNC_INNER_CUT",

                "operation": "PROFILE_INSIDE"

            },

            {

                "order": 30,

                "layer": "CNC_OUTER_CUT",

                "operation": "PROFILE_OUTSIDE"

            }

        ],

        "sheets": []

    }


    for sheet in nested_sheets:


        sheet_data = {

            "sheet_id":

            sheet["sheet_id"],

            "parts": []

        }


        for p_node in sheet["parts"]:


            part = p_node["part_ref"]


            sheet_data["parts"].append(

                {

                    "name":

                    part["name"],

                    "x":

                    p_node["dx"],

                    "y":

                    p_node["dy"],

                    "rotation":

                    p_node["angle"],

                    "width":

                    part["width"],

                    "height":

                    part["height"],

                    "operations":

                    analyze_part_operations(

                        part

                    )

                }

            )


        data["sheets"].append(

            sheet_data

        )


    return data


# ============================================================
# 24. VẼ 2D CHI TIẾT
# ============================================================

def plot_single_part_with_dimensions(

    part

):

    fig, ax = plt.subplots(

        figsize=(5, 4)

    )


    poly = build_shapely_polygon_fixed(

        part

    )


    min_x, min_y, _, _ = poly.bounds


    poly_norm = translate(

        poly,

        xoff=-min_x,

        yoff=-min_y

    )


    if isinstance(

        poly_norm,

        Polygon

    ):


        xs, ys = poly_norm.exterior.xy


        ax.fill(

            xs,

            ys,

            alpha=0.2,

            fc="#0F766E",

            ec="#115E59",

            lw=2

        )


        for interior in poly_norm.interiors:


            ixs, iys = interior.xy


            ax.fill(

                ixs,

                iys,

                fc="white",

                ec="#B91C1C",

                lw=1

            )


    w = part["width"]

    h = part["height"]


    ax.annotate(

        "",

        xy=(

            0,

            -h * 0.1

        ),

        xytext=(

            w,

            -h * 0.1

        ),

        arrowprops=dict(

            arrowstyle="<->",

            color="blue",

            lw=1.2

        )

    )


    ax.text(

        w / 2,

        -h * 0.08,

        f"{w:.1f} mm",

        color="blue",

        fontsize=9,

        ha="center",

        weight="bold"

    )


    ax.annotate(

        "",

        xy=(

            -w * 0.1,

            0

        ),

        xytext=(

            -w * 0.1,

            h

        ),

        arrowprops=dict(

            arrowstyle="<->",

            color="red",

            lw=1.2

        )

    )


    ax.text(

        -w * 0.08,

        h / 2,

        f"{h:.1f} mm",

        color="red",

        fontsize=9,

        va="center",

        ha="right",

        rotation=90,

        weight="bold"

    )


    ax.set_aspect(

        "equal"

    )


    padding = max(

        w,

        h

    ) * 0.15


    ax.set_xlim(

        -padding,

        w + padding

    )


    ax.set_ylim(

        -padding,

        h + padding

    )


    plt.axis(

        "off"

    )


    return fig


# ============================================================
# 25. VẼ SƠ ĐỒ NESTING
# ============================================================

def plot_nesting_sheet(

    sheet,

    sheet_W,

    sheet_H

):

    fig, ax = plt.subplots(

        figsize=(

            12,

            5

        )

    )


    # Khung tấm ván

    ax.add_patch(

        mpatches.Rectangle(

            (

                0,

                0

            ),

            sheet_W,

            sheet_H,

            linewidth=1.2,

            edgecolor="black",

            facecolor="#F5F5F5"

        )

    )


    for p_info in sheet["parts"]:


        raw_poly = p_info["placed_polygon"]


        if isinstance(

            raw_poly,

            Polygon

        ):


            xs, ys = raw_poly.exterior.xy


            ax.fill(

                xs,

                ys,

                alpha=0.8,

                fc="#0F766E",

                ec="#115E59",

                lw=1

            )


            for interior in raw_poly.interiors:


                ax.fill(

                    interior.xy[0],

                    interior.xy[1],

                    fc="#F5F5F5",

                    ec="#B91C1C",

                    lw=0.8

                )


            ax.text(

                raw_poly.centroid.x,

                raw_poly.centroid.y,

                p_info["part_ref"]["name"],

                color="white",

                weight="bold",

                fontsize=6,

                ha="center"

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


    plt.axis(

        "off"

    )


    return fig


# ============================================================
# 26. FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(

    "📂 TẢI LÊN FILE STEP 3D",

    type=[

        "step",

        "stp"

    ],

    accept_multiple_files=True

)


# ============================================================
# 27. XỬ LÝ FILE
# ============================================================

if uploaded_files:


    parts_db = []


    with st.spinner(

        "⚡ Đang phân tích dữ liệu CAD..."

    ):


        for f in uploaded_files:


            try:


                extracted_data = (

                    process_cad_file_with_occ(

                        f.read(),

                        f.name,

                        hole_drill_diameter

                    )

                )


                parts_db.append(

                    extracted_data

                )


            except Exception as e:


                st.error(

                    f"Lỗi đọc file {f.name}: {str(e)}"

                )


    if parts_db:


        st.success(

            f"Đã phân tích thành công {len(parts_db)} chi tiết CAD."

        )


        # ====================================================
        # 28. XEM TRƯỚC CHI TIẾT
        # ====================================================

        st.markdown("---")

        st.subheader(

            "🔍 KIỂM TRA CHI TIẾT GỐC"

        )


        for idx, part in enumerate(

            parts_db

        ):


            st.write(

                f"### Chi tiết {idx + 1}: {part['name']}"

            )


            col_2d, col_3d = st.columns(

                [1, 1]

            )


            # ------------------------------------------------
            # 2D
            # ------------------------------------------------

            with col_2d:


                st.write(

                    "📐 Biên dạng 2D"

                )


                st.caption(

                    f"

                    Dài: {part['width']:.1f} mm |

                    Rộng: {part['height']:.1f} mm |

                    Dày: {part['thickness']:.1f} mm |

                    Số lỗ: {len(part['holes'])}

                    "

                )


                fig_2d = (

                    plot_single_part_with_dimensions(

                        part

                    )

                )


                st.pyplot(

                    fig_2d

                )


            # ------------------------------------------------
            # 3D
            # ------------------------------------------------

            with col_3d:


                st.write(

                    "📦 Mô hình 3D"

                )


                try:


                    plotter = pv.Plotter(

                        window_size=[

                            400,

                            300

                        ]

                    )


                    plotter.background_color = (

                        "#EAEDE9"

                    )


                    mesh = pv.read(

                        part["stl_path"]

                    )


                    plotter.add_mesh(

                        mesh,

                        color="#0F766E",

                        edge_color="#115E59",

                        show_edges=True,

                        specular=0.2

                    )


                    plotter.view_isometric()


                    stpyvista(

                        plotter,

                        key=f"pv_preview_{idx}"

                    )


                except Exception:


                    st.warning(

                        "Không thể hiển thị 3D."

                    )


            # ------------------------------------------------
            # Bảng tác vụ
            # ------------------------------------------------

            st.write(

                "🧠 Tác vụ CNC nhận diện"

            )


            operation_df = pd.DataFrame(

                analyze_part_operations(

                    part

                )

            )


            st.dataframe(

                operation_df,

                use_container_width=True

            )


            # Xóa STL tạm

            if os.path.exists(

                part["stl_path"]

            ):


                os.remove(

                    part["stl_path"]

                )


            st.markdown("---")


        # ====================================================
        # 29. NESTING
        # ====================================================

        if st.button(

            "🧠 BẮT ĐẦU NESTING"

        ):


            with st.spinner(

                "Đang tối ưu hóa sơ đồ cắt..."

            ):


                nesting_results = (

                    perform_advanced_best_fit_nesting(

                        parts_db,

                        sheet_W,

                        sheet_H,

                        total_offset,

                        margin

                    )

                )


            st.session_state[

                "nesting_results"

            ] = nesting_results


        # ====================================================
        # 30. HIỂN THỊ NESTING
        # ====================================================

        if (

            "nesting_results"

            in st.session_state

        ):


            nesting_results = st.session_state[

                "nesting_results"

            ]


            st.markdown("---")


            st.subheader(

                f"📐 KẾT QUẢ NESTING: {len(nesting_results)} TẤM VÁN"

            )


            for sheet in nesting_results:


                st.write(

                    f"### 🟫 TẤM VÁN SỐ {sheet['sheet_id']}"

                )


                fig = plot_nesting_sheet(

                    sheet,

                    sheet_W,

                    sheet_H

                )


                st.pyplot(

                    fig

                )


            # =================================================
            # 31. MAPPING TOOLPATH
            # =================================================

            st.markdown("---")


            st.subheader(

                "🧭 MAPPING TOOLPATH CHO ASPIRE"

            )


            mapping_df = create_toolpath_mapping(

                nesting_results

            )


            st.dataframe(

                mapping_df,

                use_container_width=True

            )


            # =================================================
            # 32. XUẤT DXF
            # =================================================

            st.markdown("---")


            st.subheader(

                "💾 XUẤT DXF THEO LAYER CNC"

            )


            doc = ezdxf.new(

                "R2010"

            )


            msp = doc.modelspace()


            create_cnc_layers(

                doc

            )


            export_original_geometry_to_dxf(

                msp,

                nesting_results,

                sheet_W,

                sheet_H

            )


            # ------------------------------------------------
            # Ghi DXF vào memory
            # ------------------------------------------------

            dxf_stream = io.StringIO()


            doc.write(

                dxf_stream

            )


            st.download_button(

                label="📥 TẢI DXF CNC",

                data=dxf_stream.getvalue(),

                file_name="cnc_nesting_output.dxf",

                mime="application/dxf"

            )


            # =================================================
            # 33. XUẤT CSV TOOLPATH MAPPING
            # =================================================

            csv_data = mapping_df.to_csv(

                index=False

            )


            st.download_button(

                label="📊 TẢI TOOLPATH MAPPING CSV",

                data=csv_data,

                file_name="toolpath_mapping.csv",

                mime="text/csv"

            )


            # =================================================
            # 34. XUẤT JSON DỮ LIỆU SẢN XUẤT
            # =================================================

            json_data = create_toolpath_json(

                nesting_results,

                sheet_W,

                sheet_H,

                sheet_thickness

            )


            json_string = json.dumps(

                json_data,

                indent=4,

                ensure_ascii=False

            )


            st.download_button(

                label="🧠 TẢI CNC JOB JSON",

                data=json_string,

                file_name="cnc_job_mapping.json",

                mime="application/json"

            )


            # =================================================
            # 35. HƯỚNG DẪN ASPIRE
            # =================================================

            st.markdown("---")


            st.subheader(

                "🛠 QUY TRÌNH TRONG ASPIRE"

            )


            st.markdown(

                """

                ### Bước 1

                Import file:

                `cnc_nesting_output.dxf`

                ---

                ### Bước 2

                Kiểm tra các layer:

                `CNC_INNER_DRILL`

                → Khoan

                ---

                `CNC_INNER_CUT`

                → Profile Inside

                ---

                `CNC_OUTER_CUT`

                → Profile Outside

                ---

                ### Bước 3

                Tạo Toolpath theo thứ tự:

                1. Khoan lỗ

                2. Cắt biên dạng trong

                3. Cắt biên dạng ngoài

                ---

                ### Bước 4

                Chọn dao và thông số cắt.

                ---

                ### Bước 5

                Preview đường dao.

                ---

                ### Bước 6

                Post Processor.

                ---

                ### Bước 7

                Xuất G-code.

                """

            )


            # =================================================
            # 36. KIỂM TRA LAYER
            # =================================================

            st.markdown("---")


            st.subheader(

                "📋 CẤU TRÚC LAYER CNC"

            )


            layer_table = pd.DataFrame(

                [

                    {

                        "Layer":

                        "CNC_INNER_DRILL",

                        "Tác vụ":

                        "Khoan",

                        "Toolpath":

                        "Drill",

                        "Thứ tự":

                        1

                    },

                    {

                        "Layer":

                        "CNC_INNER_CUT",

                        "Tác vụ":

                        "Cắt trong",

                        "Toolpath":

                        "Profile Inside",

                        "Thứ tự":

                        2

                    },

                    {

                        "Layer":

                        "CNC_OUTER_CUT",

                        "Tác vụ":

                        "Cắt ngoài",

                        "Toolpath":

                        "Profile Outside",

                        "Thứ tự":

                        3

                    }

                ]

            )


            st.dataframe(

                layer_table,

                use_container_width=True

            )
