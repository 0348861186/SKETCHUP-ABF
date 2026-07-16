import io
import os
import math
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

--- CAD / HÌNH HỌC ---

import cadquery as cq
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE

try:
from shapely.validation import make_valid
except ImportError:
make_valid = None

import ezdxf

============================================================
1. CẤU HÌNH GIAO DIỆN
============================================================

st.set_page_config(
page_title="Auto-CNC Industrial: True Shape Nesting Pro",
layout="wide"
)

st.markdown(
"""
<h2 style='text-align: center; color: #1E3A8A;'>
🏭 HỆ THỐNG CNC NỘI THẤT THÔNG MINH v3.0
</h2>
""",
unsafe_allow_html=True
)

st.write(
"""
Giải pháp tự động hóa:
STEP 3D → Tách biên dạng → True-Shape Nesting →
Bù bán kính dao → Xuất DXF → Vectric Aspire
"""
)

============================================================
2. CẤU HÌNH SẢN XUẤT
============================================================

st.sidebar.header("⚙️ THÔNG SỐ KHỔ VÁN & DAO")

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

tool_diameter = st.sidebar.number_input(
"Đường kính dao cắt (mm)",
min_value=0.1,
value=6.0,
step=0.1
)

safety_spacing = st.sidebar.number_input(
"Khoảng cách an toàn giữa 2 chi tiết (mm)",
min_value=0.0,
value=4.0,
step=0.5
)

margin = st.sidebar.number_input(
"Chừa lề biên ván (mm)",
min_value=0.0,
value=15.0,
step=1.0
)

hole_drill_diameter = st.sidebar.number_input(
"Đường kính tối đa để phân loại là lỗ khoan (mm)",
min_value=0.1,
value=20.0,
step=0.5
)

============================================================
3. CẤU HÌNH BÙ DAO
============================================================

st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH BÙ BÁN KÍNH DAO")

compensation_mode = st.sidebar.selectbox(
"Phương thức bù bán kính dao:",
options=[
"Bù dao trên phần mềm CAM (Khuyên dùng)",
"Tự động bù dao trực tiếp vào DXF (Đường tâm dao)"
],
index=0
)

tool_radius = tool_diameter / 2.0

Khoảng cách mỗi chi tiết được mở rộng khi Nesting.


Ví dụ:
Dao Ø6 mm → bán kính 3 mm
Khoảng hở giữa hai chi tiết mong muốn = 4 mm


Mỗi chi tiết buffer:
3 + 4/2 = 5 mm


Khi hai chi tiết cạnh nhau:
5 + 5 = 10 mm


Trong đó:
6 mm là vùng chiếm chỗ của dao
4 mm là khoảng hở an toàn

total_offset = (tool_diameter + safety_spacing) / 2.0

============================================================
4. HÀM SỬA GEOMETRY
============================================================

def repair_geometry(geometry):
"""
Sửa Polygon lỗi nếu có.
"""

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
============================================================
5. ĐỌC FILE STEP
============================================================

def process_cad_file_with_occ(file_bytes, filename):
"""
Đọc file STEP bằng CadQuery/OCC.

Tự động:
- Tìm mặt phẳng lớn nhất
- Lấy biên ngoài
- Lấy các lỗ bên trong
- Phân loại lỗ khoan
"""

temp_path = None

try:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(filename)[1]
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    # Đọc STEP
    part = cq.importer.importStep(temp_path)

    all_faces = part.faces().vals()

    if not all_faces:
        raise ValueError(
            "Không tìm thấy bề mặt nào trong file STEP."
        )

    # Ưu tiên các mặt phẳng
    planar_faces = [
        face
        for face in all_faces
        if face.geomType() == "PLANE"
    ]

    if not planar_faces:
        planar_faces = all_faces

    # Chọn mặt có diện tích lớn nhất
    target_face = max(
        planar_faces,
        key=lambda face: face.Area()
    )

    # Biên ngoài
    outer_wire = target_face.outerWire()

    # Các biên trong
    inner_wires = target_face.innerWires()

    # Phân tích biên ngoài
    outer_edges = parse_wire_edges_high_precision(
        outer_wire
    )

    holes = []

    for inner_wire in inner_wires:

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

        hole_diameter = radius * 2

        holes.append(
            {
                "edges": hole_edges,

                "is_drill": (
                    is_pure_circle
                    and hole_diameter
                    <= hole_drill_diameter
                ),

                "radius": radius
            }
        )

    bbox = target_face.BoundingBox()

    return {
        "name": os.path.splitext(filename)[0],

        "width": bbox.xlen,

        "height": bbox.ylen,

        "outer_edges": outer_edges,

        "holes": holes
    }

finally:

    if temp_path and os.path.exists(temp_path):
        os.remove(temp_path)
============================================================
6. PHÂN TÍCH EDGE CAD
============================================================

def parse_wire_edges_high_precision(
wire,
tolerance=0.05
):
"""
Đọc các loại hình học:

LINE
CIRCLE
ARC
BSPLINE
BEZIER
OFFSET

Spline sẽ được lấy mẫu thành các đoạn LINE nhỏ.
"""

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

        circle_geom = edge.curve()

        center = circle_geom.Location()

        radius = circle_geom.Radius()

        cx = center.X()
        cy = center.Y()


        # Hình tròn khép kín
        if start.Distance(end) < 1e-4:

            edges_data.append(
                {
                    "type": "CIRCLE",

                    "center": (
                        cx,
                        cy
                    ),

                    "radius": radius
                }
            )


        # Cung tròn
        else:

            angle_start = math.degrees(
                math.atan2(
                    start.y - cy,
                    start.x - cx
                )
            )

            angle_end = math.degrees(
                math.atan2(
                    end.y - cy,
                    end.x - cx
                )
            )

            edges_data.append(
                {
                    "type": "ARC",

                    "center": (
                        cx,
                        cy
                    ),

                    "radius": radius,

                    "start_angle": angle_start,

                    "end_angle": angle_end
                }
            )


    # ----------------------------------------------------
    # SPLINE / BEZIER / OFFSET
    # ----------------------------------------------------

    elif g_type in [
        "BSPLINE",
        "BEZIER",
        "OFFSET"
    ]:

        try:

            occ_curve = edge.ToAdaptor3d()

            first_p = (
                occ_curve.FirstParameter()
            )

            last_p = (
                occ_curve.LastParameter()
            )

            edge_length = edge.Length()

            segments = max(
                16,
                min(
                    200,
                    int(
                        edge_length
                        / tolerance
                    )
                )
            )

            pts = []

            for i in range(
                segments + 1
            ):

                u = (
                    first_p
                    + (
                        last_p
                        - first_p
                    )
                    * i
                    / segments
                )

                p = edge.valueAt(u)

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
============================================================
7. TẠO POLYGON TỪ CAD
============================================================

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

points = []

for theta in np.linspace(
    start_angle,
    end_angle,
    segments
):

    rad = math.radians(theta)

    points.append(
        (
            center_x
            + radius
            * math.cos(rad),

            center_y
            + radius
            * math.sin(rad)
        )
    )

return points

def edges_to_points(
edges,
circle_segments=64,
arc_segments=32
):

points = []

for edge in edges:

    edge_type = edge["type"]


    if edge_type == "LINE":

        points.append(
            edge["start"]
        )

        points.append(
            edge["end"]
        )


    elif edge_type == "ARC":

        cx, cy = edge["center"]

        arc_points = sample_arc(
            cx,
            cy,
            edge["radius"],
            edge["start_angle"],
            edge["end_angle"],
            arc_segments
        )

        points.extend(
            arc_points
        )


    elif edge_type == "CIRCLE":

        cx, cy = edge["center"]

        r = edge["radius"]

        for theta in np.linspace(
            0,
            360,
            circle_segments,
            endpoint=False
        ):

            rad = math.radians(theta)

            points.append(
                (
                    cx
                    + r
                    * math.cos(rad),

                    cy
                    + r
                    * math.sin(rad)
                )
            )


return points

def clean_points(
points,
snap_distance=0.01
):

cleaned = []

for point in points:

    if not cleaned:

        cleaned.append(point)

        continue

    if np.allclose(
        cleaned[-1],
        point,
        atol=snap_distance
    ):

        continue

    cleaned.append(point)


if len(cleaned) > 2:

    if not np.allclose(
        cleaned[0],
        cleaned[-1],
        atol=snap_distance
    ):

        cleaned.append(
            cleaned[0]
        )


return cleaned

def build_shapely_polygon_fixed(
part,
snap_distance=0.01
):

# -----------------------------
# BIÊN NGOÀI
# -----------------------------

outer_points = edges_to_points(
    part["outer_edges"]
)

outer_points = clean_points(
    outer_points,
    snap_distance
)


if len(outer_points) < 3:

    return Polygon(
        [
            (0, 0),

            (
                part["width"],
                0
            ),

            (
                part["width"],
                part["height"]
            ),

            (
                0,
                part["height"]
            )
        ]
    )


# -----------------------------
# LỖ BÊN TRONG
# -----------------------------

interiors = []

for hole in part["holes"]:

    hole_points = edges_to_points(
        hole["edges"]
    )

    hole_points = clean_points(
        hole_points,
        snap_distance
    )

    if len(hole_points) >= 3:

        interiors.append(
            hole_points
        )


polygon = Polygon(
    outer_points,
    interiors
)

return repair_geometry(
    polygon
)
============================================================
8. NESTING
============================================================

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


# Chi tiết lớn xếp trước
sorted_parts = sorted(
    parts_list,

    key=lambda x:
    x["width"]
    * x["height"],

    reverse=True
)


nested_sheets = []


# 8 hướng xoay
angles_to_try = [
    0,
    45,
    90,
    135,
    180,
    225,
    270,
    315
]


for part in sorted_parts:


    # Polygon thật
    poly_geom = build_shapely_polygon_fixed(
        part
    )


    # Vùng tránh va chạm
    buffered_poly = poly_geom.buffer(
        offset_val,
        resolution=32,
        join_style=JOIN_STYLE.round
    )


    buffered_poly = repair_geometry(
        buffered_poly
    )


    min_x, min_y, _, _ = (
        buffered_poly.bounds
    )


    # Polygon dùng cho Nesting
    normalized_poly = translate(
        buffered_poly,

        xoff=-min_x,

        yoff=-min_y
    )


    # Polygon thật
    raw_normalized_poly = translate(
        poly_geom,

        xoff=-min_x,

        yoff=-min_y
    )


    best_position = None

    best_sheet_idx = -1

    min_waste_score = float(
        "inf"
    )


    # ====================================================
    # TÌM VỊ TRÍ TỐT NHẤT TRÊN CÁC TẤM ĐÃ CÓ
    # ====================================================

    for s_idx, sheet_info in enumerate(
        nested_sheets
    ):

        placed_polys = (
            sheet_info[
                "placed_buffered_polygons"
            ]
        )


        # Các điểm neo
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


        anchor_points = list(
            set(anchor_points)
        )


        # Thử từng góc
        for angle in angles_to_try:

            rot_poly = rotate(
                normalized_poly,
                angle,
                origin=(0, 0)
            )


            r_minx, r_miny, _, _ = (
                rot_poly.bounds
            )


            # Thử từng điểm neo
            for ax, ay in anchor_points:

                candidate_poly = translate(
                    rot_poly,

                    xoff=ax - r_minx,

                    yoff=ay - r_miny
                )


                # Kiểm tra nằm trong tấm
                if not sheet_boundary.covers(
                    candidate_poly
                ):

                    continue


                # Kiểm tra va chạm
                collision = any(
                    candidate_poly.intersects(
                        p_poly
                    )

                    for p_poly
                    in placed_polys
                )


                if collision:

                    continue


                # Tính envelope
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


                cb = candidate_poly.bounds

                all_x.extend(
                    [
                        cb[0],
                        cb[2]
                    ]
                )

                all_y.extend(
                    [
                        cb[1],
                        cb[3]
                    ]
                )


                current_envelope_area = (
                    max(all_x)
                    - min(all_x)
                ) * (
                    max(all_y)
                    - min(all_y)
                )


                if (
                    current_envelope_area
                    < min_waste_score
                ):

                    min_waste_score = (
                        current_envelope_area
                    )

                    best_sheet_idx = s_idx


                    best_position = {

                        "dx":
                        ax - r_minx,

                        "dy":
                        ay - r_miny,

                        "angle":
                        angle,

                        "candidate_poly":
                        candidate_poly,

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
    # ĐẶT VÀO TẤM ĐÃ CÓ
    # ====================================================

    if (
        best_position
        and best_sheet_idx != -1
    ):

        sheet_info = (
            nested_sheets[
                best_sheet_idx
            ]
        )


        sheet_info["parts"].append(
            {
                "part_ref": part,

                "original_offset": (
                    min_x,
                    min_y
                ),

                "placed_polygon":
                best_position[
                    "raw_poly_transformed"
                ],

                "dx":
                best_position["dx"],

                "dy":
                best_position["dy"],

                "angle":
                best_position["angle"]
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
    # TẠO TẤM MỚI
    # ====================================================

    else:

        rot_poly = rotate(
            normalized_poly,
            0,
            origin=(0, 0)
        )


        r_minx, r_miny, _, _ = (
            rot_poly.bounds
        )


        init_x = margin_val

        init_y = margin_val


        candidate_poly = translate(
            rot_poly,

            xoff=init_x - r_minx,

            yoff=init_y - r_miny
        )


        placed_raw = translate(
            rotate(
                raw_normalized_poly,
                0,
                origin=(0, 0)
            ),

            xoff=init_x - r_minx,

            yoff=init_y - r_miny
        )


        nested_sheets.append(
            {
                "sheet_id":
                len(nested_sheets) + 1,

                "parts":
                [
                    {
                        "part_ref":
                        part,

                        "original_offset":
                        (
                            min_x,
                            min_y
                        ),

                        "placed_polygon":
                        placed_raw,

                        "dx":
                        init_x - r_minx,

                        "dy":
                        init_y - r_miny,

                        "angle":
                        0
                    }
                ],

                "placed_buffered_polygons":
                [
                    candidate_poly
                ]
            }
        )


return nested_sheets
============================================================
9. TRANSFORM TỌA ĐỘ
============================================================

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

rx = (
    tx
    * math.cos(rad)
    -
    ty
    * math.sin(rad)
)

ry = (
    tx
    * math.sin(rad)
    +
    ty
    * math.cos(rad)
)

return (
    rx + dx,
    ry + dy
)
============================================================
10. HÀM VẼ VIỀN TẤM VÁN
============================================================

def add_sheet_border(
msp,
ox,
sheet_w,
sheet_h
):

layer = "CNC_SHEET_BORDER"


msp.add_line(
    (
        ox,
        0
    ),

    (
        ox + sheet_w,
        0
    ),

    dxfattribs={
        "layer": layer
    }
)


msp.add_line(
    (
        ox + sheet_w,
        0
    ),

    (
        ox + sheet_w,
        sheet_h
    ),

    dxfattribs={
        "layer": layer
    }
)


msp.add_line(
    (
        ox + sheet_w,
        sheet_h
    ),

    (
        ox,
        sheet_h
    ),

    dxfattribs={
        "layer": layer
    }
)


msp.add_line(
    (
        ox,
        sheet_h
    ),

    (
        ox,
        0
    ),

    dxfattribs={
        "layer": layer
    }
)
============================================================
11. XUẤT DXF CHẾ ĐỘ CAM
============================================================

def export_original_geometry_to_dxf(
msp,
nested_sheets,
sheet_w,
sheet_h
):

sheet_offset_x = 0


for sheet_data in nested_sheets:

    ox = sheet_offset_x


    add_sheet_border(
        msp,
        ox,
        sheet_w,
        sheet_h
    )


    for p_node in sheet_data["parts"]:

        ref = p_node["part_ref"]


        dx = (
            p_node["dx"]
            + ox
        )

        dy = p_node["dy"]

        angle = p_node["angle"]


        orig_x, orig_y = (
            p_node[
                "original_offset"
            ]
        )


        # --------------------------------------------
        # BIÊN NGOÀI
        # --------------------------------------------

        for edge in ref[
            "outer_edges"
        ]:

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
                        "layer":
                        "CNC_OUTER_CUT"
                    }
                )


            elif edge["type"] == "ARC":

                cx, cy = transform_point(
                    edge["center"][0],
                    edge["center"][1],
                    dx,
                    dy,
                    angle,
                    orig_x,
                    orig_y
                )


                msp.add_arc(
                    center=(
                        cx,
                        cy
                    ),

                    radius=edge[
                        "radius"
                    ],

                    start_angle=(
                        edge[
                            "start_angle"
                        ]
                        + angle
                    ),

                    end_angle=(
                        edge[
                            "end_angle"
                        ]
                        + angle
                    ),

                    dxfattribs={
                        "layer":
                        "CNC_OUTER_CUT"
                    }
                )


            elif edge["type"] == "CIRCLE":

                cx, cy = transform_point(
                    edge["center"][0],
                    edge["center"][1],
                    dx,
                    dy,
                    angle,
                    orig_x,
                    orig_y
                )


                msp.add_circle(
                    center=(
                        cx,
                        cy
                    ),

                    radius=edge[
                        "radius"
                    ],

                    dxfattribs={
                        "layer":
                        "CNC_OUTER_CUT"
                    }
                )


        # --------------------------------------------
        # LỖ
        # --------------------------------------------

        for hole in ref["holes"]:

            if hole["is_drill"]:

                target_layer = (
                    "CNC_INNER_DRILL"
                )

            else:

                target_layer = (
                    "CNC_INNER_CUT"
                )


            for edge in hole[
                "edges"
            ]:

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
                            "layer":
                            target_layer
                        }
                    )


                elif edge["type"] == "CIRCLE":

                    cx, cy = transform_point(
                        edge["center"][0],
                        edge["center"][1],
                        dx,
                        dy,
                        angle,
                        orig_x,
                        orig_y
                    )


                    msp.add_circle(
                        center=(
                            cx,
                            cy
                        ),

                        radius=edge[
                            "radius"
                        ],

                        dxfattribs={
                            "layer":
                            target_layer
                        }
                    )


                elif edge["type"] == "ARC":

                    cx, cy = transform_point(
                        edge["center"][0],
                        edge["center"][1],
                        dx,
                        dy,
                        angle,
                        orig_x,
                        orig_y
                    )


                    msp.add_arc(
                        center=(
                            cx,
                            cy
                        ),

                        radius=edge[
                            "radius"
                        ],

                        start_angle=(
                            edge[
                                "start_angle"
                            ]
                            + angle
                        ),

                        end_angle=(
                            edge[
                                "end_angle"
                            ]
                            + angle
                        ),

                        dxfattribs={
                            "layer":
                            target_layer
                        }
                    )


    sheet_offset_x += (
        sheet_w
        \+ 300
    )
============================================================
12. XUẤT DXF ĐƯỜNG TÂM DAO
============================================================

def export_compensated_toolpath_to_dxf(
msp,
nested_sheets,
sheet_w,
sheet_h,
tool_r
):

sheet_offset_x = 0


for sheet_data in nested_sheets:

    ox = sheet_offset_x


    add_sheet_border(
        msp,
        ox,
        sheet_w,
        sheet_h
    )


    for p_node in sheet_data["parts"]:

        placed_poly = p_node[
            "placed_polygon"
        ]


        ref = p_node[
            "part_ref"
        ]


        # --------------------------------------------
        # LỖ KHOAN:
        # KHÔNG BÙ BÁN KÍNH DAO
        # --------------------------------------------

        dx = (
            p_node["dx"]
            + ox
        )

        dy = p_node["dy"]

        angle = p_node["angle"]

        orig_x, orig_y = (
            p_node[
                "original_offset"
            ]
        )


        for hole in ref["holes"]:

            if not hole[
                "is_drill"
            ]:

                continue


            for edge in hole[
                "edges"
            ]:

                if edge[
                    "type"
                ] != "CIRCLE":

                    continue


                cx, cy = transform_point(
                    edge["center"][0],
                    edge["center"][1],
                    dx,
                    dy,
                    angle,
                    orig_x,
                    orig_y
                )


                msp.add_circle(
                    center=(
                        cx,
                        cy
                    ),

                    radius=edge[
                        "radius"
                    ],

                    dxfattribs={
                        "layer":
                        "CNC_INNER_DRILL"
                    }
                )


        # --------------------------------------------
        # BÙ DAO
        #
        # Biên ngoài:
        # + R
        #
        # Lỗ bên trong:
        # - R
        #
        # Shapely buffer Polygon:
        # outer boundary mở rộng
        # inner holes thu nhỏ
        # --------------------------------------------

        compensated = placed_poly.buffer(
            tool_r,

            resolution=64,

            join_style=JOIN_STYLE.round
        )


        compensated = repair_geometry(
            compensated
        )


        if compensated.is_empty:

            continue


        if isinstance(
            compensated,
            Polygon
        ):

            polygons = [
                compensated
            ]

        elif isinstance(
            compensated,
            MultiPolygon
        ):

            polygons = list(
                compensated.geoms
            )

        else:

            continue


        for poly in polygons:


            # ----------------------------------------
            # BIÊN NGOÀI ĐƯỜNG TÂM DAO
            # ----------------------------------------

            outer_coords = list(
                poly.exterior.coords
            )


            for i in range(
                len(outer_coords) - 1
            ):

                p1 = (
                    outer_coords[i][0]
                    + ox,

                    outer_coords[i][1]
                )

                p2 = (
                    outer_coords[i + 1][0]
                    + ox,

                    outer_coords[i + 1][1]
                )


                msp.add_line(
                    p1,
                    p2,

                    dxfattribs={
                        "layer":
                        "CNC_COMPENSATED_PATH"
                    }
                )


            # ----------------------------------------
            # BIÊN TRONG ĐƯỜNG TÂM DAO
            # ----------------------------------------

            for interior in poly.interiors:

                interior_coords = list(
                    interior.coords
                )


                for i in range(
                    len(
                        interior_coords
                    ) - 1
                ):

                    p1 = (
                        interior_coords[i][0]
                        + ox,

                        interior_coords[i][1]
                    )

                    p2 = (
                        interior_coords[i + 1][0]
                        + ox,

                        interior_coords[i + 1][1]
                    )


                    msp.add_line(
                        p1,
                        p2,

                        dxfattribs={
                            "layer":
                            "CNC_COMPENSATED_PATH"
                        }
                    )


    sheet_offset_x += (
        sheet_w
        \+ 300
    )
============================================================
13. HÀM TẠO DXF
============================================================

def generate_industrial_dxf_flexible(
nested_sheets,
sheet_w,
sheet_h,
mode,
tool_r
):

doc = ezdxf.new(
    "R2010"
)

msp = doc.modelspace()


# Layer khung ván
doc.layers.new(
    name="CNC_SHEET_BORDER",

    dxfattribs={
        "color": 8
    }
)


# --------------------------------------------
# CHẾ ĐỘ CAM
# --------------------------------------------

if (
    mode
    == "Bù dao trên phần mềm CAM (Khuyên dùng)"
):

    doc.layers.new(
        name="CNC_OUTER_CUT",

        dxfattribs={
            "color": 1
        }
    )


    doc.layers.new(
        name="CNC_INNER_CUT",

        dxfattribs={
            "color": 4
        }
    )


    doc.layers.new(
        name="CNC_INNER_DRILL",

        dxfattribs={
            "color": 2
        }
    )


    export_original_geometry_to_dxf(
        msp,
        nested_sheets,
        sheet_w,
        sheet_h
    )


# --------------------------------------------
# CHẾ ĐỘ TỰ BÙ DAO
# --------------------------------------------

else:

    doc.layers.new(
        name="CNC_COMPENSATED_PATH",

        dxfattribs={
            "color": 3
        }
    )


    doc.layers.new(
        name="CNC_INNER_DRILL",

        dxfattribs={
            "color": 2
        }
    )


    export_compensated_toolpath_to_dxf(
        msp,
        nested_sheets,
        sheet_w,
        sheet_h,
        tool_r
    )


# Ghi DXF vào bộ nhớ
out_stream = io.StringIO()

doc.write(
    out_stream
)

return out_stream.getvalue()
============================================================
14. CSV MAPPING TOOLPATH
============================================================

def generate_aspire_toolpath_csv(
compensation_mode
):

if (
    compensation_mode
    == "Bù dao trên phần mềm CAM (Khuyên dùng)"
):

    data = [

        {
            "Layer_Name":
            "CNC_OUTER_CUT",

            "Toolpath_Type":
            "Profile Outside",

            "Depth":
            "Thickness + 0.3mm",

            "Purpose":
            "Cắt biên ngoài và Aspire tự bù bán kính dao ra ngoài"
        },


        {
            "Layer_Name":
            "CNC_INNER_CUT",

            "Toolpath_Type":
            "Profile Inside",

            "Depth":
            "Thickness + 0.3mm",

            "Purpose":
            "Cắt lọt lòng và Aspire tự bù bán kính dao vào trong"
        },


        {
            "Layer_Name":
            "CNC_INNER_DRILL",

            "Toolpath_Type":
            "Drilling",

            "Depth":
            "Theo chiều sâu cài đặt",

            "Purpose":
            "Khoan lỗ tròn nhỏ"
        }
    ]


else:

    data = [

        {
            "Layer_Name":
            "CNC_COMPENSATED_PATH",

            "Toolpath_Type":
            "Profile On",

            "Depth":
            "Thickness + 0.3mm",

            "Purpose":
            "DXF đã tự bù bán kính dao; không dùng Outside/Inside"
        },


        {
            "Layer_Name":
            "CNC_INNER_DRILL",

            "Toolpath_Type":
            "Drilling",

            "Depth":
            "Theo chiều sâu cài đặt",

            "Purpose":
            "Khoan lỗ tròn nhỏ; không bù bán kính dao"
        }
    ]


return pd.DataFrame(
    data
).to_csv(
    index=False
).encode(
    "utf-8-sig"
)
============================================================
15. UPLOAD FILE STEP
============================================================

uploaded_files = st.file_uploader(
"📂 TẢI LÊN FILE 3D STEP SẢN PHẨM",

type=[
    "step",
    "stp"
],

accept_multiple_files=True

)

if uploaded_files:

parts_db = []


# --------------------------------------------
# ĐỌC STEP
# --------------------------------------------

with st.spinner(
    "⚡ Đang phân tích mô hình 3D STEP..."
):

    for f in uploaded_files:

        try:

            extracted_data = (
                process_cad_file_with_occ(
                    f.read(),
                    f.name
                )
            )


            parts_db.append(
                extracted_data
            )


        except Exception as e:

            st.error(
                f"Lỗi phân tích file "
                f"{f.name}: {str(e)}"
            )


if parts_db:

    st.success(
        f"⚡ Đã phân tích thành công "
        f"{len(parts_db)} chi tiết CAD!"
    )


    # --------------------------------------------
    # NESTING
    # --------------------------------------------

    with st.spinner(
        "🧠 Đang thực hiện True-Shape Nesting Best-Fit..."
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


    st.subheader(
        f"📐 SƠ ĐỒ NESTING "
        f"({len(nesting_results)} TẤM VÁN)"
    )


    # --------------------------------------------
    # HIỂN THỊ SƠ ĐỒ
    # --------------------------------------------

    for sheet in nesting_results:

        st.write(
            f"### 🟫 Tấm ván số "
            f"{sheet['sheet_id']}"
        )


        fig, ax = plt.subplots(
            figsize=(12, 5)
        )


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

                facecolor="#F3E8EE"
            )
        )


        for p_info in sheet[
            "parts"
        ]:

            raw_poly = p_info[
                "placed_polygon"
            ]


            if isinstance(
                raw_poly,
                Polygon
            ):

                xs, ys = (
                    raw_poly.exterior.xy
                )


                ax.fill(
                    xs,
                    ys,

                    alpha=0.8,

                    fc="#0F766E",

                    ec="#115E59",

                    lw=1
                )


                # Vẽ lỗ
                for interior in (
                    raw_poly.interiors
                ):

                    ixs, iys = (
                        interior.xy
                    )


                    ax.fill(
                        ixs,
                        iys,

                        fc="#F3E8EE",

                        ec="#B91C1C",

                        lw=0.8
                    )


                centroid = (
                    raw_poly.centroid
                )


                ax.text(
                    centroid.x,

                    centroid.y,

                    p_info[
                        "part_ref"
                    ][
                        "name"
                    ],

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


        st.pyplot(
            fig
        )


    # --------------------------------------------
    # XUẤT FILE
    # --------------------------------------------

    st.markdown(
        "---"
    )


    st.subheader(
        "💾 XUẤT DỮ LIỆU GIA CÔNG CNC"
    )


    col1, col2 = st.columns(
        2
    )


    # DXF
    industrial_dxf = (
        generate_industrial_dxf_flexible(
            nesting_results,

            sheet_W,

            sheet_H,

            compensation_mode,

            tool_radius
        )
    )


    col1.download_button(

        label="📥 TẢI DXF GIA CÔNG",

        data=industrial_dxf,

        file_name=(
            "cnc_industrial_output.dxf"
        ),

        mime="application/dxf",

        use_container_width=True
    )


    # CSV
    csv_map = (
        generate_aspire_toolpath_csv(
            compensation_mode
        )
    )


    col2.download_button(

        label="📊 TẢI CẤU HÌNH TOOLPATH",

        data=csv_map,

        file_name=(
            "aspire_rule_mapping.csv"
        ),

        mime="text/csv",

        use_container_width=True
    )


    # --------------------------------------------
    # THÔNG TIN BÙ DAO
    # --------------------------------------------

    st.markdown(
        "---"
    )


    st.info(
        f"""
        🔧 **Thông tin bù dao**

        - Đường kính dao: **{tool_diameter:.2f} mm**
        - Bán kính dao: **{tool_radius:.2f} mm**
        - Khoảng hở an toàn: **{safety_spacing:.2f} mm**
        - Khoảng Nesting mỗi phía: **{total_offset:.2f} mm**
        - Chế độ: **{compensation_mode}**
        """
    )


    if (
        compensation_mode
        == "Bù dao trên phần mềm CAM (Khuyên dùng)"
    ):

        st.success(
            """
            DXF đang xuất biên dạng gốc.

            Trong Aspire:
            - CNC_OUTER_CUT → Profile Outside
            - CNC_INNER_CUT → Profile Inside
            - CNC_INNER_DRILL → Drilling
            """
        )


    else:

        st.warning(
            """
            DXF đã tự động bù bán kính dao.

            Trong Aspire:
            - CNC_COMPENSATED_PATH → Profile On
            - CNC_INNER_DRILL → Drilling

            Không dùng Profile Outside/Inside,
            vì sẽ bị bù dao hai lần.
            """
        )
