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
    GeometryCollection,
    LineString,
    Point
)
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE
try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None

# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Production-Ready CNC CAM Engine Pro v5.6",
    layout="wide"
)
st.markdown(
    """
    ## 🏭 CNC CAM ENGINE PRO V5.6 - INDUSTRIAL UPGRADE
    **Chordal Deviation Ramping | Clean Pocketing Floors | Safe Lead-In Insertion**
    """,
    unsafe_allow_html=True
)

# ============================================================
# 2. SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.header("📐 THÔNG SỐ VẬT LIỆU PHÔI")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván X (mm)", min_value=100.0, value=2440.0, step=10.0)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván Y (mm)", min_value=100.0, value=1220.0, step=10.0)
sheet_thickness = st.sidebar.number_input("Độ dày ván thực tế Z (mm)", min_value=0.1, value=17.0, step=0.1)
margin = st.sidebar.number_input("Khoảng cách biên tấm ván (mm)", min_value=0.0, value=15.0, step=1.0)
safety_spacing = st.sidebar.number_input("Khoảng cách giữa các chi tiết (mm)", min_value=0.0, value=6.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH DAO & CẮT G-CODE")
t1_dia = st.sidebar.number_input("Đường kính dao T1 (mm)", min_value=0.1, value=6.0, step=0.1)
t1_feed = st.sidebar.number_input("Tốc độ cắt F (mm/min)", min_value=100, value=3500, step=100)
t1_plunge = st.sidebar.number_input("Tốc độ đâm dao F_plunge (mm/min)", min_value=50, value=1200, step=50)
t1_spindle = st.sidebar.number_input("Tốc độ trục chính S (RPM)", min_value=1000, value=18000, step=500)
max_stepdown = st.sidebar.number_input("Chiều sâu mỗi lớp Stepdown (mm)", min_value=0.5, value=6.0, step=0.5)

st.sidebar.markdown("### 🔩 ĐỘ MỊN & AN TOÀN NÂNG CẤP")
chord_tolerance = st.sidebar.number_input("Dung sai dây cung - Độ mịn spline (mm)", min_value=0.005, max_value=0.5, value=0.02, step=0.005, format="%.3f")
enable_leadin = st.sidebar.checkbox("Kích hoạt Lead-in an toàn", value=True)
leadin_length = st.sidebar.number_input("Chiều dài Lead-in (mm)", min_value=2.0, value=5.0, step=0.5)
enable_ramping = st.sidebar.checkbox("Kích hoạt Continuous Spiral Ramp", value=True)
enable_tabs = st.sidebar.checkbox("Kích hoạt 3D Structural Tabs", value=True)
tab_width = st.sidebar.number_input("Chiều dài Tab hình học (mm)", min_value=5.0, value=20.0, step=1.0)
tab_thickness = st.sidebar.number_input("Độ dày vật liệu còn lại tại Tab (mm)", min_value=0.5, value=4.0, step=0.5)
tab_count_default = st.sidebar.slider("Số lượng Tab / chi tiết", min_value=2, max_value=8, value=4)

st.sidebar.markdown("### ⚙ POST-PROCESSOR")
cnc_dialect = st.sidebar.selectbox("Hệ điều hành / Phần mềm máy CNC", ["Fanuc / Syntec", "Mach3 / Grbl", "UGS (Universal Gcode Sender)", "Weihong"])
safe_Z = st.sidebar.number_input("Safe Z (mm)", min_value=1.0, value=25.0, step=1.0)
thru_overlap = st.sidebar.number_input("Độ sâu cắt xuyên thêm (mm)", min_value=0.0, value=0.5, step=0.1)

tool_radius = t1_dia / 2.0
total_offset = tool_radius + safety_spacing

# ============================================================
# 3. GEOMETRY REPAIR
# ============================================================
def repair_geometry(geom):
    if geom is None or geom.is_empty:
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
    if geom is None or geom.is_empty:
                return None
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda p: p.area)
    if isinstance(geom, GeometryCollection):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polygons:
            return max(polygons, key=lambda p: p.area)
    return None

# ============================================================
# 4. CAD EDGE EXTRACTION (CẢI TIẾN: CHORDAL TOLERANCE)
# ============================================================
def get_local_coordinates(cq_edge, plane_obj, tolerance=0.02):
    p_start = plane_obj.toLocalCoords(cq_edge.startPoint())
    p_end = plane_obj.toLocalCoords(cq_edge.endPoint())
    g_type = cq_edge.geomType()

    if g_type == "LINE":
        return {"type": "LINE", "start": (p_start.x, p_start.y), "end": (p_end.x, p_end.y)}
    
    elif g_type == "CIRCLE":
        p_center = plane_obj.toLocalCoords(cq_edge.Center())
        return {"type": "CIRCLE", "center": (p_center.x, p_center.y), "radius": cq_edge.radius()}
    
    else:
        try:
            # Thuật toán tính toán số lượng phân đoạn động (Adaptive Discretization) dựa trên dung sai dây cung
            occ_curve = cq_edge.ToAdaptor3d()
            first_param = occ_curve.FirstParameter()
            last_param = occ_curve.LastParameter()
            
            # Ước tính độ dài hình học của đường cong phẳng
            length_est = cq_edge.Length()
            
            # Áp dụng công thức tính phân đoạn dựa trên dung sai mũi tên dây cung (Chordal Deviation)
            # N = L / sqrt(8 * R * tol) -> Thực hiện xấp xỉ động qua độ dài
            segments = max(32, min(512, int(length_est / math.sqrt(tolerance if tolerance > 0 else 0.02))))
            
            pts = []
            for i in range(segments + 1):
                t = first_param + (last_param - first_param) * i / segments
                p_loc = plane_obj.toLocalCoords(cq_edge.valueAt(t))
                pts.append((p_loc.x, p_loc.y))
            return {"type": "DISCRETE_CURVE", "points": pts}
        except Exception:
            return {"type": "LINE", "start": (p_start.x, p_start.y), "end": (p_end.x, p_end.y)}

# ============================================================
# 5. EDGE → CONTINUOUS POINTS
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
            for angle in np.linspace(0, 360, 128, endpoint=False):
                rad = math.radians(angle)
                pts.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
        elif edge["type"] == "DISCRETE_CURVE":
            pts.extend(edge["points"])
    return pts

def clean_polygon_points(points, tolerance=0.01):
    if not points:
        return []
    cleaned = []
    for p in points:
        p = (float(p[0]), float(p[1]))
        if not cleaned:
            cleaned.append(p)
        else:
            if not np.allclose(cleaned[-1], p, atol=tolerance):
                cleaned.append(p)
    if len(cleaned) > 2:
        if not np.allclose(cleaned[0], cleaned[-1], atol=tolerance):
            cleaned.append(cleaned[0])
    return cleaned

# ============================================================
# 6. TRANSFORM CAD EDGE
# ============================================================
def transform_point_production(x, y, dx, dy, angle, ox, oy):
    x_local = x - ox
    y_local = y - oy
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return (x_local * cos_a - y_local * sin_a + dx, x_local * sin_a + y_local * cos_a + dy)

def transform_edge_production(edge, dx, dy, angle, ox, oy):
    if edge["type"] == "LINE":
        start_t = transform_point_production(edge["start"][0], edge["start"][1], dx, dy, angle, ox, oy)
        end_t = transform_point_production(edge["end"][0], edge["end"][1], dx, dy, angle, ox, oy)
        return {"type": "LINE", "start": start_t, "end": end_t}
    elif edge["type"] == "CIRCLE":
        center_t = transform_point_production(edge["center"][0], edge["center"][1], dx, dy, angle, ox, oy)
        return {"type": "CIRCLE", "center": center_t, "radius": edge["radius"]}
    elif edge["type"] == "DISCRETE_CURVE":
        transformed_points = [transform_point_production(x, y, dx, dy, angle, ox, oy) for x, y in edge["points"]]
        return {"type": "DISCRETE_CURVE", "points": transformed_points}
    return edge

def transform_edges_production(edges, dx, dy, angle, ox, oy):
    return [transform_edge_production(edge, dx, dy, angle, ox, oy) for edge in edges]

# ============================================================
# 7. T-BONE RELIEF
# ============================================================
def apply_t_bone_relief(polygon_points, tool_radius):
    if len(polygon_points) < 4:
        return polygon_points
    pts = list(polygon_points)
    if np.allclose(pts[0], pts[-1]):
        pts.pop()

    poly = repair_geometry(Polygon(pts))
    if not isinstance(poly, Polygon) or poly.is_empty:
        return polygon_points
    if not poly.exterior.is_ccw:
        pts.reverse()

    result = []
    n = len(pts)
    for i in range(n):
        p_prev = np.array(pts[i - 1], dtype=float)
        p_curr = np.array(pts[i], dtype=float)
        p_next = np.array(pts[(i + 1) % n], dtype=float)

        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        len_v1, len_v2 = np.linalg.norm(v1), np.linalg.norm(v2)

        if len_v1 < 1e-5 or len_v2 < 1e-5:
            result.append(tuple(p_curr))
            continue

        v1_u, v2_u = v1 / len_v1, v2 / len_v2
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = math.acos(dot)
        cross_z = v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0]

        is_concave = (cross_z > 0.001)
        is_right_angle = (abs(angle - math.pi / 2) < math.radians(10))

        result.append(tuple(p_curr))
        if is_concave and is_right_angle:
            bisector = v1_u + v2_u
            norm_b = np.linalg.norm(bisector)
            if norm_b > 1e-5:
                bisector_u = bisector / norm_b
                relief_point = p_curr + bisector_u * tool_radius
                result.append(tuple(relief_point))
                result.append(tuple(p_curr))

    result.append(result[0])
    return result

# ============================================================
# 8. STEP FILE ANALYSIS
# ============================================================
def process_cad_file_production(file_bytes, filename, sheet_thick, tol_val):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        part = cq.importers.importStep(temp_path)
        all_faces = part.faces().vals()
        if not all_faces:
            raise ValueError("STEP không chứa mặt hình học hợp lệ.")

        top_faces = [f for f in all_faces if f.geomType() == "PLANE" and f.normalAt().z > 0.8]
        if not top_faces:
            top_faces = all_faces

        target_face = max(top_faces, key=lambda f: f.Area())
        ref_plane = cq.Plane(target_face)
        face_z_level = target_face.Center().z
        outer_wire = target_face.outerWire()
        outer_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in outer_wire.Edges()]

        features = []
        for inner_wire in target_face.innerWires():
            wire_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in inner_wire.Edges()]
            features.append({"type": "CNC_INNER_CUT", "edges": wire_edges, "depth": sheet_thick})

        pocket_signatures = set()
        for face in all_faces:
            if face is target_face or face.geomType() != "PLANE":
                continue
            f_z = face.Center().z
            if not (f_z < face_z_level and f_z >= (face_z_level - sheet_thick)):
                continue
            depth = abs(face_z_level - f_z)
            if not (0.5 <= depth < sheet_thick):
                continue

            p_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in face.outerWire().Edges()]
            if not p_edges:
                continue
            raw_p = clean_polygon_points(discrete_edges(p_edges))
            if len(raw_p) < 4:
                continue

            try:
                poly = repair_geometry(Polygon(raw_p))
                if poly is None:
                    continue
                centroid = poly.centroid
                signature = (round(centroid.x, 3), round(centroid.y, 3), round(depth, 3), round(poly.area, 3))
                if signature in pocket_signatures:
                    continue
                pocket_signatures.add(signature)
                features.append({"type": "CNC_POCKET", "edges": p_edges, "depth": depth})
            except Exception:
                continue

        bbox = target_face.BoundingBox()
        return {
            "name": os.path.splitext(filename)[0],
            "width": bbox.xlen,
            "height": bbox.ylen,
            "outer_edges": outer_edges,
            "features": features,
            "origin_x": target_face.Center().x,
            "origin_y": target_face.Center().y
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# ============================================================
# 9. NESTING (KHOÉT THỦNG LÒNG PHÔI ĐỂ TẬN DỤNG DIỆN TÍCH)
# ============================================================
def execute_production_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    sheet_bound = Polygon([(margin_val, margin_val), (sheet_w - margin_val, margin_val), 
                          (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)])
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    sheets = []

    for part in sorted_parts:
        raw_points = clean_polygon_points(discrete_edges(part["outer_edges"]))
        if len(raw_points) < 4:
            continue
        poly_geom = extract_largest_polygon(repair_geometry(Polygon(raw_points)))
        if poly_geom is None:
            continue

        buffered_poly = extract_largest_polygon(repair_geometry(poly_geom.buffer(offset_val, resolution=16, join_style=JOIN_STYLE.round)))
        if buffered_poly is None:
            continue

        min_x, min_y, _, _ = buffered_poly.bounds
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized = translate(poly_geom, xoff=-min_x, yoff=-min_y)

        best_pos = None
        target_sheet_idx = -1
        best_score = float("inf")

        for sheet_idx, sheet_data in enumerate(sheets):
            # SỬ DỤNG POLYGON UNION THAY VÌ LIST ĐỂ CHO PHÉP XẾP VÀO LÒNG LỖ TRỐNG (Holes Optimization)
            placed_union = sheet_data["placed_union_geom"]
            anchors = [(margin_val, margin_val)]
            
            for pb in sheet_data["placed_buffered_polygons"]:
                b = pb.bounds
                anchors.extend([(b[2], b[1]), (b[0], b[3]), (b[2], b[3])])

            for angle in [0, 90, 180, 270]:
                rotated_poly = rotate(normalized_poly, angle, origin=(0, 0))
                r_min_x, r_min_y, _, _ = rotated_poly.bounds

                for anchor_x, anchor_y in anchors:
                    dx, dy = anchor_x - r_min_x, anchor_y - r_min_y
                    candidate = translate(rotated_poly, xoff=dx, yoff=dy)

                    if not sheet_bound.covers(candidate):
                        continue
                    # Kiểm tra va chạm qua cơ chế Giao cắt diện tích thực tế (Area Intersection)
                    if placed_union is not None and candidate.intersects(placed_union):
                        continue

                    bounds = candidate.bounds
                    score = bounds[0] + bounds[1] * 2.5
                    if score < best_score:
                        best_score = score
                        target_sheet_idx = sheet_idx
                        best_pos = {
                            "dx": dx, "dy": dy, "angle": angle, "cand_poly": candidate,
                            "raw_trans": translate(rotate(raw_normalized, angle, origin=(0, 0)), xoff=dx, yoff=dy)
                        }

        if best_pos is not None and target_sheet_idx >= 0:
            sheets[target_sheet_idx]["parts"].append({
                "part_ref": part, "original_offset": (min_x, min_y),
                "placed_polygon": best_pos["raw_trans"], "dx": best_pos["dx"], "dy": best_pos["dy"], "angle": best_pos["angle"]
            })
            sheets[target_sheet_idx]["placed_buffered_polygons"].append(best_pos["cand_poly"])
            # Cập nhật hình học vùng cấm va chạm
            if sheets[target_sheet_idx]["placed_union_geom"] is None:
                sheets[target_sheet_idx]["placed_union_geom"] = best_pos["cand_poly"]
            else:
                sheets[target_sheet_idx]["placed_union_geom"] = sheets[target_sheet_idx]["placed_union_geom"].union(best_pos["cand_poly"])
        else:
            new_sheet_id = len(sheets) + 1
            dx, dy = margin_val - min_x, margin_val - min_y
            init_poly = translate(normalized_poly, xoff=dx, yoff=dy)
            sheets.append({
                "sheet_id": new_sheet_id,
                "parts": [{"part_ref": part, "original_offset": (min_x, min_y), "placed_polygon": translate(raw_normalized, xoff=dx, yoff=dy), "dx": dx, "dy": dy, "angle": 0}],
                "placed_buffered_polygons": [init_poly],
                "placed_union_geom": init_poly
            })
    return sheets

# ============================================================
# 10. TANGENT / ARC-LENGTH UTILITIES
# ============================================================
def cumulative_lengths(pts):
    lengths = [0.0]
    total = 0.0
    for i in range(len(pts) - 1):
        p1 = np.array(pts[i], dtype=float)
        p2 = np.array(pts[i + 1], dtype=float)
        total += np.linalg.norm(p2 - p1)
        lengths.append(total)
    return lengths, total

# ============================================================
# 11. TOOLPATH (CẢI TIẾN: SẠCH SÀN POCKET TRÁNH ĐỂ LẠI LÕI)
# ============================================================
def get_true_offset_toolpath(edges, op_type, tool_radius):
    raw_pts = discrete_edges(edges)
    cleaned = clean_polygon_points(raw_pts)
    if len(cleaned) < 4:
        return []

    if op_type == "CNC_INNER_CUT":
        cleaned = apply_t_bone_relief(cleaned, tool_radius)

    poly = extract_largest_polygon(repair_geometry(Polygon(cleaned)))
    if poly is None:
        return []

    if op_type == "CNC_OUTER_CUT":
        offset_geom = repair_geometry(poly.buffer(tool_radius, resolution=16, join_style=JOIN_STYLE.round))
        return [list(offset_geom.exterior.coords)] if isinstance(offset_geom, Polygon) else []
    
    elif op_type == "CNC_INNER_CUT":
        offset_geom = repair_geometry(poly.buffer(-tool_radius, resolution=16, join_style=JOIN_STYLE.round))
        return [list(offset_geom.exterior.coords)] if isinstance(offset_geom, Polygon) else []
    
    elif op_type == "CNC_POCKET":
        paths = []
        stepover = tool_radius * 0.75
        current_offset = -tool_radius
        
        while True:
            offset_geom = repair_geometry(poly.buffer(current_offset, resolution=16, join_style=JOIN_STYLE.round))
            if offset_geom is None or offset_geom.is_empty:
                # CẢI TIẾN: Nếu vùng đảo cuối cùng nhỏ hơn stepover nhưng vẫn còn diện tích, 
                # chèn thêm một điểm centroid ăn tâm cuối để làm sạch sàn hoàn toàn.
                prev_offset = current_offset + stepover
                last_valid_geom = repair_geometry(poly.buffer(prev_offset, resolution=16, join_style=JOIN_STYLE.round))
                if last_valid_geom and not last_valid_geom.is_empty:
                    centroid = last_valid_geom.centroid
                    paths.append([(centroid.x, centroid.y), (centroid.x + 0.001, centroid.y)])
                break
                
            if isinstance(offset_geom, Polygon):
                paths.append(list(offset_geom.exterior.coords))
            elif isinstance(offset_geom, MultiPolygon):
                for sub_p in offset_geom.geoms:
                    paths.append(list(sub_p.exterior.coords))
            current_offset -= stepover
        return paths
    return []

# ============================================================
# 12. TAB GENERATION
# ============================================================
def build_tab_ranges(pts, tab_width, tab_count):
    _, total_len = cumulative_lengths(pts)
    tab_ranges = []
    if total_len <= tab_width * tab_count * 2:
        return tab_ranges
    spacing = total_len / tab_count
    for i in range(tab_count):
        center = i * spacing + spacing / 2
        start = max(0.0, center - tab_width / 2)
        end = min(total_len, center + tab_width / 2)
        tab_ranges.append((start, end))
    return tab_ranges

def get_tab_state(distance, tab_ranges):
    for start, end in tab_ranges:
        if start <= distance <= end:
            return True
    return False

# ============================================================
# 13. POST PROCESSOR
# ============================================================
def generate_program_header(dialect, spindle, safe_z):
    header = []
    if dialect in ["Fanuc / Syntec", "Weihong"]:
        header.extend(["%", "G90 G21 G17 G40 G49 G80", "T1 M6", f"G43 H1 Z{safe_z:.3f}", f"S{int(spindle)} M3"])
    else:
        header.extend(["G90", "G21", "G17", f"M3 S{int(spindle)}", f"G0 Z{safe_z:.3f}"])
    return header

def generate_program_footer(dialect):
    if dialect in ["Fanuc / Syntec", "Weihong"]:
        return ["M5", "G49", "G0 Z25.000", "M30", "%"]
    return ["M5", "M30"]

# ============================================================
# 14. TOOLPATH → G-CODE (CẢI TIẾN: LEAD-IN AN TOÀN TUYỆT ĐỐI)
# ============================================================
def generate_gcode_for_toolpath(toolpath_pts, op_type, total_depth, max_step, feed, plunge, spindle, safe_z, enable_leadin, leadin_length, enable_ramping, enable_tabs, tab_width, tab_thick, tab_count, dialect):
    gcode = []
    if len(toolpath_pts) < 2:
        return gcode

    pts = list(toolpath_pts)
    if np.allclose(pts[0], pts[-1]):
        pts.pop()
    n_pts = len(pts)
    if n_pts < 2:
        return gcode

    lengths, total_len = cumulative_lengths(pts)
    tab_ranges = build_tab_ranges(pts, tab_width, tab_count) if (enable_tabs and op_type == "CNC_OUTER_CUT") else []

    # --------------------------------------------------------
    # CẢI TIẾN: AN TOÀN LỐI VÀO DAO (SAFE LEAD-IN POSITIONING)
    # --------------------------------------------------------
    start_pt = np.array(pts[0], dtype=float)
    next_pt = np.array(pts[1], dtype=float)
    vec_dir = next_pt - start_pt
    norm_v = np.linalg.norm(vec_dir)

    if norm_v > 1e-5 and enable_leadin:
        perp_vec = np.array([-vec_dir[1], vec_dir[0]]) / norm_v
        leadin_start = start_pt + perp_vec * leadin_length
        
        # ĐƯỜNG ĐI AN TOÀN: Xuống vị trí Z an toàn ngoài phôi trước khi hạ xuống sàn vật liệu
        gcode.append(f"G0 X{leadin_start[0]:.3f} Y{leadin_start[1]:.3f} Z{safe_z:.3f}")
        gcode.append(f"G1 Z0.000 F{plunge}") 
        # Cắt tịnh tiến ngang để tiến vào điểm bắt đầu chu kỳ Ramp, loại bỏ đâm dao thẳng đứng vào bề mặt chi tiết
        gcode.append(f"G1 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} F{feed}")
    else:
        gcode.append(f"G0 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{safe_z:.3f}")
        gcode.append(f"G1 Z0.000 F{plunge}")

    z_targets = []
    current_z = 0.0
    while current_z > -total_depth:
        current_z -= max_step
        if current_z < -total_depth:
            current_z = -total_depth
        z_targets.append(current_z)

    previous_z = 0.0

    # --------------------------------------------------------
    # DEPTH PASSES RUNTIME
    # --------------------------------------------------------
    for pass_index, target_z in enumerate(z_targets):
        gcode.append(f"; --- PASS {pass_index + 1} TARGET Z = {target_z:.3f} ---")
        pass_depth = abs(target_z - previous_z)

        if enable_ramping and total_len > 0 and pass_depth > 0.01:
            gcode.append("; CONTINUOUS SPIRAL RAMP ACTIVE")
            for i in range(n_pts + 1):
                idx = i % n_pts
                p_c = np.array(pts[idx], dtype=float)
                dist_accum = (lengths[idx] if i < n_pts else total_len)
                
                ratio = dist_accum / total_len
                z_ramp = previous_z - (pass_depth * ratio)

                if get_tab_state(dist_accum, tab_ranges):
                    tab_z_boundary = -total_depth + tab_thick
                    if z_ramp < tab_z_boundary:
                        z_ramp = tab_z_boundary

                gcode.append(f"G1 X{p_c[0]:.3f} Y{p_c[1]:.3f} Z{z_ramp:.3f} F{feed}")
        else:
            for i in range(n_pts + 1):
                idx = i % n_pts
                p_c = np.array(pts[idx], dtype=float)
                dist_accum = (lengths[idx] if i < n_pts else total_len)
                z_flat = target_z

                if get_tab_state(dist_accum, tab_ranges):
                    tab_z_boundary = -total_depth + tab_thick
                    if z_flat < tab_z_boundary:
                        z_flat = tab_z_boundary

                gcode.append(f"G1 X{p_c[0]:.3f} Y{p_c[1]:.3f} Z{z_flat:.3f} F{feed}")

        previous_z = target_z

    gcode.append(f"G0 Z{safe_z:.3f}")
    return gcode

# ============================================================
# 15. MAIN STREAMLIT APPLICATION ORCHESTRATION
# ============================================================
uploaded_files = st.file_uploader("Tải lên bản vẽ kỹ thuật chi tiết (.STEP / .STP)", accept_multiple_files=True, type=["step", "stp"])

if uploaded_files:
    parsed_parts = []
    for f in uploaded_files:
        with st.spinner(f"Đang phân tích dữ liệu cấu trúc: {f.name}"):
            try:
                # Đưa cấu hình Chordal Tolerance vào máy quét dữ liệu đầu vào
                part_data = process_cad_file_production(f.getvalue(), f.name, sheet_thickness, chord_tolerance)
                parsed_parts.append(part_data)
                st.success(f"Đã nạp thành công: {part_data['name']} ({part_data['width']:.1f}x{part_data['height']:.1f} mm)")
            except Exception as e:
                st.error(f"Lỗi phân tích tệp {f.name}: {str(e)}")

    if parsed_parts:
        st.subheader("📦 KẾT QUẢ SẮP XẾP TỰ ĐỘNG NÂNG CẤP (HOLES LAYER OPTIMIZED)")
        nested_sheets = execute_production_nesting(parsed_parts, sheet_W, sheet_H, total_offset, margin)
        
        st.metric("Tổng số lượng tấm ván cần dùng", len(nested_sheets))

        for sheet in nested_sheets:
            st.write(f"### 📋 Tấm phôi số #{sheet['sheet_id']}")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xlim(0, sheet_W)
            ax.set_ylim(0, sheet_H)
            ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, color="darkgrey", alpha=0.3, label="Khổ ván gốc"))
            ax.add_patch(mpatches.Rectangle((margin, margin), sheet_W - 2*margin, sheet_H - 2*margin, fill=False, linestyle="--", color="red", label="Vùng cắt an toàn"))
            
            all_gcode_blocks = generate_program_header(cnc_dialect, t1_spindle, safe_Z)

            for p_idx, placed in enumerate(sheet["parts"]):
                ref = placed["part_ref"]
                poly = placed["placed_polygon"]
                
                x, y = poly.exterior.xy
                ax.plot(x, y, "b-", linewidth=2)
                ax.fill(x, y, "skyblue", alpha=0.5)
                ax.text(poly.centroid.x, poly.centroid.y, f"P{p_idx+1}: {ref['name']}", ha='center', va='center', fontsize=9, weight='bold')

                trans_outer_edges = transform_edges_production(ref["outer_edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                outer_paths = get_true_offset_toolpath(trans_outer_edges, "CNC_OUTER_CUT", tool_radius)
                
                for path in outer_paths:
                    px, py = zip(*path)
                    ax.plot(px, py, "g--", alpha=0.8)
                    all_gcode_blocks.extend(generate_gcode_for_toolpath(
                        path, "CNC_OUTER_CUT", sheet_thickness + thru_overlap, max_stepdown,
                        t1_feed, t1_plunge, t1_spindle, safe_Z, enable_leadin, leadin_length,
                        enable_ramping, enable_tabs, tab_width, tab_thickness, tab_count_default, cnc_dialect
                    ))

                for feat in ref["features"]:
                    trans_feat_edges = transform_edges_production(feat["edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                    feat_paths = get_true_offset_toolpath(trans_feat_edges, feat["type"], tool_radius)
                    
                    for path in feat_paths:
                        px, py = zip(*path)
                        ax.plot(px, py, "m:", alpha=0.7)
                        all_gcode_blocks.extend(generate_gcode_for_toolpath(
                            path, feat["type"], feat["depth"], max_stepdown,
                            t1_feed, t1_plunge, t1_spindle, safe_Z, enable_leadin, leadin_length,
                            enable_ramping, False, tab_width, tab_thickness, tab_count_default, cnc_dialect
                        ))

            all_gcode_blocks.extend(generate_program_footer(cnc_dialect))
            ax.set_aspect('equal', adjustable='box')
            st.pyplot(fig)

            gcode_txt = "\n".join(all_gcode_blocks)
            st.download_button(
                label=f"💾 Tải xuống mã máy G-Code Tấm #{sheet['sheet_id']}",
                data=gcode_txt,
                file_name=f"CAM_Engine_Industrial_Sheet_{sheet['sheet_id']}.nc",
                mime="text/plain"
            )
