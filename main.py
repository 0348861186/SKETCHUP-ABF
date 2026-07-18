import os
import math
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ============================================================
# CAD / CAM CORE (UPGRADED FOR MULTI-BODY ASSEMBLY)
# ============================================================
import cadquery as cq
from shapely.geometry import (
    Polygon, MultiPolygon, GeometryCollection, LineString, Point
)
from shapely.affinity import translate, rotate
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
    page_title="Production-Ready CNC CAM Engine Pro v7.0",
    layout="wide"
)
st.markdown(
    """
    ## 🏭 CNC CAM ENGINE PRO V7.0 - MULTI-TOOL & LAYER SEPARATION
    **Hệ thống tự động bóc tách | Hạ phẳng | Nesting | Xuất G-Code tách biệt Layer & Dao cắt riêng biệt (M6/M0)**
    """,
    unsafe_allow_html=True
)

# ============================================================
# 2. SIDEBAR CONFIGURATION
# ============================================================
st.sidebar.header("📐 THÔNG SỐ VẬT LIỆU PHÔI")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván X (mm)", min_value=100.0, value=2440.0, step=10.0)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván Y (mm)", min_value=100.0, value=1220.0, step=10.0)
sheet_thickness = st.sidebar.number_input("Độ dày ván tiêu chuẩn Z (mm)", min_value=0.1, value=17.0, step=0.1)
margin = st.sidebar.number_input("Khoảng cách biên tấm ván (mm)", min_value=0.0, value=15.0, step=1.0)
safety_spacing = st.sidebar.number_input("Khoảng cách giữa các chi tiết (mm)", min_value=0.0, value=6.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH DAO CHO TỪNG LAYER")

# Cấu hình Dao 1 - Pocket
st.sidebar.markdown("#### 🔹 Dao T1: Gia công túi/hèm (CNC_POCKET)")
t1_dia = st.sidebar.number_input("Đường kính T1 (mm)", min_value=0.1, value=6.0, step=0.1, key="t1_d")
t1_feed = st.sidebar.number_input("Tốc độ cắt F - T1 (mm/min)", min_value=100, value=3000, step=100, key="t1_f")
t1_plunge = st.sidebar.number_input("Tốc độ đâm F_plunge - T1 (mm/min)", min_value=50, value=1000, step=50, key="t1_p")
t1_spindle = st.sidebar.number_input("Tốc độ trục S - T1 (RPM)", min_value=1000, value=18000, step=500, key="t1_s")

# Cấu hình Dao 2 - Inner Cut
st.sidebar.markdown("#### 🔸 Dao T2: Cắt lỗ khoét trong (CNC_INNER_CUT)")
t2_dia = st.sidebar.number_input("Đường kính T2 (mm)", min_value=0.1, value=4.0, step=0.1, key="t2_d")
t2_feed = st.sidebar.number_input("Tốc độ cắt F - T2 (mm/min)", min_value=100, value=2500, step=100, key="t2_f")
t2_plunge = st.sidebar.number_input("Tốc độ đâm F_plunge - T2 (mm/min)", min_value=50, value=1000, step=50, key="t2_p")
t2_spindle = st.sidebar.number_input("Tốc độ trục S - T2 (RPM)", min_value=1000, value=18000, step=500, key="t2_s")

# Cấu hình Dao 3 - Outer Cut
st.sidebar.markdown("#### 🔺 Dao T3: Cắt đứt đường viền ngoài (CNC_OUTER_CUT)")
t3_dia = st.sidebar.number_input("Đường kính T3 (mm)", min_value=0.1, value=6.0, step=0.1, key="t3_d")
t3_feed = st.sidebar.number_input("Tốc độ cắt F - T3 (mm/min)", min_value=100, value=3500, step=100, key="t3_f")
t3_plunge = st.sidebar.number_input("Tốc độ đâm F_plunge - T3 (mm/min)", min_value=50, value=1200, step=50, key="t3_p")
t3_spindle = st.sidebar.number_input("Tốc độ trục S - T3 (RPM)", min_value=1000, value=18000, step=500, key="t3_s")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ THÔNG SỐ VẬT LIỆU VẬN HÀNH CHUNG")
max_stepdown = st.sidebar.number_input("Chiều sâu mỗi lớp Stepdown (mm)", min_value=0.5, value=6.0, step=0.5)
chord_tolerance = st.sidebar.number_input("Dung sai dây cung (mm)", min_value=0.005, max_value=0.5, value=0.02, step=0.005, format="%.3f")
enable_leadin = st.sidebar.checkbox("Kích hoạt Lead-in an toàn", value=True)
leadin_length = st.sidebar.number_input("Chiều dài Lead-in (mm)", min_value=2.0, value=5.0, step=0.5)
enable_ramping = st.sidebar.checkbox("Kích hoạt Continuous Spiral Ramp", value=True)
enable_tabs = st.sidebar.checkbox("Kích hoạt 3D Structural Tabs", value=True)
tab_width = st.sidebar.number_input("Chiều dài Tab hình học (mm)", min_value=5.0, value=20.0, step=1.0)
tab_thickness = st.sidebar.number_input("Độ dày vật liệu còn lại tại Tab (mm)", min_value=0.5, value=4.0, step=0.5)
tab_count_default = st.sidebar.slider("Số lượng Tab / chi tiết", min_value=2, max_value=8, value=4)

st.sidebar.markdown("### ⚙ POST-PROCESSOR")
cnc_dialect = st.sidebar.selectbox("Hệ điều hành / Phần mềm máy CNC", ["Fanuc / Syntec", "Mach3 / Grbl", "UGS", "Weihong"])
safe_Z = st.sidebar.number_input("Safe Z (mm)", min_value=1.0, value=25.0, step=1.0)
thru_overlap = st.sidebar.number_input("Độ sâu cắt xuyên thêm (mm)", min_value=0.0, value=0.5, step=0.1)

# Tính toán khoảng cách tối ưu dựa trên đường kính dao lớn nhất để tránh va chạm khi nesting
max_tool_radius = max(t1_dia, t2_dia, t3_dia) / 2.0
total_offset = max_tool_radius + safety_spacing

# ============================================================
# 3. GEOMETRY REPAIR
# ============================================================
def repair_geometry(geom):
    if geom is None or geom.is_empty: return geom
    if geom.is_valid: return geom
    if make_valid is not None:
        try:
            fixed = make_valid(geom)
            if not fixed.is_empty: return fixed
        except: pass
    try:
        fixed = geom.buffer(0)
        if not fixed.is_empty: return fixed
    except: pass
    return geom

def extract_largest_polygon(geom):
    if geom is None or geom.is_empty: return None
    if isinstance(geom, Polygon): return geom
    if isinstance(geom, MultiPolygon): return max(geom.geoms, key=lambda p: p.area)
    if isinstance(geom, GeometryCollection):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
        if polygons: return max(polygons, key=lambda p: p.area)
    return None

# ============================================================
# 4. CAD EDGE EXTRACTION WITH LOCAL PLANE
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
            occ_curve = cq_edge.ToAdaptor3d()
            first_param = occ_curve.FirstParameter()
            last_param = occ_curve.LastParameter()
            length_est = cq_edge.Length()
            segments = max(32, min(512, int(length_est / math.sqrt(tolerance if tolerance > 0 else 0.02))))
            pts = []
            for i in range(segments + 1):
                t = first_param + (last_param - first_param) * i / segments
                p_loc = plane_obj.toLocalCoords(cq_edge.valueAt(t))
                pts.append((p_loc.x, p_loc.y))
            return {"type": "DISCRETE_CURVE", "points": pts}
        except:
            return {"type": "LINE", "start": (p_start.x, p_start.y), "end": (p_end.x, p_end.y)}

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
    if not points: return []
    cleaned = []
    for p in points:
        p = (float(p[0]), float(p[1]))
        if not cleaned: cleaned.append(p)
        else:
            if not np.allclose(cleaned[-1], p, atol=tolerance): cleaned.append(p)
    if len(cleaned) > 2:
        if not np.allclose(cleaned[0], cleaned[-1], atol=tolerance): cleaned.append(cleaned[0])
    return cleaned

# ============================================================
# 5. ASSEMBLY EXPLODER (FIXED ĐỂ BÓC TÁCH KHỐI RẮN 100%)
# ============================================================
def process_full_assembly_step(file_bytes, filename, std_thickness, tol_val):
    temp_path = None
    parsed_parts = []
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Đọc file bằng bộ nạp chuẩn
        imported_shape = cq.importers.importStep(temp_path)
        
        # Lấy danh sách các khối rắn (Solids) một cách an toàn bằng phương thức cốt lõi của CadQuery
        raw_solids = []
        if hasattr(imported_shape, "solids"):
            # Lấy các phần tử thông qua đối tượng lặp (iterable) thay vì gọi .vals()
            raw_solids = [s for s in imported_shape.solids()]
        else:
            raw_solids = imported_shape.Solids()

        if not raw_solids:
            raise ValueError("Không tìm thấy khối rắn (Solids) hợp lệ trong tệp 3D.")
        
        # Bọc các khối thành cq.Solid để đảm bảo tính nhất quán của API nâng cao
        solids = []
        for s in raw_solids:
            if hasattr(s, "val"):
                solids.append(cq.Solid(s.val().wrapped))
            elif hasattr(s, "wrapped"):
                solids.append(cq.Solid(s.wrapped))
            else:
                solids.append(cq.Solid(s))
        
        st.info(f"🔎 Đã phát hiện tổng cộng **{len(solids)}** chi tiết trong mô hình lắp ráp.")

        for idx, solid in enumerate(solids):
            if solid.Area() < 500: 
                continue  

            # Lúc này solid chắc chắn là cq.Solid chuẩn, trích xuất các mặt phẳng bằng cấu trúc lặp an toàn
            faces = [f for f in solid.faces()]
            plane_faces = [f for f in faces if f.geomType() == "PLANE"]
            if not plane_faces:
                continue
            
            target_face = max(plane_faces, key=lambda f: f.Area())
            face_center = target_face.Center()
            face_normal = target_face.normalAt(face_center)

            ref_plane = cq.Plane(origin=face_center, normal=face_normal)
            outer_wire = target_face.outerWire()
            outer_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in outer_wire.Edges()]
            
            features = []
            for inner_wire in target_face.innerWires():
                wire_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in inner_wire.Edges()]
                features.append({"type": "CNC_INNER_CUT", "edges": wire_edges, "depth": std_thickness})

            pocket_signatures = set()
            
            for face in faces:
                if face is target_face or face.geomType() != "PLANE": 
                    continue
                local_center = ref_plane.toLocalCoords(face.Center())
                depth = abs(local_center.z)
                
                if 0.5 <= depth < (std_thickness + 2.0):
                    p_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in face.outerWire().Edges()]
                    if not p_edges: continue
                    raw_p = clean_polygon_points(discrete_edges(p_edges))
                    if len(raw_p) < 4: continue
                    
                    try:
                        poly = repair_geometry(Polygon(raw_p))
                        if poly is None: continue
                        centroid = poly.centroid
                        signature = (round(centroid.x, 2), round(centroid.y, 2), round(depth, 1))
                        if signature in pocket_signatures: continue
                        pocket_signatures.add(signature)
                        features.append({"type": "CNC_POCKET", "edges": p_edges, "depth": depth})
                    except:
                        continue

            raw_outer_pts = clean_polygon_points(discrete_edges(outer_edges))
            if len(raw_outer_pts) < 4: continue
            poly_outer = repair_geometry(Polygon(raw_outer_pts))
            if poly_outer is None or poly_outer.is_empty: continue
            
            min_x, min_y, max_x, max_y = poly_outer.bounds
            width_local = max_x - min_x
            height_local = max_y - min_y

            parsed_parts.append({
                "name": f"Tam_Van_{idx+1}",
                "width": width_local,
                "height": height_local,
                "outer_edges": outer_edges,
                "features": features,
                "origin_x": face_center.x,
                "origin_y": face_center.y
            })
            
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            
    return parsed_parts
# ============================================================
# 6. TRANSFORM CAD EDGE & RELIEF UTILITIES
# ============================================================
def transform_point_production(x, y, dx, dy, angle, ox, oy):
    x_local = x - ox
    y_local = y - oy
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
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

def apply_t_bone_relief(polygon_points, tool_radius):
    if len(polygon_points) < 4: return polygon_points
    pts = list(polygon_points)
    if np.allclose(pts[0], pts[-1]): pts.pop()

    poly = repair_geometry(Polygon(pts))
    if not isinstance(poly, Polygon) or poly.is_empty: return polygon_points
    if not poly.exterior.is_ccw: pts.reverse()

    result = []
    n = len(pts)
    for i in range(n):
        p_prev = np.array(pts[i - 1], dtype=float)
        p_curr = np.array(pts[i], dtype=float)
        p_next = np.array(pts[(i + 1) % n], dtype=float)
        v1, v2 = p_prev - p_curr, p_next - p_curr
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
# 7. NESTING ENGINE
# ============================================================
def execute_production_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    sheet_bound = Polygon([(margin_val, margin_val), (sheet_w - margin_val, margin_val), 
                          (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)])
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    sheets = []

    for part in sorted_parts:
        raw_points = clean_polygon_points(discrete_edges(part["outer_edges"]))
        if len(raw_points) < 4: continue
        poly_geom = extract_largest_polygon(repair_geometry(Polygon(raw_points)))
        if poly_geom is None: continue

        buffered_poly = extract_largest_polygon(repair_geometry(poly_geom.buffer(offset_val, resolution=16, join_style=JOIN_STYLE.round)))
        if buffered_poly is None: continue

        min_x, min_y, _, _ = buffered_poly.bounds
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized = translate(poly_geom, xoff=-min_x, yoff=-min_y)

        best_pos = None
        target_sheet_idx = -1
        best_score = float("inf")

        for sheet_idx, sheet_data in enumerate(sheets):
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

                    if not sheet_bound.covers(candidate): continue
                    if placed_union is not None and candidate.intersects(placed_union): continue

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
# 8. TOOLPATH ENGINE & G-CODE GENERATION
# ============================================================
def cumulative_lengths(pts):
    lengths = [0.0]
    total = 0.0
    for i in range(len(pts) - 1):
        total += np.linalg.norm(np.array(pts[i+1]) - np.array(pts[i]))
        lengths.append(total)
    return lengths, total

def get_true_offset_toolpath(edges, op_type, tool_radius):
    raw_pts = discrete_edges(edges)
    cleaned = clean_polygon_points(raw_pts)
    if len(cleaned) < 4: return []
    if op_type == "CNC_INNER_CUT": cleaned = apply_t_bone_relief(cleaned, tool_radius)

    poly = extract_largest_polygon(repair_geometry(Polygon(cleaned)))
    if poly is None: return []

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
                prev_offset = current_offset + stepover
                last_valid_geom = repair_geometry(poly.buffer(prev_offset, resolution=16, join_style=JOIN_STYLE.round))
                if last_valid_geom and not last_valid_geom.is_empty:
                    paths.append([(last_valid_geom.centroid.x, last_valid_geom.centroid.y), (last_valid_geom.centroid.x+0.001, last_valid_geom.centroid.y)])
                break
            if isinstance(offset_geom, Polygon): paths.append(list(offset_geom.exterior.coords))
            elif isinstance(offset_geom, MultiPolygon):
                for sub_p in offset_geom.geoms: paths.append(list(sub_p.exterior.coords))
            current_offset -= stepover
        return paths
    return []

def build_tab_ranges(pts, tab_width, tab_count):
    _, total_len = cumulative_lengths(pts)
    tab_ranges = []
    if total_len <= tab_width * tab_count * 2: return tab_ranges
    spacing = total_len / tab_count
    for i in range(tab_count):
        center = i * spacing + spacing / 2
        tab_ranges.append((max(0.0, center - tab_width / 2), min(total_len, center + tab_width / 2)))
    return tab_ranges

def generate_gcode_for_toolpath(toolpath_pts, op_type, total_depth, max_step, feed, plunge, safe_z, enable_leadin, leadin_length, enable_ramping, enable_tabs, tab_width, tab_thick, tab_count):
    gcode = []
    if len(toolpath_pts) < 2: return gcode
    pts = list(toolpath_pts)
    if np.allclose(pts[0], pts[-1]): pts.pop()
    n_pts = len(pts)
    if n_pts < 2: return gcode

    lengths, total_len = cumulative_lengths(pts)
    tab_ranges = build_tab_ranges(pts, tab_width, tab_count) if (enable_tabs and op_type == "CNC_OUTER_CUT") else []

    start_pt = np.array(pts[0], dtype=float)
    next_pt = np.array(pts[1], dtype=float)
    vec_dir = next_pt - start_pt
    norm_v = np.linalg.norm(vec_dir)

    if norm_v > 1e-5 and enable_leadin:
        perp_vec = np.array([-vec_dir[1], vec_dir[0]]) / norm_v
        leadin_start = start_pt + perp_vec * leadin_length
        gcode.append(f"G0 X{leadin_start[0]:.3f} Y{leadin_start[1]:.3f} Z{safe_z:.3f}")
        gcode.append(f"G1 Z0.000 F{plunge}") 
        gcode.append(f"G1 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} F{feed}")
    else:
        gcode.append(f"G0 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{safe_z:.3f}")
        gcode.append(f"G1 Z0.000 F{plunge}")

    z_targets = []
    current_z = 0.0
    while current_z > -total_depth:
        current_z -= max_step
        if current_z < -total_depth: current_z = -total_depth
        z_targets.append(current_z)

    previous_z = 0.0
    for pass_index, target_z in enumerate(z_targets):
        gcode.append(f"; --- PASS {pass_index + 1} TARGET Z = {target_z:.3f} ---")
        pass_depth = abs(target_z - previous_z)

        if enable_ramping and total_len > 0 and pass_depth > 0.01:
            for i in range(n_pts + 1):
                idx = i % n_pts
                dist_accum = (lengths[idx] if i < n_pts else total_len)
                z_ramp = previous_z - (pass_depth * (dist_accum / total_len))
                if tab_ranges and any(s <= dist_accum <= e for s, e in tab_ranges):
                    tz = -total_depth + tab_thick
                    if z_ramp < tz: z_ramp = tz
                gcode.append(f"G1 X{pts[idx][0]:.3f} Y{pts[idx][1]:.3f} Z{z_ramp:.3f} F{feed}")
        else:
            for i in range(n_pts + 1):
                idx = i % n_pts
                dist_accum = (lengths[idx] if i < n_pts else total_len)
                z_flat = target_z
                if tab_ranges and any(s <= dist_accum <= e for s, e in tab_ranges):
                    tz = -total_depth + tab_thick
                    if z_flat < tz: z_flat = tz
                gcode.append(f"G1 X{pts[idx][0]:.3f} Y{pts[idx][1]:.3f} Z{z_flat:.3f} F{feed}")
        previous_z = target_z

    gcode.append(f"G0 Z{safe_z:.3f}")
    return gcode

def generate_program_header(dialect):
    if dialect in ["Fanuc / Syntec", "Weihong"]:
        return ["%", "G90 G21 G17 G40 G49 G80"]
    return ["G90", "G21", "G17"]

def generate_tool_change_block(dialect, tool_number, spindle_speed, safe_z):
    block = [
        "",
        "; ----------------------------------------------",
        f"; TÁC VỤ: THAY DAO -> THỰC HIỆN DAO T{tool_number}",
        "; ----------------------------------------------",
        "M5",                    # Tắt trục chính trước khi đổi
        f"G0 Z{safe_z:.3f}",     # Nhấc dao lên cao an toàn tuyệt đối
    ]
    if dialect in ["Fanuc / Syntec", "Weihong"]:
        block.extend([
            f"T{tool_number} M6",           # Gọi đài dao / Dừng thay dao cơ học
            f"G43 H{tool_number} Z{safe_z:.3f}", # Bù trừ chiều dài hình học dao tương ứng
            f"S{int(spindle_speed)} M3",     # Quay lại trục chính với tốc độ cấu hình mới
            "M0"                             # Lệnh dừng chương trình tạm thời
        ])
    else:
        block.extend([
            f"T{tool_number} M6",
            f"M3 S{int(spindle_speed)}",
            "M0"
        ])
    return block

def generate_program_footer(dialect, safe_z):
    if dialect in ["Fanuc / Syntec", "Weihong"]: 
        return ["M5", "G49", f"G0 Z{safe_z:.3f}", "M30", "%"]
    return ["M5", f"G0 Z{safe_z:.3f}", "M30"]

# ============================================================
# 9. MAIN APP ORCHESTRATION
# ============================================================
uploaded_files = st.file_uploader("Tải lên bản vẽ thiết kế 3D toàn bộ khối tủ (.STEP / .STP)", accept_multiple_files=True, type=["step", "stp"])

if uploaded_files:
    all_extracted_parts = []
    for f in uploaded_files:
        with st.spinner(f"🚀 Đang rã cụm lắp ráp & hạ phẳng khối 3D: {f.name}"):
            try:
                parts = process_full_assembly_step(f.getvalue(), f.name, sheet_thickness, chord_tolerance)
                all_extracted_parts.extend(parts)
                st.success(f"✅ Rã thành công! Tìm thấy **{len(parts)}** tấm ván đủ điều kiện cắt từ tệp {f.name}.")
            except Exception as e:
                st.error(f"❌ Thất bại khi phân rã tệp {f.name}: {str(e)}")

    if all_extracted_parts:
        st.subheader("📦 KẾT QUẢ SẮP XẾP TỰ ĐỘNG - NESTING LAYOUT")
        nested_sheets = execute_production_nesting(all_extracted_parts, sheet_W, sheet_H, total_offset, margin)
        st.metric("Tổng số lượng tấm ván gốc cần sử dụng", len(nested_sheets))

        for sheet in nested_sheets:
            st.write(f"### 📋 Sơ đồ cắt tấm phôi thứ #{sheet['sheet_id']}")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_xlim(0, sheet_W)
            ax.set_ylim(0, sheet_H)
            ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, color="darkgrey", alpha=0.3, label="Khổ ván gốc"))
            ax.add_patch(mpatches.Rectangle((margin, margin), sheet_W - 2*margin, sheet_H - 2*margin, fill=False, linestyle="--", color="red"))
            
            all_gcode_blocks = generate_program_header(cnc_dialect)

            layer_pockets_todo = []
            layer_inners_todo = []
            layer_outers_todo = []

            for p_idx, placed in enumerate(sheet["parts"]):
                ref = placed["part_ref"]
                poly = placed["placed_polygon"]
                
                x, y = poly.exterior.xy
                ax.plot(x, y, "b-", linewidth=2)
                ax.fill(x, y, "skyblue", alpha=0.5)
                ax.text(poly.centroid.x, poly.centroid.y, f"P{p_idx+1}: {ref['name']}", ha='center', va='center', fontsize=8, weight='bold')

                trans_outer_edges = transform_edges_production(ref["outer_edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                outer_paths = get_true_offset_toolpath(trans_outer_edges, "CNC_OUTER_CUT", t3_dia / 2.0)
                for path in outer_paths:
                    if len(path) >= 2:
                        layer_outers_todo.append(path)

                for feat in ref["features"]:
                    trans_feat_edges = transform_edges_production(feat["edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                    
                    if feat["type"] == "CNC_POCKET":
                        pocket_paths = get_true_offset_toolpath(trans_feat_edges, "CNC_POCKET", t1_dia / 2.0)
                        for path in pocket_paths:
                            if len(path) >= 2:
                                layer_pockets_todo.append({"path": path, "depth": feat["depth"]})
                                
                    elif feat["type"] == "CNC_INNER_CUT":
                        inner_paths = get_true_offset_toolpath(trans_feat_edges, "CNC_INNER_CUT", t2_dia / 2.0)
                        for path in inner_paths:
                            if len(path) >= 2:
                                layer_inners_todo.append({"path": path, "depth": feat["depth"]})

            # LAYER 1: POCKETS
            if layer_pockets_todo:
                all_gcode_blocks.extend(generate_tool_change_block(cnc_dialect, tool_number=1, spindle_speed=t1_spindle, safe_z=safe_Z))
                for item in layer_pockets_todo:
                    path = item["path"]
                    px, py = zip(*path)
                    ax.plot(px, py, "m:", alpha=0.7)
                    all_gcode_blocks.extend(generate_gcode_for_toolpath(
                        path, "CNC_POCKET", item["depth"], max_stepdown, t1_feed, t1_plunge, safe_Z, 
                        enable_leadin, leadin_length, enable_ramping, False, tab_width, tab_thickness, tab_count_default
                    ))

            # LAYER 2: INNERS
            if layer_inners_todo:
                all_gcode_blocks.extend(generate_tool_change_block(cnc_dialect, tool_number=2, spindle_speed=t2_spindle, safe_z=safe_Z))
                for item in layer_inners_todo:
                    path = item["path"]
                    px, py = zip(*path)
                    ax.plot(px, py, "g--", alpha=0.7)
                    all_gcode_blocks.extend(generate_gcode_for_toolpath(
                        path, "CNC_INNER_CUT", item["depth"], max_stepdown, t2_feed, t2_plunge, safe_Z, 
                        enable_leadin, leadin_length, enable_ramping, False, tab_width, tab_thickness, tab_count_default
                    ))

            # LAYER 3: OUTERS
            if layer_outers_todo:
                all_gcode_blocks.extend(generate_tool_change_block(cnc_dialect, tool_number=3, spindle_speed=t3_spindle, safe_z=safe_Z))
                for path in layer_outers_todo:
                    px, py = zip(*path)
                    ax.plot(px, py, "r-", alpha=0.6, linewidth=1.5)
                    all_gcode_blocks.extend(generate_gcode_for_toolpath(
                        path, "CNC_OUTER_CUT", sheet_thickness + thru_overlap, max_stepdown, t3_feed, t3_plunge, safe_Z, 
                        enable_leadin, leadin_length, enable_ramping, enable_tabs, tab_width, tab_thickness, tab_count_default
                    ))

            all_gcode_blocks.extend(generate_program_footer(cnc_dialect, safe_Z))
            
            st.pyplot(fig)
            plt.close(fig)

            # Khu vực hiển thị kết quả G-Code
            gcode_text = "\n".join(all_gcode_blocks)
            st.download_button(
                label=f"💾 Tải xuống G-Code Tấm phôi #{sheet['sheet_id']}",
                data=gcode_text,
                file_name=f"Sheet_{sheet['sheet_id']}_CNC_Output.nc",
                mime="text/plain"
            )
            with st.expander(f"👀 Xem trước mã G-Code Tấm #{sheet['sheet_id']}"):
                st.code(gcode_text, language="gcode")
