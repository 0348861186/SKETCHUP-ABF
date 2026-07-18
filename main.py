import io
import os
import math
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches mpatches

# ============================================================
# CAD / CAM INDUSTRIAL CORE
# ============================================================
import cadquery as cq
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString
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
    page_title="Production-Ready CNC CAM Engine Pro v5.0",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#0F172A;'>
    🏭 PRODUCTION-READY CNC CAM ENGINE PRO V5.0
    </h1>
    <p style='text-align:center; color:#475569;'>
    STEP → CAD Analysis → Nesting → Advanced Toolpath (Ramp, Lead-in, Tabs, T-Bone) → G-Code
    </p>
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
max_stepdown = st.sidebar.number_input("Chiều sâu mỗi lát cắt Stepdown (mm)", min_value=0.5, value=6.0, step=0.5)

st.sidebar.markdown("### 🔩 LEAD-IN / RAMP / TABS")
enable_leadin = st.sidebar.checkbox("Kích hoạt Lead-in (Vào dao an toàn)", value=True)
leadin_length = st.sidebar.number_input("Chiều dài đoạn Lead-in (mm)", min_value=2.0, value=5.0, step=0.5)

enable_ramping = st.sidebar.checkbox("Kích hoạt Ramp thực tế (Xuống dao chéo)", value=True)
ramp_angle = st.sidebar.slider("Góc xuống dao Ramping (độ)", min_value=2, max_value=25, value=10)

enable_tabs = st.sidebar.checkbox("Kích hoạt Tabs thực tế (Cầu giữ phôi)", value=True)
tab_width = st.sidebar.number_input("Chiều dài Tab (mm)", min_value=5.0, value=15.0, step=1.0)
tab_thickness = st.sidebar.number_input("Độ dày mối nối Tab (mm)", min_value=0.5, value=4.0, step=0.5)
tab_count_default = st.sidebar.slider("Số lượng Tab / chi tiết", min_value=2, max_value=8, value=4)

st.sidebar.markdown("### ⚙ POST-PROCESSOR")
# ĐÃ BỔ SUNG: Thêm lựa chọn "UGS (Universal Gcode Sender)" vào đây
cnc_dialect = st.sidebar.selectbox(
    "Hệ điều hành / Phần mềm máy CNC", 
    ["Fanuc / Syntec", "Mach3 / Grbl", "UGS (Universal Gcode Sender)", "Weihong"]
)
safe_Z = st.sidebar.number_input("Safe Z (mm)", min_value=1.0, value=25.0, step=1.0)
thru_overlap = st.sidebar.number_input("Độ sâu cắt xuyên thêm (mm)", min_value=0.0, value=0.5, step=0.1)

tool_radius = t1_dia / 2.0
total_offset = tool_radius + safety_spacing

# ============================================================
# 3. GEOMETRY REPAIR
# ============================================================
def repair_geometry(poly):
    if poly is None or poly.is_empty:
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
# 4. CAD EDGE → LOCAL 2D COORDINATES & DISCRETIZATION
# ============================================================
def get_local_coordinates(cq_edge, plane_obj):
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
            segments = 64
            pts = []
            for i in range(segments + 1):
                t = first_param + (last_param - first_param) * i / segments
                p_loc = plane_obj.toLocalCoords(cq_edge.valueAt(t))
                pts.append((p_loc.x, p_loc.y))
            return {"type": "DISCRETE_CURVE", "points": pts}
        except Exception:
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
            for angle in np.linspace(0, 360, 96, endpoint=False):
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
# 5. IMPROVED T-BONE RELIEF (FIXED & STABLE)
# ============================================================
def apply_t_bone_relief(polygon_points, tool_radius):
    if len(polygon_points) < 4:
        return polygon_points
        
    pts = list(polygon_points)
    if np.allclose(pts[0], pts[-1]):
        pts.pop()
        
    poly = Polygon(pts)
    poly = repair_geometry(poly)
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
        len_v1 = np.linalg.norm(v1)
        len_v2 = np.linalg.norm(v2)
        
        if len_v1 < 1e-5 or len_v2 < 1e-5:
            result.append(tuple(p_curr))
            continue
            
        v1_u = v1 / len_v1
        v2_u = v2 / len_v2
        
        dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
        angle = math.acos(dot)
        
        cross_z = v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0]
        is_concave = cross_z > 0.001
        is_right_angle = abs(angle - math.pi / 2.0) < math.radians(10)
        
        result.append(tuple(p_curr))
        
        if is_concave and is_right_angle:
            bisector = v1_u + v2_u
            norm_b = np.linalg.norm(bisector)
            if norm_b > 1e-5:
                bisector_u = bisector / norm_b
                relief_distance = tool_radius
                relief_point = p_curr + bisector_u * relief_distance
                
                result.append(tuple(relief_point))
                result.append(tuple(p_curr))
                
    result.append(result[0])
    return result

# ============================================================
# 6. STEP FILE ANALYSIS
# ============================================================
def process_cad_file_production(file_bytes, filename, sheet_thick):
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
        outer_edges = [get_local_coordinates(edge, ref_plane) for edge in outer_wire.Edges()]
        
        features = []
        for inner_wire in target_face.innerWires():
            wire_edges = [get_local_coordinates(edge, ref_plane) for edge in inner_wire.Edges()]
            features.append({"type": "CNC_INNER_CUT", "edges": wire_edges, "depth": sheet_thick})
            
        for face in all_faces:
            if face is target_face:
                continue
            if face.geomType() == "PLANE" and face.Area() < target_face.Area() * 0.9:
                f_z = face.Center().z
                if f_z < face_z_level and f_z >= (face_z_level - sheet_thick):
                    depth = abs(face_z_level - f_z)
                    if depth > 0.2:
                        p_edges = [get_local_coordinates(edge, ref_plane) for edge in face.outerWire().Edges()]
                        features.append({"type": "CNC_POCKET", "edges": p_edges, "depth": depth})
                        
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
# 7. NESTING ENGINE & TOOLPATH GENERATOR
# ============================================================
def transform_point_production(x, y, dx, dy, angle, ox, oy):
    x_local, y_local = x - ox, y - oy
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return (x_local * cos_a - y_local * sin_a + dx, x_local * sin_a + y_local * cos_a + dy)

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
            
        buffered_poly = extract_largest_polygon(repair_geometry(poly_geom.buffer(offset_val, resolution=8, join_style=JOIN_STYLE.round)))
        if buffered_poly is None:
            continue
            
        min_x, min_y, _, _ = buffered_poly.bounds
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized = translate(poly_geom, xoff=-min_x, yoff=-min_y)
        
        best_pos = None
        target_sheet_idx = -1
        best_score = float("inf")
        
        for sheet_idx, sheet_data in enumerate(sheets):
            placed_polys = sheet_data["placed_buffered_polygons"]
            anchors = [(margin_val, margin_val)]
            for pb in placed_polys:
                b = pb.bounds
                anchors.extend([(b[2], b[1]), (b[0], b[3]), (b[2], b[3])])
                
            for angle in [0, 90, 180, 270]:
                rotated_poly = rotate(normalized_poly, angle, origin=(0, 0))
                r_min_x, r_min_y, _, _ = rotated_poly.bounds
                for anchor_x, anchor_y in anchors:
                    dx = anchor_x - r_min_x
                    dy = anchor_y - r_min_y
                    candidate = translate(rotated_poly, xoff=dx, yoff=dy)
                    
                    if not sheet_bound.covers(candidate):
                        continue
                    if any(candidate.intersects(placed) for placed in placed_polys):
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
        else:
            new_sheet_id = len(sheets) + 1
            dx, dy = margin_val - min_x, margin_val - min_y
            sheets.append({
                "sheet_id": new_sheet_id,
                "parts": [{
                    "part_ref": part, "original_offset": (min_x, min_y),
                    "placed_polygon": translate(raw_normalized, xoff=dx, yoff=dy), "dx": dx, "dy": dy, "angle": 0
                }],
                "placed_buffered_polygons": [translate(normalized_poly, xoff=dx, yoff=dy)]
            })
    return sheets

def get_true_offset_toolpath(edges, op_type, tool_radius):
    raw_pts = discrete_edges(edges)
    cleaned = clean_polygon_points(raw_pts)
    if len(cleaned) < 4:
        return cleaned
        
    if op_type == "CNC_INNER_CUT":
        cleaned = apply_t_bone_relief(cleaned, tool_radius)
        
    poly = extract_largest_polygon(repair_geometry(Polygon(cleaned)))
    if poly is None:
        return cleaned
        
    offset_val = tool_radius if op_type == "CNC_OUTER_CUT" else -tool_radius
    offset_geom = extract_largest_polygon(repair_geometry(poly.buffer(offset_val, resolution=8, join_style=JOIN_STYLE.round)))
    if offset_poly := offset_geom:
        return list(offset_poly.exterior.coords)
    return cleaned

# ============================================================
# 🔥 8 & 9 & 10. REAL RAMP, LEAD-IN & TABS G-CODE GENERATOR
# ============================================================
def generate_gcode_for_toolpath(toolpath_pts, op_type, total_depth, max_step, feed, plunge, spindle, safe_z, r_angle, t_width, t_thick, t_count):
    gcode = []
    if len(toolpath_pts) < 2:
        return gcode

    pts = list(toolpath_pts)
    if np.allclose(pts[0], pts[-1]):
        pts.pop()
    
    n_pts = len(pts)
    
    tab_segments = []
    if enable_tabs and op_type == "CNC_OUTER_CUT" and n_pts > 4:
        step_dist = n_pts // t_count
        for idx in range(t_count):
            t_idx = (idx * step_dist) % n_pts
            tab_segments.append(t_idx)

    gcode.append(f"; --- BAT DAU CHUONG TRINH {op_type} ---")
    gcode.append(f"M3 S{int(spindle)}")
    
    start_pt = np.array(pts[0])
    next_pt = np.array(pts[1])
    vec_dir = next_pt - start_pt
    norm_v = np.linalg.norm(vec_dir)
    
    if norm_v > 1e-5 and enable_leadin:
        perp_vec = np.array([-vec_dir[1], vec_dir[0]]) / norm_v
        leadin_start = start_pt + perp_vec * leadin_length
        gcode.append(f"G0 X{leadin_start[0]:.3f} Y{leadin_start[1]:.3f} Z{safe_z:.3f}")
        gcode.append(f"G1 Z0.0 F{plunge}")
        gcode.append(f"G1 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} F{feed}")
    else:
        gcode.append(f"G0 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{safe_z:.3f}")
        gcode.append(f"G1 Z0.0 F{plunge}")

    z_layers = []
    current_z = 0.0
    while current_z > -total_depth:
        current_z -= max_step
        if current_z < -total_depth:
            current_z = -total_depth
        z_layers.append(current_z)

    for l_idx, target_z in enumerate(z_layers):
        gcode.append(f"; --- Lat cat lop Z = {target_z:.3f} ---")
        
        if enable_ramping and l_idx > 0:
            prev_z = z_layers[l_idx - 1]
            z_dist = abs(target_z - prev_z)
            ramp_dist = z_dist / math.tan(math.radians(r_angle))
            
            accum_dist = 0.0
            r_i = 0
            while accum_dist < ramp_dist and r_i < n_pts:
                p_c = np.array(pts[r_i % n_pts])
                p_n = np.array(pts[(r_i + 1) % n_pts])
                d = np.linalg.norm(p_n - p_c)
                accum_dist += d
                r_i += 1
            
            gcode.append(f"; Thuc hien xuong dao Ramp cheo (Goc {ramp_angle} do)")
            gcode.append(f"G1 X{pts[min(r_i, n_pts-1)][0]:.3f} Y{pts[min(r_i, n_pts-1)][1]:.3f} Z{target_z:.3f} F{feed}")
        else:
            gcode.append(f"G1 Z{target_z:.3f} F{plunge}")

        for i in range(n_pts + 1):
            curr_idx = i % n_pts
            pt = pts[curr_idx]
            
            if curr_idx in tab_segments and target_z <= -(total_depth - t_thick):
                tab_z = -(total_depth - t_thick)
                gcode.append(f"; --- Bat dau cau giu khoi (3D TAB) ---")
                gcode.append(f"G1 Z{tab_z:.3f} F{plunge}")
                gcode.append(f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} F{feed}")
                gcode.append(f"G1 Z{target_z:.3f} F{plunge}")
            else:
                gcode.append(f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} F{feed}")
                
    gcode.append(f"G0 Z{safe_z:.3f}")
    return gcode

# ============================================================
# 11. STREAMLIT APPLICATION ORCHESTRATION LAYER
# ============================================================
uploaded_files = st.file_uploader("Tải lên các tệp STEP chi tiết cần gia công CNC (.step / .stp)", type=["step", "stp"], accept_multiple_files=True)

if uploaded_files:
    loaded_parts = []
    st.info(f"Đang phân tích cú pháp dữ liệu cấu trúc hình học của {len(uploaded_files)} tệp tin...")
    
    for f in uploaded_files:
        try:
            p_data = process_cad_file_production(f.getvalue(), f.name, sheet_thickness)
            loaded_parts.append(p_data)
            st.success(f"Đã nạp và giải mã thành công hình học của: {p_data['name']}")
        except Exception as e:
            st.error(f"Lỗi khi đọc file {f.name}: {str(e)}")

    if loaded_parts:
        st.markdown("---")
        st.subheader("📦 KẾT QUẢ XẾP PHÔI TỰ ĐỘNG (AUTOMATIC NESTING)")
        
        sheets_result = execute_production_nesting(loaded_parts, sheet_W, sheet_H, total_offset, margin)
        
        st.metric("Tổng số tấm ván (Plates) yêu cầu", len(sheets_result))
        
        all_gcode_blocks = []
        
        for sh in sheets_result:
            st.markdown(f"### 📋 TẤM VÁN SỐ: {sh['sheet_id']}")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlim(0, sheet_W)
            ax.set_ylim(0, sheet_H)
            ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, color="#E2E8F0", ec="#94A3B8"))
            
            sheet_gcode = [
                f"; --- SHEET ID: {sh['sheet_id']} ---",
                "G21 ; Don vi mm",
                "G90 ; Toa do tuyet doi"
            ]
            
            for p_idx, p_info in enumerate(sh["parts"]):
                ref = p_info["part_ref"]
                dx, dy, angle = p_info["dx"], p_info["dy"], p_info["angle"]
                ox, oy = ref["origin_x"], ref["origin_y"]
                
                ext_coords = list(p_info["placed_polygon"].exterior.coords)
                xs, ys = zip(*ext_coords)
                ax.plot(xs, ys, label=f"{ref['name']} ({p_idx})", lw=1.5)
                ax.fill(xs, ys, alpha=0.3)
                
                for feat in ref["features"]:
                    if feat["type"] == "CNC_INNER_CUT":
                        trans_edges = []
                        for edge in feat["edges"]:
                            if edge["type"] == "LINE":
                                start_t = transform_point_production(edge["start"][0], edge["start"][1], dx, dy, angle, ox, oy)
                                end_t = transform_point_production(edge["end"][0], edge["end"][1], dx, dy, angle, ox, oy)
                                trans_edges.append({"type": "LINE", "start": start_t, "end": end_t})
                                
                        toolpath_pts = get_true_offset_toolpath(trans_edges, "CNC_INNER_CUT", tool_radius)
                        f_gcode = generate_gcode_for_toolpath(
                            toolpath_pts, "CNC_INNER_CUT", feat["depth"], max_stepdown,
                            t1_feed, t1_plunge, t1_spindle, safe_Z, ramp_angle, tab_width, tab_thickness, tab_count_default
                        )
                        sheet_gcode.extend(f_gcode)
                
                outer_trans_edges = []
                for edge in ref["outer_edges"]:
                    if edge["type"] == "LINE":
                        start_t = transform_point_production(edge["start"][0], edge["start"][1], dx, dy, angle, ox, oy)
                        end_t = transform_point_production(edge["end"][0], edge["end"][1], dx, dy, angle, ox, oy)
                        outer_trans_edges.append({"type": "LINE", "start": start_t, "end": end_t})
                        
                outer_toolpath = get_true_offset_toolpath(outer_trans_edges, "CNC_OUTER_CUT", tool_radius)
                
                if outer_toolpath:
                    txs, tys = zip(*outer_toolpath)
                    ax.plot(txs, tys, 'r--', lw=1.0, alpha=0.7)
                
                o_gcode = generate_gcode_for_toolpath(
                    outer_toolpath, "CNC_OUTER_CUT", sheet_thickness + thru_overlap, max_stepdown,
                    t1_feed, t1_plunge, t1_spindle, safe_Z, ramp_angle, tab_width, tab_thickness, tab_count_default
                )
                sheet_gcode.extend(o_gcode)
                
            sheet_gcode.extend(["M5", "M30 ; Ket thuc chuong trinh"])
            full_sheet_gcode_str = "\n".join(sheet_gcode)
            all_gcode_blocks.append(full_sheet_gcode_str)
            
            st.pyplot(fig)
            
            st.download_button(
                label=f"💾 Tải xuống mã ATC G-Code cho Tấm ván số {sh['sheet_id']}",
                data=full_sheet_gcode_str,
                file_name=f"CNC_CAM_ENGINE_SHEET_{sh['sheet_id']}.nc",
                mime="text/plain"
            )
            
            with st.expander(f"🔍 Xem trước khối lệnh G-Code (200 dòng đầu tiên) của tấm số {sh['sheet_id']}"):
                st.code("\n".join(sheet_gcode[:200]), language="gcode")
