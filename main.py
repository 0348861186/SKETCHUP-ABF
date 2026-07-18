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
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
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
st.set_page_config(page_title="CNC CAM Engine Pro ATC v7.0", layout="wide")
st.markdown("## 🏭 CNC CAM ENGINE PRO V7.0 - AUTOMATIC ATC & MULTI-TOOL MANAGEMENT")

# ============================================================
# 2. SIDEBAR - ĐỔI THÀNH QUẢN LÝ 3 LOẠI DAO RIÊNG BIỆT
# ============================================================
st.sidebar.header("📐 THÔNG SỐ VẬT LIỆU PHÔI")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván X (mm)", min_value=100.0, value=2440.0, step=10.0)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván Y (mm)", min_value=100.0, value=1220.0, step=10.0)
sheet_thickness = st.sidebar.number_input("Độ dày ván tiêu chuẩn Z (mm)", min_value=0.1, value=17.0, step=0.1)
margin = st.sidebar.number_input("Khoảng cách biên tấm ván (mm)", min_value=0.0, value=15.0, step=1.0)
safety_spacing = st.sidebar.number_input("Khoảng cách giữa các chi tiết (mm)", min_value=0.0, value=6.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🧰 CẤU HÌNH ĐÀI DAO (ATC SPINDLE)")

# Cấu hình Dao 1: Cắt đứt
st.sidebar.markdown("### 🔴 DAO 1 (Cắt đứt biên dạng)")
t1_slot = st.sidebar.number_input("Vị trí ổ dao (T)", min_value=1, value=1, step=1)
t1_dia = st.sidebar.number_input("Đường kính dao 1 (mm)", min_value=0.1, value=6.0, step=0.1)
t1_feed = st.sidebar.number_input("Tốc độ F1 (mm/min)", min_value=100, value=3500, step=100)

# Cấu hình Dao 2: Khoét lỗ thủng
st.sidebar.markdown("### 🟢 DAO 2 (Khoét lỗ thủng bên trong)")
t2_slot = st.sidebar.number_input("Vị trí ổ dao (T) ", min_value=1, value=2, step=1)
t2_dia = st.sidebar.number_input("Đường kính dao 2 (mm)", min_value=0.1, value=4.0, step=0.1)
t2_feed = st.sidebar.number_input("Tốc độ F2 (mm/min)", min_value=100, value=2500, step=100)

# Cấu hình Dao 3: Hạ nền / Làm túi
st.sidebar.markdown("### 🔵 DAO 3 (Phá lòng / Hạ nền / Pocket)")
t3_slot = st.sidebar.number_input("Vị trí ổ dao (T)  ", min_value=1, value=3, step=1)
t3_dia = st.sidebar.number_input("Đường kính dao 3 (mm)", min_value=0.1, value=10.0, step=0.1)
t3_feed = st.sidebar.number_input("Tốc độ F3 (mm/min)", min_value=100, value=4000, step=100)

st.sidebar.markdown("---")
st.sidebar.header("⚙ THÔNG SỐ VẬN HÀNH CHUNG")
t1_plunge = st.sidebar.number_input("Tốc độ đâm dao F_plunge (mm/min)", min_value=50, value=1200, step=50)
t1_spindle = st.sidebar.number_input("Tốc độ trục chính S (RPM)", min_value=1000, value=18000, step=500)
max_stepdown = st.sidebar.number_input("Chiều sâu mỗi lớp Stepdown (mm)", min_value=0.5, value=6.0, step=0.5)
safe_Z = st.sidebar.number_input("Safe Z (mm)", min_value=1.0, value=25.0, step=1.0)
thru_overlap = st.sidebar.number_input("Độ sâu cắt xuyên thêm (mm)", min_value=0.0, value=0.5, step=0.1)
chord_tolerance = 0.02

# Gom từ điển quản lý dao để truyền vào hàm tính toán
TOOLS_CONFIG = {
    "CNC_OUTER_CUT": {"slot": int(t1_slot), "radius": t1_dia / 2.0, "feed": t1_feed},
    "CNC_INNER_CUT": {"slot": int(t2_slot), "radius": t2_dia / 2.0, "feed": t2_feed},
    "CNC_POCKET": {"slot": int(t3_slot), "radius": t3_dia / 2.0, "feed": t3_feed}
}

# Tận dụng lại bộ lề an toàn tối đa của dao lớn nhất để chống va chạm phôi khi Nesting
max_tool_radius = max(t1_dia, t2_dia, t3_dia) / 2.0
total_offset = max_tool_radius + safety_spacing

# ============================================================
# CÁC HÀM XỬ LÝ HÌNH HỌC (Giữ nguyên logic từ v6.0)
# ============================================================
def repair_geometry(geom):
    if geom is None or geom.is_empty: return geom
    if geom.is_valid: return geom
    if make_valid is not None:
        try:
            fixed = make_valid(geom)
            if not fixed.is_empty: return fixed
        except: pass
    return geom.buffer(0)

def extract_largest_polygon(geom):
    if geom is None or geom.is_empty: return None
    if isinstance(geom, Polygon): return geom
    if isinstance(geom, MultiPolygon): return max(geom.geoms, key=lambda p: p.area)
    return None

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
            f_p, l_p = occ_curve.FirstParameter(), occ_curve.LastParameter()
            pts = []
            for i in range(65):
                t = f_p + (l_p - f_p) * i / 64
                p_loc = plane_obj.toLocalCoords(cq_edge.valueAt(t))
                pts.append((p_loc.x, p_loc.y))
            return {"type": "DISCRETE_CURVE", "points": pts}
        except:
            return {"type": "LINE", "start": (p_start.x, p_start.y), "end": (p_end.x, p_end.y)}

def discrete_edges(edges):
    pts = []
    for edge in edges:
        if edge["type"] == "LINE":
            pts.extend([edge["start"], edge["end"]])
        elif edge["type"] == "CIRCLE":
            cx, cy = edge["center"]
            for a in np.linspace(0, 360, 64, endpoint=False):
                pts.append((cx + edge["radius"] * math.cos(math.radians(a)), cy + edge["radius"] * math.sin(math.radians(a))))
        elif edge["type"] == "DISCRETE_CURVE":
            pts.extend(edge["points"])
    return pts

def clean_polygon_points(points):
    cleaned = []
    for p in points:
        if not cleaned or not np.allclose(cleaned[-1], p, atol=0.01):
            cleaned.append((float(p[0]), float(p[1])))
    if len(cleaned) > 2 and not np.allclose(cleaned[0], cleaned[-1], atol=0.01):
        cleaned.append(cleaned[0])
    return cleaned

def process_full_assembly_step(file_bytes, filename, std_thickness, tol_val):
    temp_path = None
    parsed_parts = []
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name
        imported_shape = cq.importers.importStep(temp_path)
        solids = imported_shape.solids().vals()
        for idx, solid in enumerate(solids):
            if solid.Area() < 500: continue 
            faces = solid.faces().vals()
            plane_faces = [f for f in faces if f.geomType() == "PLANE"]
            if not plane_faces: continue
            target_face = max(plane_faces, key=lambda f: f.Area())
            face_center = target_face.Center()
            ref_plane = cq.Plane(origin=face_center, normal=target_face.normalAt(face_center))
            
            outer_wire = target_face.outerWire()
            outer_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in outer_wire.Edges()]
            
            features = []
            for inner_wire in target_face.innerWires():
                wire_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in inner_wire.Edges()]
                features.append({"type": "CNC_INNER_CUT", "edges": wire_edges, "depth": std_thickness})

            for face in faces:
                if face is target_face or face.geomType() != "PLANE": continue
                local_center = ref_plane.toLocalCoords(face.Center())
                depth = abs(local_center.z)
                if 0.5 <= depth < (std_thickness + 2.0):
                    p_edges = [get_local_coordinates(edge, ref_plane, tol_val) for edge in face.outerWire().Edges()]
                    if p_edges: features.append({"type": "CNC_POCKET", "edges": p_edges, "depth": depth})

            raw_outer_pts = clean_polygon_points(discrete_edges(outer_edges))
            if len(raw_outer_pts) < 4: continue
            poly_outer = repair_geometry(Polygon(raw_outer_pts))
            if poly_outer is None or poly_outer.is_empty: continue
            min_x, min_y, max_x, max_y = poly_outer.bounds
            
            parsed_parts.append({
                "name": f"Tam_{idx+1}", "width": max_x - min_x, "height": max_y - min_y,
                "outer_edges": outer_edges, "features": features, "origin_x": face_center.x, "origin_y": face_center.y
            })
    finally:
        if temp_path and os.path.exists(temp_path): os.remove(temp_path)
    return parsed_parts

def transform_point_production(x, y, dx, dy, angle, ox, oy):
    x_l, y_l = x - ox, y - oy
    rad = math.radians(angle)
    return (x_l * math.cos(rad) - y_l * math.sin(rad) + dx, x_l * math.sin(rad) + y_l * math.cos(rad) + dy)

def transform_edge_production(edge, dx, dy, angle, ox, oy):
    if edge["type"] == "LINE":
        return {"type": "LINE", "start": transform_point_production(edge["start"][0], edge["start"][1], dx, dy, angle, ox, oy), "end": transform_point_production(edge["end"][0], edge["end"][1], dx, dy, angle, ox, oy)}
    elif edge["type"] == "CIRCLE":
        return {"type": "CIRCLE", "center": transform_point_production(edge["center"][0], edge["center"][1], dx, dy, angle, ox, oy), "radius": edge["radius"]}
    elif edge["type"] == "DISCRETE_CURVE":
        return {"type": "DISCRETE_CURVE", "points": [transform_point_production(x, y, dx, dy, angle, ox, oy) for x, y in edge["points"]]}
    return edge

def transform_edges_production(edges, dx, dy, angle, ox, oy):
    return [transform_edge_production(e, dx, dy, angle, ox, oy) for e in edges]

# ============================================================
# NESTING ENGINE
# ============================================================
def execute_production_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    sheet_bound = Polygon([(margin_val, margin_val), (sheet_w - margin_val, margin_val), (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)])
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    sheets = []
    for part in sorted_parts:
        raw_points = clean_polygon_points(discrete_edges(part["outer_edges"]))
        poly_geom = extract_largest_polygon(repair_geometry(Polygon(raw_points)))
        if poly_geom is None: continue
        buffered_poly = extract_largest_polygon(repair_geometry(poly_geom.buffer(offset_val, resolution=8, join_style=JOIN_STYLE.round)))
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
                    score = candidate.bounds[0] + candidate.bounds[1] * 2.5
                    if score < best_score:
                        best_score = score
                        target_sheet_idx = sheet_idx
                        best_pos = {"dx": dx, "dy": dy, "angle": angle, "cand_poly": candidate, "raw_trans": translate(rotate(raw_normalized, angle, origin=(0, 0)), xoff=dx, yoff=dy)}
        if best_pos is not None and target_sheet_idx >= 0:
            sheets[target_sheet_idx]["parts"].append({"part_ref": part, "original_offset": (min_x, min_y), "placed_polygon": best_pos["raw_trans"], "dx": best_pos["dx"], "dy": best_pos["dy"], "angle": best_pos["angle"]})
            sheets[target_sheet_idx]["placed_buffered_polygons"].append(best_pos["cand_poly"])
            sheets[target_sheet_idx]["placed_union_geom"] = sheets[target_sheet_idx]["placed_union_geom"].union(best_pos["cand_poly"])
        else:
            dx, dy = margin_val - min_x, margin_val - min_y
            init_poly = translate(normalized_poly, xoff=dx, yoff=dy)
            sheets.append({"sheet_id": len(sheets) + 1, "parts": [{"part_ref": part, "original_offset": (min_x, min_y), "placed_polygon": translate(raw_normalized, xoff=dx, yoff=dy), "dx": dx, "dy": dy, "angle": 0}], "placed_buffered_polygons": [init_poly], "placed_union_geom": init_poly})
    return sheets

# ============================================================
# TOOLPATH GENERATION WITH TARGETED TOOL RADIUS
# ============================================================
def get_true_offset_toolpath(edges, op_type, tool_radius):
    raw_pts = discrete_edges(edges)
    cleaned = clean_polygon_points(raw_pts)
    if len(cleaned) < 4: return []
    poly = extract_largest_polygon(repair_geometry(Polygon(cleaned)))
    if poly is None: return []

    if op_type == "CNC_OUTER_CUT":
        offset_geom = repair_geometry(poly.buffer(tool_radius, resolution=8, join_style=JOIN_STYLE.round))
        return [list(offset_geom.exterior.coords)] if isinstance(offset_geom, Polygon) else []
    elif op_type == "CNC_INNER_CUT":
        offset_geom = repair_geometry(poly.buffer(-tool_radius, resolution=8, join_style=JOIN_STYLE.round))
        return [list(offset_geom.exterior.coords)] if isinstance(offset_geom, Polygon) else []
    elif op_type == "CNC_POCKET":
        paths = []
        stepover = tool_radius * 0.75
        current_offset = -tool_radius
        while True:
            offset_geom = repair_geometry(poly.buffer(current_offset, resolution=8, join_style=JOIN_STYLE.round))
            if offset_geom is None or offset_geom.is_empty: break
            if isinstance(offset_geom, Polygon): paths.append(list(offset_geom.exterior.coords))
            current_offset -= stepover
        return paths
    return []

# ============================================================
# NEW ATC G-CODE ENGINE (TỰ ĐỘNG PHÂN LỚP VÀ CHÈN LỆNH THAY DAO)
# ============================================================
def generate_gcode_for_toolpath(toolpath_pts, target_z, max_step, feed, plunge, safe_z):
    gcode = []
    if len(toolpath_pts) < 2: return gcode
    pts = list(toolpath_pts)
    start_pt = pts[0]
    
    gcode.append(f"G0 X{start_pt[0]:.3f} Y{start_pt[1]:.3f} Z{safe_z:.3f}")
    
    z_targets = []
    curr_z = 0.0
    while curr_z > -target_z:
        curr_z -= max_step
        if curr_z < -target_z: curr_z = -target_z
        z_targets.append(curr_z)

    for tz in z_targets:
        gcode.append(f"G1 Z{tz:.3f} F{plunge}")
        for p in pts[1:]:
            gcode.append(f"G1 X{p[0]:.3f} Y{p[1]:.3f} F{feed}")
    gcode.append(f"G0 Z{safe_z:.3f}")
    return gcode

# ============================================================
# MAIN APP ORCHESTRATION
# ============================================================
uploaded_files = st.file_uploader("Tải lên bản vẽ thiết kế 3D (.STEP / .STP)", accept_multiple_files=True)

if uploaded_files:
    all_extracted_parts = []
    for f in uploaded_files:
        with st.spinner(f"🚀 Đang phân rã khối 3D: {f.name}"):
            parts = process_full_assembly_step(f.getvalue(), f.name, sheet_thickness, chord_tolerance)
            all_extracted_parts.extend(parts)

    if all_extracted_parts:
        nested_sheets = execute_production_nesting(all_extracted_parts, sheet_W, sheet_H, total_offset, margin)

        for sheet in nested_sheets:
            st.write(f"### 📋 Sơ đồ cắt & Cấu hình đài dao Tấm #{sheet['sheet_id']}")
            
            # Khởi tạo các mảng G-code phân lớp nhiệm vụ
            pocket_gcode_layer = []
            inner_gcode_layer = []
            outer_gcode_layer = []

            # QUÉT TOÀN BỘ CHI TIẾT TRÊN TẤM VÁN ĐỂ GOM NHÓM THEO THỨ TỰ CÔNG NGHỆ GIA CÔNG
            for placed in sheet["parts"]:
                ref = placed["part_ref"]
                
                # LỚP 1: Gom Tác vụ Hạ Nền (Sử dụng Dao 3)
                for feat in [f for f in ref["features"] if f["type"] == "CNC_POCKET"]:
                    t_edges = transform_edges_production(feat["edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                    paths = get_true_offset_toolpath(t_edges, "CNC_POCKET", TOOLS_CONFIG["CNC_POCKET"]["radius"])
                    for p in paths:
                        pocket_gcode_layer.extend(generate_gcode_for_toolpath(p, feat["depth"], max_stepdown, TOOLS_CONFIG["CNC_POCKET"]["feed"], t1_plunge, safe_Z))

                # LỚP 2: Gom Tác vụ Khoét Lỗ Thủng (Sử dụng Dao 2)
                for feat in [f for f in ref["features"] if f["type"] == "CNC_INNER_CUT"]:
                    t_edges = transform_edges_production(feat["edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                    paths = get_true_offset_toolpath(t_edges, "CNC_INNER_CUT", TOOLS_CONFIG["CNC_INNER_CUT"]["radius"])
                    for p in paths:
                        inner_gcode_layer.extend(generate_gcode_for_toolpath(p, sheet_thickness + thru_overlap, max_stepdown, TOOLS_CONFIG["CNC_INNER_CUT"]["feed"], t1_plunge, safe_Z))

                # LỚP 3: Gom Tác vụ Cắt Đứt Biên Dạng (Sử dụng Dao 1)
                trans_outer_edges = transform_edges_production(ref["outer_edges"], placed["dx"], placed["dy"], placed["angle"], ref["origin_x"], ref["origin_y"])
                outer_paths = get_true_offset_toolpath(trans_outer_edges, "CNC_OUTER_CUT", TOOLS_CONFIG["CNC_OUTER_CUT"]["radius"])
                for p in outer_paths:
                    outer_gcode_layer.extend(generate_gcode_for_toolpath(p, sheet_thickness + thru_overlap, max_stepdown, TOOLS_CONFIG["CNC_OUTER_CUT"]["feed"], t1_plunge, safe_Z))

            # COMPINE HÀO TRÌNH G-CODE TOÀN DIỆN VÀ CHÈN ATC TỰ ĐỘNG
            final_gcode = [
                "%", "G90 G21 G17 G40 G49 G80", f"G0 Z{safe_Z:.3f}"
            ]

            # Thực thi Lớp 1 (Nếu có tác vụ hạ nền) -> Gọi dao T3
            if pocket_gcode_layer:
                final_gcode.append(f"; >>> BẮT ĐẦU TÁC VỤ HẠ NỀN - GỌI DAO T{TOOLS_CONFIG['CNC_POCKET']['slot']} <<<")
                final_gcode.append(f"T{TOOLS_CONFIG['CNC_POCKET']['slot']} M6") # Lệnh Thay dao tự động ATC
                final_gcode.append(f"G43 H{TOOLS_CONFIG['CNC_POCKET']['slot']} Z{safe_Z:.3f}")
                final_gcode.append(f"S{int(t1_spindle)} M3")
                final_gcode.extend(pocket_gcode_layer)
                final_gcode.append("M5") # Dừng trục chính trước khi đổi dao

            # Thực thi Lớp 2 (Nếu có tác vụ cắt thủng) -> Gọi dao T2
            if inner_gcode_layer:
                final_gcode.append(f"; >>> BẮT ĐẦU TÁC VỤ CẮT LỖ THỦNG - GỌI DAO T{TOOLS_CONFIG['CNC_INNER_CUT']['slot']} <<<")
                final_gcode.append(f"T{TOOLS_CONFIG['CNC_INNER_CUT']['slot']} M6") # Lệnh Thay dao tự động ATC
                final_gcode.append(f"G43 H{TOOLS_CONFIG['CNC_INNER_CUT']['slot']} Z{safe_Z:.3f}")
                final_gcode.append(f"S{int(t1_spindle)} M3")
                final_gcode.extend(inner_gcode_layer)
                final_gcode.append("M5")

            # Thực thi Lớp 3 (Luôn có: Cắt đứt phôi) -> Gọi dao T1
            if outer_gcode_layer:
                final_gcode.append(f"; >>> BẮT ĐẦU TÁC VỤ CẮT ĐỨT BIÊN DẠNG - GỌI DAO T{TOOLS_CONFIG['CNC_OUTER_CUT']['slot']} <<<")
                final_gcode.append(f"T{TOOLS_CONFIG['CNC_OUTER_CUT']['slot']} M6") # Lệnh Thay dao tự động ATC
                final_gcode.append(f"G43 H{TOOLS_CONFIG['CNC_OUTER_CUT']['slot']} Z{safe_Z:.3f}")
                final_gcode.append(f"S{int(t1_spindle)} M3")
                final_gcode.extend(outer_gcode_layer)
                final_gcode.append("M5")

            final_gcode.extend(["G0 Z50.000", "G28 G91 X0 Y0", "M30", "%"])

            # Render đồ họa minh họa sơ đồ (giản lược để tối ưu hiển thị)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_xlim(0, sheet_W)
            ax.set_ylim(0, sheet_H)
            ax.add_patch(mpatches.Rectangle((0,0), sheet_W, sheet_H, color="grey", alpha=0.2))
            for placed in sheet["parts"]:
                x, y = placed["placed_polygon"].exterior.xy
                ax.plot(x, y, "b-")
            st.pyplot(fig)
            plt.close(fig)

            st.download_button(
                label=f"💾 Tải xuống G-Code ATC Tấm #{sheet['sheet_id']}",
                data="\n".join(final_gcode),
                file_name=f"ATC_Production_Sheet_{sheet['sheet_id']}.nc",
                mime="text/plain"
            )
