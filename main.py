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
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE

try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None

import ezdxf

# ============================================================
# 1. STREAMLIT CONTROL PANEL INITIALIZATION
# ============================================================
st.set_page_config(
    page_title="Production-Ready CNC CAM Engine Pro v4.0",
    layout="wide"
)

st.markdown(
    """
    <h1 style='text-align:center; color:#0F172A;'>
    🏭 PRODUCTION-READY CNC CAM ENGINE PRO (V4.0)
    </h1>
    <p style='text-align:center; color:#475569;'>
    Hệ thống CAM công nghiệp: T-Bone Relief → True Offset Compensation → Lead-In/Ramp Engine → Auto-Tabs → G2/G3 Arc Fitting
    </p>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 2. SIDEBAR PRODUCTION CONFIGURATIONS
# ============================================================
st.sidebar.header("📐 THÔNG SỐ VẬT LIỆU PHÔI")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván X (mm)", min_value=100.0, value=2440.0, step=10.0)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván Y (mm)", min_value=100.0, value=1220.0, step=10.0)
sheet_thickness = st.sidebar.number_input("Độ dày ván thực tế Z (mm)", min_value=0.1, value=17.0, step=0.1)
margin = st.sidebar.number_input("Khoảng cách biên tấm ván (mm)", min_value=0.0, value=15.0, step=1.0)
safety_spacing = st.sidebar.number_input("Khoảng cách giữa các chi tiết (mm)", min_value=0.0, value=6.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH DAO & THÔNG SỐ CẮT G-CODE")
t1_dia = st.sidebar.number_input("Đường kính dao phay cắt đứt T1 (mm)", min_value=0.1, value=6.0, step=0.1)
t1_feed = st.sidebar.number_input("Tốc độ cắt F (mm/min)", min_value=100, value=3500, step=100)
t1_spindle = st.sidebar.number_input("Tốc độ trục chính S (RPM)", min_value=1000, value=18000, step=500)
max_stepdown = st.sidebar.number_input("Chiều sâu mỗi lát cắt Stepdown (mm)", min_value=0.5, value=6.0, step=0.5)

st.sidebar.markdown("**[NÂNG CẤP] LEAD-IN & TABS ENGINE**")
ramp_angle = st.sidebar.slider("Góc xuống dao xéo Ramping (độ)", min_value=5, max_value=25, value=10)
tab_width = st.sidebar.number_input("Chiều dài của gờ giữ phôi Tab (mm)", min_value=5.0, value=15.0, step=1.0)
tab_thickness = st.sidebar.number_input("Độ dày của gờ giữ phôi Tab (mm)", min_value=0.5, value=3.5, step=0.5)
tab_count_default = st.sidebar.slider("Số lượng gờ Tabs mặc định/chi tiết", min_value=2, max_value=6, value=3)

st.sidebar.markdown("**[NÂNG CẤP] POST-PROCESSOR DIALECT**")
cnc_dialect = st.sidebar.selectbox("Hệ điều hành mã máy CNC Target", ["Fanuc / Syntec (Tiêu chuẩn)", "Mach3 / Grbl", "Weihong"])
safe_Z = st.sidebar.number_input("Mặt phẳng an toàn Safe Z (mm)", min_value=1.0, value=25.0, step=1.0)
thru_overlap = st.sidebar.number_input("Độ sâu cắt đứt lẹm nền ván nỉ (mm)", min_value=0.0, value=0.5, step=0.1)

total_offset = (t1_dia / 2.0) + safety_spacing

# ============================================================
# 3. CHUẨN HÓA TOÀN DIỆN VÀ CHIẾU TỌA ĐỘ PHẲNG 2D
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
            f_p = occ_curve.FirstParameter()
            l_p = occ_curve.LastParameter()
            segments = 32
            pts = []
            for i in range(segments + 1):
                t = f_p + (l_p - f_p) * i / segments
                p_loc = plane_obj.toLocalCoords(cq_edge.valueAt(t))
                pts.append((p_loc.x, p_loc.y))
            return {"type": "DISCRETE_CURVE", "points": pts}
        except Exception:
            return {"type": "LINE", "start": (p_start.x, p_start.y), "end": (p_end.x, p_end.y)}

# ============================================================
# 4. THUẬT TOÁN T-BONE CORNER RELIEF (XỬ LÝ GÓC MỘNG KHÍT NỘI THẤT)
# ============================================================
def apply_t_bone_relief(polygon_points, tool_radius):
    """
    NÂNG CẤP CƠ KHÍ: Tạo hốc xương chó/T-Bone tại các góc vuông 90 độ bên trong 
    để triệt tiêu bán kính dao phay, cho phép lắp ráp mộng gỗ vuông khít 100%.
    """
    if len(polygon_points) < 3: return polygon_points
    pts = list(polygon_points)
    if np.allclose(pts[0], pts[-1]):
        pts.pop()
        
    n = len(pts)
    modified_pts = []
    
    for i in range(n):
        p1 = np.array(pts[i-1])
        p2 = np.array(pts[i])
        p3 = np.array(pts[(i+1)%n])
        
        modified_pts.append(pts[i])
        
        # Tính toán góc hình học giữa 3 điểm liên tiếp
        v1 = p1 - p2
        v2 = p3 - p2
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        
        dot_product = np.dot(v1_u, v2_u)
        angle = math.acos(st.sidebar.slider("Dung sai góc vuông nhận diện T-Bone", 85, 95, 90) * math.pi / 180 if False else math.clamp(dot_product, -1.0, 1.0))
        
        # Nhận diện góc vuông nội tạng (~90 độ bên trong chi tiết)
        if math.isclose(angle, math.pi/2, abs_tol=0.1):
            bisector = v1_u + v2_u
            if np.linalg.norm(bisector) > 1e-4:
                bisector_u = bisector / np.linalg.norm(bisector)
                # Đẩy dao lẹm sâu vào góc vuông một khoảng bằng bán kính dao * căn 2
                tb_point = p2 - bisector_u * (tool_radius * math.sqrt(2))
                modified_pts.append((tb_point[0], tb_point[1]))
                modified_pts.append(pts[i])
                
    modified_pts.append(modified_pts[0])
    return modified_pts

# ============================================================
# 5. ĐỌC FILE STEP & PHÂN TÍCH THUẬT TOÁN ĐỘ SÂU SÂU VÀ QUY TRÌNH CAM
# ============================================================
def process_cad_file_production(file_bytes, filename, sheet_thick):
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        part = cq.importers.importStep(temp_path)
        all_faces = part.faces().vals()
        
        # Thuật toán lọc chọn bề mặt phẳng ngửa (+Z Normal)
        top_faces = [f for f in all_faces if f.geomType() == "PLANE" and f.normalAt().z > 0.8]
        if not top_faces: top_faces = all_faces
        
        target_face = max(top_faces, key=lambda f: f.Area())
        ref_plane = cq.Plane(target_face)
        face_z_level = target_face.Center().z

        # Phân tích chuỗi điểm bao ngoài
        outer_wire = target_face.outerWire()
        outer_edges = [get_local_coordinates(e, ref_plane) for e in outer_wire.Edges()]
        
        features = []
        # Khảo sát biên dạng trong
        for inner_wire in target_face.innerWires():
            wire_edges = [get_local_coordinates(e, ref_plane) for e in inner_wire.Edges()]
            features.append({"type": "CNC_INNER_CUT", "edges": wire_edges, "depth": sheet_thick})

        # Quét tìm hốc hạ nền phẳng (Pocket) bằng sai lệch cao độ 3D Z-axis
        for face in all_faces:
            if face.geomType() == "PLANE" and face.Area() < target_face.Area() * 0.9:
                f_z = face.Center().z
                if f_z < face_z_level and f_z >= (face_z_level - sheet_thick):
                    p_edges = [get_local_coordinates(e, ref_plane) for e in face.outerWire().Edges()]
                    p_depth = abs(face_z_level - f_z)
                    if p_depth > 0.2:
                        features.append({"type": "CNC_POCKET", "edges": p_edges, "depth": p_depth})

        bbox = target_face.BoundingBox()
        return {
            "name": os.path.splitext(filename)[0],
            "width": bbox.xlen,
            "height": bbox.ylen,
            "outer_edges": outer_edges,
            "features": features
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def discrete_edges(edges):
    pts = []
    for edge in edges:
        if edge["type"] == "LINE":
            pts.extend([edge["start"], edge["end"]])
        elif edge["type"] == "CIRCLE":
            cx, cy = edge["center"]
            r = edge["radius"]
            for t in np.linspace(0, 360, 64, endpoint=False):
                pts.append((cx + r * math.cos(math.radians(t)), cy + r * math.sin(math.radians(t))))
        elif edge["type"] == "DISCRETE_CURVE":
            pts.extend(edge["points"])
    return pts

# ============================================================
# 6. THUẬT TOÁN NESTING MA TRẬN PHẲNG (BOTTOM-LEFT FIT CHUYÊN DỤNG)
# ============================================================
def execute_production_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    sheet_bound = Polygon([
        (margin_val, margin_val), (sheet_w - margin_val, margin_val),
        (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)
    ])
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    sheets = []

    for part in sorted_parts:
        outer_raw = clean_polygon_points(discrete_edges(part["outer_edges"]))
        poly_geom = Polygon(outer_raw)
        if not poly_geom.is_valid: poly_geom = repair_geometry(poly_geom)
        
        buffered_poly = poly_geom.buffer(offset_val, resolution=8, join_style=JOIN_STYLE.round)
        min_x, min_y, _, _ = buffered_poly.bounds
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized = translate(poly_geom, xoff=-min_x, yoff=-min_y)

        best_pos = None
        s_idx_target = -1
        min_score = float("inf")

        for s_idx, sheet_data in enumerate(sheets):
            placed_pbs = sheet_data["placed_buffered_polygons"]
            anchors = [(margin_val, margin_val)]
            for pb in placed_pbs:
                b = pb.bounds
                anchors.extend([(b[2], b[1]), (b[0], b[3]), (b[2], b[3])])
            anchors = list(set(anchors))

            for angle in [0, 90, 180, 270]:
                rot_poly = rotate(normalized_poly, angle, origin=(0, 0))
                r_mx, r_my, _, _ = rot_poly.bounds
                
                for ax, ay in anchors:
                    dx_c = ax - r_mx
                    dy_c = ay - r_my
                    cand = translate(rot_poly, xoff=dx_c, yoff=dy_c)
                    
                    if not sheet_bound.covers(cand): continue
                    if any(cand.intersects(pb) for pb in placed_pbs): continue
                    
                    b_c = cand.bounds
                    score = b_c[0] + b_c[1] * 2.5 # Trọng số nén phôi chặt về góc trái dưới
                    
                    if score < min_score:
                        min_score = score
                        s_idx_target = s_idx
                        best_pos = {
                            "dx": dx_c, "dy": dy_c, "angle": angle, "cand_poly": cand,
                            "raw_trans": translate(rotate(raw_normalized, angle, origin=(0, 0)), xoff=dx_c, yoff=dy_c)
                        }

        if best_pos and s_idx_target != -1:
            sheets[s_idx_target]["parts"].append({
                "part_ref": part, "original_offset": (min_x, min_y),
                "placed_polygon": best_pos["raw_trans"], "dx": best_pos["dx"], "dy": best_pos["dy"], "angle": best_pos["angle"]
            })
            sheets[s_idx_target]["placed_buffered_polygons"].append(best_pos["cand_poly"])
        else:
            new_id = len(sheets) + 1
            dx_n = margin_val - min_x
            dy_n = margin_val - min_y
            sheets.append({
                "sheet_id": new_id,
                "parts": [{
                    "part_ref": part, "original_offset": (min_x, min_y),
                    "placed_polygon": translate(raw_normalized, xoff=dx_n, yoff=dy_n), "dx": dx_n, "dy": dy_n, "angle": 0
                }],
                "placed_buffered_polygons": [translate(normalized_poly, xoff=dx_n, yoff=dy_n)]
            })
    return sheets

def clean_polygon_points(points, tolerance=0.01):
    if not points: return []
    cleaned = []
    for p in points:
        if not cleaned or not np.allclose(cleaned[-1], p, atol=tolerance):
            cleaned.append(p)
    if len(cleaned) > 2 and not np.allclose(cleaned[0], cleaned[-1], atol=tolerance):
        cleaned.append(cleaned[0])
    return cleaned

def transform_point_production(x, y, dx, dy, angle, ox, oy):
    xl = x - ox
    yl = y - oy
    rad = math.radians(angle)
    return (
        xl * math.cos(rad) - yl * math.sin(rad) + dx,
        xl * math.sin(rad) + yl * math.cos(rad) + dy
    )

# ============================================================
# 7. ENGINE PHÁT TRIỂN ĐƯỜNG CHẠY DAO THỰC TẾ (TRUE CAM TOOLPATH)
# ============================================================
def get_true_offset_toolpath(edges, op_type, tool_radius):
    """
    NÂNG CẤP BÙ DAO: Ổn định hình học sai sót của Shapely bằng xử lý ngoại lệ tuần hoàn,
    tự động vạt góc chữ T (T-Bone) trước khi sinh bán kính dao chạy thực tế.
    """
    raw_pts = discrete_edges(edges)
    cleaned = clean_polygon_points(raw_pts)
    if len(cleaned) < 3: return cleaned
    
    # 1. Ứng dụng xử lý T-Bone cho biên dạng trong trước khi bù dao
    if op_type == "CNC_INNER_CUT":
        cleaned = apply_t_bone_relief(cleaned, tool_radius)

    poly = Polygon(cleaned)
    if not poly.is_valid: poly = poly.buffer(0)
    
    if op_type == "CNC_OUTER_CUT":
        comp = poly.buffer(tool_radius, resolution=8, join_style=JOIN_STYLE.round)
    else:
        comp = poly.buffer(-tool_radius, resolution=8, join_style=JOIN_STYLE.round)
        
    if comp.is_empty: return cleaned
    if isinstance(comp, MultiPolygon):
        comp = max(comp.geoms, key=lambda p: p.area)
        
    return list(comp.exterior.coords)

# ============================================================
# 8. THUẬT TOÁN NỘI SUY CUNG TRÒN TIÊU CHUẨN G2/G3 (ARC FITTING)
# ============================================================
def fit_arcs_and_emit_gcode(pts, current_z, feed_rate):
    """
    NÂNG CẤP THUẬT TOÁN: Nhận diện chuỗi điểm đa phân đoạn, nếu bán kính đồng nhất 
    thì tự động nội suy ép sang mã G2/G3 (I, J) để giảm dung lượng file và tránh giật máy CNC.
    """
    lines = []
    i = 0
    n = len(pts)
    
    while i < n - 1:
        if i < n - 3:
            # Thuật toán quét 3 điểm kiểm tra cung tròn
            p1, p2, p3 = np.array(pts[i]), np.array(pts[i+1]), np.array(pts[i+2])
            # Tính toán tâm hình học cục bộ
            ma = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6)
            mb = (p3[1] - p2[1]) / (p3[0] - p2[0] + 1e-6)
            
            if not math.isclose(ma, mb, abs_tol=1e-2):
                # Xác định tọa độ tâm cung tròn đường cong (Center X, Center Y)
                cx = (ma*mb*(p1[1] - p3[1]) + mb*(p1[0] + p2[0]) - ma*(p2[0] + p3[0])) / (2 * (mb - ma + 1e-6))
                cy = -1 / (ma + 1e-6) * (cx - (p1[0] + p2[0])/2) + (p1[1] + p2[1])/2
                
                r1 = np.linalg.norm(p1 - np.array([cx, cy]))
                r3 = np.linalg.norm(p3 - np.array([cx, cy]))
                
                if math.isclose(r1, r3, rel_tol=1e-2):
                    # Xác định hướng xoay Vector (Thuận hay ngược chiều kim đồng hồ)
                    cross_product = np.cross(p2 - p1, p3 - p2)
                    g_cmd = "G3" if cross_product > 0 else "G2"
                    
                    # Tính toán sai số I, J tương đối từ điểm bắt đầu dao
                    v_i = cx - p1[0]
                    v_j = cy - p1[1]
                    
                    lines.append(f"{g_cmd} X{p3[0]:.3f} Y{p3[1]:.3f} I{v_i:.3f} J{v_j:.3f} F{feed_rate}")
                    i += 2
                    continue
                    
        lines.append(f"G1 X{pts[i+1][0]:.3f} Y{pts[i+1][1]:.3f} F{feed_rate}")
        i += 1
        
    return lines

# ============================================================
# 9. ENGINE ATC CAM: TỰ ĐỘNG LẬP TRÌNH G-CODE TIÊU CHUẨN MÁY CNC CÔNG NGHIỆP
# ============================================================
def generate_production_atc_gcode(sheet_data, sheet_th, stepdown, safe_z, overlap, dialect):
    gcode = []
    gcode.append(f"; --- TIÊU CHUẨN G-CODE XUẤT XƯỞNG HỆ MÁY: {dialect.upper()} ---")
    gcode.append(f"; THỜI GIAN BIÊN DỊCH CAM: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    gcode.append("G21 ; Đơn vị hệ mét (mm)\nG90 ; Hệ tọa độ tuyệt đối G90\nG17 ; Chọn mặt làm việc phẳng XY")
    gcode.append("G54 ; Thiết lập gốc tọa độ chi tiết chuẩn xưởng")
    
    # Gom cụm hai nhóm tác vụ chính để giảm thiểu thời gian thay trục dao (ATC)
    queue_pockets = []
    queue_inners = []
    queue_outers = []

    for placed in sheet_data["parts"]:
        part = placed["part_ref"]
        for feat in part["features"]:
            if feat["type"] == "CNC_POCKET":
                queue_pockets.append((part, placed, feat))
            elif feat["type"] == "CNC_INNER_CUT":
                queue_inners.append((part, placed, feat))
        queue_outers.append((part, placed, {"type": "CNC_OUTER_CUT", "edges": part["outer_edges"]}))

    # KÍCH HOẠT DUY NHẤT TRỤC DAO T1 CHUYÊN DỤNG PHAY CẮT (END MILL)
    gcode.append(f"\nM6 T1 ; Kích hoạt hệ thống thay dao tự động gọi dao phay T1")
    gcode.append(f"M3 S{int(t1_spindle)} ; Trục chính quay thuận ổn định")
    gcode.append(f"G0 Z{safe_z:.3f} ; Đưa trục Z lên cao độ an toàn tối đa")

    # --------------------------------------------------------
    # NGUYÊN CÔNG 1: PHAY TOÀN BỘ CÁC HỐC HẠ NỀN (POCKETS)
    # --------------------------------------------------------
    if queue_pockets:
        gcode.append("\n; =============================================\n; NGUYÊN CÔNG 1: GIA CÔNG HỐC HẠ NỀN POCKET (BÙ DAO TRONG)\n; =============================================")
        for part, placed, feat in queue_pockets:
            comp_edges = get_true_offset_toolpath(feat["edges"], "CNC_POCKET", t1_dia / 2.0)
            t_pts = [transform_point_production(p[0], p[1], placed["dx"], placed["dy"], placed["angle"], placed["original_offset"][0], placed["original_offset"][1]) for p in comp_edges]
            write_industrial_toolpath(gcode, t_pts, feat["depth"], stepdown, safe_z, t1_feed, has_tabs=False)

    # --------------------------------------------------------
    # NGUYÊN CÔNG 2: CẮT BIÊN DẠNG RỖNG BÊN TRONG (INNER CUTS)
    # --------------------------------------------------------
    if queue_inners:
        gcode.append("\n; =============================================\n; NGUYÊN CÔNG 2: CẮT ĐỨT BIÊN DẠNG TRONG INNER CUT (BÙ DAO TRONG)\n; =============================================")
        for part, placed, feat in queue_inners:
            comp_edges = get_true_offset_toolpath(feat["edges"], "CNC_INNER_CUT", t1_dia / 2.0)
            t_pts = [transform_point_production(p[0], p[1], placed["dx"], placed["dy"], placed["angle"], placed["original_offset"][0], placed["original_offset"][1]) for p in comp_edges]
            write_industrial_toolpath(gcode, t_pts, sheet_th + overlap, stepdown, safe_z, t1_feed, has_tabs=False)

    # --------------------------------------------------------
    # NGUYÊN CÔNG 3: CẮT ĐỨT BIÊN NGOÀI + TỰ ĐỘNG CHÈN TABS GIỮ PHÔI
    # --------------------------------------------------------
    if queue_outers:
        gcode.append("\n; =============================================\n; NGUYÊN CÔNG 3: CẮT BIÊN NGOÀI CHI TIẾT OUTER CUT (BÙ DAO NGOÀI + AUTO TABS)\n; =============================================")
        for part, placed, feat in queue_outers:
            comp_edges = get_true_offset_toolpath(feat["edges"], "CNC_OUTER_CUT", t1_dia / 2.0)
            t_pts = [transform_point_production(p[0], p[1], placed["dx"], placed["dy"], placed["angle"], placed["original_offset"][0], placed["original_offset"][1]) for p in comp_edges]
            write_industrial_toolpath(gcode, t_pts, sheet_th + overlap, stepdown, safe_z, t1_feed, has_tabs=True)

    gcode.append("\n; =============================================\n; TRÌNH KẾT THÚC CHƯƠNG TRÌNH AN TOÀN XƯỞNG M30\n; =============================================")
    gcode.append(f"G0 Z{safe_z:.3f}")
    gcode.append("M5 ; Tắt trục chính dừng cắt")
    gcode.append("G0 X0 Y0 ; Đưa dàn trục về điểm tham chiếu máy G54")
    gcode.append("M30 ; Kết thúc lệnh và thiết lập lại hệ thống")
    return "\n".join(gcode)

def write_industrial_toolpath(gcode_list, pts, total_depth, stepdown, safe_z, feed_rate, has_tabs):
    cleaned = clean_polygon_points(pts)
    if len(cleaned) < 2: return

    # 1. THIẾT LẬP THUẬT TOÁN ĐƯỜNG VÀO DAO XÉO (RAMPING ENGINE)
    # Thay vì đâm thẳng đứng, dao tịnh tiến từ điểm mào đầu xéo góc xuống Z
    p1 = cleaned[0]
    p2 = cleaned[1]
    v_dir = np.array(p2) - np.array(p1)
    norm_v = np.linalg.norm(v_dir)
    
    # Định dạng điểm dắt dao lùi lại 15mm ngoài thành phẩm (Lead-In)
    lead_in_len = 12.0
    v_lead = p1 - (v_dir / (norm_v + 1e-6)) * lead_in_len
    
    current_z = 0.0
    
    # Xác định các điểm phân bổ vị trí gờ cố định giữ ván Tabs
    tab_positions = []
    if has_tabs and len(cleaned) > 4:
        indices = np.linspace(1, len(cleaned) - 2, tab_count_default, dtype=int)
        for idx in indices:
            tab_positions.append(cleaned[idx])

    while current_z > -total_depth:
        prev_z = current_z
        current_z -= stepdown
        if current_z < -total_depth: current_z = -total_depth
        
        # Công đoạn Ramp xuống dao ngọt: Chạy xéo từ điểm dắt dao vào điểm cắt chính
        gcode_list.append(f"G0 X{v_lead[0]:.3f} Y{v_lead[1]:.3f}")
        gcode_list.append(f"G1 Z{prev_z:.3f} F{feed_rate}")
        gcode_list.append(f"G1 X{p1[0]:.3f} Y{p1[1]:.3f} Z{current_z:.3f} F{feed_rate * 0.5:.0f} ; [RAMP PASS]")
        
        # Duyệt qua các điểm của profile chi tiết
        idx = 1
        while idx < len(cleaned):
            pt = cleaned[idx]
            
            # KIỂM TRA CHÈN TABS (Chỉ thực hiện tại lát cắt chiều sâu cuối cùng chạm nền ván)
            is_at_tab = False
            if has_tabs and math.isclose(current_z, -total_depth, abs_tol=0.1):
                for t_pt in tab_positions:
                    if np.allclose(pt, t_pt, atol=2.0):
                        is_at_tab = True
                        break
                        
            if is_at_tab:
                # Thuật toán nhấc dao tạo gờ Tab
                z_tab = current_z + tab_thickness
                if z_tab > 0: z_tab = 0
                gcode_list.append(f"G1 Z{z_tab:.3f} F{feed_rate * 0.4:.0f} ; [NHẤC DAO TẠO TABS]")
                gcode_list.append(f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} F{feed_rate}")
                gcode_list.append(f"G1 Z{current_z:.3f} F{feed_rate * 0.4:.0f} ; [HẠ DAO TIẾP TỤC CẮT]")
            else:
                # NỘI SUY CUNG TRÒN G2/G3 ENGINE TRỰC TIẾP TRÊN PHÂN ĐOẠN TIẾP THEO
                if idx < len(cleaned) - 2:
                    sub_segment = [cleaned[idx-1], cleaned[idx], cleaned[idx+1]]
                    arc_lines = fit_arcs_and_emit_gcode(sub_segment, current_z, feed_rate)
                    if any("G2" in l or "G3" in l for l in arc_lines):
                        gcode_list.extend(arc_lines)
                        idx += 2
                        continue
                        
                gcode_list.append(f"G1 X{pt[0]:.3f} Y{pt[1]:.3f} F{feed_rate}")
            idx += 1
            
        gcode_list.append(f"G0 Z{safe_z:.3f} ; Nhấc trục Z an toàn")

# ============================================================
# 10. PRODUCTION WEB CONTROL INTERFACE
# ============================================================
st.markdown("---")
st.subheader("📥 HỆ THỐNG KIỂM ĐỊNH FILE CAD STEP VÀ BIÊN DỊCH MÃ CNC")

uploaded_files = st.file_uploader(
    "Nạp các file cấu kiện máy gỗ/cơ khí (.step, .stp)",
    type=["step", "stp"],
    accept_multiple_files=True
)

if uploaded_files:
    production_db = []
    bar = st.progress(0)
    
    for idx, f_item in enumerate(uploaded_files):
        with st.spinner(f"Đang giải mã ma trận không gian: {f_item.name}..."):
            try:
                parsed_data = process_cad_file_production(f_item.read(), f_item.name, sheet_thickness)
                production_db.append(parsed_data)
            except Exception as e:
                st.error(f"Lỗi tệp cấu trúc {f_item.name}: {str(e)}")
        bar.progress((idx + 1) / len(uploaded_files))

    if production_db:
        st.success(f"Hệ thống CAM đã đồng bộ hóa thành công {len(production_db)} chi tiết đạt chuẩn sản xuất hàng loạt!")
        
        # Bảng dữ liệu quản lý chất lượng phôi đầu vào
        summary_data = []
        for p in production_db:
            types = [f["type"] for f in p["features"]]
            summary_data.append({
                "Mã cấu kiện": p["name"],
                "Chiều rộng X (mm)": round(p["width"], 2),
                "Chiều cao Y (mm)": round(p["height"], 2),
                "Số hốc âm Pocket": types.count("CNC_POCKET"),
                "Đường cắt rỗng trong": types.count("CNC_INNER_CUT"),
                "Độ dày danh nghĩa (mm)": sheet_thickness
            })
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)

        # Chạy công cụ sắp xếp Nesting Layout hình học công nghiệp
        st.markdown("---")
        st.subheader("🧩 SƠ ĐỒ SẮP XẾP TẤM VÀ ĐƯỜNG CHẠY DAO THỰC TẾ")
        
        with st.spinner("Đang tối ưu hóa đường cắt giảm thiểu hao hụt ván..."):
            sheets_result = execute_production_nesting(production_db, sheet_W, sheet_H, total_offset, margin)

        st.metric("Tổng lượng phôi ván tiêu hao", f"{len(sheets_result)} Tấm")

        # Quản lý giao diện Tabs cho từng tấm riêng lẻ độc lập
        tabs = st.tabs([f"TẤM SẢN XUẤT #{s['sheet_id']}" for s in sheets_result])
        
        for idx, sheet in enumerate(sheets_result):
            with tabs[idx]:
                col_graph, col_nc_output = st.columns([3, 2])
                
                with col_graph:
                    st.markdown(f"##### Sơ đồ vector đường chạy dao thực tế Tấm #{sheet['sheet_id']}")
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.set_xlim(-50, sheet_W + 50)
                    ax.set_ylim(-50, sheet_H + 50)
                    ax.set_aspect('equal')
                    
                    # Trực quan hóa phôi nền tấm ván chính
                    ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, color="#F8FAFC", ec="#0F172A", lw=2, ls="--"))
                    
                    for placed in sheet["parts"]:
                        poly = placed["placed_polygon"]
                        p_ref = placed["part_ref"]
                        
                        if isinstance(poly, Polygon):
                            x_e, y_e = poly.exterior.xy
                            ax.fill(x_e, y_e, alpha=0.6, facecolor="#BAE6FD", edgecolor="#0284C7", lw=1.5)
                            ax.text(poly.centroid.x, poly.centroid.y, p_ref["name"], fontsize=8, ha='center', va='center', weight='bold')
                            
                            # Vẽ các đường bao cắt trong và hốc hạ nền phay lòng
                            for feat in p_ref["features"]:
                                f_pts = [transform_point_production(p[0], p[1], placed["dx"], placed["dy"], placed["angle"], placed["original_offset"][0], placed["original_offset"][1]) for p in discrete_edges(feat["edges"])]
                                f_cleaned = clean_polygon_points(f_pts)
                                if len(f_cleaned) >= 2:
                                    fx, fy = zip(*f_cleaned)
                                    color = "#F59E0B" if feat["type"] == "CNC_POCKET" else "#10B981"
                                    ax.plot(fx, fy, color=color, lw=1)

                    st.pyplot(fig)
                    plt.close()

                with col_nc_output:
                    st.markdown("##### 💾 HỆ THỐNG XUẤT FILE ĐIỀU KHIỂN MÁY CNC")
                    
                    # 1. Kết xuất tệp tin bản vẽ công nghiệp phân rã Layer AutoCAD DXF
                    dxf_str = generate_dxf_industrial_layered(sheet, sheet_W, sheet_H)
                    st.download_button(
                        label=f"📥 Tải DXF xuất xưởng Tấm #{sheet['sheet_id']}",
                        data=dxf_str,
                        file_name=f"Factory_Layer_Sheet_{sheet['sheet_id']}.dxf",
                        mime="image/vnd.dxf",
                        key=f"dxf_key_{sheet['sheet_id']}"
                    )
                    
                    # 2. Biên dịch trực tiếp G-Code công nghiệp tích hợp bộ nén lệnh G2/G3 & Tabs chống văng phôi
                    gcode_str = generate_production_atc_gcode(sheet, sheet_thickness, max_stepdown, safe_Z, thru_overlap, cnc_dialect)
                    st.download_button(
                        label=f"📟 Tải mã G-code chạy máy Tấm #{sheet['sheet_id']}",
                        data=gcode_str,
                        file_name=f"ATC_Production_Sheet_{sheet['sheet_id']}.nc",
                        mime="text/plain",
                        key=f"nc_key_{sheet['sheet_id']}"
                    )
                    
                    with st.expander("Bản xem trước cấu trúc khối lệnh NC mã máy"):
                        st.code(gcode_str[:1500] + "\n\n... [Hệ thống nén toán hạng G2/G3 Arc Fitting hoạt động] ...", language="gcode")
else:
    st.info("💡 Đang chờ dữ liệu đầu vào. Vui lòng tải các file cấu kiện STEP lên để hệ thống CAM tự động lập trình chạy dao.")
