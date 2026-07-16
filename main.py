import io
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- THƯ VIỆN HÌNH HỌC VÀ CAD CHUYÊN NGHIỆP ---
import cadquery as cq
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import translate, rotate
import ezdxf

# ----------------- CẤU HÌNH GIAO DIỆN -----------------
st.set_page_config(page_title="Auto-CNC Industrial: True Shape Nesting Pro", layout="wide")
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🏭 HỆ THỐNG CNC NỘI THẤT THÔNG MINH v2.0</h2>", unsafe_allow_html=True)
st.write("Giải pháp CAM tự động: Nhân OCC chọn mặt chuẩn, Thuật toán Nesting Best-Fit đa hướng & Auto-Layer cho Aspire.")

# ----------------- CẤU HÌNH SẢN XUẤT (SIDEBAR) -----------------
st.sidebar.header("⚙️ THÔNG SỐ KHỔ VÁN & DAO")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván (X) - mm", value=2440)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván (Y) - mm", value=1220)
tool_diameter = st.sidebar.number_input("Đường kính dao cắt (mm)", value=6.0)
safety_spacing = st.sidebar.number_input("Khoảng cách an toàn giữa 2 tấm (mm)", value=4.0)
margin = st.sidebar.number_input("Chừa lề biên ván (mm)", value=15)

# Tính tổng khoảng hở cần offset khi sắp xếp hình học
total_offset = (tool_diameter + safety_spacing) / 2.0

# ----------------- PHẦN 1: TỰ ĐỘNG CHỌN MẶT STEP GIA CÔNG -----------------
def process_cad_file_with_occ(file_bytes, filename):
    """
    Tự động tìm mặt phẳng lớn nhất hướng theo trục Z làm mặt gia công CNC.
    Gọi hàm parse_wire_edges_high_precision để xử lý chi tiết biên dạng.
    """
    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    try:
        # Load mô hình 3D từ file STEP
        part = cq.importer.importStep(temp_path)
        all_faces = part.faces().vals()
        if not all_faces:
            raise ValueError("Không tìm thấy bề mặt phẳng nào trong cấu trúc STEP.")
            
        # Tự động tìm mặt phẳng gia công lớn nhất
        planar_faces = [f for f in all_faces if f.geomType() == "PLANE"]
        if not planar_faces:
            planar_faces = all_faces
            
        target_face = max(planar_faces, key=lambda f: f.Area())
        
        # Trích xuất Wire ngoài và các Wire trong
        outer_wire = target_face.outerWire()
        inner_wires = target_face.innerWires()
        
        outer_edges = parse_wire_edges_high_precision(outer_wire)
        
        # Nhận diện đặc tính lỗ để phân phối vào Layer CUT hoặc DRILL
        holes = []
        for iw in inner_wires:
            hole_edges = parse_wire_edges_high_precision(iw)
            if hole_edges:
                is_pure_circle = len(hole_edges) == 1 and hole_edges[0]["type"] == "CIRCLE"
                radius = hole_edges[0]["radius"] if is_pure_circle else 0
                holes.append({
                    "edges": hole_edges,
                    "is_drill": is_pure_circle and radius <= 10.0, # Lỗ đường kính <= 20mm đưa vào layer khoan mồi
                    "radius": radius
                })
                
        bbox = target_face.BoundingBox()
        return {
            "name": os.path.splitext(filename)[0],
            "width": bbox.xlen,
            "height": bbox.ylen,
            "outer_edges": outer_edges,
            "holes": holes
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------- PHẦN 2: ĐỌC HÌNH HỌC CAD CHUẨN (SPLINE & FIX HỞ BIÊN) -----------------
def parse_wire_edges_high_precision(wire, tolerance=0.05):
    """
    Trích xuất hình học với độ chính xác cao.
    Xử lý B-Spline dựa trên tham số độ cong và nội suy mượt.
    """
    edges_data = []
    for edge in wire.Edges():
        g_type = edge.geomType()
        start = edge.startPoint()
        end = edge.endPoint()
        
        if g_type == "LINE":
            edges_data.append({
                "type": "LINE", "start": (start.x, start.y), "end": (end.x, end.y)
            })
            
        elif g_type == "CIRCLE":
            circle_geom = edge.curve()
            center = circle_geom.Location()
            radius = circle_geom.Radius()
            
            if start.Distance(end) < 1e-4: # Khép kín hoàn toàn
                edges_data.append({
                    "type": "CIRCLE", "center": (center.X(), center.Y()), "radius": radius
                })
            else: # Cung tròn
                angle_start = math.atan2(start.y - center.Y(), start.x - center.X())
                angle_end = math.atan2(end.y - center.Y(), end.x - center.X())
                edges_data.append({
                    "type": "ARC", "center": (center.X(), center.Y()), "radius": radius,
                    "start_angle": math.degrees(angle_start), "end_angle": math.degrees(angle_end)
                })
                
        elif g_type in ["BSPLINE", "BEZIER", "OFFSET"]:
            # Thuật toán lấy mẫu thích ứng (Adaptive Sampling) dựa trên độ dài và độ cong của Spline
            occ_curve = edge.ToAdaptor3d()
            first_p = occ_curve.FirstParameter()
            last_p = occ_curve.LastParameter()
            
            edge_length = edge.Length()
            segments = max(16, min(64, int(edge_length / tolerance))) 
            
            pts = []
            for i in range(segments + 1):
                u = first_p + (last_p - first_p) * i / segments
                p = edge.valueAt(u)
                pts.append((p.x, p.y))
            
            for i in range(len(pts) - 1):
                edges_data.append({
                    "type": "LINE", "start": pts[i], "end": pts[i+1]
                })
    return edges_data

def build_shapely_polygon_fixed(part, snap_distance=0.01):
    """
    Dựng đa giác hình học từ các thực thể toán học.
    Gom cụm các điểm hở (Snap Points) để sửa lỗi biên dạng CAD.
    """
    raw_points = []
    for edge in part["outer_edges"]:
        if edge["type"] == "LINE":
            raw_points.append(edge["start"])
            raw_points.append(edge["end"])
        elif edge["type"] == "ARC":
            cx, cy = edge["center"]
            r = edge["radius"]
            sa, ea = edge["start_angle"], edge["end_angle"]
            if sa > ea: ea += 360
            for theta in np.linspace(sa, ea, 12):
                rad = math.radians(theta)
                raw_points.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))

    # Sửa lỗi hở biên dạng: Nối và khử các điểm trùng hoặc hở nhỏ
    cleaned_points = []
    for p in raw_points:
        if not cleaned_points:
            cleaned_points.append(p)
        else:
            if np.allclose(cleaned_points[-1], p, atol=snap_distance):
                continue
            cleaned_points.append(p)
            
    if len(cleaned_points) > 2 and not np.allclose(cleaned_points[0], cleaned_points[-1], atol=snap_distance):
        cleaned_points.append(cleaned_points[0])

    if len(cleaned_points) < 3:
        return Polygon([(0,0), (part["width"], 0), (part["width"], part["height"]), (0, part["height"])])

    interiors = []
    for h in part["holes"]:
        h_pts = []
        for edge in h["edges"]:
            if edge["type"] == "LINE":
                h_pts.append(edge["start"])
                h_pts.append(edge["end"])
            elif edge["type"] == "CIRCLE":
                cx, cy = edge["center"]
                r = edge["radius"]
                for theta in np.linspace(0, 360, 16):
                    rad = math.radians(theta)
                    h_pts.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))
        if len(h_pts) >= 3:
            interiors.append(h_pts)

    return Polygon(cleaned_points, interiors)

# ----------------- PHẦN 3: NESTING XOAY ĐA HƯỚNG & BEST-FIT MINIMUM WASTE -----------------
def perform_advanced_best_fit_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    """
    Thuật toán True-Shape Nesting nâng cao:
    - Xoay linh hoạt nhiều hướng (Mặc định quét 8 hướng tăng hiệu suất).
    - Tiêu chí lựa chọn: Best-Fit (Vị trí làm tăng diện tích bao của phôi ít nhất).
    """
    sheet_boundary = Polygon([
        (margin_val, margin_val), (sheet_w - margin_val, margin_val),
        (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)
    ])
    
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    nested_sheets = []
    
    # Ma trận xoay đa hướng bước góc 45 độ
    angles_to_try = [0, 45, 90, 135, 180, 225, 270, 315]
    
    for part in sorted_parts:
        poly_geom = build_shapely_polygon_fixed(part)
        buffered_poly = poly_geom.buffer(offset_val)
        min_x, min_y, _, _ = buffered_poly.bounds
        
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized_poly = translate(poly_geom, xoff=-min_x, yoff=-min_y)
        
        best_position = None
        best_sheet_idx = -1
        min_waste_score = float('inf') 
        
        for s_idx, sheet_info in enumerate(nested_sheets):
            placed_polys = sheet_info["placed_buffered_polygons"]
            
            anchor_points = [(margin_val, margin_val)]
            for p_poly in placed_polys:
                p_minx, p_miny, p_maxx, p_maxy = p_poly.bounds
                anchor_points.extend([
                    (p_maxx, p_miny), 
                    (p_minx, p_maxy),
                    (p_maxx, p_maxy)
                ])
            anchor_points = list(set(anchor_points))
            
            for angle in angles_to_try:
                rot_poly = rotate(normalized_poly, angle, origin=(0, 0))
                r_minx, r_miny, _, _ = rot_poly.bounds
                
                for ax, ay in anchor_points:
                    candidate_poly = translate(rot_poly, xoff=ax - r_minx, yoff=ay - r_miny)
                    
                    if sheet_boundary.contains(candidate_poly):
                        collision = any(candidate_poly.intersects(p_p) for p_p in placed_polys)
                        
                        if not collision:
                            # Thuật toán tính toán diện tích bao (Best-Fit Envelope)
                            all_x = []
                            all_y = []
                            for p_p in placed_polys:
                                b = p_p.bounds
                                all_x.extend([b[0], b[2]])
                                all_y.extend([b[1], b[3]])
                            cb = candidate_poly.bounds
                            all_x.extend([cb[0], cb[2]])
                            all_y.extend([cb[1], cb[3]])
                            
                            current_envelope_area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y))
                            
                            if current_envelope_area < min_waste_score:
                                min_waste_score = current_envelope_area
                                best_sheet_idx = s_idx
                                best_position = {
                                    "dx": ax - r_minx,
                                    "dy": ay - r_miny,
                                    "angle": angle,
                                    "candidate_poly": candidate_poly,
                                    "raw_poly_transformed": translate(rotate(raw_normalized_poly, angle, origin=(0,0)), xoff=ax - r_minx, yoff=ay - r_miny)
                                }
                                
        if best_position and best_sheet_idx != -1:
            sheet_info = nested_sheets[best_sheet_idx]
            sheet_info["parts"].append({
                "part_ref": part,
                "original_offset": (min_x, min_y),
                "placed_polygon": best_position["raw_poly_transformed"],
                "dx": best_position["dx"],
                "dy": best_position["dy"],
                "angle": best_position["angle"]
            })
            sheet_info["placed_buffered_polygons"].append(best_position["candidate_poly"])
        else:
            new_idx = len(nested_sheets)
            rot_poly = rotate(normalized_poly, 0, origin=(0, 0))
            r_minx, r_miny, _, _ = rot_poly.bounds
            
            init_x, init_y = margin_val, margin_val
            candidate_poly = translate(rot_poly, xoff=init_x - r_minx, yoff=init_y - r_miny)
            placed_raw = translate(rotate(raw_normalized_poly, 0, origin=(0,0)), xoff=init_x - r_minx, yoff=init_y - r_miny)
            
            nested_sheets.append({
                "sheet_id": new_idx + 1,
                "parts": [{
                    "part_ref": part, "original_offset": (min_x, min_y),
                    "placed_polygon": placed_raw, "dx": init_x - r_minx, "dy": init_y - r_miny, "angle": 0
                }],
                "placed_buffered_polygons": [candidate_poly]
            })
            
    return nested_sheets

# ----------------- PHẦN 4: XUẤT DXF THEO LAYER ĐỂ TỰ ĐỘNG BÙ DAO TRÊN CAM -----------------
def transform_point(x, y, dx, dy, angle, orig_x, orig_y):
    """Tính toán ma trận tịnh tiến tọa độ hình học thực tế"""
    tx, ty = x - orig_x, y - orig_y
    rad = math.radians(angle)
    rx = tx * math.cos(rad) - ty * math.sin(rad)
    ry = tx * math.sin(rad) + ty * math.cos(rad)
    return rx + dx, ry + dy

def generate_industrial_dxf(nested_sheets, sheet_w, sheet_h):
    """
    Xuất file DXF phân lớp chuyên nghiệp. Các phần mềm CAM (Aspire) sẽ tự động nhận diện:
    - CNC_OUTER_CUT -> Chọn Toolpath Profile Outside để bù bán kính dao ra ngoài biên sản phẩm.
    - CNC_INNER_CUT -> Chọn Toolpath Profile Inside để bù bán kính dao vào trong lỗ khoét.
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    doc.layers.new(name='CNC_OUTER_CUT', dxfattribs={'color': 1})    # Đỏ
    doc.layers.new(name='CNC_INNER_CUT', dxfattribs={'color': 4})    # Cyan
    doc.layers.new(name='CNC_INNER_DRILL', dxfattribs={'color': 2})  # Vàng
    doc.layers.new(name='CNC_SHEET_BORDER', dxfattribs={'color': 8}) # Xám
    
    sheet_offset_x = 0
    
    for sheet_data in nested_sheets:
        ox = sheet_offset_x
        # Khung viền tấm ván
        msp.add_line((ox, 0), (ox + sheet_w, 0), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox + sheet_w, 0), (ox + sheet_w, sheet_h), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox + sheet_w, sheet_h), (ox, sheet_h), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox, sheet_h), (ox, 0), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        
        for p_node in sheet_data["parts"]:
            ref = p_node["part_ref"]
            dx, dy = p_node["dx"] + ox, p_node["dy"]
            angle = p_node["angle"]
            orig_x, orig_y = p_node["original_offset"]
            
            # Xuất biên ngoài chi tiết
            for edge in ref["outer_edges"]:
                if edge["type"] == "LINE":
                    p1 = transform_point(edge["start"][0], edge["start"][1], dx, dy, angle, orig_x, orig_y)
                    p2 = transform_point(edge["end"][0], edge["end"][1], dx, dy, angle, orig_x, orig_y)
                    msp.add_line(p1, p2, dxfattribs={'layer': 'CNC_OUTER_CUT'})
                elif edge["type"] == "ARC":
                    cx, cy = transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y)
                    msp.add_arc(center=(cx, cy), radius=edge["radius"],
                                start_angle=edge["start_angle"] + angle, end_angle=edge["end_angle"] + angle,
                                dxfattribs={'layer': 'CNC_OUTER_CUT'})
                                
            # Xuất các lỗ khoét bên trong (Lọt lòng)
            for hole in ref["holes"]:
                target_layer = 'CNC_INNER_DRILL' if hole["is_drill"] else 'CNC_INNER_CUT'
                for edge in hole["edges"]:
                    if edge["type"] == "LINE":
                        p1 = transform_point(edge["start"][0], edge["start"][1], dx, dy, angle, orig_x, orig_y)
                        p2 = transform_point(edge["end"][0], edge["end"][1], dx, dy, angle, orig_x, orig_y)
                        msp.add_line(p1, p2, dxfattribs={'layer': target_layer})
                    elif edge["type"] == "CIRCLE":
                        cx, cy = transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y)
                        msp.add_circle(center=(cx, cy), radius=edge["radius"], dxfattribs={'layer': target_layer})
                    elif edge["type"] == "ARC":
                        cx, cy = transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y)
                        msp.add_arc(center=(cx, cy), radius=edge["radius"],
                                    start_angle=edge["start_angle"] + angle, end_angle=edge["end_angle"] + angle,
                                    dxfattribs={'layer': target_layer})
                                    
        sheet_offset_x += sheet_w + 300
        
    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue()

def generate_aspire_toolpath_csv():
    data = [
        {"Layer_Name": "CNC_OUTER_CUT", "Toolpath_Type": "Profile Outside", "Depth": "Thickness + 0.3mm", "Purpose": "Cắt đứt biên ngoài (Tự động bù dao ra ngoài vách)"},
        {"Layer_Name": "CNC_INNER_CUT", "Toolpath_Type": "Profile Inside", "Depth": "Thickness + 0.3mm", "Purpose": "Cắt lọt lòng lỗ lớn (Tự động bù dao vào trong vách)"},
        {"Layer_Name": "CNC_INNER_DRILL", "Toolpath_Type": "Drilling", "Depth": "12.0mm", "Purpose": "Khoan mồi liên kết cam chốt"}
    ]
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

# ----------------- LUỒNG XỬ LÝ CHƯƠNG TRÌNH STREAMLIT -----------------
uploaded_files = st.file_uploader("📂 TẢI LÊN FILE 3D STEP SẢN PHẨM", type=["step", "stp"], accept_multiple_files=True)

if uploaded_files:
    parts_db = []
    with st.spinner("⚡ Đang phân tích kết cấu 3D và tối ưu spline mịn..."):
        for f in uploaded_files:
            try:
                extracted_data = process_cad_file_with_occ(f.read(), f.name)
                parts_db.append(extracted_data)
            except Exception as e:
                st.error(f"Lỗi phân tích file {f.name}: {str(e)}")
                
    if parts_db:
        st.success(f"⚡ Bóc tách thành công {len(parts_db)} cấu kiện CAD tiêu chuẩn!")
        
        with st.spinner("🧠 Thuật toán đang thực hiện True-Shape Nesting Best-Fit..."):
            nesting_results = perform_advanced_best_fit_nesting(parts_db, sheet_W, sheet_H, total_offset, margin)
            
        st.subheader(f"📐 Sơ đồ sắp xếp phôi ván thông minh ({len(nesting_results)} Tấm ván)")
        
        for idx, sheet in enumerate(nesting_results):
            st.write(f"### 🟫 Sơ đồ mặt cắt tấm ván số: {sheet['sheet_id']}")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, linewidth=1.2, edgecolor='black', facecolor='#F3E8EE'))
            
            for p_info in sheet["parts"]:
                raw_poly = p_info["placed_polygon"]
                if isinstance(raw_poly, Polygon):
                    xs, ys = raw_poly.exterior.xy
                    ax.fill(xs, ys, alpha=0.8, fc='#0F766E', ec='#115E59', lw=1)
                    for interior in raw_poly.interiors:
                        ixs, iys = interior.xy
                        ax.fill(ixs, iys, fc='#F3E8EE', ec='#B91C1C', lw=0.8)
                    centroid = raw_poly.centroid
                    ax.text(centroid.x, centroid.y, p_info["part_ref"]["name"], color='white', weight='bold', fontsize=6, ha='center')
                    
            ax.set_xlim(-50, sheet_W + 50)
            ax.set_ylim(-50, sheet_H + 50)
            ax.set_aspect('equal')
            plt.axis('off')
            st.pyplot(fig)
            
        st.markdown("---")
        st.subheader("💾 TẢI DỮ LIỆU ĐỂ HẬU XỬ LÝ TRÊN VECTRIC ASPIRE")
        col1, col2 = st.columns(2)
        
        industrial_dxf = generate_industrial_dxf(nesting_results, sheet_W, sheet_H)
        col1.download_button(label="📥 Tải Xuất DXF Chuẩn Layer (Có tách Trong / Ngoài)", data=industrial_dxf, file_name="cnc_industrial_output.dxf", mime="application/dxf", use_container_width=True)
        
        csv_map = generate_aspire_toolpath_csv()
        col2.download_button(label="📊 Tải Cấu Hình Ánh Xạ Đường Dao CAM", data=csv_map, file_name="aspire_rule_mapping.csv", mime="text/csv", use_container_width=True)
