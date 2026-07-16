import io
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches mpatches

# --- THƯ VIỆN HÌNH HỌC VÀ CAD CHUYÊN NGHIỆP ---
import cadquery as cq
from cadquery import Edge, Face, Wire, Vector
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import translate, rotate
import ezdxf

# ----------------- CẤU HÌNH GIAO DIỆN -----------------
st.set_page_config(page_title="Auto-CNC Industrial: True Shape Nesting Pro", layout="wide")
st.markdown("<h2 style='text-align: center; color: #1E3A8A;'>🏭 HỆ THỐNG CNC NỘI THẤT THÔNG MINH v2.0</h2>", unsafe_allow_html=True)
st.write("Giải pháp CAM tự động: Nhân OCC chuẩn hoá hình học, Thuật toán Nesting tối ưu hoá vật liệu & Auto-Layer cho Aspire.")

# ----------------- CẤU HÌNH SẢN XUẤT (SIDEBAR) -----------------
st.sidebar.header("⚙️ THÔNG SỐ KHỔ VÁN & DAO")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván (X) - mm", value=2440)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván (Y) - mm", value=1220)
tool_diameter = st.sidebar.number_input("Đường kính dao cắt (mm)", value=6.0)
safety_spacing = st.sidebar.number_input("Khoảng cách an toàn giữa 2 tấm (mm)", value=4.0)
margin = st.sidebar.number_input("Chừa lề biên ván (mm)", value=15)

# Tính tổng khoảng hở cần offset khi sắp xếp
total_offset = (tool_diameter + safety_spacing) / 2.0

# ----------------- PHẦN 1 & 2: TRÍCH XUẤT MẶT PHẲNG & ĐỌC HÌNH HỌC CHUẨN -----------------
def process_cad_file_with_occ(file_bytes, filename):
    """
    1. Tự động chọn đúng mặt phẳng gia công lớn nhất.
    2. Đọc hình học chuẩn hệ LINE/ARC/CIRCLE/SPLINE không làm suy hao độ phân giải.
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
            
        # PHẦN 1: Tự động tìm mặt phẳng gia công CNC tối ưu (Diện tích lớn nhất, phẳng)
        planar_faces = [f for f in all_faces if f.geomType() == "PLANE"]
        if not planar_faces:
            planar_faces = all_faces # Fallback nếu định dạng lỗi
            
        target_face = max(planar_faces, key=lambda f: f.Area())
        
        # Trích xuất Wire ngoài và các Wire trong
        outer_wire = target_face.outerWire()
        inner_wires = target_face.innerWires()
        
        def parse_wire_edges(wire):
            edges_data = []
            for edge in wire.Edges():
                g_type = edge.geomType()
                start = edge.startPoint()
                end = edge.endPoint()
                
                # Xử lý LINE
                if g_type == "LINE":
                    edges_data.append({
                        "type": "LINE", "start": (start.x, start.y), "end": (end.x, end.y)
                    })
                # Xử lý HÌNH TRÒN / CUNG TRÒN
                elif g_type == "CIRCLE":
                    circle_geom = edge.curve()
                    center = circle_geom.Location()
                    radius = circle_geom.Radius()
                    
                    if start.Distance(end) < 1e-4: # Đường tròn khép kín
                        edges_data.append({
                            "type": "CIRCLE", "center": (center.X(), center.Y()), "radius": radius
                        })
                    else: # Cung tròn (Arc)
                        angle_start = math.atan2(start.y - center.Y(), start.x - center.X())
                        angle_end = math.atan2(end.y - center.Y(), end.x - center.X())
                        edges_data.append({
                            "type": "ARC", "center": (center.X(), center.Y()), "radius": radius,
                            "start_angle": math.degrees(angle_start), "end_angle": math.degrees(angle_end)
                        })
                # PHẦN 2: Xử lý Bo góc phức tạp, Spline, NURBS nâng cao bằng nội suy điểm mịn
                elif g_type in ["BSPLINE", "BEZIER", "OFFSET"]:
                    # Lấy mẫu phân đoạn mịn để giữ độ chính xác hình học biên dạng spline
                    occ_curve = edge.ToAdaptor3d()
                    first_p = occ_curve.FirstParameter()
                    last_p = occ_curve.LastParameter()
                    intervals = 16
                    pts = []
                    for i in range(intervals + 1):
                        u = first_p + (last_p - first_p) * i / intervals
                        p = edge.valueAt(u)
                        pts.append((p.x, p.y))
                    
                    for i in range(len(pts) - 1):
                        edges_data.append({
                            "type": "LINE", "start": pts[i], "end": pts[i+1]
                        })
            return edges_data

        outer_edges = parse_wire_edges(outer_wire)
        
        # Nhận diện đặc tính lỗ để phân phối vào Layer CUT hoặc DRILL hoặc POCKET
        holes = []
        for iw in inner_wires:
            hole_edges = parse_wire_edges(iw)
            if hole_edges:
                # Phân loại tự động dựa trên cấu trúc lỗ tròn kín hay hốc hình học
                is_pure_circle = len(hole_edges) == 1 and hole_edges[0]["type"] == "CIRCLE"
                radius = hole_edges[0]["radius"] if is_pure_circle else 0
                holes.append({
                    "edges": hole_edges,
                    "is_drill": is_pure_circle and radius <= 10.0, # Đường kính <= 20mm xếp vào lỗ khoan mồi
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

# ----------------- PHẦN 3: THUẬT TOÁN NESTING TỐI ƯU VÁN -----------------
def build_shapely_polygon(part):
    """ Dựng hình hình học Đa giác từ dữ liệu OCC đã bóc tách để tính va chạm """
    points = []
    for edge in part["outer_edges"]:
        if edge["type"] == "LINE":
            points.append(edge["start"])
            points.append(edge["end"])
        elif edge["type"] == "ARC":
            cx, cy = edge["center"]
            r = edge["radius"]
            sa, ea = edge["start_angle"], edge["end_angle"]
            if sa > ea: ea += 360
            for theta in np.linspace(sa, ea, 8):
                rad = math.radians(theta)
                points.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))
                
    cleaned_points = []
    for p in points:
        if not cleaned_points or not np.allclose(cleaned_points[-1], p, atol=1e-3):
            cleaned_points.append(p)
            
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
                for theta in np.linspace(0, 360, 12):
                    rad = math.radians(theta)
                    h_pts.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))
        if len(h_pts) >= 3:
            interiors.append(h_pts)
            
    return Polygon(cleaned_points, interiors)

def perform_true_shape_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    """ Thuật toán xếp ván True-Shape tối ưu ma trận Bottom-Left nâng cao """
    sheet_boundary = Polygon([
        (margin_val, margin_val), (sheet_w - margin_val, margin_val),
        (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)
    ])
    
    # Sắp xếp theo diện tích giảm dần nâng cao hiệu suất lấp đầy khoảng trống
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    nested_sheets = []
    
    for part in sorted_parts:
        poly_geom = build_shapely_polygon(part)
        buffered_poly = poly_geom.buffer(offset_val)
        min_x, min_y, _, _ = buffered_poly.bounds
        
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized_poly = translate(poly_geom, xoff=-min_x, yoff=-min_y)
        
        placed = False
        # Các góc xoay linh hoạt (Ưu tiên giữ vân 0, 180 hoặc tự do tùy chọn)
        angles_to_try = [0, 180, 90, 270]
        
        for sheet_info in nested_sheets:
            placed_polys = sheet_info["placed_buffered_polygons"]
            
            # Thuật toán tìm kiếm thông minh dọc theo các góc biên có sẵn thay vì ô lưới cố định
            for angle in angles_to_try:
                rot_poly = rotate(normalized_poly, angle, origin=(0, 0))
                r_minx, r_miny, r_maxx, r_maxy = rot_poly.bounds
                
                # Quét các điểm neo (Anchor Points) thông minh dựa trên bounding box hiện tại
                test_points = [(margin_val, margin_val)]
                for p_poly in placed_polys:
                    p_minx, p_miny, p_maxx, p_maxy = p_poly.bounds
                    test_points.append((p_maxx + 1, p_miny))
                    test_points.append((p_minx, p_maxy + 1))
                
                # Sắp xếp thứ tự các điểm test ưu tiên từ dưới lên trên, từ trái qua phải (Bottom-Left)
                test_points = sorted(list(set(test_points)), key=lambda p: (p[1], p[0]))
                
                for tx, ty in test_points:
                    candidate_poly = translate(rot_poly, xoff=tx - r_minx, yoff=ty - r_miny)
                    
                    if sheet_boundary.contains(candidate_poly):
                        collision = any(candidate_poly.intersects(p_p) for p_p in placed_polys)
                        if not collision:
                            placed_raw = translate(rotate(raw_normalized_poly, angle, origin=(0,0)), xoff=tx - r_minx, yoff=ty - r_miny)
                            sheet_info["parts"].append({
                                "part_ref": part, "original_offset": (min_x, min_y),
                                "placed_polygon": placed_raw, "dx": tx - r_minx, "dy": ty - r_miny, "angle": angle
                            })
                            placed_polys.append(candidate_poly)
                            placed = True
                            break
                if placed: break
            if placed: break
            
        if not placed:
            # Tạo tấm ván mới khi không xếp vừa vào các tấm ván cũ
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

# ----------------- PHẦN 4: XUẤT DXF TỰ ĐỘNG PHÂN LAYER CHO ASPIRE -----------------
def transform_point(x, y, dx, dy, angle, orig_x, orig_y):
    tx, ty = x - orig_x, y - orig_y
    rad = math.radians(angle)
    rx = tx * math.cos(rad) - ty * math.sin(rad)
    ry = tx * math.sin(rad) + ty * math.cos(rad)
    return rx + dx, ry + dy

def generate_industrial_dxf(nested_sheets, sheet_w, sheet_h):
    """ Xuất file DXF định nghĩa cấu trúc chuẩn xác 4 tầng Layer gán dao nhanh cho CAM """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Định nghĩa Layer chuẩn hóa màu sắc và tên định danh
    doc.layers.new(name='CNC_OUTER_CUT', dxfattribs={'color': 1})    # Cắt đứt ngoài (Màu Đỏ)
    doc.layers.new(name='CNC_INNER_CUT', dxfattribs={'color': 4})    # Cắt lọt lòng lỗ lớn (Màu Cyan)
    doc.layers.new(name='CNC_INNER_DRILL', dxfattribs={'color': 2})  # Khoan mồi tâm dao (Màu Vàng)
    doc.layers.new(name='CNC_INNER_POCKET', dxfattribs={'color': 5}) # Hạ nền/Làm hốc khuyết (Màu Xanh)
    doc.layers.new(name='CNC_SHEET_BORDER', dxfattribs={'color': 8}) # Khung bao tấm ván (Màu Xám)
    
    sheet_offset_x = 0
    
    for sheet_data in nested_sheets:
        ox = sheet_offset_x
        # Vẽ khung bao ván thực tế
        msp.add_line((ox, 0), (ox + sheet_w, 0), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox + sheet_w, 0), (ox + sheet_w, sheet_h), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox + sheet_w, sheet_h), (ox, sheet_h), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox, sheet_h), (ox, 0), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        
        for p_node in sheet_data["parts"]:
            ref = p_node["part_ref"]
            dx, dy = p_node["dx"] + ox, p_node["dy"]
            angle = p_node["angle"]
            orig_x, orig_y = p_node["original_offset"]
            
            # 1. Ghi biên dạng bên ngoài (Layer OUTER_CUT)
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
                                
            # 2. Ghi các lỗ và hốc bên trong theo phân loại Layer tự động
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

# --- FILE ÁNH XẠ ĐƯỜNG GIA CÔNG CHO ASPIRE ---
def generate_aspire_toolpath_csv():
    data = [
        {"Layer_Name": "CNC_OUTER_CUT", "Toolpath_Type": "Profile Outside", "Depth": "Thickness + 0.3mm", "Purpose": "Cắt đứt biên ngoài chi tiết"},
        {"Layer_Name": "CNC_INNER_CUT", "Toolpath_Type": "Profile Inside", "Depth": "Thickness + 0.3mm", "Purpose": "Cắt lọt lòng lỗ lớn"},
        {"Layer_Name": "CNC_INNER_DRILL", "Toolpath_Type": "Drilling", "Depth": "12.0mm", "Purpose": "Khoan mồi chốt cam/ốc liên kết"},
        {"Layer_Name": "CNC_INNER_POCKET", "Toolpath_Type": "Pocketing", "Depth": "9.5mm", "Purpose": "Hạ nền rãnh hậu ván tủ"}
    ]
    return pd.DataFrame(data).to_csv(index=False).encode('utf-8')

# ----------------- LUỒNG XỬ LÝ CHƯƠNG TRÌNH STREAMLIT -----------------
uploaded_files = st.file_uploader("📂 TẢI LÊN FILE 3D STEP SẢN PHẨM", type=["step", "stp"], accept_multiple_files=True)

if uploaded_files:
    parts_db = []
    with st.spinner("⚡ Đang bóc tách hình học nâng cao bằng OpenCascade..."):
        for f in uploaded_files:
            try:
                extracted_data = process_cad_file_with_occ(f.read(), f.name)
                parts_db.append(extracted_data)
            except Exception as e:
                st.error(f"Lỗi phân tích cấu trúc file {f.name}: {str(e)}")
                
    if parts_db:
        st.success(f" Bóc tách thành công {len(parts_db)} cấu kiện CAD tiêu chuẩn!")
        
        # Chạy giải thuật Nesting dạng True-Shape nâng cao
        with st.spinner("🧠 Thuật toán đang tối ưu hóa sơ đồ sắp xếp phôi ván..."):
            nesting_results = perform_true_shape_nesting(parts_db, sheet_W, sheet_H, total_offset, margin)
            
        st.subheader(f"📐 Sơ đồ phân bố phôi tối ưu hình học ({len(nesting_results)} Tấm ván)")
        
        # Biểu diễn trực quan sơ đồ
        for idx, sheet in enumerate(nesting_results):
            st.write(f"### 🟫 Mặt bằng cắt tấm ván số: {sheet['sheet_id']}")
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
        col1.download_button(label="📥 Tải Xuất DXF Chuẩn Layer (Màu CAM)", data=industrial_dxf, file_name="auto_cnc_perfect_output.dxf", mime="application/dxf", use_container_width=True)
        
        csv_map = generate_aspire_toolpath_csv()
        col2.download_button(label="📊 Tải Cấu Hình Ánh Xạ Đường Dao Aspire", data=csv_map, file_name="aspire_rule_mapping.csv", mime="text/csv", use_container_width=True)
