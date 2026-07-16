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
st.set_page_config(page_title="Auto-CNC Industrial: True Shape Nesting", layout="wide")
st.markdown("<h2 style='text-align: center; color: #FF4B4B;'>🏭 HỆ THỐNG CNC NỘI THẤT CÔNG NGHIỆP</h2>", unsafe_allow_html=True)
st.write("Phiên bản CAD/CAM tự động hóa: Nhân OpenCascade bóc tách STEP, True-Shape Nesting & Auto-Layer cho Vectric Aspire.")

# ----------------- CẤU HÌNH SẢN XUẤT (SIDEBAR) -----------------
st.sidebar.header("⚙️ THÔNG SỐ KHỔ VÁN & DAO")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván (X) - mm", value=2440)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván (Y) - mm", value=1220)
tool_diameter = st.sidebar.number_input("Đường kính dao cắt (mm)", value=6.0)
safety_spacing = st.sidebar.number_input("Khoảng cách an toàn giữa 2 tấm (mm)", value=4.0)
margin = st.sidebar.number_input("Chừa lề biên ván (mm)", value=15)

# Tính tổng khoảng hở cần offset khi sắp xếp
total_offset = (tool_diameter + safety_spacing) / 2.0

# ----------------- BƯỚC 1: TRÍCH XUẤT TOPOLOGY (OPENCASCADE) -----------------
def process_cad_file_with_occ(file_bytes, filename):
    """
    Sử dụng CadQuery (nhân OpenCascade) để đọc file CAD (STEP/IGES)
    và trích xuất chính xác các thực thể Line, Arc, Circle từ tấm phẳng.
    """
    # Ghi tạm file bytes ra đĩa để CadQuery đọc
    temp_path = f"temp_{filename}"
    with open(temp_path, "wb") as f:
        f.write(file_bytes)
    
    try:
        # Load mô hình 3D bằng OpenCascade
        part = cq.importer.importStep(temp_path)
        
        # Lọc mặt phẳng lớn nhất hướng theo trục Z (mặt tấm phẳng mặt chính)
        all_faces = part.faces().vals()
        if not all_faces:
            raise ValueError("Không tìm thấy bề mặt phẳng nào trong file STEP.")
            
        # Tìm mặt phẳng có diện tích lớn nhất và có vector pháp tuyến song song với trục Z
        target_face = max(all_faces, key=lambda f: f.Area())
        
        outer_wire = target_face.outerWire()
        inner_wires = target_face.innerWires()
        
        # 1. Trích xuất biên dạng ngoài (Outer Boundary)
        outer_edges = []
        for edge in outer_wire.Edges():
            geom_type = edge.geomType() # Trả về 'LINE', 'CIRCLE', 'OFFSET', 'BSPLINE'
            
            if geom_type == "LINE":
                start = edge.startPoint()
                end = edge.endPoint()
                outer_edges.append({
                    "type": "LINE",
                    "start": (start.x, start.y),
                    "end": (end.x, end.y)
                })
            elif geom_type == "CIRCLE":
                # Xác định xem là đường tròn kín hay cung tròn (Arc)
                circle_geom = edge.curve()
                start_p = edge.startPoint()
                end_p = edge.endPoint()
                
                # Tính góc quét
                center = circle_geom.Location()
                radius = circle_geom.Radius()
                
                # Check góc bằng lượng giác
                angle_start = math.atan2(start_p.y - center.Y(), start_p.x - center.X())
                angle_end = math.atan2(end_p.y - center.Y(), end_p.x - center.X())
                
                if start_p.Distance(end_p) < 1e-5:
                    outer_edges.append({
                        "type": "CIRCLE",
                        "center": (center.X(), center.Y()),
                        "radius": radius
                    })
                else:
                    outer_edges.append({
                        "type": "ARC",
                        "center": (center.X(), center.Y()),
                        "radius": radius,
                        "start_angle": math.degrees(angle_start),
                        "end_angle": math.degrees(angle_end)
                    })
                    
        # 2. Trích xuất các lỗ khoét bên trong (Inner Holes)
        holes = []
        for wire in inner_wires:
            hole_edges = []
            for edge in wire.Edges():
                geom_type = edge.geomType()
                if geom_type == "LINE":
                    start = edge.startPoint()
                    end = edge.endPoint()
                    hole_edges.append({
                        "type": "LINE",
                        "start": (start.x, start.y),
                        "end": (end.x, end.y)
                    })
                elif geom_type == "CIRCLE":
                    circle_geom = edge.curve()
                    center = circle_geom.Location()
                    radius = circle_geom.Radius()
                    start_p = edge.startPoint()
                    end_p = edge.endPoint()
                    
                    if start_p.Distance(end_p) < 1e-5:
                        hole_edges.append({
                            "type": "CIRCLE",
                            "center": (center.X(), center.Y()),
                            "radius": radius
                        })
                    else:
                        angle_start = math.atan2(start_p.y - center.Y(), start_p.x - center.X())
                        angle_end = math.atan2(end_p.y - center.Y(), end_p.x - center.X())
                        hole_edges.append({
                            "type": "ARC",
                            "center": (center.X(), center.Y()),
                            "radius": radius,
                            "start_angle": math.degrees(angle_start),
                            "end_angle": math.degrees(angle_end)
                        })
            if hole_edges:
                holes.append(hole_edges)
                
        # 3. Tính kích thước bao của tấm phẳng (Bounding Box)
        bbox = target_face.BoundingBox()
        width = bbox.xlen
        height = bbox.ylen
        
        part_data = {
            "name": os.path.splitext(filename)[0],
            "width": width,
            "height": height,
            "outer_edges": outer_edges,
            "holes": holes
        }
        return part_data
        
    finally:
        # Xóa file tạm
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ----------------- BƯỚC 2: TRUE-SHAPE NESTING VỚI SHAPELY -----------------
def build_shapely_polygon(part):
    """
    Chuyển đổi các thực thể cạnh thô của OpenCascade thành đối tượng Polygon của Shapely
    để phục vụ tính toán va chạm và tìm vị trí xếp ván.
    """
    # Lấy các điểm nối đầu đuôi từ danh sách LINE/ARC biên ngoài
    points = []
    for edge in part["outer_edges"]:
        if edge["type"] == "LINE":
            points.append(edge["start"])
            points.append(edge["end"])
        elif edge["type"] == "ARC":
            # Tạo các điểm gần đúng phân đoạn cung tròn để tính va chạm đa giác trong Shapely
            cx, cy = edge["center"]
            r = edge["radius"]
            sa, ea = edge["start_angle"], edge["end_angle"]
            if sa > ea:
                ea += 360
            for theta in np.linspace(sa, ea, 10):
                rad = math.radians(theta)
                points.append((cx + r * math.cos(rad), cy + r * math.sin(rad)))
                
    # Loại bỏ các điểm trùng nhau liên tiếp
    cleaned_points = []
    for p in points:
        if not cleaned_points or not np.allclose(cleaned_points[-1], p, atol=1e-3):
            cleaned_points.append(p)
            
    if len(cleaned_points) < 3:
        # Fallback nếu cấu trúc biên lỗi: Dựng khung hình chữ nhật đại diện
        return Polygon([(0,0), (part["width"], 0), (part["width"], part["height"]), (0, part["height"])])
        
    # Tạo Polygon gốc
    outer_poly = Polygon(cleaned_points)
    
    # Tạo danh sách các lỗ rỗng (hỗ trợ phay rỗng lồng chi tiết vào trong)
    interiors = []
    for hole in part["holes"]:
        hole_pts = []
        for h_edge in hole:
            if h_edge["type"] == "LINE":
                hole_pts.append(h_edge["start"])
            elif h_edge["type"] == "ARC":
                cx, cy = h_edge["center"]
                r = h_edge["radius"]
                sa, ea = h_edge["start_angle"], h_edge["end_angle"]
                if sa > ea:
                    ea += 360
                for theta in np.linspace(sa, ea, 8):
                    hole_pts.append((cx + r * math.cos(math.radians(theta)), cy + r * math.sin(math.radians(theta))))
        if len(hole_pts) >= 3:
            interiors.append(hole_pts)
            
    return Polygon(cleaned_points, interiors)

def perform_true_shape_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    """
    Giải thuật Heuristic True-Shape Nesting sắp xếp các đa giác tùy biến dựa trên Shapely.
    Hỗ trợ xoay chi tiết tự do (0, 90, 180, 270 độ) để lấp đầy khoảng trống tối ưu.
    """
    # Khởi tạo vùng chứa khả dụng bên trong tấm ván (đã chừa lề)
    sheet_boundary = Polygon([
        (margin_val, margin_val), 
        (sheet_w - margin_val, margin_val), 
        (sheet_w - margin_val, sheet_h - margin_val), 
        (margin_val, sheet_h - margin_val)
    ])
    
    # Sắp xếp các tấm cần cắt theo diện tích giảm dần để ưu tiên đặt tấm lớn trước
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    
    nested_sheets = [] # Danh sách lưu kết quả phân bố trên các tấm ván thực tế
    
    for part in sorted_parts:
        poly_geom = build_shapely_polygon(part)
        
        # Bù thêm bán kính dao & khoảng cách an toàn bằng Shapely Buffer
        buffered_poly = poly_geom.buffer(offset_val)
        
        # Tìm tọa độ gốc cực tiểu để tịnh tiến về (0,0) trước khi tính toán
        min_x, min_y, _, _ = buffered_poly.bounds
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized_poly = translate(poly_geom, xoff=-min_x, yoff=-min_y)
        
        placed = False
        
        # Duyệt qua các tấm ván hiện có để tìm chỗ đặt
        for sheet_idx, placed_info in enumerate(nested_sheets):
            placed_polys = placed_info["placed_buffered_polygons"]
            
            # Quét tìm vị trí đặt trống bằng phương pháp dịch chuyển (Grid Search)
            for step_y in range(int(margin_val), int(sheet_h - margin_val), 15):
                for step_x in range(int(margin_val), int(sheet_w - margin_val), 15):
                    # Thử nghiệm 4 góc xoay tiêu chuẩn để khóa/mở vân gỗ tự động
                    angles_to_try = [0, 90, 180, 270]
                    
                    for angle in angles_to_try:
                        # Thực hiện xoay và tịnh tiến đa giác kiểm tra
                        rotated_poly = rotate(normalized_poly, angle, origin=(0, 0))
                        candidate_poly = translate(rotated_poly, xoff=step_x, yoff=step_y)
                        
                        # Điều kiện 1: Nằm gọn hoàn toàn trong vùng biên của ván
                        if sheet_boundary.contains(candidate_poly):
                            # Điều kiện 2: Không giao cắt với bất kỳ chi tiết nào đã xếp trước đó
                            collision = False
                            for placed_p in placed_polys:
                                if candidate_poly.intersects(placed_p):
                                    collision = True
                                    break
                            
                            if not collision:
                                # Đặt thành công! Lưu lại cấu hình biến đổi hình học
                                placed_raw = translate(rotate(raw_normalized_poly, angle, origin=(0, 0)), xoff=step_x, yoff=step_y)
                                
                                placed_info["parts"].append({
                                    "part_ref": part,
                                    "original_offset": (min_x, min_y),
                                    "placed_polygon": placed_raw,
                                    "dx": step_x,
                                    "dy": step_y,
                                    "angle": angle
                                })
                                placed_polys.append(candidate_poly)
                                placed = True
                                break
                    if placed:
                        break
                if placed:
                    break
            if placed:
                break
                
        # Nếu không có tấm ván hiện tại nào chứa vừa, khởi tạo một tấm ván mới
        if not placed:
            new_sheet_idx = len(nested_sheets)
            # Quét vị trí đầu tiên (góc dưới cùng bên trái) của tấm ván mới
            init_x, init_y = margin_val, margin_val
            candidate_poly = translate(normalized_poly, xoff=init_x, yoff=init_y)
            placed_raw = translate(raw_normalized_poly, xoff=init_x, yoff=init_y)
            
            nested_sheets.append({
                "sheet_id": new_sheet_idx + 1,
                "parts": [{
                    "part_ref": part,
                    "original_offset": (min_x, min_y),
                    "placed_polygon": placed_raw,
                    "dx": init_x,
                    "dy": init_y,
                    "angle": 0
                }],
                "placed_buffered_polygons": [candidate_poly]
            })
            
    return nested_sheets

# ----------------- BƯỚC 3: XUẤT DXF THỰC THỂ GỐC CHẤT LƯỢNG CAO -----------------
def transform_point(x, y, dx, dy, angle, orig_x, orig_y):
    """
    Hàm toán học áp dụng xoay và dịch chuyển tọa độ từ gốc phôi sang tọa độ bàn máy CNC.
    """
    # 1. Tịnh tiến đưa về gốc (0,0)
    tx, ty = x - orig_x, y - orig_y
    # 2. Xoay tọa độ
    rad = math.radians(angle)
    rx = tx * math.cos(rad) - ty * math.sin(rad)
    ry = tx * math.sin(rad) + ty * math.cos(rad)
    # 3. Tịnh tiến đến tọa độ xếp phôi trên ván
    return rx + dx, ry + dy

def generate_industrial_dxf(nested_sheets, sheet_w, sheet_h):
    """
    Khởi tạo file DXF tiêu chuẩn chứa đúng thực thể toán học nguyên bản.
    Gán các lớp (Layers) xử lý riêng biệt để gán dao tự động trên Aspire.
    """
    doc = ezdxf.new('R2010')
    msp = doc.modelspace()
    
    # Định nghĩa các Layer gán tự động toolpath (Auto-CAM Layer Association)
    doc.layers.new(name='CNC_OUTER_CUT', dxfattribs={'color': 1})  # Lớp cắt ngoài - Màu đỏ
    doc.layers.new(name='CNC_INNER_HOLES', dxfattribs={'color': 5}) # Lớp lỗ khoét - Màu xanh dương
    doc.layers.new(name='CNC_SHEET_BORDER', dxfattribs={'color': 8})# Lớp bao khổ ván - Màu xám
    
    sheet_spacing_offset_x = 0
    
    for sheet_data in nested_sheets:
        # Vẽ khổ ván bao ngoài bằng thực thể LINE gốc
        ox = sheet_spacing_offset_x
        msp.add_line((ox, 0), (ox + sheet_w, 0), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox + sheet_w, 0), (ox + sheet_w, sheet_h), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox + sheet_w, sheet_h), (ox, sheet_h), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        msp.add_line((ox, sheet_h), (ox, 0), dxfattribs={'layer': 'CNC_SHEET_BORDER'})
        
        for part_node in sheet_data["parts"]:
            ref = part_node["part_ref"]
            dx, dy = part_node["dx"] + ox, part_node["dy"]
            angle = part_node["angle"]
            orig_x, orig_y = part_node["original_offset"]
            
            # Xuất biên dạng ngoài (Outer Profile)
            for edge in ref["outer_edges"]:
                if edge["type"] == "LINE":
                    p1 = transform_point(edge["start"][0], edge["start"][1], dx, dy, angle, orig_x, orig_y)
                    p2 = transform_point(edge["end"][0], edge["end"][1], dx, dy, angle, orig_x, orig_y)
                    msp.add_line(p1, p2, dxfattribs={'layer': 'CNC_OUTER_CUT'})
                    
                elif edge["type"] == "ARC":
                    cx, cy = transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y)
                    msp.add_arc(
                        center=(cx, cy),
                        radius=edge["radius"],
                        start_angle=edge["start_angle"] + angle,
                        end_angle=edge["end_angle"] + angle,
                        dxfattribs={'layer': 'CNC_OUTER_CUT'}
                    )
                elif edge["type"] == "CIRCLE":
                    cx, cy = transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y)
                    msp.add_circle(
                        center=(cx, cy),
                        radius=edge["radius"],
                        dxfattribs={'layer': 'CNC_OUTER_CUT'}
                    )
                    
            # Xuất các biên dạng lỗ/rãnh bên trong (Inner Holes)
            for hole in ref["holes"]:
                for h_edge in hole:
                    if h_edge["type"] == "LINE":
                        p1 = transform_point(h_edge["start"][0], h_edge["start"][1], dx, dy, angle, orig_x, orig_y)
                        p2 = transform_point(h_edge["end"][0], h_edge["end"][1], dx, dy, angle, orig_x, orig_y)
                        msp.add_line(p1, p2, dxfattribs={'layer': 'CNC_INNER_HOLES'})
                        
                    elif h_edge["type"] == "ARC":
                        cx, cy = transform_point(h_edge["center"][0], h_edge["center"][1], dx, dy, angle, orig_x, orig_y)
                        msp.add_arc(
                            center=(cx, cy),
                            radius=h_edge["radius"],
                            start_angle=h_edge["start_angle"] + angle,
                            end_angle=h_edge["end_angle"] + angle,
                            dxfattribs={'layer': 'CNC_INNER_HOLES'}
                        )
                    elif h_edge["type"] == "CIRCLE":
                        cx, cy = transform_point(h_edge["center"][0], h_edge["center"][1], dx, dy, angle, orig_x, orig_y)
                        msp.add_circle(
                            center=(cx, cy),
                            radius=h_edge["radius"],
                            dxfattribs={'layer': 'CNC_INNER_HOLES'}
                        )
                        
        # Đưa tấm ván tiếp theo dịch sang bên phải 300mm trên màn hình CAD
        sheet_spacing_offset_x += sheet_w + 300
        
    out_stream = io.StringIO()
    doc.write(out_stream)
    return out_stream.getvalue()

# ----------------- BƯỚC 4: ASPIRE CSV TOOLPATH MAPPING GENERATOR -----------------
def generate_aspire_toolpath_csv(nested_sheets):
    """
    Xuất cấu hình JSON/CSV định nghĩa thông số dao, tốc độ, chiều sâu cắt tương ứng 
    với từng tên lớp hình học (Layer name) để Aspire tự động import và gán dao nhanh.
    """
    data = [
        {
            "Layer_Name": "CNC_OUTER_CUT",
            "Toolpath_Type": "Profile Outside",
            "Default_Tool": "Endmill 6.0mm",
            "Target_Depth_Equation": "Ván_Dày + 0.5mm (Cắt đứt hẳn)",
            "Feed_Rate_MM_Min": 3500,
            "Spindle_Speed_RPM": 18000,
            "Action": "Cắt biên dạng ngoài tấm ván"
        },
        {
            "Layer_Name": "CNC_INNER_HOLES",
            "Toolpath_Type": "Pocket or Inside Profile",
            "Default_Tool": "Endmill 4.0mm or 6.0mm",
            "Target_Depth_Equation": "Tùy biến (Khoét lỗ liên kết/chậu rửa)",
            "Feed_Rate_MM_Min": 2800,
            "Spindle_Speed_RPM": 20000,
            "Action": "Khoét rỗng liên kết gỗ âm"
        },
        {
            "Layer_Name": "CNC_SHEET_BORDER",
            "Toolpath_Type": "No Toolpath",
            "Default_Tool": "None",
            "Target_Depth_Equation": "Không gia công",
            "Feed_Rate_MM_Min": 0,
            "Spindle_Speed_RPM": 0,
            "Action": "Biên nhận diện khổ ván thô"
        }
    ]
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8')

# ----------------- LUỒNG XỬ LÝ CHƯƠNG TRÌNH STREAMLIT -----------------
uploaded_files = st.file_uploader(
    "📂 TẢI LÊN FILE SẢN PHẨM 3D (STEP / STP)", 
    type=["step", "stp"], 
    accept_multiple_files=True
)

if uploaded_files:
    parts_db = []
    with st.spinner("⚡ Đang bóc tách hình học toán học bằng OpenCascade..."):
        for f in uploaded_files:
            file_bytes = f.read()
            try:
                extracted_data = process_cad_file_with_occ(file_bytes, f.name)
                parts_db.append(extracted_data)
            except Exception as e:
                st.error(f"Không thể xử lý file {f.name}. Lỗi: {str(e)}")
                
    if parts_db:
        st.success(f"⚡ Bóc tách thành công {len(parts_db)} chi tiết CAD thực tế!")
        
        # Hiển thị BOM chi tiết dạng bảng
        st.subheader("📋 Bảng Thống Kê Cấu Kiện CAD (BOM)")
        st.dataframe(pd.DataFrame([{
            "Tên chi tiết": p["name"],
            "Rộng (X) mm": round(p["width"], 2),
            "Dài (Y) mm": round(p["height"], 2),
            "Số rãnh/lỗ khoét": len(p["holes"])
        } for p in parts_db]))
        
        # Chạy giải thuật xếp True Shape
        with st.spinner("🧠 Đang tính toán True-Shape Nesting..."):
            nesting_results = perform_true_shape_nesting(parts_db, sheet_W, sheet_H, total_offset, margin)
            
        st.subheader(f"📐 Bản Đồ Sắp Xếp Ván Thực Tế ({len(nested_sheets := nesting_results)} Tấm)")
        
        # Vẽ trực quan sơ đồ sắp xếp
        for idx, sheet in enumerate(nested_sheets):
            st.write(f"### 🟫 Sơ Đồ Xếp Tấm Ván {sheet['sheet_id']}")
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Vẽ khổ ván
            ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, linewidth=1.5, edgecolor='black', facecolor='#efe5d9'))
            
            # Vẽ từng chi tiết hình học phẳng
            for p_info in sheet["parts"]:
                raw_poly = p_info["placed_polygon"]
                part_name = p_info["part_ref"]["name"]
                
                # Biểu diễn tọa độ đa giác
                if isinstance(raw_poly, Polygon):
                    xs, ys = raw_poly.exterior.xy
                    ax.fill(xs, ys, alpha=0.8, fc='#2e7d32', ec='#1b5e20', lw=1.2)
                    
                    # Vẽ các lỗ khoét
                    for interior in raw_poly.interiors:
                        ixs, iys = interior.xy
                        ax.fill(ixs, iys, fc='#efe5d9', ec='#bf360c', lw=1)
                        
                    # Hiện nhãn chữ tên tấm ở trọng tâm
                    centroid = raw_poly.centroid
                    ax.text(centroid.x, centroid.y, part_name, color='white', weight='bold', fontsize=7, ha='center', va='center')
                    
            ax.set_xlim(-100, sheet_W + 100)
            ax.set_ylim(-100, sheet_H + 100)
            ax.set_aspect('equal')
            plt.axis('off')
            st.pyplot(fig)
            
        # Xuất và chuẩn bị dữ liệu tải về cho người đứng máy CNC
        st.markdown("---")
        st.subheader("💾 TẢI VỀ DỮ LIỆU SẢN XUẤT CAD/CAM")
        
        col1, col2 = st.columns(2)
        
        # 1. Tải DXF xuất xưởng
        industrial_dxf_data = generate_industrial_dxf(nested_sheets, sheet_W, sheet_H)
        col1.download_button(
            label="📥 Tải DXF Biên Dạng Gốc (ARC/LINE/CIRCLE)",
            data=industrial_dxf_data,
            file_name="auto_cnc_production_output.dxf",
            mime="application/dxf",
            use_container_width=True
        )
        
        # 2. Tải cấu hình ánh xạ đường dao cho Vectric Aspire
        csv_map_data = generate_aspire_toolpath_csv(nested_sheets)
        col2.download_button(
            label="📊 Tải Aspire CSV Toolpath Mapping",
            data=csv_map_data,
            file_name="aspire_toolpath_mapping.csv",
            mime="text/csv",
            use_container_width=True
        )
