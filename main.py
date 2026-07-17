import io
import os
import math
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --- CAD / 3D GRAPHICS ---
import cadquery as cq
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate
from shapely.geometry import JOIN_STYLE
try:
    from shapely.validation import make_valid
except ImportError:
    make_valid = None
import ezdxf

# Thư viện hỗ trợ hiển thị 3D tương tác
import pyvista as pv
from stpyvista import stpyvista

# ============================================================
# 1. CẤU HÌNH GIAO DIỆN
# ============================================================
st.set_page_config(
    page_title="Auto-CNC Industrial: True Shape Nesting Pro",
    layout="wide"
)
st.markdown(
    """
    <h2 style='text-align: center; color: #1E3A8A;'>
    🏭 HỆ THỐNG CNC NỘI THẤT THÔNG MINH v3.5 (Tích hợp 3D View)
    </h2>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 2. THANH THÔNG SỐ (SIDEBAR)
# ============================================================
st.sidebar.header("⚙️ THÔNG SỐ KHỔ VÁN & DAO")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván X (mm)", min_value=100.0, value=2440.0, step=1.0)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván Y (mm)", min_value=100.0, value=1220.0, step=1.0)
tool_diameter = st.sidebar.number_input("Đường kính dao cắt (mm)", min_value=0.1, value=6.0, step=0.1)
safety_spacing = st.sidebar.number_input("Khoảng cách an toàn chi tiết (mm)", min_value=0.0, value=4.0, step=0.5)
margin = st.sidebar.number_input("Chừa lề biên ván (mm)", min_value=0.0, value=15.0, step=1.0)
hole_drill_diameter = st.sidebar.number_input("Đường kính lỗ khoan tối đa (mm)", min_value=0.1, value=20.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("🔧 CẤU HÌNH BÙ BÁN KÍNH DAO")
compensation_mode = st.sidebar.selectbox(
    "Phương thức bù bán kính dao:",
    options=["Bù dao trên phần mềm CAM (Khuyên dùng)", "Tự động bù dao trực tiếp vào DXF (Đường tâm dao)"]
)
tool_radius = tool_diameter / 2.0
total_offset = (tool_diameter + safety_spacing) / 2.0

# ============================================================
# 3. CÁC HÀM XỬ LÝ HÌNH HỌC & ĐỌC STEP
# ============================================================
def repair_geometry(geometry):
    if geometry is None or geometry.is_empty: return geometry
    if geometry.is_valid: return geometry
    if make_valid is not None:
        try: return make_valid(geometry)
        except Exception: pass
    try: return geometry.buffer(0)
    except Exception: return geometry

def process_cad_file_with_occ(file_bytes, filename):
    temp_path = None
    temp_stl_path = None
    try:
        # Ghi file tạm STEP
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Đọc hình học CAD bằng CadQuery
        part = cq.importers.importStep(temp_path)
        all_faces = part.faces().vals()

        if not all_faces:
            raise ValueError("Không tìm thấy bề mặt nào trong file STEP.")

        # Xuất thêm một file STL tạm để phục vụ render 3D
        with tempfile.NamedTemporaryFile(delete=False, suffix=".stl") as temp_stl:
            temp_stl_path = temp_stl.name
        cq.exporters.export(part, temp_stl_path, cq.exporters.ExportTypes.STL)

        planar_faces = [face for face in all_faces if face.geomType() == "PLANE"]
        if not planar_faces: planar_faces = all_faces
        target_face = max(planar_faces, key=lambda face: face.Area())

        outer_wire = target_face.outerWire()
        inner_wires = target_face.innerWires()
        outer_edges = parse_wire_edges_high_precision(outer_wire)
        holes = []

        for inner_wire in inner_wires:
            hole_edges = parse_wire_edges_high_precision(inner_wire)
            if not hole_edges: continue
            is_pure_circle = (len(hole_edges) == 1 and hole_edges[0]["type"] == "CIRCLE")
            radius = hole_edges[0]["radius"] if is_pure_circle else 0
            holes.append({
                "edges": hole_edges,
                "is_drill": is_pure_circle and (radius * 2) <= hole_drill_diameter,
                "radius": radius
            })

        bbox = target_face.BoundingBox()
        
        # Bbox tổng thể của khối 3D để lấy chiều dày ván
        total_bbox = part.val().BoundingBox()
        thickness = total_bbox.zlen if total_bbox.zlen > 0 else 17.0 # mặc định nếu lỗi phẳng

        return {
            "name": os.path.splitext(filename)[0],
            "width": bbox.xlen,
            "height": bbox.ylen,
            "thickness": thickness,
            "outer_edges": outer_edges,
            "holes": holes,
            "stl_path": temp_stl_path  # Lưu đường dẫn để vẽ 3D
        }
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

def parse_wire_edges_high_precision(wire, tolerance=0.05):
    edges_data = []
    for edge in wire.Edges():
        g_type = edge.geomType()
        start = edge.startPoint()
        end = edge.endPoint()
        if g_type == "LINE":
            edges_data.append({"type": "LINE", "start": (start.x, start.y), "end": (end.x, end.y)})
        elif g_type == "CIRCLE":
            radius = edge.radius()
            center = edge.Center()
            if (start - end).Length < 1e-4:
                edges_data.append({"type": "CIRCLE", "center": (center.x, center.y), "radius": radius})
            else:
                edges_data.append({
                    "type": "ARC", "center": (center.x, center.y), "radius": radius,
                    "start_angle": math.degrees(math.atan2(start.y - center.y, start.x - center.x)),
                    "end_angle": math.degrees(math.atan2(end.y - center.y, end.x - center.x))
                })
        elif g_type in ["BSPLINE", "BEZIER", "OFFSET"]:
            try:
                occ_curve = edge.ToAdaptor3d()
                first_p, last_p = occ_curve.FirstParameter(), occ_curve.LastParameter()
                segments = max(16, min(200, int(edge.Length() / tolerance)))
                pts = [(edge.valueAt(first_p + (last_p - first_p) * i / segments).x, edge.valueAt(first_p + (last_p - first_p) * i / segments).y) for i in range(segments + 1)]
                for i in range(len(pts) - 1):
                    edges_data.append({"type": "LINE", "start": pts[i], "end": pts[i + 1]})
            except Exception: pass
    return edges_data

# ============================================================
# 4. CÁC HÀM VẼ 2D CHI TIẾT
# ============================================================
def sample_arc(center_x, center_y, radius, start_angle, end_angle, segments=32):
    if end_angle < start_angle: end_angle += 360
    return [(center_x + radius * math.cos(math.radians(t)), center_y + radius * math.sin(math.radians(t))) for t in np.linspace(start_angle, end_angle, segments)]

def edges_to_points(edges, circle_segments=64, arc_segments=32):
    points = []
    for edge in edges:
        if edge["type"] == "LINE": points.extend([edge["start"], edge["end"]])
        elif edge["type"] == "ARC": points.extend(sample_arc(edge["center"][0], edge["center"][1], edge["radius"], edge["start_angle"], edge["end_angle"], arc_segments))
        elif edge["type"] == "CIRCLE":
            cx, cy, r = edge["center"][0], edge["center"][1], edge["radius"]
            points.extend([(cx + r * math.cos(math.radians(t)), cy + r * math.sin(math.radians(t))) for t in np.linspace(0, 360, circle_segments, endpoint=False)])
    return points

def clean_points(points, snap_distance=0.01):
    cleaned = []
    for p in points:
        if not cleaned or not np.allclose(cleaned[-1], p, atol=snap_distance): cleaned.append(p)
    if len(cleaned) > 2 and not np.allclose(cleaned[0], cleaned[-1], atol=snap_distance): cleaned.append(cleaned[0])
    return cleaned

def build_shapely_polygon_fixed(part, snap_distance=0.01):
    outer_points = clean_points(edges_to_points(part["outer_edges"]), snap_distance)
    if len(outer_points) < 3: return Polygon([(0, 0), (part["width"], 0), (part["width"], part["height"]), (0, part["height"])])
    interiors = [clean_points(edges_to_points(h["edges"]), snap_distance) for h in part["holes"] if len(clean_points(edges_to_points(h["edges"]), snap_distance)) >= 3]
    return repair_geometry(Polygon(outer_points, interiors))

def plot_single_part_with_dimensions(part):
    fig, ax = plt.subplots(figsize=(5, 4))
    poly = build_shapely_polygon_fixed(part)
    min_x, min_y, _, _ = poly.bounds
    poly_norm = translate(poly, xoff=-min_x, yoff=-min_y)
    
    xs, ys = poly_norm.exterior.xy
    ax.fill(xs, ys, alpha=0.2, fc='#0F766E', ec='#115E59', lw=2)
    for interior in poly_norm.interiors:
        ixs, iys = interior.xy
        ax.fill(ixs, iys, fc='white', ec='#B91C1C', lw=1)
        
    w, h = part["width"], part["height"]
    ax.annotate('', xy=(0, -h*0.1), xytext=(w, -h*0.1), arrowprops=dict(arrowstyle='<->', color='blue', lw=1.2))
    ax.text(w/2, -h*0.08, f"{w:.1f} mm", color='blue', fontsize=9, ha='center', weight='bold')
    ax.annotate('', xy=(-w*0.1, 0), xytext=(-w*0.1, h), arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
    ax.text(-w*0.08, h/2, f"{h:.1f} mm", color='red', fontsize=9, va='center', ha='right', rotation=90, weight='bold')

    ax.set_aspect('equal')
    padding = max(w, h) * 0.15
    ax.set_xlim(-padding, w + padding)
    ax.set_ylim(-padding, h + padding)
    plt.axis('off')
    return fig

# ============================================================
# 5. THUẬT TOÁN NESTING & DXF EXPORT (Giữ nguyên logic cũ)
# ============================================================
def perform_advanced_best_fit_nesting(parts_list, sheet_w, sheet_h, offset_val, margin_val):
    sheet_boundary = Polygon([(margin_val, margin_val), (sheet_w - margin_val, margin_val), (sheet_w - margin_val, sheet_h - margin_val), (margin_val, sheet_h - margin_val)])
    sorted_parts = sorted(parts_list, key=lambda x: x["width"] * x["height"], reverse=True)
    nested_sheets = []
    angles_to_try = [0, 90, 180, 270]

    for part in sorted_parts:
        poly_geom = build_shapely_polygon_fixed(part)
        buffered_poly = repair_geometry(poly_geom.buffer(offset_val, resolution=16, join_style=JOIN_STYLE.round))
        min_x, min_y, _, _ = buffered_poly.bounds
        normalized_poly = translate(buffered_poly, xoff=-min_x, yoff=-min_y)
        raw_normalized_poly = translate(poly_geom, xoff=-min_x, yoff=-min_y)

        best_position, best_sheet_idx, min_waste_score = None, -1, float("inf")

        for s_idx, sheet_info in enumerate(nested_sheets):
            placed_polys = sheet_info["placed_buffered_polygons"]
            anchor_points = [(margin_val, margin_val)]
            for p_poly in placed_polys:
                p_minx, p_miny, p_maxx, p_maxy = p_poly.bounds
                anchor_points.extend([(p_maxx, p_miny), (p_minx, p_maxy), (p_maxx, p_maxy)])
            anchor_points = list(set(anchor_points))

            for angle in angles_to_try:
                rot_poly = rotate(normalized_poly, angle, origin=(0, 0))
                r_minx, r_miny, _, _ = rot_poly.bounds
                for ax, ay in anchor_points:
                    candidate_poly = translate(rot_poly, xoff=ax - r_minx, yoff=ay - r_miny)
                    if not sheet_boundary.covers(candidate_poly): continue
                    if any(candidate_poly.intersects(p_poly) for p_poly in placed_polys): continue

                    all_x = [b.bounds[0] for b in placed_polys] + [b.bounds[2] for b in placed_polys] + [candidate_poly.bounds[0], candidate_poly.bounds[2]]
                    all_y = [b.bounds[1] for b in placed_polys] + [b.bounds[3] for b in placed_polys] + [candidate_poly.bounds[1], candidate_poly.bounds[3]]
                    current_envelope_area = (max(all_x) - min(all_x)) * (max(all_y) - min(all_y))
                    
                    if current_envelope_area < min_waste_score:
                        min_waste_score = current_envelope_area
                        best_sheet_idx = s_idx
                        best_position = {
                            "dx": ax - r_minx, "dy": ay - r_miny, "angle": angle, "candidate_poly": candidate_poly,
                            "raw_poly_transformed": translate(rotate(raw_normalized_poly, angle, origin=(0, 0)), xoff=ax - r_minx, yoff=ay - r_miny)
                        }

        if best_position and best_sheet_idx != -1:
            sheet_info = nested_sheets[best_sheet_idx]
            sheet_info["parts"].append({"part_ref": part, "original_offset": (min_x, min_y), "placed_polygon": best_position["raw_poly_transformed"], "dx": best_position["dx"], "dy": best_position["dy"], "angle": best_position["angle"]})
            sheet_info["placed_buffered_polygons"].append(best_position["candidate_poly"])
        else:
            new_idx = len(nested_sheets) + 1
            r_minx, r_miny, _, _ = rotate(normalized_poly, 0, origin=(0, 0)).bounds
            nested_sheets.append({
                "sheet_id": new_idx,
                "parts": [{"part_ref": part, "original_offset": (min_x, min_y), "placed_polygon": translate(rotate(raw_normalized_poly, 0, origin=(0, 0)), xoff=margin_val - r_minx, yoff=margin_val - r_miny), "dx": margin_val - r_minx, "dy": margin_val - r_miny, "angle": 0}],
                "placed_buffered_polygons": [translate(rotate(normalized_poly, 0, origin=(0, 0)), xoff=margin_val - r_minx, yoff=margin_val - r_miny)]
            })
    return nested_sheets

def transform_point(x, y, dx, dy, angle, orig_x, orig_y):
    tx, ty = x - orig_x, y - orig_y
    rad = math.radians(angle)
    return tx * math.cos(rad) - ty * math.sin(rad) + dx, tx * math.sin(rad) + ty * math.cos(rad) + dy

def add_sheet_border(msp, ox, sheet_w, sheet_h):
    for p1, p2 in [((ox, 0), (ox + sheet_w, 0)), ((ox + sheet_w, 0), (ox + sheet_w, sheet_h)), ((ox + sheet_w, sheet_h), (ox, sheet_h)), ((ox, sheet_h), (ox, 0))]:
        msp.add_line(p1, p2, dxfattribs={"layer": "CNC_SHEET_BORDER"})

def export_original_geometry_to_dxf(msp, nested_sheets, sheet_w, sheet_h):
    sheet_offset_x = 0
    for sheet_data in nested_sheets:
        ox = sheet_offset_x
        add_sheet_border(msp, ox, sheet_w, sheet_h)
        for p_node in sheet_data["parts"]:
            ref = p_node["part_ref"]
            dx, dy, angle = p_node["dx"] + ox, p_node["dy"], p_node["angle"]
            orig_x, orig_y = p_node["original_offset"]
            for edge in ref["outer_edges"]:
                if edge["type"] == "LINE": msp.add_line(transform_point(edge["start"][0], edge["start"][1], dx, dy, angle, orig_x, orig_y), transform_point(edge["end"][0], edge["end"][1], dx, dy, angle, orig_x, orig_y), dxfattribs={"layer": "CNC_OUTER_CUT"})
                elif edge["type"] == "CIRCLE": msp.add_circle(center=transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y), radius=edge["radius"], dxfattribs={"layer": "CNC_OUTER_CUT"})
            for hole in ref["holes"]:
                layer = "CNC_INNER_DRILL" if hole["is_drill"] else "CNC_INNER_CUT"
                for edge in hole["edges"]:
                    if edge["type"] == "LINE": msp.add_line(transform_point(edge["start"][0], edge["start"][1], dx, dy, angle, orig_x, orig_y), transform_point(edge["end"][0], edge["end"][1], dx, dy, angle, orig_x, orig_y), dxfattribs={"layer": layer})
                    elif edge["type"] == "CIRCLE": msp.add_circle(center=transform_point(edge["center"][0], edge["center"][1], dx, dy, angle, orig_x, orig_y), radius=edge["radius"], dxfattribs={"layer": layer})
        sheet_offset_x += (sheet_w + 300)

# ============================================================
# 6. KỊCH BẢN CHẠY STREAMLIT WORKFLOW
# ============================================================
uploaded_files = st.file_uploader("📂 TẢI LÊN FILE STEP 3D CHI TIẾT SẢN PHẨM", type=["step", "stp"], accept_multiple_files=True)

if uploaded_files:
    parts_db = []
    with st.spinner("⚡ Đang bóc tách dữ liệu khối 3D và biên dạng..."):
        for f in uploaded_files:
            try:
                extracted_data = process_cad_file_with_occ(f.read(), f.name)
                parts_db.append(extracted_data)
            except Exception as e:
                st.error(f"Lỗi đọc file {f.name}: {str(e)}")

    if parts_db:
        st.success(f"⚡ Đã bóc tách thành công {len(parts_db)} chi tiết CAD!")
        
        # --- KHU VỰC NÂNG CẤP: BẢNG XEM TRƯỚC HÌNH HỌC 3D VÀ KÍCH THƯỚC 2D ---
        st.markdown("---")
        st.subheader("🔍 KHU VỰC THẨM ĐỊNH CHI TIẾT GỐC (3D Tương tác & Kích thước)")
        
        for idx, part in enumerate(parts_db):
            st.write(f"### Chi tiết {idx + 1}: **{part['name']}**")
            
            # Chia làm 2 cột: Cột trái xem Phôi 2D + kích thước, Cột phải xoay Khối 3D
            col_preview_2d, col_preview_3d = st.columns([1, 1])
            
            with col_preview_2d:
                st.write("📐 **Biên dạng phẳng & Kích thước bao**")
                st.caption(f"Dài: {part['width']:.1f} mm | Rộng: {part['height']:.1f} mm | Dày ván: {part['thickness']:.1f} mm")
                fig_2d = plot_single_part_with_dimensions(part)
                st.pyplot(fig_2d)
                
            with col_preview_3d:
                st.write("📦 **Khung xem 3D tương tác (Dùng chuột xoay/phóng to)**")
                try:
                    # Khởi tạo cửa sổ vẽ 3D ẩn bằng PyVista
                    plotter = pv.Plotter(window_size=[400, 300])
                    plotter.background_color = "#EAEDE9"
                    
                    # Đọc file lưới STL tạm thời đã xuất từ CadQuery
                    mesh = pv.read(part["stl_path"])
                    
                    # Thêm mesh vào không gian 3D
                    plotter.add_mesh(mesh, color="#0F766E", edge_color="#115E59", show_edges=True, specular=0.2)
                    plotter.view_isometric()
                    
                    # Đưa khung nhìn 3D lên Streamlit WebGL thông qua stpyvista
                    stpyvista(plotter, key=f"pv_preview_{idx}")
                except Exception as ex:
                    st.warning("Không thể hiển thị 3D trên môi trường này. Vui lòng kiểm tra thư viện đồ họa.")
            
            # Xóa file stl tạm sau khi đã hiển thị xong để giải phóng bộ nhớ
            if os.path.exists(part["stl_path"]):
                os.remove(part["stl_path"])
                
            st.markdown("---")
        
        # --- TIẾP TỤC QUY TRÌNH SẮP XẾP NESTING ---
        if st.button("🧠 Bắt đầu tối ưu hóa sắp xếp ván (Nesting)"):
            with st.spinner("Đang tính toán sơ đồ sắp xếp tốt nhất..."):
                nesting_results = perform_advanced_best_fit_nesting(parts_db, sheet_W, sheet_H, total_offset, margin)

            st.subheader(f"📐 Sơ đồ sắp xếp phôi ván thông minh ({len(nesting_results)} Tấm ván)")
            for sheet in nesting_results:
                st.write(f"### 🟫 Sơ đồ tấm ván số: {sheet['sheet_id']}")
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.add_patch(mpatches.Rectangle((0, 0), sheet_W, sheet_H, linewidth=1.2, edgecolor='black', facecolor='#F5F5F5'))
                for p_info in sheet["parts"]:
                    raw_poly = p_info["placed_polygon"]
                    if isinstance(raw_poly, Polygon):
                        xs, ys = raw_poly.exterior.xy
                        ax.fill(xs, ys, alpha=0.8, fc='#0F766E', ec='#115E59', lw=1)
                        for interior in raw_poly.interiors:
                            ax.fill(interior.xy[0], interior.xy[1], fc='#F5F5F5', ec='#B91C1C', lw=0.8)
                        ax.text(raw_poly.centroid.x, raw_poly.centroid.y, p_info["part_ref"]["name"], color='white', weight='bold', fontsize=6, ha='center')
                ax.set_xlim(-50, sheet_W + 50); ax.set_ylim(-50, sheet_H + 50); ax.set_aspect('equal'); plt.axis('off')
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("💾 TẢI FILE DXF & MAPPING HƯỚNG DẪN ĐƯỜNG DAO")
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            doc.layers.new(name='CNC_SHEET_BORDER', dxfattribs={'color': 8})
            doc.layers.new(name='CNC_OUTER_CUT', dxfattribs={'color': 1})
            doc.layers.new(name='CNC_INNER_CUT', dxfattribs={'color': 4})
            doc.layers.new(name='CNC_INNER_DRILL', dxfattribs={'color': 2})
            
            export_original_geometry_to_dxf(msp, nesting_results, sheet_W, sheet_H)
            
            out_stream = io.StringIO()
            doc.write(out_stream)
            
            st.download_button(label="📥 Tải xuống file DXF sản xuất", data=out_stream.getvalue(), file_name="cnc_nesting_output.dxf", mime="application/dxf")
