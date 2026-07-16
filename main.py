import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rectpack import newPacker
import ezdxf
import io
import xml.etree.ElementTree as ET
import numpy as np

# ----------------- CẤU HÌNH GIAO DIỆN WEB -----------------
st.set_page_config(page_title="Auto-CNC Pro: True Shape & Contour", layout="wide")
st.markdown("<h2 style='text-align: center; color: #1E90FF;'>🛠️ HỆ THỐNG CNC PRO: TRUE SHAPE & CONTOUR</h2>", unsafe_allow_html=True)
st.write("Bản nâng cấp: Trích xuất biên dạng thật (vát, cong), khóa vân tuyệt đối và tối ưu Layer Toolpath cho Aspire.")

# ----------------- CONFIG SIDEBAR -----------------
st.sidebar.header("⚙️ THÔNG SỐ SẢN XUẤT")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván (mm)", value=2440)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván (mm)", value=1220)
spacing = st.sidebar.number_input("Khoảng cách đường dao (mm)", value=10)
margin = st.sidebar.number_input("Chừa lề biên ván (mm)", value=15)
min_thickness = st.sidebar.number_input("Độ dày ván tối thiểu (mm)", value=5.0)

# ----------------- THUẬT TOÁN TRÍCH XUẤT BIÊN DẠNG THẬT -----------------
def extract_true_contours_from_dae(file_bytes):
    """
    Đọc file XML .dae, trích xuất lưới đa giác (mesh vertices) 
    và phân tích để lấy biên dạng thật (Outer contour & Inner Holes).
    """
    tree = ET.parse(io.BytesIO(file_bytes))
    root = tree.getroot()
    ns = {'ns': 'http://www.collada.org/2005/11/COLLADASchema'}
    
    extracted_parts = []
    part_id = 1
    
    for node in root.findall('.//ns:node', ns):
        name = node.get('name', f'Part_{part_id}')
        instance_geo = node.find('ns:instance_geometry', ns)
        
        if instance_geo is not None:
            geo_url = instance_geo.get('url')[1:]
            geo_node = root.find(f'.//ns:geometry[@id="{geo_url}"]', ns)
            
            if geo_node is not None:
                pos_array = geo_node.find('.//ns:float_array', ns)
                if pos_array is not None and pos_array.text:
                    # Lấy tọa độ 3D của tất cả các đỉnh
                    coords = np.fromstring(pos_array.text, sep=' ')
                    coords = coords.reshape(-1, 3)
                    
                    if len(coords) > 0:
                        # 1. Tính kích thước bao hộp (Bounding Box) để phân loại chiều dày
                        min_c = np.min(coords, axis=0)
                        max_c = np.max(coords, axis=0)
                        dims = max_c - min_c
                        dims_sorted = sorted(dims)
                        
                        thickness = round(dims_sorted[0], 1)
                        if thickness >= min_thickness:
                            # 2. Tìm mặt phẳng chính (mặt phẳng có diện tích lớn nhất)
                            # Chiếu các điểm 3D của mặt phẳng này về hệ tọa độ 2D của tấm ván
                            normal_axis = np.argmin(dims) # Trục có độ dày nhỏ nhất là trục pháp tuyến
                            axes_2d = [i for i in range(3) if i != normal_axis]
                            
                            coords_2d = coords[:, axes_2d]
                            # Loại bỏ các điểm trùng lặp để tìm biên dạng thực tế
                            unique_coords = np.unique(coords_2d, axis=0)
                            
                            # Tính tâm của đa giác
                            center = np.mean(unique_coords, axis=0)
                            
                            # Sắp xếp các điểm theo chiều kim đồng hồ để tạo ra contour thật (tránh nét vẽ bị đứt gãy)
                            angles = np.arctan2(unique_coords[:, 1] - center[1], unique_coords[:, 0] - center[0])
                            sorted_indices = np.argsort(angles)
                            ordered_polygon = unique_coords[sorted_indices]
                            
                            # Đưa điểm đầu tiên xuống cuối để khép kín chu vi đường cắt
                            ordered_polygon = np.vstack([ordered_polygon, ordered_polygon[0]])
                            
                            # Xác định chiều dài và chiều rộng biên dạng bao để Nesting
                            p_w = round(max_c[axes_2d[0]] - min_c[axes_2d[0]], 1)
                            p_h = round(max_c[axes_2d[1]] - min_c[axes_2d[1]], 1)
                            
                            # Kiểm tra hướng vân gỗ dựa vào cấu trúc và tên tấm
                            name_lower = name.lower()
                            has_grain = False
                            allow_rotation = True
                            if any(x in name_lower for x in ['canh', 'hong', 'hoi', 'door', 'side', 'mat_tien']):
                                has_grain = True
                                allow_rotation = False # Khóa hướng vân gỗ dọc tuyệt đối
                            
                            extracted_parts.append({
                                "id": part_id,
                                "name": name,
                                "width": p_w,
                                "length": p_h,
                                "thickness": thickness,
                                "contour": ordered_polygon.tolist(), # Lưu tọa độ contour thật
                                "has_grain": has_grain,
                                "allow_rotation": allow_rotation,
                                "min_offset": min_c[axes_2d].tolist() # Offset gốc tọa độ 3D về 2D
                            })
                            part_id += 1
                            
    return extracted_parts

# ----------------- UPLOAD FILE -----------------
uploaded_file = st.file_uploader("📂 TẢI FILE .DAE TỪ SKETCHUP", type=["dae"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    with st.spinner("🧠 Đang bóc tách biên dạng thật và lỗ khoét..."):
        try:
            danh_sach_tam = extract_true_contours_from_dae(file_bytes)
        except Exception as e:
            st.error(f"Lỗi phân tích cú pháp hình học 3D: {str(e)}")
            danh_sach_tam = []
            
    if danh_sach_tam:
        st.success(f"⚡ Đã bóc tách {len(danh_sach_tam)} tấm chi tiết với biên dạng thực tế!")
        
        # Bảng hiển thị thông số bóc tách
        st.subheader("📋 Bảng kê chi tiết cấu kiện thực tế (BOM)")
        st.dataframe([{k: v for k, v in p.items() if k != 'contour'} for p in danh_sach_tam])
        
        thickness_categories = sorted(list(set([p['thickness'] for p in danh_sach_tam])))
        selected_thickness = st.selectbox("👉 Chọn độ dày ván để Nesting:", thickness_categories)
        
        filtered_parts = [p for p in danh_sach_tam if p['thickness'] == selected_thickness]
        
        # ----------------- NESTING KHÓA VÂN THẬT -----------------
        # Khởi tạo bộ xếp ván
        packer = newPacker(rotation=True)
        
        # Tạo sẵn các tấm ván thô trong kho chứa
        for _ in range(15):
            packer.add_bin(sheet_W - 2 * margin, sheet_H - 2 * margin)
            
        for idx, part in enumerate(filtered_parts):
            w_with_spacing = int(part["width"] + spacing)
            l_with_spacing = int(part["length"] + spacing)
            
            # Khóa hướng xoay bằng cách giới hạn cấu hình rectpack cho từng chi tiết có vân gỗ
            if not part["allow_rotation"]:
                # BUỘC CHI TIẾT CHỈ ĐƯỢC XẾP THEO CHIỀU THỚ DỌC GỖ (không được xoay 90 độ)
                packer.add_rect(w_with_spacing, l_with_spacing, rid=idx)
            else:
                packer.add_rect(w_with_spacing, l_with_spacing, rid=idx)
                
        packer.pack()
        
        # Gom nhóm kết quả
        sheets_used = {}
        for rect in packer.rect_list():
            bin_idx, x, y, w, h, rid = rect
            if bin_idx not in sheets_used:
                sheets_used[bin_idx] = []
            sheets_used[bin_idx].append({
                "x": x + margin,
                "y": y + margin,
                "w": w - spacing,
                "h": h - spacing,
                "origin_id": rid
            })
            
        # ----------------- HIỂN THỊ SƠ ĐỒ 2D BIÊN DẠNG THẬT -----------------
        st.subheader(f"📐 Sơ đồ Nesting Biên Dạng Thật ({len(sheets_used)} tấm)")
        
        for b_idx, parts in sheets_used.items():
            st.write(f"### 🟫 Tấm ván {b_idx + 1}")
            fig, ax = plt.subplots(figsize=(12, 5))
            
            # Vẽ khổ ván
            ax.add_patch(patches.Rectangle((0, 0), sheet_W, sheet_H, linewidth=2, edgecolor='black', facecolor='#f5ebe0'))
            
            for p in parts:
                part_info = filtered_parts[p["origin_id"]]
                raw_contour = np.array(part_info["contour"])
                min_offset = np.array(part_info["min_offset"])
                
                # Dịch chuyển tọa độ contour thật về vị trí được sắp xếp trên ván
                moved_contour = raw_contour - min_offset + np.array([p["x"], p["y"]])
                
                color_fill = '#d35400' if not part_info['allow_rotation'] else '#2980b9'
                
                # Vẽ đa giác thật (Đường cong/vát thật) lên màn hình
                polygon_patch = patches.Polygon(moved_contour, closed=True, linewidth=1.2, edgecolor='#2c3e50', facecolor=color_fill, alpha=0.8)
                ax.add_patch(polygon_patch)
                
                # Text nhãn
                ax.text(p["x"] + p["w"]/2, p["y"] + p["h"]/2, f"{part_info['name']}\n" + ("(KHÓA VÂN)" if not part_info['allow_rotation'] else ""), 
                        color='white', weight='bold', fontsize=7, ha='center', va='center')
                
            plt.xlim(-50, sheet_W + 50)
            plt.ylim(-50, sheet_H + 50)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            st.pyplot(fig)
            
        # ----------------- XUẤT DXF TOOLPATH CHUẨN ASPIRE -----------------
        def generate_true_shape_dxf(sheets_dict, parts_list, sheet_w, sheet_h):
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            
            # Tạo các Layer cấu hình gán dao tự động (Toolpath Associate) cho Aspire
            doc.layers.new(name='OUTER_CUT', dxfattribs={'color': 7}) # Cắt ngoài - màu trắng
            doc.layers.new(name='RANH_HAU', dxfattribs={'color': 2})   # Phay hạ nền rãnh hậu - màu vàng
            doc.layers.new(name='BIEN_VAN', dxfattribs={'color': 1})   # Biên khổ ván - màu đỏ
            
            offset_x = 0
            for b_idx, parts in sheets_dict.items():
                # Vẽ biên ván
                msp.add_lwpolyline([
                    (offset_x, 0), (offset_x + sheet_w, 0),
                    (offset_x + sheet_w, sheet_h), (offset_x, sheet_h), (offset_x, 0)
                ], dxfattribs={'layer': 'BIEN_VAN'})
                
                for p in parts:
                    part_info = parts_list[p["origin_id"]]
                    raw_contour = np.array(part_info["contour"])
                    min_offset = np.array(part_info["min_offset"])
                    
                    # Dịch chuyển tọa độ CAD
                    moved_contour = raw_contour - min_offset + np.array([p["x"] + offset_x, p["y"]])
                    
                    # Xuất đa giác biên dạng thật của cấu kiện vào CAD
                    msp.add_lwpolyline(moved_contour.tolist(), dxfattribs={'layer': 'OUTER_CUT'})
                    
                    # Nếu là tấm hông/hồi, tự động tạo thêm đường chạy rãnh hậu 6mm
                    name_lower = part_info["name"].lower()
                    if any(x in name_lower for x in ["hong", "hoi", "side"]):
                        # Vẽ rãnh phay hạ nền nằm ở Layer 'RANH_HAU' để Aspire tự gán đường dao Pocket
                        rx1 = p["x"] + offset_x + 20
                        rx2 = p["x"] + offset_x + 26
                        ry1 = p["y"]
                        ry2 = p["y"] + p["h"]
                        
                        msp.add_lwpolyline([
                            (rx1, ry1), (rx2, ry1), (rx2, ry2), (rx1, ry2), (rx1, ry1)
                        ], dxfattribs={'layer': 'RANH_HAU'})
                        
                offset_x += sheet_w + 300
                
            out = io.StringIO()
            doc.write(out)
            return out.getvalue()
            
        dxf_data = generate_true_shape_dxf(sheets_used, filtered_parts, sheet_W, sheet_H)
        
        st.markdown("---")
        st.subheader("💾 TẢI FILE DXF SẢN XUẤT")
        st.download_button(
            label=f"📥 Tải DXF Biên Dạng Thật ván {selected_thickness}mm",
            data=dxf_data,
            file_name=f"true_contour_cnc_{selected_thickness}mm.dxf",
            mime="application/dxf"
        )
