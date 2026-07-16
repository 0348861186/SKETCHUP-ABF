import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rectpack import newPacker
import ezdxf
import io
import xml.etree.ElementTree as ET
import numpy as np

# ----------------- CẤU HÌNH TRANG WEB -----------------
st.set_page_config(page_title="Auto-CNC: Đọc file 3D Khách hàng", layout="wide")
st.title("🤖 Hệ thống Tự động hóa CNC từ File 3D Khách hàng")
st.write("Hỗ trợ định dạng file 3D `.dae` (Khách hàng chỉ cần vào SketchUp -> Export -> 3D Model -> Chọn đuôi .dae)")

# ----------------- THANH ĐIỀU CHỈNH -----------------
st.sidebar.header("⚙️ Cấu hình xưởng")
sheet_W = st.sidebar.number_input("Chiều rộng khổ ván (mm)", value=2440)
sheet_H = st.sidebar.number_input("Chiều cao khổ ván (mm)", value=1220)
spacing = st.sidebar.number_input("Khoảng cách giữa các tấm khi cắt (mm)", value=10)
min_thickness = st.sidebar.number_input("Độ dày ván tối thiểu để bóc tách (mm)", value=5.0)

# ----------------- HÀM XỬ LÝ BÓC TÁCH FILE 3D (.DAE) -----------------
def extract_parts_from_dae(file_bytes):
    """
    Đọc cấu trúc XML của file .dae để lấy kích thước Box (Bounding Box) 
    của từng Group/Component trong bản vẽ 3D.
    """
    tree = ET.parse(io.BytesIO(file_bytes))
    root = tree.getroot()
    
    # Namespace mặc định của file Collada (.dae)
    ns = {'ns': 'http://www.collada.org/2005/11/COLLADASchema'}
    
    parts = []
    part_id = 1
    
    # Tìm kiếm các node chứa thông tin hình học (Geometries)
    for node in root.findall('.//ns:node', ns):
        name = node.get('name', f'Tam_gỗ_{part_id}')
        
        # Tìm tọa độ các đỉnh của tấm ván để tính kích thước dài, rộng, cao thực tế
        instance_geo = node.find('ns:instance_geometry', ns)
        if instance_geo is not None:
            geo_url = instance_geo.get('url')[1:] # Bỏ dấu # ở đầu
            
            # Tìm danh sách các đỉnh (vertices) của hình học này
            geo_node = root.find(f'.//ns:geometry[@id="{geo_url}"]', ns)
            if geo_node is not None:
                pos_array = geo_node.find('.//ns:float_array', ns)
                if pos_array is not None and pos_array.text:
                    coords = np.fromstring(pos_array.text, sep=' ')
                    coords = coords.reshape(-1, 3) # Chuyển thành ma trận tọa độ X, Y, Z
                    
                    if len(coords) > 0:
                        # Tính Bounding Box (Kích thước bao ngoài của tấm ván)
                        min_coords = np.min(coords, axis=0)
                        max_coords = np.max(coords, axis=0)
                        dims = max_coords - min_coords # Trả về mảng [Dài, Rộng, Cao] tính bằng mm
                        
                        # Sắp xếp kích thước từ lớn đến bé để xác định: Dài, Rộng, Dày
                        dims_sorted = sorted(dims)
                        thickness = dims_sorted[0]  # Kích thước nhỏ nhất là độ dày tấm
                        width = dims_sorted[1]      # Kích thước trung bình là chiều rộng
                        length = dims_sorted[2]     # Kích thước lớn nhất là chiều dài
                        
                        # Chỉ lấy những tấm có độ dày lớn hơn độ dày tối thiểu cấu hình (bỏ qua đinh ốc, tay nắm...)
                        if thickness >= min_thickness:
                            parts.append({
                                "id": part_id,
                                "name": name,
                                "length": round(length, 1),
                                "width": round(width, 1),
                                "thickness": round(thickness, 1)
                            })
                            part_id += 1
    return parts

# ----------------- GIAO DIỆN UPLOAD FILE -----------------
uploaded_file = st.file_uploader("📂 Tải lên file 3D (.dae) của khách hàng", type=["dae"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    
    with st.spinner("Đang bóc tách dữ liệu 3D..."):
        try:
            danh_sach_tam = extract_parts_from_dae(file_bytes)
        except Exception as e:
            st.error(f"Lỗi khi đọc file 3D: {str(e)}")
            danh_sach_tam = []
            
    if danh_sach_tam:
        st.success(f"Bóc tách thành công! Phát hiện {len(danh_sach_tam)} tấm ván gỗ.")
        
        # Hiển thị bảng danh sách vật tư đã bóc tách
        st.subheader("📋 Danh sách tấm ván bóc tách tự động từ 3D")
        st.dataframe(danh_sach_tam)
        
        # Phân loại ván theo độ dày (ví dụ ván hậu 6mm, ván thùng 17mm)
        thickness_categories = list(set([p['thickness'] for p in danh_sach_tam]))
        selected_thickness = st.selectbox("Chọn độ dày ván để tiến hành Nesting:", sorted(thickness_categories))
        
        # Lọc danh sách tấm theo độ dày đã chọn
        filtered_parts = [p for p in danh_sach_tam if p['thickness'] == selected_thickness]
        
        # ----------------- THUẬT TOÁN NESTING -----------------
        packer = newPacker(rotation=True)
        packer.add_bin(sheet_W, sheet_H)
        
        for idx, part in enumerate(filtered_parts):
            w_with_spacing = int(part["width"] + spacing)
            l_with_spacing = int(part["length"] + spacing)
            packer.add_rect(w_with_spacing, l_with_spacing, rid=idx)
            
        packer.pack()
        all_rects = packer.rect_list()
        
        # ----------------- TRỰC QUAN HÓA SƠ ĐỒ XẾP VÁN -----------------
        st.subheader(f"📐 Sơ đồ xếp ván tối ưu cho ván dày {selected_thickness}mm")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.add_patch(patches.Rectangle((0, 0), sheet_W, sheet_H, linewidth=2, edgecolor='black', facecolor='#f5f5f5'))
        
        for rect in all_rects:
            b, x, y, w, h, rid = rect
            actual_w = w - spacing
            actual_h = h - spacing
            part_name = filtered_parts[rid]["name"]
            
            # Vẽ hình chữ nhật đại diện tấm ván
            ax.add_patch(patches.Rectangle((x, y), actual_w, actual_h, linewidth=1, edgecolor='#d35400', facecolor='#3498db', alpha=0.7))
            # Ghi thông số kích thước lên hình vẽ
            ax.text(x + actual_w/2, y + actual_h/2, f"{part_name}\n{int(actual_w)}x{int(actual_h)}", 
                    color='white', weight='bold', fontsize=7, ha='center', va='center', clip_on=True)
            
        plt.xlim(-50, sheet_W + 50)
        plt.ylim(-50, sheet_H + 50)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis('off')
        st.pyplot(fig)
        
        # ----------------- XUẤT FILE DXF -----------------
        def generate_dxf(rects, parts_list, sheet_w, sheet_h):
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()
            doc.layers.new(name='CUT_LINE', dxfattribs={'color': 7}) # Màu trắng làm đường cắt ngoài
            doc.layers.new(name='SHEET_BORDER', dxfattribs={'color': 1}) # Màu đỏ làm biên ván
            
            # Vẽ biên khổ ván
            msp.add_lwpolyline([(0,0), (sheet_w,0), (sheet_w, sheet_h), (0, sheet_h), (0,0)], dxfattribs={'layer': 'SHEET_BORDER'})
            
            for rect in rects:
                b, x, y, w, h, rid = rect
                actual_w = w - spacing
                actual_h = h - spacing
                
                points = [(x, y), (x + actual_w, y), (x + actual_w, y + actual_h), (x, y + actual_h), (x, y)]
                msp.add_lwpolyline(points, dxfattribs={'layer': 'CUT_LINE'})
                msp.add_text(parts_list[rid]["name"], dxfattribs={'height': 25, 'layer': 'CUT_LINE'}).set_placement((x + 10, y + 10))
                
            out = io.StringIO()
            doc.write(out)
            return out.getvalue()
            
        dxf_data = generate_dxf(all_rects, filtered_parts, sheet_W, sheet_H)
        
        st.download_button(
            label=f"📥 Tải File DXF ván {selected_thickness}mm cho Aspire",
            data=dxf_data,
            file_name=f"nesting_{selected_thickness}mm.dxf",
            mime="application/dxf"
        )
    else:
        st.warning("Không tìm thấy tấm ván hợp lệ nào trong file 3D. Hãy kiểm tra lại bản vẽ của khách.")
