import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import gmsh
import os
import meshio

# 设置中文字体（解决中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

class FiberBraggGratingModel:
    """光纤布拉格光栅(FBG)与碳纤维复合材料封装建模"""
    
    def __init__(self):
        # 初始化Gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 1)
        self.model = gmsh.model
        self.model.add("FBG_CarbonFiber")
        
        # 默认参数设置
        self.params = {
            # FBG参数(mm)
            "fiber_radius": 0.0625,       # 光纤半径
            "grating_length": 5.0,        # 光栅长度
            "grating_period": 0.00055,    # 光栅周期(550nm)
            "grating_modulation": 0.0001, # 光栅折射率调制
            
            # 封装参数(mm)
            "encapsulation_thickness": 1.0,     # 封装厚度
            "encapsulation_length": 20.0,       # 封装长度
            "encapsulation_width": 2.0,         # 封装宽度(矩形截面)
            "encapsulation_height": 2.0,        # 封装高度(矩形截面)
            
            # 碳纤维复合材料参数
            "carbon_layers": 4,                 # 碳层数量
            "carbon_ply_thickness": 0.125,      # 每层厚度(mm)
            "carbon_orientation": [0, 90, -45, 45], # 各层方向(度)
            
            # 网格参数
            "mesh_size_fbg": 0.05,              # FBG区域网格大小
            "mesh_size_interface": 0.02,        # 界面区域网格大小
            "mesh_size_encapsulation": 0.1,     # 封装区域网格大小
        }
        
        # 实体标签
        self.entities = {
            "fiber": None,
            "grating": None,
            "encapsulation": [],
            "interfaces": []
        }
        
    def set_parameters(self, params):
        """设置模型参数"""
        self.params.update(params)
    
    def create_geometry(self):
        """创建几何模型"""
        # 创建光纤
        fiber_length = self.params["encapsulation_length"]
        fiber = self.model.occ.addCylinder(0, 0, 0, fiber_length, 0, 0, self.params["fiber_radius"])
        self.entities["fiber"] = fiber
        self.model.occ.synchronize()  # 同步几何模型
        if fiber is None:
            print("错误：光纤实体创建失败")
            return
        print(f"创建光纤，标签: {fiber}")  # 添加调试信息
        
        # 创建光栅区域
        grating = self.model.occ.addCylinder(
            (fiber_length - self.params["grating_length"]) / 2, 0, 0,
            self.params["grating_length"], 0, 0, self.params["fiber_radius"]
        )
        self.entities["grating"] = grating
        self.model.occ.synchronize()  # 同步几何模型
        if grating is None:
            print("错误：光栅实体创建失败")
            return
        print(f"创建光栅区域，标签: {grating}")  # 添加调试信息
        
        # 从光纤中切割出光栅区域
        try:
            self.model.occ.cut([(3, fiber)], [(3, grating)], removeTool=False)
            self.model.occ.synchronize()  # 同步几何模型
            print("完成光纤和光栅区域的切割操作")  # 添加调试信息
        except Exception as e:
            print(f"光纤和光栅区域切割失败: {e}")
            return
        
        # 创建碳纤维封装
        encapsulation_base = self.model.occ.addBox(
            -self.params["encapsulation_width"]/2,
            -self.params["encapsulation_height"]/2,
            0,
            self.params["encapsulation_width"],
            self.params["encapsulation_height"],
            self.params["encapsulation_length"]
        )
        self.model.occ.synchronize()  # 同步几何模型
        if encapsulation_base is None:
            print("错误：碳纤维封装实体创建失败")
            return
        print(f"创建碳纤维封装，标签: {encapsulation_base}")  # 添加调试信息
        
        # 从封装中减去光纤区域
        try:
            self.model.occ.cut([(3, encapsulation_base)], [(3, fiber)], removeTool=False)
            self.model.occ.synchronize()  # 同步几何模型
            print("完成封装和光纤区域的切割操作")  # 添加调试信息
        except Exception as e:
            print(f"封装和光纤区域切割失败: {e}")
            return
        
        # 分层创建碳纤维复合材料
        layer_thickness = self.params["carbon_ply_thickness"]
        total_thickness = self.params["carbon_layers"] * layer_thickness
        
        # 确保总厚度不超过封装厚度
        if total_thickness > self.params["encapsulation_height"]:
            layer_thickness = self.params["encapsulation_height"] / self.params["carbon_layers"]
            self.params["carbon_ply_thickness"] = layer_thickness
        
        for i in range(self.params["carbon_layers"]):
            z1 = -self.params["encapsulation_height"]/2 + i * layer_thickness
            z2 = z1 + layer_thickness
            
            layer = self.model.occ.addBox(
                -self.params["encapsulation_width"]/2,
                -self.params["encapsulation_height"]/2,
                z1,
                self.params["encapsulation_width"],
                self.params["encapsulation_height"],
                z2 - z1
            )
            self.model.occ.synchronize()  # 同步几何模型
            if layer is None:
                print(f"错误：碳层 {i+1} 实体创建失败")
                continue
            print(f"创建碳层 {i+1}，标签: {layer}")  # 添加调试信息
            
            # 从层中减去光纤区域
            try:
                self.model.occ.cut([(3, layer)], [(3, fiber)], removeTool=False)
                self.entities["encapsulation"].append(layer)
                self.model.occ.synchronize()  # 同步几何模型
                print(f"完成碳层 {i+1} 和光纤区域的切割操作")  # 添加调试信息
            except Exception as e:
                print(f"碳层 {i+1} 和光纤区域切割失败: {e}")
        
        # 最终同步几何模型
        self.model.occ.synchronize()
        print("几何模型创建完成")  # 添加调试信息
        
    def define_materials(self):
        """定义材料属性"""
        # FBG材料属性
        self.model.addPhysicalGroup(3, [self.entities["fiber"]], 1)
        self.model.setPhysicalName(3, 1, "Silica_Fiber")
        
        self.model.addPhysicalGroup(3, [self.entities["grating"]], 2)
        self.model.setPhysicalName(3, 2, "FBG_Grating")
        
        # 碳纤维复合材料层
        for i, layer in enumerate(self.entities["encapsulation"]):
            layer_id = i + 3
            self.model.addPhysicalGroup(3, [layer], layer_id)
            self.model.setPhysicalName(3, layer_id, f"Carbon_Layer_{i+1}")
    
    def generate_mesh(self):
        """生成网格"""
        try:
            # 设置网格尺寸
            # FBG区域
            self.model.mesh.setSize([(0, self.entities["fiber"])], self.params["mesh_size_fbg"])
            self.model.mesh.setSize([(0, self.entities["grating"])], self.params["mesh_size_fbg"])
            
            # 界面区域
            for layer in self.entities["encapsulation"]:
                surfaces = self.model.getBoundary([(3, layer)], oriented=False)
                for surface in surfaces:
                    self.model.mesh.setSize([surface], self.params["mesh_size_interface"])
            
            # 封装区域
            for layer in self.entities["encapsulation"]:
                self.model.mesh.setSize([(3, layer)], self.params["mesh_size_encapsulation"])
            
            print("开始生成网格...")
            # 生成网格
            self.model.mesh.generate(3)
            print("网格生成成功")
        except Exception as e:
            print(f"网格生成失败: {e}")
            # 尝试调整网格尺寸并重新生成
            print("尝试调整网格尺寸并重新生成...")
            self.params["mesh_size_interface"] *= 1.5
            self.params["mesh_size_encapsulation"] *= 1.5
            
            # 重新设置网格尺寸
            # FBG区域
            self.model.mesh.setSize([(0, self.entities["fiber"])], self.params["mesh_size_fbg"])
            self.model.mesh.setSize([(0, self.entities["grating"])], self.params["mesh_size_fbg"])
            
            # 界面区域
            for layer in self.entities["encapsulation"]:
                surfaces = self.model.getBoundary([(3, layer)], oriented=False)
                for surface in surfaces:
                    self.model.mesh.setSize([surface], self.params["mesh_size_interface"])
            
            # 封装区域
            for layer in self.entities["encapsulation"]:
                self.model.mesh.setSize([(3, layer)], self.params["mesh_size_encapsulation"])
            
            try:
                self.model.mesh.generate(3)
                print("调整网格尺寸后，网格生成成功")
            except Exception as e:
                print(f"调整网格尺寸后，网格仍然生成失败: {e}")
    
    def export_mesh(self, filename="fbg_carbon_mesh.msh"):
        """导出网格"""
        try:
            gmsh.write(filename)
            if os.path.exists(filename):
                print(f"网格已导出至 {filename}")
            else:
                print(f"警告：文件 {filename} 未成功导出。")
                return

            # 转换为其他格式(如果需要)
            if filename.endswith(".msh"):
                new_filename = filename.replace(".msh", ".vtk")
                if os.path.exists(filename):
                    try:
                        print(f"尝试读取文件 {filename}...")
                        mesh = meshio.read(filename)
                        print(f"成功读取文件 {filename}，开始写入 {new_filename}...")
                        meshio.write(new_filename, mesh)
                        print(f"网格已转换并导出至 {new_filename}")
                    except SystemExit as se:
                        print(f"读取文件 {filename} 时触发 SystemExit 异常，退出码: {se.code}")
                    except Exception as e:
                        print(f"转换文件 {filename} 到 {new_filename} 时出错: {e}")
                else:
                    print(f"警告：源文件 {filename} 不存在，无法进行转换。")
        except Exception as e:
            print(f"导出网格时出错: {e}")
    
    def visualize_model(self):
        """可视化模型(简化版)"""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制光纤(简化为圆柱体)
        z = np.linspace(0, self.params["encapsulation_length"], 100)
        theta = np.linspace(0, 2*np.pi, 100)
        Z, THETA = np.meshgrid(z, theta)
        X = self.params["fiber_radius"] * np.cos(THETA)
        Y = self.params["fiber_radius"] * np.sin(THETA)
        
        ax.plot_surface(X, Y, Z, color='blue', alpha=0.3)
        
        # 绘制光栅区域(简化为光纤上的一段)
        z_grating = np.linspace(
            (self.params["encapsulation_length"] - self.params["grating_length"]) / 2,
            (self.params["encapsulation_length"] + self.params["grating_length"]) / 2,
            100
        )
        Z_GRATING, THETA_GRATING = np.meshgrid(z_grating, theta)
        X_GRATING = self.params["fiber_radius"] * np.cos(THETA_GRATING)
        Y_GRATING = self.params["fiber_radius"] * np.sin(THETA_GRATING)
        
        # 绘制封装(简化为矩形)
        x = np.linspace(-self.params["encapsulation_width"]/2, self.params["encapsulation_width"]/2, 100)  # 修改为 100 以匹配 Z_GRATING 的形状
        y = np.linspace(-self.params["encapsulation_height"]/2, self.params["encapsulation_height"]/2, 100)  # 修改为 100 以匹配 Z_GRATING 的形状
        X, Y = np.meshgrid(x, y)
        
        # 顶部和底部
        Z_top = np.ones_like(X) * self.params["encapsulation_length"]
        Z_bottom = np.zeros_like(X)
        
        ax.plot_surface(X, Y, Z_top, color='gray', alpha=0.2)
        ax.plot_surface(X, Y, Z_bottom, color='gray', alpha=0.2)
        
        # 侧面
        for x_val in [x[0], x[-1]]:
            X_side = np.ones_like(Y) * x_val
            ax.plot_surface(X_side, Y, Z_GRATING, color='gray', alpha=0.2)
        
        for y_val in [y[0], y[-1]]:
            Y_side = np.ones_like(X) * y_val
            ax.plot_surface(X, Y_side, Z_GRATING, color='gray', alpha=0.2)
        
        # 设置图表属性
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('碳纤维复合材料封装FBG模型')
        
        plt.show()
    
    def generate_ansys_input(self, filename="fbg_carbon_ansys.inp"):
        """生成ANSYS输入文件"""
        with open(filename, 'w') as f:
            # 写入标题
            f.write(f"! 碳纤维复合材料封装FBG模型\n")
            f.write(f"! 模型参数\n")
            for key, value in self.params.items():
                f.write(f"! {key}: {value}\n")
            f.write("\n")
            
            # 定义材料属性
            f.write("! 定义材料属性\n")
            # 二氧化硅光纤
            f.write("MP,EX,1,72E3 ! 弹性模量(GPa)\n")
            f.write("MP,NUXY,1,0.17 ! 泊松比\n")
            f.write("MP,DENS,1,2.2E-9 ! 密度(kg/mm^3)\n")
            
            # FBG光栅(与光纤类似，但可能有不同的光学特性)
            f.write("MP,EX,2,72E3\n")
            f.write("MP,NUXY,2,0.17\n")
            f.write("MP,DENS,2,2.2E-9\n")
            
            # 碳纤维复合材料(每层可能不同)
            for i in range(self.params["carbon_layers"]):
                mat_id = i + 3
                orientation = self.params["carbon_orientation"][i % len(self.params["carbon_orientation"])]
                
                f.write(f"! 碳层 {i+1}, 方向: {orientation}度\n")
                f.write(f"MP,EX,{mat_id},140E3 ! 纵向弹性模量(GPa)\n")
                f.write(f"MP,EY,{mat_id},10E3 ! 横向弹性模量(GPa)\n")
                f.write(f"MP,EZ,{mat_id},10E3 ! 横向弹性模量(GPa)\n")
                f.write(f"MP,NUXY,{mat_id},0.3 ! 泊松比\n")
                f.write(f"MP,NUYZ,{mat_id},0.4 ! 泊松比\n")
                f.write(f"MP,NUXZ,{mat_id},0.3 ! 泊松比\n")
                f.write(f"MP,GXY,{mat_id},5E3 ! 剪切模量(GPa)\n")
                f.write(f"MP,GYZ,{mat_id},3.5E3 ! 剪切模量(GPa)\n")
                f.write(f"MP,GXZ,{mat_id},5E3 ! 剪切模量(GPa)\n")
                f.write(f"MP,DENS,{mat_id},1.6E-9 ! 密度(kg/mm^3)\n")
                f.write("\n")
            
            # 这里可以继续添加网格和边界条件定义
            f.write("! 注意: 此文件仅包含材料定义\n")
            f.write("! 网格和边界条件需要根据具体的网格划分和分析需求添加\n")
        
        print(f"ANSYS输入文件已生成至 {filename}")
    
    def finalize(self):
        """完成并关闭模型"""
        gmsh.finalize()


if __name__ == "__main__":
    # 创建模型实例
    model = FiberBraggGratingModel()
    
    # 可选: 修改默认参数
    custom_params = {
        "encapsulation_thickness": 1.5,
        "carbon_layers": 6,
        "carbon_orientation": [0, 90, -45, 45, 90, 0]
    }
    model.set_parameters(custom_params)
    
    # 创建几何模型
    model.create_geometry()
    
    # 定义材料
    model.define_materials()
    
    # 生成网格
    model.generate_mesh()
    
    # 导出网格
    mesh_filename = "fbg_carbon_mesh.msh"
    print(f"尝试导出网格到: {os.path.abspath(mesh_filename)}")
    model.export_mesh(mesh_filename)
    
    # 生成ANSYS输入文件
    model.generate_ansys_input("fbg_carbon_ansys.inp")
    
    # 可视化模型
    model.visualize_model()
    
    # 完成并关闭
    model.finalize()