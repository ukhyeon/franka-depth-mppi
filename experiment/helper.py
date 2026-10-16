#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from typing import Dict, List

# Third Party
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import cm
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid
from omni.isaac.core.robots import Robot
from pxr import UsdPhysics
from omni.isaac.core.articulations import Articulation  # 추가
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.robots import RobotView

# CuRobo
from curobo.util.logger import log_warn
from curobo.util.usd_helper import set_prim_transform

ISAAC_SIM_23 = False
ISAAC_SIM_45 = False
try:
    # Third Party
    from omni.isaac.urdf import _urdf  # isaacsim 2022.2
except ImportError:
    # Third Party
    try:
        from omni.importer.urdf import _urdf  # isaac sim 2023.1 or above
    except ImportError:
        from isaacsim.asset.importer.urdf import _urdf  # isaac sim 4.5+

        ISAAC_SIM_45 = True
    ISAAC_SIM_23 = True

try:
    # import for older isaacsim installations
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    # import for isaac sim 4.5+
    from isaacsim.core.api.materials import OmniPBR


# Standard Library
from typing import Optional

# Third Party
from omni.isaac.core.utils.extensions import enable_extension

# CuRobo
from curobo.util_file import get_assets_path, get_filename, get_path_of_dir, join_path


def add_extensions(simulation_app, headless_mode: Optional[str] = None):
    ext_list = [
        "omni.kit.asset_converter",
        "omni.kit.tool.asset_importer",
        "omni.isaac.asset_browser",
    ]
    if headless_mode is not None:
        log_warn("Running in headless mode: " + headless_mode)
        ext_list += ["omni.kit.livestream." + headless_mode]
    [enable_extension(x) for x in ext_list]
    simulation_app.update()

    return True


############################################################
def add_robot_to_scene(
    robot_config: Dict,  # 로봇 URDF/YAML 에서 파싱한 설정 딕셔너리
    my_world: World,    # isaac sim 에서 생성한 World 객체
    load_from_usd: bool = False,    # USD 파일에서 직접 로드
    subroot: str = "", # USD 파일 삽입할 경로
    robot_name: str = "robot", # 로봇 이름
    position: np.array = np.array([0, 0, 0]), # 초기 배치 위치
    initialize_world: bool = True,  # 물리 시물레이션 초기화 여부
):

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()    # URDF -> USD 변환시 물리적/ 시물레이션 속성을 설정
    import_config.merge_fixed_joints = False # 고정 조인트 병합 x
    import_config.convex_decomp = False  
    import_config.fix_base = True  # 베이스 링크 고정
    import_config.make_default_prim = True  
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 1047.19751 # 위치, 속도 제어에 따라 변경, 1047.19751
    import_config.default_position_drive_damping = 52.35988 # 
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION       #  드라이브 모드 JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            os.path.join(os.environ["FRANKA_DEMO_ROOT"], "assets"), get_filename(file_name, remove_extension=True) + "_temp.usd"
        )
        # URDF 를 USD로 변환
        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:

        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )
    # Robot 객체 생성, my_world 에 추가하여 시물레이션 
    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path) 
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    # 물리 시물레이션 초기화
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    robot_view = RobotView(prim_paths_expr=robot_path + "/" + base_link_name,
        name=robot_name)
    robot_view.initialize()

    return robot, robot_path, robot_view

def add_robot_to_scene_vel(
    robot_config: Dict,  # 로봇 URDF/YAML 에서 파싱한 설정 딕셔너리
    my_world: World,    # isaac sim 에서 생성한 World 객체
    load_from_usd: bool = False,    # USD 파일에서 직접 로드
    subroot: str = "", # USD 파일 삽입할 경로
    robot_name: str = "robot", # 로봇 이름
    position: np.array = np.array([0, 0, 0]), # 초기 배치 위치
    initialize_world: bool = True,  # 물리 시물레이션 초기화 여부
):

    urdf_interface = _urdf.acquire_urdf_interface()
    # Set the settings in the import config
    import_config = _urdf.ImportConfig()    # URDF -> USD 변환시 물리적/ 시물레이션 속성을 설정
    import_config.merge_fixed_joints = False # 고정 조인트 병합 x
    import_config.convex_decomp = False  
    import_config.fix_base = True  # 베이스 링크 고정
    import_config.make_default_prim = True  
    import_config.self_collision = False
    import_config.create_physics_scene = True
    import_config.import_inertia_tensor = False
    import_config.default_drive_strength = 0.0 # 위치, 속도 제어에 따라 변경, 1047.19751
    import_config.default_position_drive_damping = 52.35988 # 
    import_config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_VELOCITY       #  드라이브 모드 JOINT_DRIVE_POSITION
    import_config.distance_scale = 1
    import_config.density = 0.0

    asset_path = get_assets_path()
    if (
        "external_asset_path" in robot_config["kinematics"]
        and robot_config["kinematics"]["external_asset_path"] is not None
    ):
        asset_path = robot_config["kinematics"]["external_asset_path"]

    # urdf_path:
    # meshes_path:
    # meshes path should be a subset of urdf_path
    full_path = join_path(asset_path, robot_config["kinematics"]["urdf_path"])
    # full path contains the path to urdf
    # Get meshes path
    robot_path = get_path_of_dir(full_path)
    filename = get_filename(full_path)
    if ISAAC_SIM_45:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        import omni.kit.commands
        import omni.usd

        # Retrieve the path of the URDF file from the extension
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        root_path = robot_path
        file_name = filename

        # Parse the robot's URDF file to generate a robot model

        dest_path = join_path(
            os.path.join(os.environ["FRANKA_DEMO_ROOT"], "assets"), get_filename(file_name, remove_extension=True) + "_temp.usd"
        )
        # URDF 를 USD로 변환
        result, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile",
            urdf_path="{}/{}".format(root_path, file_name),
            import_config=import_config,
            dest_path=dest_path,
        )
        prim_path = omni.usd.get_stage_next_free_path(
            my_world.scene.stage,
            str(my_world.scene.stage.GetDefaultPrim().GetPath()) + robot_path,
            False,
        )
        robot_prim = my_world.scene.stage.OverridePrim(prim_path)
        robot_prim.GetReferences().AddReference(dest_path)
        robot_path = prim_path
    else:

        imported_robot = urdf_interface.parse_urdf(robot_path, filename, import_config)
        dest_path = subroot

        robot_path = urdf_interface.import_robot(
            robot_path,
            filename,
            imported_robot,
            import_config,
            dest_path,
        )

    base_link_name = robot_config["kinematics"]["base_link"]

    robot_p = Robot(
        prim_path=robot_path + "/" + base_link_name,
        name=robot_name,
    )
    # Robot 객체 생성, my_world 에 추가하여 시물레이션 
    robot_prim = robot_p.prim
    stage = robot_prim.GetStage()
    linkp = stage.GetPrimAtPath(robot_path) 
    set_prim_transform(linkp, [position[0], position[1], position[2], 1, 0, 0, 0])

    robot = my_world.scene.add(robot_p)
    # 물리 시물레이션 초기화
    if initialize_world:
        if ISAAC_SIM_45:
            my_world.initialize_physics()
            robot.initialize()

    robot_view = RobotView(prim_paths_expr=robot_path + "/" + base_link_name,
        name=robot_name)
    robot_view.initialize()

    return robot, robot_path, robot_view

import torch
def cal_Mq(mass, inertia, Jp, Jr):
    """
    mass: (14,)             # shape = [14]
    inertia: (1, 14, 9)     # shape = [1, 14, 9]
    Jp: (1, 14, 3, 9)
    Jr: (1, 14, 3, 9)
    """
    #print(f" mass : {mass.shape}")
    #print(f" inertia : {inertia.shape}")
    #print(f" J_p : {Jp.shape}")
    
    B, N, D = inertia.shape
    inertia = inertia.view(B, N, 3, 3)

    # mass: (1, 14, 1, 1)
    mass = mass.view(1, N, 1, 1)

    # Linear term: Jp^T * m * Jp
    Jp_T = Jp.transpose(2, 3)   # (1, 14, 9, 3)
    Mq_linear = torch.einsum('bnij,bnjk->bnik', Jp_T, Jp) * mass  # (1, 14, 9, 9)

    # Angular term: Jr^T * I * Jr
    Jr_T = Jr.transpose(2, 3)   # (1, 14, 9, 3)
    Mq_angular = torch.einsum('bnij,bnjk,bnkl->bnil', Jr_T, inertia, Jr)  # (1, 14, 9, 9)

    # Sum over all links (dim=1)
    Mq = Mq_linear.sum(dim=1) + Mq_angular.sum(dim=1)  # (1, 9, 9)
    return Mq

def compute_all_joint_meff_cholesky_fast(Mq, J_p):
    """
    효율적이고 안정적인 유효 질량 계산 (Cholesky 기반, 빠른 버전)
    Mq가 양의 정부호(PD)인 경우에만 사용.

    Parameters:
        - Mq: (1, 9, 9) - 일반화된 질량 행렬
        - J_p: (1, 14, 3, 9) - 선형 자코비안
        - u: (B, 24, 3) - 단위 방향 벡터 (normalized) 생성

    Returns:
        - Meff: (B, 24)
    """
    B, J, _, D = J_p.shape
    epsilon = 1e-6
    u = torch.zeros((1, 14, 3),device=Mq.device)  # 먼저 0으로 초기화
    u[:, :, 0] = 1.0  # x축 방향만 1.0  # x축 방향만 1로 설정 # frames , 24, 3 
    u = u / torch.norm(u, dim=-1, keepdim=True)  # ← 반드시 정규화
    # Cholesky 분해 (B, 75, 75)
    Mq = Mq + epsilon * torch.eye(D, device=Mq.device).unsqueeze(0)
    L = torch.linalg.cholesky(Mq)  # (B, 75, 75)

    # J^T u 계산
    Jp_T = J_p.transpose(-2, -1)         # (B, 24, 75, 3)
    u_vec = u.unsqueeze(-1)              # (B, 24, 3, 1)
    rhs = torch.matmul(Jp_T, u_vec)      # (B, 24, 75, 1)
    rhs = rhs.view(-1, D, 1)             # (B*24, 75, 1)

    # L 반복 확장
    L_ = L.unsqueeze(1).expand(B, J, D, D).reshape(-1, D, D)  # (B*24, 75, 75)

    # Cholesky solve
    x = torch.cholesky_solve(rhs, L_)    # (B*24, 75, 1)
    x = x.view(B, J, D, 1)

    # Lambda_u = J x 계산
    Lambda_u = torch.matmul(J_p, x)       # (B, 24, 3, 1)
    denom = torch.matmul(u_vec.transpose(2, 3), Lambda_u).squeeze(-1).squeeze(-1)  # (B, 24)

    # 유효 질량 계산
    Meff = 1.0 / (denom + epsilon)
    return Meff

def plot_vel(link_names, link_velocities, save_dir="plots_vel"):
    os.makedirs(save_dir, exist_ok=True)
    link_velocities = np.array(link_velocities)  # shape: (frames, 14, 6)
    #print(f" link_velocity shape : {link_velocities.shape}")
    frames = np.arange(link_velocities.shape[0])
    #print(f" frames : {frames}")
    for i, link in enumerate(link_names):
        # 선형 속도만 추출 (vx, vy, vz)
        linear_vel = link_velocities[:, i, :3]  # shape: (frames, 3)
        speed_norm = np.linalg.norm(linear_vel, axis=1)  # shape: (frames,)
        #print(f" spedd_norm {speed_norm}")
        # 그래프 그리기
        plt.figure(figsize=(10, 4))
        plt.plot(frames, speed_norm, label=f"Speed of {link}", color='blue')
        plt.title(f"Linear Velocity Magnitude of {link}")
        plt.xlabel("Frames")
        plt.ylabel("Speed (m/s)")
        plt.legend()

        # 저장
        save_path = os.path.join(save_dir, f"V_{link}.png")
        plt.savefig(save_path)
        plt.close()
        #print(f"Saved: {save_path}")
             
def plot_joint_vel(joint_vel, save_dir ="plots_joint_vel"):
    os.makedirs(save_dir, exist_ok=True)

    joint_vel = np.array(joint_vel)  # shape: (frames, 9)
    #print(f" link_velocity shape : {joint_vel.shape}")
    frames = np.arange(joint_vel.shape[0])
    #print(f" frames : {frames}")
    for i in range(joint_vel.shape[2]):
        # 선형 속도만 추출 (vx, vy, vz)
        joint_vel_i = joint_vel[:, 0, i]  # shape: (frames, 1, 9)
        # speed_norm = np.linalg.norm(linear_vel, axis=1)  # shape: (frames,)
        #print(f" spedd_norm {speed_norm}")
        # 그래프 그리기
        plt.figure(figsize=(10, 4))
        plt.plot(frames, joint_vel_i, label=f"Speed of joint {i}", color='blue')
        plt.title(f"speed of joint {i}")
        plt.xlabel("Frames")
        plt.ylabel("Speed (rad/s)")
        plt.legend()

        # 저장
        save_path = os.path.join(save_dir, f"V_{i}.png")
        plt.savefig(save_path)
        plt.close()
        #print(f"Saved: {save_path}")



class VoxelManager:
    def __init__(
        self,
        num_voxels: int = 5000,
        size: float = 0.02,
        color: List[float] = [1, 1, 1],
        prefix_path: str = "/World/curobo/voxel_",
        material_path: str = "/World/looks/v_",
    ) -> None:
        self.cuboid_list = []
        self.cuboid_material_list = []
        self.disable_idx = num_voxels
        for i in range(num_voxels):
            target_material = OmniPBR("/World/looks/v_" + str(i), color=np.ravel(color))

            cube = cuboid.VisualCuboid(
                prefix_path + str(i),
                position=np.array([0, 0, -10]),
                orientation=np.array([1, 0, 0, 0]),
                size=size,
                visual_material=target_material,
            )
            self.cuboid_list.append(cube)
            self.cuboid_material_list.append(target_material)
            cube.set_visibility(True)

    def update_voxels(self, voxel_position: np.ndarray, color_axis: int = 0):
        max_index = min(voxel_position.shape[0], len(self.cuboid_list))

        jet = cm.get_cmap("hot")  # .reversed()
        z_val = voxel_position[:, 0]

        jet_colors = jet(z_val)

        for i in range(max_index):
            self.cuboid_list[i].set_visibility(True)

            self.cuboid_list[i].set_local_pose(translation=voxel_position[i])
            self.cuboid_material_list[i].set_color(jet_colors[i][:3])

        for i in range(max_index, len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))

            # self.cuboid_list[i].set_visibility(False)

    def clear(self):
        for i in range(len(self.cuboid_list)):
            self.cuboid_list[i].set_local_pose(translation=np.ravel([0, 0, -10.0]))
