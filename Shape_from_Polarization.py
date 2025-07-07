import os
import argparse
from utils import *

eta = 1.5 # 折射率默认值
theta_tab = np.linspace(0, np.pi/2, 5000) # 角度表，0到90度
rho_tab = dolp_specular(theta_tab, eta) # 计算specular DoLP与zenith角的关系表

### 利用偏振信息获取法线向量
def SfP(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    I_dirs = [os.path.join(input_dir, f'pol{i:03d}') for i in [0, 45, 90, 135]]
    normal_dir = os.path.join(input_dir, 'normal')
    mask_dir = os.path.join(input_dir, 'mask')

    outdirs = [
        'AoLP_visualize', 'DoLP_visualize',
        'SfP_normal', 'SfP_normal_error_visualize',
    ]
    ### 注意AoLP的范围是[0, 180]，而DoLP的范围是[0, 1]，因此在可视化时需要注意色图的选择和范围设置。

    for d in outdirs:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    for filename in os.listdir(I_dirs[0]):
        ### 只处理特定格式的图像
        if not filename.lower().endswith(('.png', '.tif', '.exr')):
            continue
        name = os.path.splitext(filename)[0]

        # 读取偏振图像
        I000, I045, I090, I135 = [load_average(os.path.join(d, filename)) for d in I_dirs]

        # mask
        mask = np.ones_like(I000, dtype=np.uint8)
        mask_path = os.path.join(mask_dir, f'{name}.png')
        if os.path.isfile(mask_path):
            raw_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if raw_mask is not None:
                if raw_mask.ndim == 3: raw_mask = raw_mask.mean(axis=2)
                if raw_mask.dtype == np.uint16: raw_mask = (raw_mask / 256).astype(np.uint8)
                mask = (raw_mask > 0).astype(np.uint8)
        mask3 = mask[..., None]



        # Stokes参数（请补充代码）
        s0 =
        s1 =
        s2 =
        aolp =
        dolp =



        # GT法线
        normal_path = os.path.join(normal_dir, f'{name}.png')
        normal = cv2.imread(normal_path, cv2.IMREAD_UNCHANGED)
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB) if normal.ndim == 3 else normal
        if normal is None: raise FileNotFoundError(f"Cannot find normal image: {normal_path}")
        normal = normal.astype(np.float32) / (65535.0 if normal.max() > 255 else 255.0)
        normal_gt = normal * 2 - 1  # [-1,1]

        # Diffuse/specular azimuth/zenith（请补充代码）


        diffuse_azimuths =
        diffuse_zenith =
        specular_azimuths =


        dolp_clip = dolp.clip(np.nanmin(rho_tab), np.nanmax(rho_tab))
        spec_zen = find_two_zenith_for_each_rho(dolp_clip, rho_tab, theta_tab, tol=1e-3)

        diff_normal_1 = get_normal(diffuse_zenith, diffuse_azimuths[0]) * mask3
        diff_normal_2 = get_normal(diffuse_zenith, diffuse_azimuths[1]) * mask3
        spec_normal_1 = get_normal(spec_zen[0], specular_azimuths[0]) * mask3
        spec_normal_2 = get_normal(spec_zen[0], specular_azimuths[1]) * mask3
        spec_normal_3 = get_normal(spec_zen[1], specular_azimuths[0]) * mask3
        spec_normal_4 = get_normal(spec_zen[1], specular_azimuths[1]) * mask3

        candidates = np.stack([
            diff_normal_1,
            diff_normal_2,
            spec_normal_1,
            spec_normal_2,
            spec_normal_3,
            spec_normal_4
        ], axis=0)

        # 最优normal
        dot = np.clip((candidates * normal_gt[None, ...]).sum(-1), -1, 1)


        ### 计算每个候选normal与GT法线的点积,选取最合适的（请补充代码）






        best_normal =

        # 误差
        best_dot = np.clip((best_normal * normal_gt).sum(-1), -1, 1)
        error_map = np.rad2deg(np.arccos(best_dot))
        mean_error = error_map[mask > 0].mean() if (mask > 0).any() else 0.0

        # 保存
        best_normal = best_normal[:, :, ::-1]
        cv2.imwrite(os.path.join(output_dir, 'SfP_normal', f'{name}.png'), ((best_normal + 1) / 2 * 255).clip(0,255).astype(np.uint8))

        aolp_deg = np.rad2deg(aolp) % 180

        # 可视化
        visualize_and_save(aolp_deg, os.path.join(output_dir, 'AoLP_visualize', f'{name}.png'), cmap='hsv', vmin=0, vmax=180, mask=mask)
        visualize_and_save(dolp, os.path.join(output_dir, 'DoLP_visualize', f'{name}.png'), cmap='GnBu', vmin=0, vmax=1, mask=mask)
        visualize_error_with_title(
            error_map,
            os.path.join(output_dir, 'SfP_normal_error_visualize', f'{name}.png'),
            cmap='jet', vmin=0, vmax=30, mask=mask,
            title=f"Mean error={mean_error:.2f}°"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Polarization image processing")
    parser.add_argument('--input_dir', type=str, default=r'./data', help='输入路径，包含 pol000-135/normal/mask')
    parser.add_argument('--output_dir', type=str, default='./output', help='输出路径')
    args = parser.parse_args()
    SfP(args.input_dir, args.output_dir)
