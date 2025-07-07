import numpy as np
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import cv2

### Normal MAP与Azimuth和Zenith角的转换
def get_normal(zenith, azimuth):
    nx = np.sin(zenith) * np.cos(azimuth)
    ny = np.sin(zenith) * np.sin(azimuth)
    nz = np.cos(zenith)
    return np.stack([nx, ny, nz], axis=-1)


### 可视化法线图
def visualize_normal(normal, save_path, mask=None):
    # normal: (H, W, 3) in [-1, 1]
    norm_img = ((normal + 1) / 2).clip(0, 1)  # 映射到[0,1]
    if mask is not None:
        mask = mask.astype(bool)
        norm_img[~mask] = 0
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(norm_img)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


### 根据观测的rho值和查找表中的rho值，找到对应的两个Zenith角
def find_two_zenith_for_each_rho(rho_obs, rho_tab, theta_tab, tol=1e-3):
    rho_obs = np.asarray(rho_obs)
    shape = rho_obs.shape
    z1 = np.full(shape, np.nan)
    z2 = np.full(shape, np.nan)
    # 拉平处理，便于遍历
    flat_rho = rho_obs.flatten()
    flat_z1 = z1.flatten()
    flat_z2 = z2.flatten()
    for i, val in enumerate(flat_rho):
        idx = np.where(np.abs(rho_tab - val) < tol)[0]
        if len(idx) == 0:
            flat_z1[i] = np.nan
            flat_z2[i] = np.nan
        elif len(idx) == 1:
            flat_z1[i] = theta_tab[idx[0]]
            flat_z2[i] = theta_tab[idx[0]]
        else:
            flat_z1[i] = theta_tab[idx[0]]
            flat_z2[i] = theta_tab[idx[-1]]
    # 恢复原始shape
    z1 = flat_z1.reshape(shape)
    z2 = flat_z2.reshape(shape)
    return z1, z2


### 根据入射角和折射率计算DoLP
def dolp_specular(theta, eta):
    s = np.sin(theta)
    c = np.cos(theta)
    numerator = 2 * s**2 * c * np.sqrt(eta**2 - s**2)
    denominator = eta**2 - (1 + eta**2) * s**2 + 2 * s**4
    return numerator / denominator


### 根据入射角和折射率计算AoLP
def poly_fit(params, x):
    v = np.zeros_like(x)+params[-1]
    c_size = params.shape[0]
    for i in range(0, c_size-1):
        v += params[i]*x**(c_size-1-i)
    return v

### 读取图像如果是三通道计算平均值，如果是单通道则直接返回
def load_average(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    return img.mean(-1).astype(np.float64) if img.ndim == 3 else img.astype(np.float64)



### 读取npy
def read_npy(file_path):
    array = np.load(file_path)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    elif array.ndim == 2:
        return np.stack([array] * 3, axis=-1)
    else:
        raise ValueError(f"{file_path} 图像维度不支持: {array.shape}")


### 读取EXR文件中的原始RGB通道
def read_exr_raw_channels(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    def read_channel(c):
        return np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape((height, width))

    r = read_channel('R')
    g = read_channel('G')
    b = read_channel('B')
    return np.stack([r, g, b], axis=-1)


### 保存EXR文件
def save_exr(image, save_path):
    header = OpenEXR.Header(image.shape[1], image.shape[0])
    float_chan = Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
    header['channels'] = dict(R=float_chan, G=float_chan, B=float_chan)
    exr = OpenEXR.OutputFile(save_path, header)
    exr.writePixels({
        'R': image[:, :, 0].astype(np.float32).tobytes(),
        'G': image[:, :, 1].astype(np.float32).tobytes(),
        'B': image[:, :, 2].astype(np.float32).tobytes()
    })
    exr.close()



### 可视化和保存图像
def visualize_and_save(image, save_path, cmap='hsv', vmin=0, vmax=360, mask=None):
    print('visualize_and_save: image shape =', image.shape)  # <-- debug
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    if mask is not None:
        image = np.where(mask > 0, image, np.nan)
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()



### 可视化误差图像并添加标题
def visualize_error_with_title(image, save_path, cmap='jet',
                               vmin=0, vmax=180, title='', mask=None):
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    if mask is not None:
        image = np.where(mask > 0, image, np.nan)
    plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=14)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()