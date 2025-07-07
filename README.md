# Shape_from_Polarization_Tutorial
**测试提交要求**

- **test1/**

  - `Shape_from_Polarization.py`

    > 主程序文件，实现偏振图像处理主流程。

  - `utils.py`

    > 工具函数文件，包含辅助计算与可视化函数。

  - `data/`

    > 存放输入数据，具体结构如下：

    - `mask/`

      > 掩码图像文件夹

    - `normal/`

      > 法线图像文件夹

    - `pol000/`

    - `pol045/`

    -  ……

  - `output/`

    > 程序输出结果目录，结构如下：

    - `AoLP_visualize/`

      > AoLP 可视化结果

    - `DoLP_visualize/`

      > DoLP 可视化结果

    - `SfP_normal/`

      > 重建得到的法线图像

    - `SfP_normal_error_visualize/`

      > 法线误差可视化结果

- **test2/**

  - `result.pptx`

    > 演示文稿，要求如下：
    >
    > - **第一页**：回答小测2的问题1
    > - **第二页**：补全小测2的问题2

------

## 注意事项

- **提交格式**：请将所有内容打包成 `姓名+学号+学院+年级.zip`，保持上述目录结构。
- **代码要求**：`Shape_from_Polarization.py` 为主程序入口，`utils.py` 实现所需的辅助函数。
- **PPT要求**：`result.pptx` 第一页回答小测2的第1题，第二页补全第2题。
