import numpy as np



def calculate_distance(point1, point2):
    # 计算两个点之间的欧氏距离
    return np.linalg.norm(np.array(point1) - np.array(point2))


def map_curve_to_bbox(curve_color_positions, bboxes):
    # 计算每个 bbox 的中心位置
    bbox_centers = np.mean(bboxes, axis=1)  # bboxes: [num_bboxes, 4, 2]，每个 bbox 是四个角点
    curve_bbox_mapping = []

    for i, bbox in enumerate(bboxes):
        min_distance_dict = {}  # 存储当前 bbox 到每条曲线的最近距离

        # 计算 bbox 的所有关键点（四个角点和中心点）
        bbox_points = np.vstack((bbox, bbox_centers[i]))  # [5, 2]，包括四个角和中心点

        # 遍历每条曲线
        for curve_id, points in curve_color_positions.items():
            points = np.array(points)  # 矢量化处理

            # 计算 bbox 的每个点到曲线的最小距离
            distances = np.linalg.norm(points[:, None, :] - bbox_points, axis=2)  # [num_points, 5]
            min_distance = np.min(distances)  # 当前 bbox 的关键点到曲线的最小距离

            # 更新最近距离
            min_distance_dict[curve_id] = min_distance

        # 获取距离最小的曲线 ID 和距离
        nearest_curve_id = min(min_distance_dict, key=min_distance_dict.get)
        curve_bbox_mapping.append((i, nearest_curve_id, min_distance_dict[nearest_curve_id]))

    return curve_bbox_mapping

