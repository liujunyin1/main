import math
import numpy as np
from collections import defaultdict
import traceback


def _calculate_coeffs(point, origin, basis_v1, basis_v2):
    """计算一个点相对于给定的原点和基矢量的投影系数 (c1, c2)。"""
    p_vec = point - origin
    A = np.array([basis_v1, basis_v2]).T
    if np.linalg.det(A) == 0: return None, None
    try:
        coeffs = np.linalg.solve(A, p_vec)
        return coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        return None, None


def _construct_rectangular_neighbors(distances_pool, center_point_np, all_other_atoms):
    """根据给定的邻居点池，执行几何构造法来寻找四方晶格的邻居。"""
    if not distances_pool: return []
    cx, cy = center_point_np[0], center_point_np[1]
    core_neighbors, p1 = [], distances_pool[0][1]
    v1 = np.array([p1["midpoint"]["x"], p1["midpoint"]["y"]]) - center_point_np
    core_neighbors.append(p1)
    p2, v2 = None, None
    min_angle, max_angle = math.pi / 4, 3 * math.pi / 4
    if len(distances_pool) > 1:
        for dist, p_candidate in distances_pool[1:]:
            if p_candidate["id"] == p1["id"]: continue
            v_candidate = np.array([p_candidate["midpoint"]["x"], p_candidate["midpoint"]["y"]]) - center_point_np
            dot_prod, norm_prod = np.dot(v1, v_candidate), np.linalg.norm(v1) * np.linalg.norm(v_candidate)
            if norm_prod < 1e-9: continue
            angle = math.acos(np.clip(dot_prod / norm_prod, -1.0, 1.0))
            if min_angle <= angle <= max_angle:
                p2, v2 = p_candidate, v_candidate
                break
    if not p2: return core_neighbors
    core_neighbors.append(p2)
    if v1 is not None and v2 is not None:
        p3_ideal, p4_ideal = center_point_np - v1, center_point_np - v2
        search_space = [p for p in all_other_atoms if p["id"] not in [n["id"] for n in core_neighbors]]
        if search_space: core_neighbors.append(
            min(search_space, key=lambda p: math.dist((p["midpoint"]["x"], p["midpoint"]["y"]), p3_ideal)))
        search_space = [p for p in all_other_atoms if p["id"] not in [n["id"] for n in core_neighbors]]
        if search_space: core_neighbors.append(
            min(search_space, key=lambda p: math.dist((p["midpoint"]["x"], p["midpoint"]["y"]), p4_ideal)))
    quadrants = {1: [], 2: [], 3: [], 4: []}
    core_neighbor_ids = {n["id"] for n in core_neighbors}
    remaining_atoms = [p for p in all_other_atoms if p["id"] not in core_neighbor_ids]
    if v1 is not None and v2 is not None:
        for atom in remaining_atoms:
            point_np = np.array([atom["midpoint"]["x"], atom["midpoint"]["y"]])
            c1, c2 = _calculate_coeffs(point_np, center_point_np, v1, v2)
            if c1 is None: continue
            if c1 > 0 and c2 > 0:
                quadrants[1].append(atom)
            elif c1 < 0 and c2 > 0:
                quadrants[2].append(atom)
            elif c1 < 0 and c2 < 0:
                quadrants[3].append(atom)
            elif c1 > 0 and c2 < 0:
                quadrants[4].append(atom)
    quadrant_representatives = [
        min(quadrants[q_idx], key=lambda p: math.dist((p["midpoint"]["x"], p["midpoint"]["y"]), (cx, cy))) for q_idx in
        range(1, 5) if quadrants[q_idx]]
    return core_neighbors + quadrant_representatives


def analyze_central_structure(inlier_data, image_height, image_width, forced_center_id=None):
    """分析中心原子的局部晶格结构，并使用余弦定理计算中心角。"""

    if not inlier_data:
        return {}

    K_NEAREST = 15
    img_center_x, img_center_y = image_width / 2, image_height / 2

    central_atom = None
    analysis_candidates = []

    # --- 1. 确定中心原子和候选邻居 ---
    if forced_center_id:
        # 尝试找到强制指定的中心原子
        for p in inlier_data:
            if p["id"] == forced_center_id:
                central_atom = p
                break

    if central_atom:
        # 如果指定了中心，按“到该原子的距离”排序寻找邻居
        cx, cy = central_atom["midpoint"]["x"], central_atom["midpoint"]["y"]
        sorted_by_dist_sq = sorted(
            inlier_data,
            key=lambda p: (p["midpoint"]["x"] - cx) ** 2 + (p["midpoint"]["y"] - cy) ** 2
        )
        # 排除自身后的 K 个最近邻 (取 K+1 包括自身)
        analysis_candidates = sorted_by_dist_sq[:K_NEAREST + 1]
    else:
        # 默认逻辑：按“到图片中心距离”排序
        sorted_by_dist_sq = sorted(
            inlier_data,
            key=lambda p: (p["midpoint"]["x"] - img_center_x) ** 2 + (p["midpoint"]["y"] - img_center_y) ** 2
        )
        num_to_select = min(K_NEAREST, len(inlier_data))
        analysis_candidates = sorted_by_dist_sq[:num_to_select]

        # 再次确认中心点（在候选集里离图片中心最近的）
        if analysis_candidates:
            central_atom = min(analysis_candidates,
                               key=lambda p: math.dist((p["midpoint"]["x"], p["midpoint"]["y"]),
                                                       (img_center_x, img_center_y)))

    # --- 2. 检查数据量 ---
    if len(analysis_candidates) < 9 or not central_atom:
        return {}

    # --- 3. 准备数据进行几何分析 ---
    cx, cy, center_point_np = central_atom["midpoint"]["x"], central_atom["midpoint"]["y"], np.array(
        [central_atom["midpoint"]["x"], central_atom["midpoint"]["y"]])

    other_atoms = [p for p in analysis_candidates if p["id"] != central_atom["id"]]
    if not other_atoms: return {}

    distances = sorted([(math.dist((p["midpoint"]["x"], p["midpoint"]["y"]), (cx, cy)), p) for p in other_atoms],
                       key=lambda x: x[0])

    lattice_info, final_neighbors = {}, []

    radius_multiplier = 1.8
    while radius_multiplier >= 1.5:
        if not distances: break
        search_radius = distances[0][0] * radius_multiplier
        local_distances = [d for d in distances if d[0] <= search_radius]
        if len(local_distances) < 4:
            radius_multiplier -= 0.1
            continue

        def _temp_calc_angles(sorted_points):
            angles = []
            num_points = len(sorted_points)
            if num_points < 3: return []
            points_np = np.array([[p["midpoint"]["x"], p["midpoint"]["y"]] for p in sorted_points])
            for i in range(num_points):
                p_prev, p_curr, p_next = points_np[(i - 1 + num_points) % num_points], points_np[i], points_np[
                    (i + 1) % num_points]
                v1, v2 = p_prev - p_curr, p_next - p_curr
                dot_prod, norm_prod = np.dot(v1, v2), np.linalg.norm(v1) * np.linalg.norm(v2)
                angle_rad = math.pi if norm_prod < 1e-9 else math.acos(np.clip(dot_prod / norm_prod, -1.0, 1.0))
                angles.append(math.degrees(angle_rad))
            return angles

        neighbors_in_radius = [d[1] for d in local_distances]
        if len(neighbors_in_radius) == 6:
            potential_neighbors = sorted(neighbors_in_radius, key=lambda atom: (math.atan2(atom["midpoint"]["y"] - cy,
                                                                                           atom["midpoint"][
                                                                                               "x"] - cx) + math.pi / 2) % (
                                                                                       2 * math.pi))
            if not any(angle < 90 for angle in _temp_calc_angles(potential_neighbors)):
                final_neighbors, lattice_info["lattice_type"] = potential_neighbors, "hexagonal"
                break

        current_neighbors = _construct_rectangular_neighbors(local_distances, center_point_np, other_atoms)
        rect_angles = _temp_calc_angles(current_neighbors)
        if rect_angles and not any(angle < 80 for angle in rect_angles):
            final_neighbors, lattice_info["lattice_type"] = current_neighbors, "rectangular"
            break

        radius_multiplier -= 0.1

    if not final_neighbors and distances:
        lattice_info["lattice_type"] = "rectangular (fallback)"
        geometric_neighbors = _construct_rectangular_neighbors(distances, center_point_np, other_atoms)
        final_neighbors, final_neighbor_ids = geometric_neighbors, {n["id"] for n in geometric_neighbors}
        for dist, atom in distances:
            if len(final_neighbors) >= 8: break
            if atom["id"] not in final_neighbor_ids:
                final_neighbors.append(atom)
                final_neighbor_ids.add(atom["id"])

    if final_neighbors:
        final_neighbors.sort(
            key=lambda atom: (math.atan2(atom["midpoint"]["y"] - cy, atom["midpoint"]["x"] - cx) + math.pi / 2) % (
                    2 * math.pi))
        num_neighbors = len(final_neighbors)

        r_values = {f"r1{i + 2}": math.dist(central_atom['midpoint'].values(), n['midpoint'].values()) for i, n in
                    enumerate(final_neighbors)}

        d_values = {}
        if num_neighbors > 1:
            for i in range(num_neighbors):
                p1, p2 = final_neighbors[i], final_neighbors[(i + 1) % num_neighbors]
                idx1, idx2 = i + 2, ((i + 1) % num_neighbors) + 2
                d_values[f"d{idx1}{idx2}"] = math.dist(p1['midpoint'].values(), p2['midpoint'].values())

        a_values = {}
        if num_neighbors > 1:
            for i in range(num_neighbors):
                idx_curr, idx_next = i + 2, ((i + 1) % num_neighbors) + 2
                r_curr, r_next = r_values[f"r1{idx_curr}"], r_values[f"r1{idx_next}"]
                d_side = d_values[f"d{idx_curr}{idx_next}"]
                numerator = r_curr ** 2 + r_next ** 2 - d_side ** 2
                denominator = 2 * r_curr * r_next
                angle_rad = 0.0
                if denominator > 1e-9:
                    cos_angle = np.clip(numerator / denominator, -1.0, 1.0)
                    angle_rad = math.acos(cos_angle)
                a_values[f"a{idx_curr}1{idx_next}"] = math.degrees(angle_rad)

        lattice_info.update({
            "ordered_points": [central_atom] + final_neighbors,
            "r_distances_center_to_neighbor": r_values,
            "d_distances_neighbor_to_neighbor": d_values,
            "a_center_angles_degrees": a_values
        })

    return lattice_info