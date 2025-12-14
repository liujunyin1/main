import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import least_squares
import traceback


def calculate_projection_coeffs(point, origin, basis_a1, basis_a2):
    """
    计算点 P 相对于原点 O 和基矢量 a1, a2 的投影系数 (c1, c2)。
    """
    P_minus_O = point - origin
    A = np.array([basis_a1, basis_a2]).T
    B = P_minus_O

    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    if abs(det) < 1e-6:
        return None, None

    try:
        coeffs = np.linalg.solve(A, B)
        return coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        return None, None


def generate_lattice_points(origin, basis_a1, basis_a2, min_x, max_x, min_y, max_y):
    """
    根据晶格参数生成一个覆盖指定范围的理想晶格点网格，用于可视化。
    """
    lattice_nodes = []
    len_a1 = np.linalg.norm(basis_a1)
    len_a2 = np.linalg.norm(basis_a2)

    if len_a1 < 1e-6 or len_a2 < 1e-6:
        return []

    span_x = max_x - min_x
    span_y = max_y - min_y

    max_n_approx = max(int(span_x / len_a1) if len_a1 > 0 else 0,
                       int(span_y / len_a2) if len_a2 > 0 else 0,
                       5) + 5

    for n1 in range(-max_n_approx, max_n_approx + 1):
        for n2 in range(-max_n_approx, max_n_approx + 1):
            p_ideal = origin + n1 * basis_a1 + n2 * basis_a2
            if (min_x - len_a1 * 2 <= p_ideal[0] <= max_x + len_a1 * 2 and
                    min_y - len_a2 * 2 <= p_ideal[1] <= max_y + len_a2 * 2):
                lattice_nodes.append(p_ideal)
    return np.array(lattice_nodes)


def filter_anomalies(image_targets_data, params):
    """
    根据RANSAC算法拟合2D点云中的周期性晶格结构，并过滤异常值。
    """
    # (已移除所有 print 语句)

    if not image_targets_data:
        return [], [], {"message": "No points to process."}

    num_iterations = params.get("num_iterations", 1000)
    inlier_threshold = params.get("inlier_threshold", 10.0)
    min_inliers_ratio = params.get("min_inliers_ratio", 0.8)
    k_neighbors_for_basis = params.get("k_neighbors_for_basis", 15)
    basis_angle_tolerance_deg = params.get("basis_angle_tolerance_deg", 10)
    min_basis_len_override = params.get("min_basis_len_override", 5.0)
    max_basis_len_ratio = params.get("max_basis_len_ratio", 3.0)
    adaptive_min_basis_len_neighbors = params.get("adaptive_min_basis_len_neighbors", 5)
    adaptive_min_basis_len_percentile = params.get("adaptive_min_basis_len_percentile", 10)

    points_np = np.array([[d["midpoint"]["x"], d["midpoint"]["y"]] for d in image_targets_data])

    if len(points_np) < 3:
        # print("Warning: Not enough points for RANSAC. Returning all points as inliers.") # <-- 已注释
        return image_targets_data, [], {"message": "Not enough points for RANSAC."}

    min_x, min_y = np.min(points_np, axis=0)
    max_x, max_y = np.max(points_np, axis=0)

    kdtree = KDTree(points_np)

    calculated_min_basis_len = min_basis_len_override

    if len(points_np) > adaptive_min_basis_len_neighbors:
        all_neighbor_distances = []
        for p_idx in range(len(points_np)):
            distances, _ = kdtree.query(points_np[p_idx], k=adaptive_min_basis_len_neighbors + 1)
            filtered_distances = [d for d in distances if d > 1e-6]
            all_neighbor_distances.extend(filtered_distances)

        if len(all_neighbor_distances) > 0:
            percentile_dist = np.percentile(all_neighbor_distances, adaptive_min_basis_len_percentile)
            calculated_min_basis_len = max(min_basis_len_override,
                                           percentile_dist,
                                           inlier_threshold * 1.5)

    best_inliers_count = -1
    best_model = None
    best_inlier_indices = []

    for i in range(num_iterations):
        idx_p0 = np.random.randint(0, len(points_np))
        P0 = points_np[idx_p0]
        distances, indices = kdtree.query(P0, k=k_neighbors_for_basis + 1)

        neighbor_indices = [idx for idx in indices if idx != idx_p0]
        if not neighbor_indices:
            continue

        neighbors = points_np[neighbor_indices]
        relative_vectors = neighbors - P0

        valid_vectors = [v for v in relative_vectors if np.linalg.norm(v) > calculated_min_basis_len]
        if len(valid_vectors) < 2:
            continue

        sorted_vectors = sorted(valid_vectors, key=lambda v: np.linalg.norm(v))

        candidate_basis_a1 = None
        candidate_basis_a2 = None

        for j in range(len(sorted_vectors)):
            v1 = sorted_vectors[j]
            for k in range(j + 1, len(sorted_vectors)):
                v2 = sorted_vectors[k]

                len_v1 = np.linalg.norm(v1)
                len_v2 = np.linalg.norm(v2)

                if len_v1 < 1e-6 or len_v2 < 1e-6:
                    continue

                dot_product = np.dot(v1, v2)
                clipped_dot_product = np.clip(dot_product / (len_v1 * len_v2), -1.0, 1.0)
                angle = np.degrees(np.arccos(clipped_dot_product))

                if basis_angle_tolerance_deg < angle < (180 - basis_angle_tolerance_deg):
                    if max(len_v1, len_v2) / min(len_v1, len_v2) <= max_basis_len_ratio:
                        candidate_basis_a1 = v1
                        candidate_basis_a2 = v2
                        break
            if candidate_basis_a1 is not None:
                break

        if candidate_basis_a1 is None or candidate_basis_a2 is None:
            continue

        current_model = (P0, candidate_basis_a1, candidate_basis_a2)

        potential_inliers_map = {}

        for point_idx, Pj in enumerate(points_np):
            c1, c2 = calculate_projection_coeffs(Pj, P0, candidate_basis_a1, candidate_basis_a2)

            if c1 is None or c2 is None:
                continue

            n1, n2 = round(c1), round(c2)
            P_ideal = P0 + n1 * candidate_basis_a1 + n2 * candidate_basis_a2

            distance = np.linalg.norm(Pj - P_ideal)
            if distance <= inlier_threshold:
                ideal_coords_key = (int(n1), int(n2))
                if ideal_coords_key not in potential_inliers_map:
                    potential_inliers_map[ideal_coords_key] = []
                potential_inliers_map[ideal_coords_key].append((distance, point_idx))

        current_inlier_indices = []
        for ideal_coords_key, candidates in potential_inliers_map.items():
            best_candidate = min(candidates, key=lambda x: x[0])
            current_inlier_indices.append(best_candidate[1])

        current_inliers_count = len(current_inlier_indices)
        if current_inliers_count > best_inliers_count and \
                (len(points_np) == 0 or current_inliers_count / len(points_np) >= min_inliers_ratio):
            best_inliers_count = current_inliers_count
            best_model = current_model
            best_inlier_indices = current_inlier_indices

    if best_model is None:
        # print("Warning: RANSAC failed to find a robust lattice model. Returning all points as inliers.") # <-- 已注释
        return image_targets_data, [], {"message": "RANSAC failed to find a robust model."}

    refined_origin, refined_basis_a1, refined_basis_a2 = best_model

    if len(best_inlier_indices) < 3:
        pass
        # print("Warning: Not enough inliers for refinement. Using best found model without refinement.") # <-- 已注释
    else:
        def objective_func(params_flat, inliers_for_refinement, current_inlier_threshold):
            origin_ref = np.array([params_flat[0], params_flat[1]])
            a1_ref = np.array([params_flat[2], params_flat[3]])
            a2_ref = np.array([params_flat[4], params_flat[5]])

            residuals = []
            for Pj_ref in inliers_for_refinement:
                c1_ref, c2_ref = calculate_projection_coeffs(Pj_ref, origin_ref, a1_ref, a2_ref)
                if c1_ref is None or c2_ref is None:
                    residuals.extend([current_inlier_threshold * 2, current_inlier_threshold * 2])
                    continue
                n1_ref, n2_ref = round(c1_ref), round(c2_ref)
                P_ideal_ref = origin_ref + n1_ref * a1_ref + n2_ref * a2_ref
                residuals.extend((Pj_ref - P_ideal_ref).tolist())

            return np.array(residuals)

        initial_params = np.concatenate((refined_origin, refined_basis_a1, refined_basis_a2))

        try:
            result = least_squares(objective_func, initial_params,
                                   args=(points_np[best_inlier_indices], inlier_threshold),
                                   method='lm', ftol=1e-3, xtol=1e-3, max_nfev=500)

            if result.success:
                optimized_params = result.x
                refined_origin_candidate = np.array([optimized_params[0], optimized_params[1]])
                refined_basis_a1_candidate = np.array([optimized_params[2], optimized_params[3]])
                refined_basis_a2_candidate = np.array([optimized_params[4], optimized_params[5]])

                det_refined = refined_basis_a1_candidate[0] * refined_basis_a2_candidate[1] - \
                              refined_basis_a1_candidate[1] * refined_basis_a2_candidate[0]
                if abs(det_refined) < 1e-6:
                    pass
                    # print("Warning: Refined basis vectors became collinear. Reverting to unrefined model.") # <-- 已注释
                else:
                    refined_potential_inliers_map = {}
                    for point_idx, Pj in enumerate(points_np):
                        c1, c2 = calculate_projection_coeffs(Pj, refined_origin_candidate, refined_basis_a1_candidate,
                                                             refined_basis_a2_candidate)
                        if c1 is None or c2 is None:
                            continue
                        n1, n2 = round(c1), round(c2)
                        P_ideal = refined_origin_candidate + n1 * refined_basis_a1_candidate + n2 * refined_basis_a2_candidate
                        distance = np.linalg.norm(Pj - P_ideal)
                        if distance <= inlier_threshold:
                            ideal_coords_key = (int(n1), int(n2))
                            if ideal_coords_key not in refined_potential_inliers_map:
                                refined_potential_inliers_map[ideal_coords_key] = []
                            refined_potential_inliers_map[ideal_coords_key].append((distance, point_idx))

                    final_inlier_indices_refined = []
                    for ideal_coords_key, candidates in refined_potential_inliers_map.items():
                        best_candidate = min(candidates, key=lambda x: x[0])
                        final_inlier_indices_refined.append(best_candidate[1])

                    if len(final_inlier_indices_refined) >= len(best_inlier_indices) * 0.9:
                        refined_origin = refined_origin_candidate
                        refined_basis_a1 = refined_basis_a1_candidate
                        refined_basis_a2 = refined_basis_a2_candidate
                        best_inlier_indices = final_inlier_indices_refined
                    else:
                        pass
                        # print("Warning: Refinement led to significant inlier reduction. Reverting to unrefined model.") # <-- 已注释
            else:
                pass
                # print("Warning: Least squares refinement failed. Using best found model without refinement.") # <-- 已注释
        except Exception as e:
            # (捕捉错误，但什么也不打印)
            # print(f"Error during least squares refinement: {e}. Using best found model without refinement.") # <-- 已注释
            pass

    final_inliers_data = [image_targets_data[idx] for idx in best_inlier_indices]

    lattice_nodes_for_viz = generate_lattice_points(refined_origin, refined_basis_a1, refined_basis_a2, min_x, max_x,
                                                    min_y, max_y).tolist()

    debug_info = {
        "best_model_origin": refined_origin.tolist(),
        "best_model_basis_a1": refined_basis_a1.tolist(),
        "best_model_basis_a2": refined_basis_a2.tolist(),
        "final_inliers_count": len(final_inliers_data),
        "total_points": len(points_np),
        "inlier_ratio": len(final_inliers_data) / len(points_np) if len(points_np) > 0 else 0,
        "calculated_min_basis_len": calculated_min_basis_len
    }

    return final_inliers_data, lattice_nodes_for_viz, debug_info


if __name__ == '__main__':
    # (此处的 print 是安全的)
    pass
    # (为简洁起见，移除了 __main__ 中的示例代码)