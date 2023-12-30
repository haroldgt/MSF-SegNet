import os
import numpy as np
from torch.utils import data
import yaml
import pickle
# import numba as nb
from scipy import spatial
import torch
from val_tools.metric import var_save_txt

# load Semantic KITTI class info
def get_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        dataset_properties = yaml.safe_load(stream)
    label_name = dict()
    for i in sorted(list(dataset_properties['learning_map'].keys()))[::-1]:
        label_name[dataset_properties['learning_map'][i]] = dataset_properties['labels'][i]

    return label_name


class SemKITTI_sk(data.Dataset):
    def __init__(self, data_path, imageset='train',
                 return_ref=False, label_mapping="semantic-kitti.yaml",labelData_bits=32):
        self.return_ref = return_ref
        self.labelData_bits = labelData_bits
        with open(label_mapping, 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        self.learning_map = semkittiyaml['learning_map']
        self.color_map = semkittiyaml['color_map'] # A*
        self.imageset = imageset
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'ytest':
            split = semkittiyaml['split']['test']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/ytest/test')

        self.im_idx = []
        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            if self.labelData_bits == 32:
                annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.uint32).reshape((-1, 1))
                annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
                annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            elif self.labelData_bits == 16:
                annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                            dtype=np.uint16).reshape((-1, 1))
                annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data)
            else :
                raise ValueError("bits number of label must be 16 or 32!")
        # find zero vectors in xyz, and delete it.
        zero_rows_index = np.where(~raw_data[:, :3].any(axis=1))[0]
        raw_data = np.delete(raw_data, zero_rows_index, axis=0)
        annotated_data = np.delete(annotated_data, zero_rows_index, axis=0)

        # data_tuple = (raw_data[:26240, :3], annotated_data[:26240, :].astype(np.uint8))
        data_tuple = (raw_data[:, :3], annotated_data.astype(np.uint8))
        if self.return_ref:
            # data_tuple += (raw_data[:26240, 3],)
            data_tuple += (raw_data[:, 3],)
        # var_save_txt("data_tuple", data_tuple)

        # A* -- rgb features compution
        # bgr_feat = bgr_features(annotated_data.astype(np.uint8), self.color_map)
        # data_tuple += (bgr_feat,)

        return data_tuple


class cylinder_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space
        self.transform = transform_aug
        self.trans_std = trans_std

        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):
        # points: [N, 3]
        rot_sin = np.sin(angle)
        rot_cos = np.cos(angle)
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)
        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)
        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)
        else:
            raise ValueError("axis should in range")

        return points @ rot_mat_T

    def __getitem__(self, index):
        # 'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2: sig = np.squeeze(sig)
        # elif len(data) == 4: # A*
        #     xyz, labels, sig, bgr_feat = data
        #     if len(sig.shape) == 2: sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        # convert coordinate into polar coordinates

        if self.transform:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T

            xyz[:, 0:3] += noise_translate

        # A* -- compute volume feature
        v_feature = Volume_feature(xyz, self.max_volume_space[0], self.min_volume_space[0], self.grid_size[0])

        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any(): print("Zero interval!")
        grid_ind = (np.floor((np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort((grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label, xyz)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate((return_xyz, sig[..., np.newaxis]), axis=1)
        # elif len(data) == 4: # A*
        #     return_fea = np.concatenate((return_xyz, sig[..., np.newaxis], bgr_feat), axis=1)
            # return_fea = np.concatenate((return_xyz, sig[..., np.newaxis], grid_ind), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            # data_tuple += (grid_ind, labels, return_fea)
            data_tuple += (grid_ind, labels, return_fea, v_feature) # A*
        return data_tuple


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


# A*
def bgr_features(labels, color_map):
        # labels: [N, 1], color_map: {0: [0, 255, 0], 1:[0, 180, 250], ...}.
        # bgr_features: [N, 3].
        existClass_in_frame = np.unique(labels)
        bgr_features = np.zeros((labels.shape[0], 3), dtype=np.uint8)
        for ith_class in existClass_in_frame:
            ithClass_rows_index = np.nonzero(labels == ith_class)[0]
            bgr_features[ithClass_rows_index, :] = color_map[ith_class]
        return bgr_features


def Volume_feature(points, max_volume_space = 50, min_volume_space = 0, grid_size = 480):
    # points: [N, 3].
    # v&nVectors_features: [N, 4]. 1st cols - Volume, 2ed-4th cols - nVectors.
    
    # knn search.
    K = 10
    tree = spatial.KDTree(points)
    n = tree.query(points, K)[1]

    """
        level1: Compute V and nVectors.
    """

    original_xyz = points
    neigh_1st_xyz = points[n[:, 1], :]
    neigh_2ed_xyz = points[n[:, 2], :]
    neigh_3rd_xyz = points[n[:, 3], :]
    neigh_4th_xyz = points[n[:, 4], :]
    neigh_5th_xyz = points[n[:, 5], :]
    neigh_6th_xyz = points[n[:, 6], :]
    neigh_7th_xyz = points[n[:, 7], :]
    neigh_8th_xyz = points[n[:, 8], :]
    neigh_9th_xyz = points[n[:, 9], :]

    # level2:Compute V

    PM = original_xyz
    QM = neigh_1st_xyz
    RM = neigh_2ed_xyz
    AM = neigh_3rd_xyz

    PQM = QM - PM
    PRM = RM - PM
    PQ_PRM = np.cross(PQM, PRM)
    APM = PM - AM
    APxPQPRM = APM * PQ_PRM
    AP_PQPRM = APxPQPRM[:, 0] + APxPQPRM[:, 1] + APxPQPRM[:, 2]
    AP_PQPRM = AP_PQPRM.reshape(-1, 1)  # (n, ) -> (n, 1)
    V_fea = (1/6) * np.abs(AP_PQPRM)

    min_Vthres = (max_volume_space - min_volume_space) / (grid_size * 1000)
    Idn = np.where(V_fea <= min_Vthres)[0]
    V_fea[Idn, :] = 0

    v_features = np.arctan(V_fea)*max_volume_space

    # level3:Compute nVectors

    neigh_mean_xyz = (original_xyz+neigh_1st_xyz+neigh_2ed_xyz+neigh_3rd_xyz+neigh_4th_xyz+neigh_5th_xyz+neigh_6th_xyz
                      + neigh_7th_xyz + neigh_8th_xyz + neigh_9th_xyz)/10
    neigh_mean_xyz[:, 0] = 0
    neigh_mean_xyz[:, 1] = 0

    Pn = neigh_9th_xyz - original_xyz
    Pn_1 = neigh_8th_xyz - original_xyz
    VectsM = np.cross(Pn, Pn_1)

    # compute theta between VectsM and PMM.
    PMM = neigh_mean_xyz - original_xyz

    Vects_PMM = VectsM * PMM
    Vects_PMM_dot = Vects_PMM[:, 0] + Vects_PMM[:, 1] + Vects_PMM[:, 2]
    Vects_PMM_dot = Vects_PMM_dot.reshape(-1, 1)  # (n, ) -> (n, 1)

    VectsM2 = VectsM * VectsM
    VectsM_dot = VectsM2[:, 0] + VectsM2[:, 1] + VectsM2[:, 2]
    VectsM_dot = VectsM_dot.reshape(-1, 1)  # (n, ) -> (n, 1)
    VectsM_norm = np.sqrt(VectsM_dot)

    PMM2 = PMM * PMM
    PMM_dot = PMM2[:, 0] + PMM2[:, 1] + PMM2[:, 2]
    PMM_dot = PMM_dot.reshape(-1, 1)  # (n, ) -> (n, 1)
    PMM_norm = np.sqrt(PMM_dot)

    VectsMxPMM_norm = VectsM_norm * PMM_norm
    cos_theta = Vects_PMM_dot / VectsMxPMM_norm
    # check over range value
    over_down_index1 = np.where(cos_theta < -1)[0]
    cos_theta[over_down_index1, :] = -1
    over_up_index2 = np.where(cos_theta > 1)[0]
    cos_theta[over_up_index2, :] = 1
    theta = np.arccos(cos_theta) * 180 / np.pi

    # Vectors direction
    Vects_flag = np.zeros((theta.shape[0], 1))
    rows_index1 = np.where(theta <= 90)[0]
    Vects_flag[rows_index1, :] = -1
    rows_index2 = np.where(theta > 90)[0]
    Vects_flag[rows_index2, :] = 1
    # Vects_flag1 = np.concatenate((Vects_flag, Vects_flag, Vects_flag), axis=1)

    VectsM = VectsM * Vects_flag

    concat_v_Vects = np.concatenate((v_features, VectsM), axis=1)

    return concat_v_Vects


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    point_xyz = [d[2] for d in data]
    grid_ind_stack = [d[3] for d in data]
    point_label = [d[4] for d in data]
    xyz_feature = [d[5] for d in data]
    v_feature = [d[6] for d in data] # A*
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), point_xyz, grid_ind_stack, point_label, xyz_feature, v_feature


def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):
    # print(input_xyz_polar.shape)
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


# @nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1], cur_sear_ind[2]] = np.argmax(counter)
    return processed_label
