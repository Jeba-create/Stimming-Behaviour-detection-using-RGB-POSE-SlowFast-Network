from torch.nn.modules.utils import _pair
import numpy as np

class GeneratePoseTarget:
    """Generate pseudo heatmaps based on joint coordinates and confidence.

    Required keys are "keypoint", "img_shape", "keypoint_score" (optional),
    added or modified keys are "imgs".

    Args:
        sigma (float): The sigma of the generated gaussian map. Default: 0.6.
        use_score (bool): Use the confidence score of keypoints as the maximum
            of the gaussian maps. Default: True.
        with_kp (bool): Generate pseudo heatmaps for keypoints. Default: True.
        with_limb (bool): Generate pseudo heatmaps for limbs. At least one of
            'with_kp' and 'with_limb' should be True. Default: False.
        skeletons (tuple[tuple]): The definition of human skeletons.
            Default: ((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7), (7, 9),
                      (0, 6), (6, 8), (8, 10), (5, 11), (11, 13), (13, 15),
                      (6, 12), (12, 14), (14, 16), (11, 12)),
            which is the definition of COCO-17p skeletons.
        double (bool): Output both original heatmaps and flipped heatmaps.
            Default: False.
        left_kp (tuple[int]): Indexes of left keypoints, which is used when
            flipping heatmaps. Default: (1, 3, 5, 7, 9, 11, 13, 15),
            which is left keypoints in COCO-17p.
        right_kp (tuple[int]): Indexes of right keypoints, which is used when
            flipping heatmaps. Default: (2, 4, 6, 8, 10, 12, 14, 16),
            which is right keypoints in COCO-17p.
    """

    def __init__(self,
                 sigma=0.6,
                 use_score=True,
                 with_kp=True,
                 with_limb=False,
                 skeletons=((0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (5, 7),
                            (7, 9), (0, 6), (6, 8), (8, 10), (5, 11), (11, 13),
                            (13, 15), (6, 12), (12, 14), (14, 16), (11, 12)),
                 double=False,
                 left_kp=(1, 3, 5, 7, 9, 11, 13, 15),
                 right_kp=(2, 4, 6, 8, 10, 12, 14, 16)):

        self.sigma = sigma
        self.use_score = use_score
        self.with_kp = with_kp
        self.with_limb = with_limb
        self.double = double

        # an auxiliary const
        self.eps = 1e-4

        assert self.with_kp or self.with_limb, (
            'At least one of "with_limb" '
            'and "with_kp" should be set as True.')
        self.left_kp = left_kp
        self.right_kp = right_kp
        self.skeletons = skeletons

    def generate_a_heatmap(self, img_h, img_w, centers, sigma, max_values):
        """Generate pseudo heatmap for one keypoint in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            centers (np.ndarray): The coordinates of corresponding keypoints
                (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The max values of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for center, max_value in zip(centers, max_values):
            mu_x, mu_y = center[0], center[1]
            if max_value < self.eps:
                continue

            st_x = max(int(mu_x - 3 * sigma), 0)
            ed_x = min(int(mu_x + 3 * sigma) + 1, img_w)
            st_y = max(int(mu_y - 3 * sigma), 0)
            ed_y = min(int(mu_y + 3 * sigma) + 1, img_h)
            x = np.arange(st_x, ed_x, 1, np.float32)
            y = np.arange(st_y, ed_y, 1, np.float32)

            # if the keypoint not in the heatmap coordinate system
            if not (len(x) and len(y)):
                continue
            y = y[:, None]

            patch = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / 2 / sigma**2)
            patch = patch * max_value
            heatmap[st_y:ed_y,
                    st_x:ed_x] = np.maximum(heatmap[st_y:ed_y, st_x:ed_x],
                                            patch)

        return heatmap

    def generate_a_limb_heatmap(self, img_h, img_w, starts, ends, sigma,
                                start_values, end_values):
        """Generate pseudo heatmap for one limb in one frame.

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            starts (np.ndarray): The coordinates of one keypoint in the
                corresponding limbs (of multiple persons).
            ends (np.ndarray): The coordinates of the other keypoint in the
                corresponding limbs (of multiple persons).
            sigma (float): The sigma of generated gaussian.
            start_values (np.ndarray): The max values of one keypoint in the
                corresponding limbs.
            end_values (np.ndarray): The max values of the other keypoint in
                the corresponding limbs.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmap = np.zeros([img_h, img_w], dtype=np.float32)

        for start, end, start_value, end_value in zip(starts, ends,
                                                      start_values,
                                                      end_values):
            value_coeff = min(start_value, end_value)
            if value_coeff < self.eps:
                continue

            min_x, max_x = min(start[0], end[0]), max(start[0], end[0])
            min_y, max_y = min(start[1], end[1]), max(start[1], end[1])

            min_x = max(int(min_x - 3 * sigma), 0)
            max_x = min(int(max_x + 3 * sigma) + 1, img_w)
            min_y = max(int(min_y - 3 * sigma), 0)
            max_y = min(int(max_y + 3 * sigma) + 1, img_h)

            x = np.arange(min_x, max_x, 1, np.float32)
            y = np.arange(min_y, max_y, 1, np.float32)

            if not (len(x) and len(y)):
                continue

            y = y[:, None]
            x_0 = np.zeros_like(x)
            y_0 = np.zeros_like(y)

            # distance to start keypoints
            d2_start = ((x - start[0])**2 + (y - start[1])**2)

            # distance to end keypoints
            d2_end = ((x - end[0])**2 + (y - end[1])**2)

            # the distance between start and end keypoints.
            d2_ab = ((start[0] - end[0])**2 + (start[1] - end[1])**2)

            if d2_ab < 1:
                full_map = self.generate_a_heatmap(img_h, img_w, [start],
                                                   sigma, [start_value])
                heatmap = np.maximum(heatmap, full_map)
                continue

            coeff = (d2_start - d2_end + d2_ab) / 2. / d2_ab

            a_dominate = coeff <= 0
            b_dominate = coeff >= 1
            seg_dominate = 1 - a_dominate - b_dominate

            position = np.stack([x + y_0, y + x_0], axis=-1)
            projection = start + np.stack([coeff, coeff], axis=-1) * (
                end - start)
            d2_line = position - projection
            d2_line = d2_line[:, :, 0]**2 + d2_line[:, :, 1]**2
            d2_seg = (
                a_dominate * d2_start + b_dominate * d2_end +
                seg_dominate * d2_line)

            patch = np.exp(-d2_seg / 2. / sigma**2)
            patch = patch * value_coeff

            heatmap[min_y:max_y, min_x:max_x] = np.maximum(
                heatmap[min_y:max_y, min_x:max_x], patch)

        return heatmap

    def generate_heatmap(self, img_h, img_w, kps, sigma, max_values):
        """Generate pseudo heatmap for all keypoints and limbs in one frame (if
        needed).

        Args:
            img_h (int): The height of the heatmap.
            img_w (int): The width of the heatmap.
            kps (np.ndarray): The coordinates of keypoints in this frame.
            sigma (float): The sigma of generated gaussian.
            max_values (np.ndarray): The confidence score of each keypoint.

        Returns:
            np.ndarray: The generated pseudo heatmap.
        """

        heatmaps = []
        if self.with_kp:
            num_kp = kps.shape[1]
            for i in range(num_kp):
                heatmap = self.generate_a_heatmap(img_h, img_w, kps[:, i],
                                                  sigma, max_values[:, i])
                heatmaps.append(heatmap)

        if self.with_limb:
            for limb in self.skeletons:
                start_idx, end_idx = limb
                starts = kps[:, start_idx]
                ends = kps[:, end_idx]

                start_values = max_values[:, start_idx]
                end_values = max_values[:, end_idx]
                heatmap = self.generate_a_limb_heatmap(img_h, img_w, starts,
                                                       ends, sigma,
                                                       start_values,
                                                       end_values)
                heatmaps.append(heatmap)
        #print("heatmap_poseloading;;;;;;;;;;")
        return np.stack(heatmaps, axis=-1)

    def gen_an_aug(self, results):
        """Generate pseudo heatmaps for all frames.

        Args:
            results (dict): The dictionary that contains all info of a sample.

        Returns:
            list[np.ndarray]: The generated pseudo heatmaps.
        """

        all_kps = results['keypoint']
        kp_shape = all_kps.shape

        if 'keypoint_score' in results:
            all_kpscores = results['keypoint_score']
        else:
            all_kpscores = np.ones(kp_shape[:-1], dtype=np.float32)

        img_h, img_w = results['img_shape']
        num_frame = kp_shape[1]

        imgs = []
        for i in range(num_frame):
            sigma = self.sigma
            kps = all_kps[:, i]
            kpscores = all_kpscores[:, i]

            max_values = np.ones(kpscores.shape, dtype=np.float32)
            if self.use_score:
                max_values = kpscores

            hmap = self.generate_heatmap(img_h, img_w, kps, sigma, max_values)
            imgs.append(hmap)

        return imgs

    def __call__(self, results):
        if not self.double:
            results['imgs'] = np.stack(self.gen_an_aug(results))
        else:
            results_ = cp.deepcopy(results)
            flip = Flip(
                flip_ratio=1, left_kp=self.left_kp, right_kp=self.right_kp)
            results_ = flip(results_)
            results['imgs'] = np.concatenate(
                [self.gen_an_aug(results),
                 self.gen_an_aug(results_)])
        return results

class UniformSampler:
  def  __init__(self,NoOfFrms2slct,mode):
    self.NoOfFrms2slct=NoOfFrms2slct
    self.mode='train'

  def closestNumber(self,NoFrms, NoOfFrms2slct) :
    # Find the quotient
    q = int(NoFrms /self.NoOfFrms2slct)
    # 1st possible closest number
    NoofFrmsUprBound = self.NoOfFrms2slct * q
    return NoofFrmsUprBound 

  def __call__(self, pose):
    Frmstoselect=[]
    i=0
    Nofrms=pose['total_frames']
    NoofFrmss=self.closestNumber(pose['total_frames'], self.NoOfFrms2slct)
    if NoofFrmss>self.NoOfFrms2slct:
      rangeIntrvl=int(NoofFrmss/self.NoOfFrms2slct)
      for _ in range(self.NoOfFrms2slct):
        if self.mode=='train':
          Frmstoselect.append(randrange(i, i+rangeIntrvl)) 
        elif self.mode=='val':
          Frmstoselect.append(i)
        i+=rangeIntrvl 
    else:
      start = 0
      inds = np.arange(start, start + self.NoOfFrms2slct)
      #print("inds",inds)
      Frmstoselect = np.mod(inds, Nofrms) 
    #print("Frmstoselect..",Nofrms,Frmstoselect) 
    if 'Frmstoselect' not in pose:
      pose['Frmstoselect']=Frmstoselect
    return pose

class  PoseCompact:
  def  __init__(self,padding=0.25,threshold=10,hw_ratio=None,allow_imgpad=True):
    self.padding=padding
    self.threshold=threshold
    self.hw_ratio=hw_ratio
    self.allow_imgpad=allow_imgpad

  def _combine_quadruple(self,a, b):
    return (a[0] + a[2] * b[0], a[1] + a[3] * b[1], a[2] * b[2], a[3] * b[3])

  def __call__(self, pose):
    if self.hw_ratio is not None:
      self.hw_ratio = _pair(self.hw_ratio)
    img_shape = pose['img_shape']
    h, w = img_shape
    kp = pose['keypoint']
    
    # Make NaN zero
    kp[np.isnan(kp)] = 0.
    kp_x = kp[..., 0]
    kp_y = kp[..., 1]
    
    min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
    min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
    max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
    max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)
    
    # The compact area is too small
    if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
      return pose     
    
    center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
    half_width = (max_x - min_x) / 2 * (1 + self.padding)
    half_height = (max_y - min_y) / 2 * (1 + self.padding)
    
    if self.hw_ratio is not None:
      half_height = max(self.hw_ratio[0] * half_width, half_height)
      half_width = max(1 / self.hw_ratio[1] * half_height, half_width)
      
    min_x, max_x = center[0] - half_width, center[0] + half_width
    min_y, max_y = center[1] - half_height, center[1] + half_height

    # hot update
    if not self.allow_imgpad:
      min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
      max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
    else:
      min_x, min_y = int(min_x), int(min_y)
      max_x, max_y = int(max_x), int(max_y)

    kp_x[kp_x != 0] -= min_x
    kp_y[kp_y != 0] -= min_y

    #print("min_x,min_y",min_x,max_x,min_y,max_y)

    new_shape = (max_y - min_y, max_x - min_x)
    pose['img_shape'] = new_shape

    # the order is x, y, w, h (in [0, 1]), a tuple
    crop_quadruple = pose.get('crop_quadruple', (0., 0., 1., 1.))

    new_crop_quadruple = (min_x / w, min_y / h, (max_x - min_x) / w,
                              (max_y - min_y) / h)
    #print("crop_quadruple",crop_quadruple,new_crop_quadruple,max_y,min_y, max_x,min_x)
    crop_quadruple =self._combine_quadruple(crop_quadruple, new_crop_quadruple)
    pose['crop_quadruple'] = crop_quadruple
    return pose
 
class resize:
  def  __init__(self,scale):
    self.scale=scale
  def  __call__(self,pose):
    img_h, img_w = pose['img_shape']
    new_w, new_h =  self.scale

    scale_factor = np.array([new_w / img_w, new_h / img_h],
                                         dtype=np.float32)
    if 'scale_factor' not in pose:
      pose['scale_factor'] = np.array([1, 1], dtype=np.float32)

    pose['img_shape'] = (new_h, new_w)
    pose['scale_factor'] = pose['scale_factor'] * scale_factor
    pose['keypoint']= pose['keypoint'] * scale_factor
    return pose

class flip:
  def __init__(self,left_kp=None,right_kp=None):
    self.left_kp=left_kp
    self.right_kp=right_kp
  def __call__(self,pose):  
    img_width = pose['img_shape'][1]
    kps = pose['keypoint']
    kpscores = pose.get('keypoint_score', None)
    kp_x = kps[..., 0]
    kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
    new_order = list(range(kps.shape[2]))
    if self.left_kp is not None and self.right_kp is not None:
      for left, right in zip(self.left_kp, self.right_kp):
        new_order[left] = right
        new_order[right] = left
    kps = kps[:, :, new_order]
    if kpscores is not None:
      kpscores = kpscores[:, :, new_order]
      pose['keypoint'] = kps
    if 'keypoint_score' in pose:
      pose['keypoint_score'] = kpscores
    return pose

class CenterCrop:
    def __init__(self,size):
        self.size=size
    def get_Crop_Bbox(self,img_shape,size):
        #print("img_shape,size",img_shape,size)
        img_h, img_w =img_shape
        cntr_h,cntr_w=int(img_h)/2,int(img_h)/2
        min_x=max(0,cntr_w-int(size/2))
        min_y=max(0,cntr_h-int(size/2))
        max_x=min(cntr_w+int(size/2),img_w)
        max_y=min(cntr_h+int(size/2),img_h)

        return min_x,min_y,max_x,max_y

    def _crop_kps(self,kps, crop_bbox):
        return kps - crop_bbox[:2]

    def __call__(self,pose): 
        img_h, img_w = pose['img_shape']
        left, top, right, bottom = self.get_Crop_Bbox((img_h, img_w), self.size)
        new_h, new_w = bottom - top, right - left

        crop_bbox = np.array([left, top, right, bottom])
        pose['crop_bbox'] = crop_bbox
        pose['img_shape'] = (new_h, new_w)

        if 'keypoint' in pose:
            pose['keypoint'] = self._crop_kps(pose['keypoint'],crop_bbox)
        return pose
             

class RandomResizedCrop:
  def __init__(self,area_range):
    self.area_range=area_range

  def get_crop_bbox(self,img_shape,area_range,aspect_ratio_range=(3 / 4, 4 / 3),max_attempts=10):

    assert 0 < self.area_range[0] <= self.area_range[1] <= 1
    assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

    img_h, img_w =img_shape
    area = img_h * img_w

    min_ar, max_ar = aspect_ratio_range
    aspect_ratios = np.exp(np.random.uniform(np.log(min_ar), np.log(max_ar), size=max_attempts))
    target_areas = np.random.uniform(*self.area_range, size=max_attempts) * area
    candidate_crop_w = np.round(np.sqrt(target_areas *aspect_ratios)).astype(np.int32)
    candidate_crop_h = np.round(np.sqrt(target_areas /aspect_ratios)).astype(np.int32)
    
    for i in range(max_attempts):
      crop_w = candidate_crop_w[i]
      crop_h = candidate_crop_h[i]
      if crop_h <= img_h and crop_w <= img_w:
        x_offset = random.randint(0, img_w - crop_w)
        y_offset = random.randint(0, img_h - crop_h)
        return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

    # Fallback
    crop_size = min(img_h, img_w)
    x_offset = (img_w - crop_size) // 2
    y_offset = (img_h - crop_size) // 2
    
    return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size

  def _crop_kps(self,kps, crop_bbox):
    return kps - crop_bbox[:2]

  def __call__(self,pose): 

    img_h, img_w = pose['img_shape']
    left, top, right, bottom = self.get_crop_bbox((img_h, img_w), self.area_range)
    new_h, new_w = bottom - top, right - left
    
    if 'crop_quadruple' not in pose:
      pose['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32) # x, y, w, h

    x_ratio, y_ratio = left / img_w, top / img_h
    w_ratio, h_ratio = new_w / img_w, new_h / img_h
    
    old_crop_quadruple = pose['crop_quadruple']
    old_x_ratio, old_y_ratio = old_crop_quadruple[0], old_crop_quadruple[1]
    old_w_ratio, old_h_ratio = old_crop_quadruple[2], old_crop_quadruple[3]
    new_crop_quadruple = [old_x_ratio + x_ratio * old_w_ratio,old_y_ratio + y_ratio * old_h_ratio, w_ratio * old_w_ratio,h_ratio * old_h_ratio]
    pose['crop_quadruple'] = np.array(new_crop_quadruple, dtype=np.float32)
    
    crop_bbox = np.array([left, top, right, bottom])
    pose['crop_bbox'] = crop_bbox
    pose['img_shape'] = (new_h, new_w)

    if 'keypoint' in pose:
      pose['keypoint'] = self._crop_kps(pose['keypoint'],crop_bbox)
      
    return pose
