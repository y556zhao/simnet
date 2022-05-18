import dataclasses
import IPython
import copy
import sys
import io
import operator
import time
import pathlib
import pickle
import shortuuid
import urllib3

from PIL import Image
import cv2
import boto3
import numpy as np
import zstandard as zstd
import os
import OpenEXR
import Imath
import torch
import matplotlib.pyplot as plt


def get_uid():
  return shortuuid.uuid()


# Struct for Pose Prediction
@dataclasses.dataclass
class Pose:
  heat_map: np.ndarray
  vertex_target: np.ndarray
  z_centroid: np.ndarray


# Struct for Keypoint Prediction
@dataclasses.dataclass
class Keypoint:
  heat_map: np.ndarray


# Struct for Oriented Bounding Box Prediction
@dataclasses.dataclass
class OBB:
  heat_map: np.ndarray
  vertex_target: np.ndarray
  z_centroid: np.ndarray
  cov_matrices: np.ndarray
  compressed: bool = False

  def compress(self):
    if self.compressed:
      return
    # Heat map scale by 4x and quantize
    height, width = self.heat_map.shape
    self.heat_map = cv2.resize(
        self.heat_map, (width // 4, height // 4), interpolation=cv2.INTER_CUBIC
    ).astype(np.float16)

    # Vertex field, quantize and transpose to make vertex field smooth in memory order (makes
    # downstream compression 50x more effective)
    self.vertex_target = self.vertex_target.transpose(2, 0, 1).astype(np.float16)

    self.compressed = True

  def decompress(self):
    if not self.compressed:
      return

    #print('OBB.decompress')
    #def _debug(name, x):
    #  print(name, x.dtype, x.shape, x.min(), x.max())
    #_debug('OBB.decompress.heat_map', self.heat_map)
    #_debug('OBB.decompress.z_centroid', self.z_centroid)
    #_debug('OBB.decompress.vertex_target', self.vertex_target)
    #_debug('OBB.decompress.cov_matrices', self.cov_matrices)
    #print('----------------------------------------------------')

    # Heat map scale by 4x and quantize
    height, width = self.heat_map.shape
    self.heat_map = cv2.resize(
        self.heat_map.astype(np.float32), (width * 4, height * 4), interpolation=cv2.INTER_CUBIC
    )
    #print('heat_map:', self.heat_map.dtype, self.heat_map.shape)

    # Vertex field, quantize and transpose to make vertex field smooth in memory order (makes
    # downstream compression 50x more effective)
    self.vertex_target = self.vertex_target.astype(np.float32).transpose(1, 2, 0)
    #print('vertex_target:', self.vertex_target.dtype, self.vertex_target.shape)

    self.compressed = False


def compress_color_image(img, quality=90):
  with io.BytesIO() as buf:
    img = Image.fromarray(img)
    img.save(buf, format='jpeg', quality=quality)
    return buf.getvalue()


def decompress_color_image(img_bytes):
  with io.BytesIO(img_bytes) as buf:
    img = Image.open(buf)
    return np.array(img)


#Struct for Stereo Representation
@dataclasses.dataclass
class Stereo:
  left_color: np.ndarray
  right_color: np.ndarray
  compressed: bool = False

  def compress(self):
    if self.compressed:
      return
    self.left_color = compress_color_image(self.left_color)
    self.right_color = compress_color_image(self.right_color)
    self.compressed = True

  def decompress(self):
    if not self.compressed:
      return
    #print('Stereo.decompress')

    self.left_color = decompress_color_image(self.left_color)
    self.right_color = decompress_color_image(self.right_color)
    self.compressed = False


# Application Specific Datapoints Should be specified here.
@dataclasses.dataclass
class Panoptic:
  stereo: Stereo
  depth: np.ndarray
  segmentation: np.ndarray
  object_poses: list
  boxes: list
  detections: list
  keypoints: list = dataclasses.field(default_factory=list)
  instance_mask: np.ndarray = None
  scene_name: str = 'sim'
  uid: str = dataclasses.field(default_factory=get_uid)
  compressed: bool = False

  def compress(self):
    self.stereo.compress()
    for object_pose in self.object_poses:
      object_pose.compress()

    if self.compressed:
      return

    # Depth scale by 4x and quantize
    height, width = self.depth.shape
    self.depth = cv2.resize(self.depth, (width // 4, height // 4),
                            interpolation=cv2.INTER_CUBIC).astype(np.float16)

    self.compressed = True

  def decompress(self):
    self.stereo.decompress()
    for object_pose in self.object_poses:
      object_pose.decompress()

    if not self.compressed:
      return
    #print('Panoptic.decompress')

    # Depth scale by 4x and quantize
    height, width = self.depth.shape
    self.depth = cv2.resize(
        self.depth.astype(np.float32), (width * 4, height * 4), interpolation=cv2.INTER_CUBIC
    )

    self.compressed = False


# Application Specific Datapoints Should be specified here.
@dataclasses.dataclass
class RobotMask:
  stereo: Stereo
  depth: np.ndarray
  segmentation: np.ndarray
  uid: str = dataclasses.field(default_factory=get_uid)


# End Applications Here
def compress_datapoint(x):
  x = copy.deepcopy(x)
  x.compress()
  buf = pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL)
  cctx = zstd.ZstdCompressor()
  cbuf = cctx.compress(buf)
  return cbuf


def decompress_datapoint(cbuf, disable_final_decompression=False):
  cctx = zstd.ZstdDecompressor()
  buf = cctx.decompress(cbuf)
  x = pickle.loads(buf)
  if not disable_final_decompression:
    x.decompress()
  return x


def make_dataset(uri):
  if ',' in uri:
    datasets = []
    for uri in uri.split(','):
      datasets.append(make_one_dataset(uri))
    return ConcatDataset(datasets)
  return make_one_dataset(uri)


def make_one_dataset(uri):
  # parse parameters
  uri, _, raw_params = uri.partition('?')
  dataset = make_one_simple_dataset(uri)
  if not raw_params:
    return dataset

  params = {}
  for raw_param in raw_params.split('&'):
    k, _, v = raw_param.partition('=')
    assert k and v
    assert k not in params
    params[k] = v

  return FilterDataset(dataset, params)


def make_one_simple_dataset(uri):
  if uri.startswith('s3://'):
    path = uri.partition('s3://')[2]
    bucket, _, dataset_path = path.partition('/')
    return RemoteDataset(bucket, dataset_path)

  if uri.startswith('file://'):
    path = uri.partition('file://')[2]
    dataset_path = pathlib.Path(path)
    return KeyposeDataset(dataset_path)
    #return LocalDataset(dataset_path)

  raise ValueError(f'uri must start with `s3://` or `file://`. uri={uri}')


def _datapoint_path(dataset_path, uid):
  return f'{dataset_path}/{uid}.pickle.zstd'


class FilterDataset:

  def __init__(self, dataset, params):
    self.dataset = dataset
    self.params = params
    self.samples = None
    for key in params:
      if key == 'samples':
        self.samples = int(params[key])
      else:
        raise ValueError(f'Unknown param in dataset args: {key}')

  def list(self):
    handles = self.dataset.list()
    if self.samples is not None:
      handles = handles * (self.samples // len(handles) + 1)
      handles = handles[:self.samples]
    return handles

  def write(self, datapoint):
    raise ValueError('Cannot write to concat dataset')


class ConcatDataset:

  def __init__(self, datasets):
    self.datasets = datasets

  def list(self):
    handles = []
    for dataset in self.datasets:
      handles.extend(dataset.list())
    return handles

  def write(self, datapoint):
    raise ValueError('Cannot write to concat dataset')


class RemoteDataset:

  def __init__(self, bucket, path):
    self.s3 = boto3.resource('s3')
    self.bucket = bucket
    self.dataset_path = path
    assert not path.endswith('/')
    self._cache_list = None

  def list(self):
    if self._cache_list is not None:
      return self._cache_list
    bucket = self.s3.Bucket(self.bucket)
    handles = []
    for obj in bucket.objects.filter(Prefix=self.dataset_path + '/'):
      path = obj.key
      if not path.endswith('.pickle.zstd'):
        continue
      uid = path.rpartition('/')[2].partition('.pickle.zstd')[0]
      handles.append(RemoteReadHandle(self.bucket, self.dataset_path, uid))
    x = sorted(handles, key=operator.attrgetter('uid'))
    self._cache_list = x
    return x

  def write(self, datapoint):
    buf = compress_datapoint(datapoint)
    path = _datapoint_path(self.dataset_path, datapoint.uid)
    self.s3.Bucket(self.bucket).put_object(Key=path, Body=buf)


class RemoteDataset:

  def __init__(self, bucket, path):
    self.s3 = boto3.resource('s3')
    self.bucket = bucket
    self.dataset_path = path
    assert not path.endswith('/')
    self._cache_list = None

  def list(self):
    if self._cache_list is not None:
      return self._cache_list
    bucket = self.s3.Bucket(self.bucket)
    handles = []
    for obj in bucket.objects.filter(Prefix=self.dataset_path + '/'):
      path = obj.key
      if not path.endswith('.pickle.zstd'):
        continue
      uid = path.rpartition('/')[2].partition('.pickle.zstd')[0]
      handles.append(RemoteReadHandle(self.bucket, self.dataset_path, uid))
    x = sorted(handles, key=operator.attrgetter('uid'))
    self._cache_list = x
    return x

  def write(self, datapoint):
    buf = compress_datapoint(datapoint)
    path = _datapoint_path(self.dataset_path, datapoint.uid)
    self.s3.Bucket(self.bucket).put_object(Key=path, Body=buf)


class RemoteReadHandle:

  def __init__(self, bucket, dataset_path, uid):
    self.bucket = bucket
    self.dataset_path = dataset_path
    self.uid = uid

  def read(self, disable_final_decompression=False):
    # Lazily initialize s3 resource due to pytorch data loaders
    attempt = 0
    while True:
      success = False
      try:
        if False:
          session = boto3.session.Session()
          s3 = session.resource('s3')
        else:
          s3 = boto3.resource('s3')
        path = _datapoint_path(self.dataset_path, self.uid)
        rsp = s3.Object(self.bucket, path).get()
        buf = rsp['Body'].read()
        success = True
      except urllib3.exceptions.ProtocolError as exc:
        print('RETRY: urllib3.exceptions.ProtocolError, attempt:', attempt)
        attempt += 1
        delay = retry_delay(attempt)
        print('Sleeping before retry:', delay)
        time.sleep(delay)
      except KeyError as exc:
        print('RETRY: KeyError, attempt:', attempt)
        attempt += 1
        delay = retry_delay(attempt)
        print('Sleeping before retry:', delay)
        time.sleep(delay)
      if success:
        break
    dp = decompress_datapoint(buf, disable_final_decompression=disable_final_decompression)
    # TODO: remove this, once old datasets without UID are out of use
    if not hasattr(dp, 'uid'):
      dp.uid = self.uid
    assert dp.uid == self.uid
    return dp


def retry_delay(attempt):
  """Exponential backoff with maximum delay and partial jitter."""
  delay = 100e-3 * (2.**attempt)  # exponential back off
  delay = min(10.0, delay)  # with maximum
  delay = delay * np.random.uniform(0.5, 1)  # partial jitter
  return delay


class LocalDataset:
  def __init__(self, dataset_path):
    if not dataset_path.exists():
      print('New dataset directory:', dataset_path)
      dataset_path.mkdir(parents=True)
    assert dataset_path.is_dir()
    self.dataset_path = dataset_path

  def list(self):
    handles = []
    for path in self.dataset_path.glob('*.pickle.zstd'):
      uid = path.name.partition('.')[0]
      handles.append(LocalReadHandle(self.dataset_path, uid))
    return sorted(handles, key=operator.attrgetter('uid'))

  def write(self, datapoint):
    path = _datapoint_path(self.dataset_path, datapoint.uid)
    buf = compress_datapoint(datapoint)
    with open(path, 'wb') as fh:
      fh.write(buf)


class KeyposeDataset:
  def __init__(self, dataset_path):
    if not dataset_path.exists():
      print('New dataset directory:', dataset_path)
      dataset_path.mkdir(parents=True)
    assert dataset_path.is_dir()
    self.dataset_path = dataset_path

  def list(self):
    handles = []
    obj_names = [dr for dr in os.listdir(self.dataset_path)]
    obj_name = obj_names[0]
    data_directory = f'{self.dataset_path}/{obj_name}/data/{obj_name}'
    scenes = [dr for dr in os.listdir(data_directory)]
    for scene in scenes:
      scene_directory = f'{data_directory}/{scene}'
      ids = []
      for scene_id in os.listdir(scene_directory):
        if scene_id[:6] not in ids and scene_id[0] == '0':
          ids.append(scene_id[:6])
          handles.append(KeyposeReadHandle(scene_directory, scene_id[:6]))
    return sorted(handles, key=operator.attrgetter('uid'))

  def write(self, datapoint):
    path = _datapoint_path(self.dataset_path, datapoint.uid)
    buf = compress_datapoint(datapoint)
    with open(path, 'wb') as fh:
      fh.write(buf)


class Keypose:
  def __init__(self, stereo=None, depth=None, segmentation=None, file_id=None):
    self.stereo=stereo
    self.depth=depth
    self.segmentation=segmentation
    self.file_id=file_id
    self.scene_name=file_id
    self.detections=None
    #object_poses: list
    #boxes: list
    #detections: list
    #keypoints: list = dataclasses.field(default_factory=list)
    #instance_mask: np.ndarray = None
    #scene_name: str = 'sim'
    #uid: str = dataclasses.field(default_factory=get_uid)
    #compressed: bool = False

def load_depth(file):
  def exr_loader(EXR_PATH, ndim=3):
    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
      # read channels indivudally
      allchannels = []
      for c in ['R', 'G', 'B']:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
        channel.shape = (size[1], size[0])
        allchannels.append(channel)

      # create array and transpose dimensions to match tensor style
      exr_arr = np.array(allchannels).transpose((0, 1, 2))
      return exr_arr

    if ndim == 1:
      # transform data to numpy
      channel = np.frombuffer(exr_file.channel('D', pt), dtype=np.float32)
      channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
      exr_arr = np.array(channel)
      return exr_arr

  gt_depth = exr_loader(file, ndim=1)
  vmin = min(gt_depth.min(), gt_depth.min())
  vmax = max(gt_depth.max(), gt_depth.max())
  dim = (640, 512)
  gt_depth = cv2.resize(gt_depth, dim, interpolation=cv2.INTER_CUBIC)
  #gt_depth = np.expand_dims(gt_depth, axis=0)
  #gt_depth = torch.from_numpy(gt_depth).float()
  return gt_depth

def load_img(file):
  img = plt.imread(file)
  dim = (640, 512)
  img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
  return np.array(img)

def load_segmentation_mask(file):
  img = cv2.imread(file)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  dim = (640, 512)
  img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)
  gray_scale_img = np.zeros(img.shape)
  return np.array(gray_scale_img)

def decompress_keypose_datapoint(dataset_path, uid):

  depth=load_depth(f'{dataset_path}/{uid}_Do.exr')
  left_img=load_img(f'{dataset_path}/{uid}_L.png')
  right_img = load_img(f'{dataset_path}/{uid}_R.png')
  stereo=Stereo(left_color=left_img,right_color=right_img,compressed= False)
  segmentation=load_segmentation_mask(f'{dataset_path}/{uid}_mask.png')
  file_id=f'{dataset_path}/{uid}'
  dp = Keypose(stereo=stereo, depth=depth, segmentation=segmentation, file_id=file_id)
  return dp

class KeyposeReadHandle:
  def __init__(self, dataset_path, uid):
    self.dataset_path = dataset_path
    self.uid = uid

  def read(self, disable_final_decompression=False):
    dp = decompress_keypose_datapoint(self.dataset_path, self.uid)
    # TODO: remove this, once old datasets without UID are out of use
    #if not hasattr(dp, 'uid'):
    #  dp.uid = self.uid
    #assert dp.uid == self.uid
    print(dp.file_id)
    return dp

class LocalReadHandle:
  def __init__(self, dataset_path, uid):
    self.dataset_path = dataset_path
    self.uid = uid

  def read(self, disable_final_decompression=False):
    path = _datapoint_path(self.dataset_path, self.uid)
    with open(path, 'rb') as fh:
      dp = decompress_datapoint(fh.read(), disable_final_decompression=disable_final_decompression)
    # TODO: remove this, once old datasets without UID are out of use
    if not hasattr(dp, 'uid'):
      dp.uid = self.uid
    assert dp.uid == self.uid
    return dp


if __name__ == '__main__':
  ds = make_dataset(sys.argv[1])
  for dph in ds.list():
    dp = dph.read()
    IPython.embed()
    break
