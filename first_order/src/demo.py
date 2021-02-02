import matplotlib
import os, sys
import yaml
import paddle.fluid.dygraph as dygraph
import paddle.fluid as fluid
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte

from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from animate import normalize_kp


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path):

    with open(config_path) as f:
        config = yaml.load(f)
    pretrain_model = config['ckpt_model']
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if pretrain_model['generator'] is not None:
        if pretrain_model['generator'][-3:] == 'npz':
            G_param = np.load(pretrain_model['generator'], allow_pickle=True)['arr_0'].item()
            G_param_clean = [(i, G_param[i]) for i in G_param if 'num_batches_tracked' not in i]
            parameter_clean = generator.parameters()
            del(parameter_clean[65])  # The parameters in AntiAliasInterpolation2d is not in dict_set and should be ignore.
            for p, v in zip(parameter_clean, G_param_clean):
                p.set_value(v[1])
        else:
            a, b = fluid.load_dygraph(pretrain_model['generator'])
            generator.set_dict(a)
        print('Restore Pre-trained Generator')
    if pretrain_model['kp'] is not None:
        if pretrain_model['kp'][-3:] == 'npz':
            KD_param = np.load(pretrain_model['kp'], allow_pickle=True)['arr_0'].item()
            KD_param_clean = [(i, KD_param[i]) for i in KD_param if 'num_batches_tracked' not in i]
            parameter_clean = kp_detector.parameters()
            for p, v in zip(parameter_clean, KD_param_clean):
                p.set_value(v[1])
        else:
            a, b = fluid.load_dygraph(pretrain_model['kp'])
            kp_detector.set_dict(a)
        print('Restore Pre-trained KD')
    generator.eval()
    kp_detector.eval()
    
    return generator, kp_detector

def make_animation(source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True):
    with dygraph.no_grad():
        predictions = []
        source = dygraph.to_variable(np.transpose(source_image[np.newaxis], (0, 3, 1, 2)).astype(np.float32))
        driving = dygraph.to_variable(np.transpose(np.array(driving_video)[np.newaxis], (0, 4, 1, 2, 3)).astype(np.float32))
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].numpy(), [0, 2, 3, 1])[0])
    return predictions

# def find_best_frame(source, driving, cpu=False):
#     import face_alignment  #需要pytorch
#     from scipy.spatial import ConvexHull

#     def normalize_kp(kp):
#         kp = kp - kp.mean(axis=0, keepdims=True)
#         area = ConvexHull(kp[:, :2]).volume
#         area = np.sqrt(area)
#         kp[:, :2] = kp[:, :2] / area
#         return kp

#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
#                                       device='cpu' if cpu else 'cuda')
#     kp_source = fa.get_landmarks(255 * source)[0]
#     kp_source = normalize_kp(kp_source)
#     norm  = float('inf')
#     frame_num = 0
#     for i, image in tqdm(enumerate(driving)):
#         kp_driving = fa.get_landmarks(255 * image)[0]
#         kp_driving = normalize_kp(kp_driving)
#         new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
#         if new_norm < norm:
#             norm = new_norm
#             frame_num = i
#     return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--source_image", default='sup-mat/source.png', help="path to source image")
    parser.add_argument("--driving_video", default='sup-mat/source.png', help="path to driving video")
    parser.add_argument("--result_video", default='result.mp4', help="path to output")
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    # parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
    #                     help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    # parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
    #                     help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    try:
        fps = reader.get_meta_data()['fps']
        video_mode = True
    except KeyError:
        video_mode = False
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    if opt.cpu:
        plac = fluid.CPUPlace()
    else:
        plac = fluid.CUDAPlace(0)
    with dygraph.guard(plac):
        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
        generator, kp_detector = load_checkpoints(config_path=opt.config)

        # if opt.find_best_frame or opt.best_frame is not None:
        #     i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        #     print ("Best frame: " + str(i))
        #     driving_forward = driving_video[i:]
        #     driving_backward = driving_video[:(i+1)][::-1]
        #     predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        #     predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        #     predictions = predictions_backward[::-1] + predictions_forward[1:]
        # else:
        #     predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, cpu=opt.cpu)
        
        predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=opt.relative, adapt_movement_scale=opt.adapt_scale)
    if video_mode:
        imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
    else:
        imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions])

