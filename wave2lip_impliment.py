import platform
import subprocess
from os import path

import argparse
import audio
import cv2
import face_detection
import numpy as np
import os
import torch
from tqdm import tqdm

from models import Wav2Lip
from utils.global_constant import config
from utils.log_utils import logger

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')


class args:
    fps = 25
    static = False
    pads = [0, 10, 0, 0]
    face_det_batch_size = 8
    wav2lip_batch_size = 256
    resize_factor = 1
    crop = [0, -1, 0, -1]
    box = [-1, -1, -1, -1]
    rotate = False
    nosmooth = False
    img_size = 96
    checkpoint_path = config.get("wave2lip","checkpoint_path")


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def face_detect(images):
    batch_size = args.face_det_batch_size
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError(
                    'Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break
    head_exist = []
    results = []
    pady1, pady2, padx1, padx2 = args.pads

    first_head_rect = None
    first_head_image = None
    for rect, image in zip(predictions, images):
        if rect is not None:
            first_head_rect = rect
            first_head_image = image
            break
    for rect, image in zip(predictions, images):
        if rect is None:
            head_exist.append(False)
            if len(results) == 0:
                y1 = max(0, first_head_rect[1] - pady1)
                y2 = min(first_head_image.shape[0], first_head_rect[3] + pady2)
                x1 = max(0, first_head_rect[0] - padx1)
                x2 = min(first_head_image.shape[1], first_head_rect[2] + padx2)
                results.append([x1, y1, x2, y2])
            else:
                results.append(results[-1])
        # cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
        # raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
        else:
            head_exist.append(True)
            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)
            results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results, head_exist


def datagen(frames, mels):
    img_batch, head_exist_batch, mel_batch, frame_batch, coords_batch = [], [], [], [], []

    # ***************************1、识别人脸对应的位置坐标，未识别的人脸的帧对应为None ***************************
    if args.box[0] == -1:
        if not args.static:
            face_det_results, head_exist = face_detect(frames)  # BGR2RGB for CNN face detection
        else:
            face_det_results, head_exist = face_detect([frames[0]])
    else:
        logger.info('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
        head_exist = [True] * len(frames)

    logger.info("face detect end")

    for i, m in enumerate(mels):
        # 获取对应的一组音频对应的帧下标idx
        idx = 0 if args.static else i % len(frames)
        # 获取对应的一组音频对应的帧
        frame_to_save = frames[idx].copy()
        # 获取对应的一组音频对应的帧对应的人脸坐标
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))
        if i in (0, 3, 5):
            cv2.imwrite(path.join("/home/guo/wave2lip/wave2lip_torch/Wav2Lip/results", '{}_resize.jpg'.format(i)), face)
        head_exist_batch.append(head_exist[idx])
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, head_exist_batch, mel_batch, frame_batch, coords_batch
            img_batch, head_exist_batch, mel_batch, frame_batch, coords_batch = [], [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, head_exist_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

model = load_model(args.checkpoint_path)

def wave2lip(face_path, audio_path,handle_num):
    if not os.path.isfile(face_path):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif face_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(face_path)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        logger.info('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    logger.info("Number of frames available for inference: {}".format(len(full_frames)))

    if not audio_path.endswith('.wav'):
        logger.info('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    # TODO 与视频对应起来，每16，理论上来说，mel_idx_multiplier与mel_step_size相等，将音频分组，并获取与音频长度相等的视频帧
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    logger.info("Length of mel chunks: {}".format(len(mel_chunks)))

    # TODO 找到视频与音频的对应关系
    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)
    # 覆盖对应的帧（脑袋部位像素）
    for i, (img_batch, exist_head_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                                                                      total=int(np.ceil(float(
                                                                                          len(mel_chunks)) / batch_size)))):
        if i == 0:
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        logger.info("batch write message:img batch: %d,frames:%d,coords:%d,exist_head_batch:%d", len(img_batch), len(frames), len(coords), len(exist_head_batch))

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        # #逐帧更新并写入到临时视频文件中去
        for p, f, c, exist in zip(pred, frames, coords, exist_head_batch):
            if exist:
                y1, y2, x1, x2 = c
                p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                f[y1:y2, x1:x2] = p
            #TODO 图像修复
            out.write(f)

    out.release()
    output_file = "temp/{}.avi".format(handle_num // 100)
    if os.path.isfile(output_file):
        os.remove(output_file)
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', output_file)
    logger.info("merge audio and video begin")
    subprocess.call(command, shell=platform.system() != 'Windows')
    logger.info("merge audio and video end")
    return output_file


if __name__ == '__main__':
    wave2lip("/home/guo/wave2lip/temp/盘春园视频test_clip.mp4", "/home/guo/wave2lip/temp/3.mp3",0)
