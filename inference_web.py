from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import streamlit as st
import uuid
from moviepy.editor import *
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from',default="checkpoints/wav2lip.pth", required=False)
# parser.add_argument('--face', type=str,
# 					help='Filepath of video/image that contains faces to use', required=True)
# parser.add_argument('--audio', type=str,
#                     help='Filepath of video/audio file to use as raw audio source', required=True)
# parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
#                     default='results/result_voice.mp4')

parser.add_argument('--static', type=bool,
                    help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
                    help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                         'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                         'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                         'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')
args = parser.parse_args()
args.img_size = 96

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

ORIGINAL_DATA_PATH = "temp/original_data"
if not os.path.exists(ORIGINAL_DATA_PATH):
    os.mkdir(ORIGINAL_DATA_PATH)
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

class Wave2lip:
    def __init__(self):

        self.detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
        print("load detector end")
        self.wave_lip_model = load_model(args.checkpoint_path)
        print("load wave_lip_mode end")

    def get_smoothened_boxes(self,boxes, T):
        for i in range(len(boxes)):
            if i + T > len(boxes):
                window = boxes[len(boxes) - T:]
            else:
                window = boxes[i: i + T]
            boxes[i] = np.mean(window, axis=0)
        return boxes

    def face_detect(self,images):
        # TODO 识别头像信息

        detector = self.detector

        batch_size = args.face_det_batch_size

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
        if not args.nosmooth: boxes = self.get_smoothened_boxes(boxes, T=5)
        results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

        del detector
        return results, head_exist

    def datagen(self,frames, mels):
        img_batch, head_exist_batch, mel_batch, frame_batch, coords_batch = [], [], [], [], []

        # ***************************1、识别人脸对应的位置坐标，未识别的人脸的帧对应为None ***************************
        if args.box[0] == -1:
            if not args.static:
                face_det_results, head_exist = self.face_detect(frames)  # BGR2RGB for CNN face detection
            else:
                face_det_results, head_exist = self.face_detect([frames[0]])
        else:
            print('Using the specified bounding box instead of face detection...')
            y1, y2, x1, x2 = args.box
            face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
            head_exist = [True] * len(frames)

        for i, m in enumerate(mels):
            # 获取对应的一组音频对应的帧下标idx
            idx = 0 if args.static else i % len(frames)
            # 获取对应的一组音频对应的帧
            frame_to_save = frames[idx].copy()
            # 获取对应的一组音频对应的帧对应的人脸坐标
            face, coords = face_det_results[idx].copy()

            face = cv2.resize(face, (args.img_size, args.img_size))
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

    def wave2lip(self,task_id,video_path,audio_path):

        if os.path.isfile(video_path) and video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
            args.static = True

        if not os.path.isfile(video_path):
            raise ValueError('--face argument must be a valid path to video/image file')

        elif video_path.split('.')[1] in ['jpg', 'png', 'jpeg']:
            full_frames = [cv2.imread(video_path)]
            fps = args.fps

        else:
            video_stream = cv2.VideoCapture(video_path)
            fps = video_stream.get(cv2.CAP_PROP_FPS)

            print('Reading video frames...')

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

        print("Number of frames available for inference: " + str(len(full_frames)))

        if not audio_path.endswith('.wav'):
            print('Extracting raw audio...')
            command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

            subprocess.call(command, shell=True)
            audio_path = 'temp/temp.wav'

        wav = audio.load_wav(audio_path, 16000)
        mel = audio.melspectrogram(wav)
        print(mel.shape)

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

        print("Length of mel chunks: {}".format(len(mel_chunks)))

        # TODO 找到视频与音频的对应关系
        full_frames = full_frames[:len(mel_chunks)]

        batch_size = args.wav2lip_batch_size
        gen = self.datagen(full_frames.copy(), mel_chunks)
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

            print("batch write message:", len(img_batch), len(frames), len(coords), len(exist_head_batch))

            with torch.no_grad():
                pred = self.wave_lip_model(mel_batch, img_batch)

            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
            # #逐帧更新并写入到临时视频文件中去
            i = 0
            for p, f, c, exist in zip(pred, frames, coords, exist_head_batch):
                i += 1
                if not exist:
                    out.write(f)
                else:
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    head_high, head_width, _ = p.shape
                    width_cut = int(head_width * 0.2)
                    f[y1:y2, x1 + width_cut:x2 - width_cut] = p[:, width_cut:head_width - width_cut]
                    out.write(f)

        out.release()
        outfile_file_name = "results/{0}_{1}_{2}.mp4".format(task_id,"temp1","temp2")
        # command = 'ffmpeg -y -i {} -i {} -c:a aac -c:v copy  -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile_file_name)
        command = 'ffmpeg -y -i {} -i {} -c:v libx264 -c:a aac -map 0:a:0 -map 1:v:0 {}'.format(audio_path, 'temp/result.avi', outfile_file_name)
        subprocess.call(command, shell=platform.system() != 'Windows')
        #self.merge_video_audio_video('temp/result.avi',audio_path,outfile_file_name)
        return outfile_file_name

    def merge_video_audio_video(self,original_video_file, original_audio_video_file, video_dst_file):

        video = VideoFileClip(original_video_file, audio=False)
        print(video.fps)
        if os.path.exists(video_dst_file):
            os.remove(video_dst_file)
        #audio = VideoFileClip(original_audio_video_file).audio
        audio = AudioFileClip(original_audio_video_file)
        video = video.set_audio(audio)
        print(video.fps)
        video.write_videofile(video_dst_file, fps=video.fps,preset="ultrafast",codec="libx264", audio_codec="aac", audio=True,
                              threads=6)
wave2lip = Wave2lip()

def main():
    # 设置标签栏
    st.set_page_config(page_title="wav2lip", page_icon="🔍")
    # 设置标题
    st.title('Welcome to wav2lip!')
    # 视频上传组件
    uploaded_video = st.file_uploader("Choose a video or img")
    video_name = None  # name of the video
    # 判断视频是否上传成功
    if uploaded_video is not None:
        # preview, delete and download the video
        video_bytes = uploaded_video.read()
        # save file to disk for later process
        video_name = uploaded_video.name
        if video_name.split('.')[1] in ['jpg', 'png', 'jpeg',"webp"]:
            st.image(video_bytes)
        else:
            st.video(video_bytes)


        with open(f"{ORIGINAL_DATA_PATH}/{video_name}", mode='wb') as f:
            f.write(video_bytes)  # save video to disk
    else:
        print("video is null")

    video_file_path = f"{ORIGINAL_DATA_PATH}/{video_name}"
    # 视频上传组件
    uploaded_audio = st.file_uploader("Choose a audio file")
    audio_name = None  # name of the video
    # 判断视频是否上传成功
    if uploaded_audio is not None:
        # preview, delete and download the video
        audio_bytes = uploaded_audio.read()
        st.audio(audio_bytes)
        # save file to disk for later process
        audio_name = uploaded_audio.name
        with open(f"{ORIGINAL_DATA_PATH}/{audio_name}", mode='wb') as f:
            f.write(audio_bytes)  # save video to disk
    else:
        print("video is null")

    audio_file_path = f"{ORIGINAL_DATA_PATH}/{audio_name}"
    uid = uuid.uuid1()
    # wave2lip
    search_button = st.button("wave2lip")
    if search_button:  # 判断是否点击搜索按钮
        if uploaded_audio is None or uploaded_video is None:
            st.warning('Please upload video and audio first!')
        else:
            output=wave2lip.wave2lip(uid,video_file_path,audio_file_path)
            st.video(output)
        st.success("Done!")

if __name__ == '__main__':
    main()
