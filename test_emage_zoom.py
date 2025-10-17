import os
import argparse
import torch
import torch.nn.functional as F
from torchvision.io import write_video

import librosa
import time
import numpy as np
from tqdm import tqdm
from emage_utils.motion_io import beat_format_save
from emage_utils import fast_render
from models.emage_audio import EmageAudioModel, EmageVQVAEConv, EmageVAEConv, EmageVQModel


def inference(model, motion_vq, audio_path, device, save_folder, sr, pose_fps, head_motion_path=None, expression_scale=1.0):
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker_id = torch.zeros(1,1).long().to(device)
    with torch.no_grad():
        # motion seed
        # motion_path = audio_path.replace("audio", "motion").replace(".wav", ".npz")
        # motion_data = np.load(motion_path, allow_pickle=True)
        # poses = torch.from_numpy(motion_data["poses"]).unsqueeze(0).to(device).float()
        # foot_contact = torch.from_numpy(np.load(motion_path.replace("smplxflame_30", "footcontact").replace(".npz", ".npy"))).unsqueeze(0).to(device).float()
        # trans = torch.from_numpy(motion_data["trans"]).unsqueeze(0).to(device).float()
        # bs, t, _ = poses.shape
        # poses_6d = rc.axis_angle_to_rotation_6d(poses.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
        # masked_motion = torch.cat([poses_6d, trans, foot_contact], dim=-1) # bs t 337
        trans = torch.zeros(1, 1, 3).to(device)

        latent_dict = model.inference(audio, speaker_id, motion_vq, masked_motion=None, mask=None)
        
        face_latent = latent_dict["rec_face"] if model.cfg.lf > 0 and model.cfg.cf == 0 else None
        upper_latent = latent_dict["rec_upper"] if model.cfg.lu > 0 and model.cfg.cu == 0 else None
        hands_latent = latent_dict["rec_hands"] if model.cfg.lh > 0 and model.cfg.ch == 0 else None
        lower_latent = None
        
        face_index = torch.max(F.log_softmax(latent_dict["cls_face"], dim=2), dim=2)[1] if model.cfg.cf > 0 else None
        upper_index = torch.max(F.log_softmax(latent_dict["cls_upper"], dim=2), dim=2)[1] if model.cfg.cu > 0 else None
        hands_index = torch.max(F.log_softmax(latent_dict["cls_hands"], dim=2), dim=2)[1] if model.cfg.ch > 0 else None
        lower_index = None

        all_pred = motion_vq.decode(
            face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,
            get_global_motion=False, ref_trans=trans[:,0])
        
    motion_pred = all_pred["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
    
    if head_motion_path and os.path.exists(head_motion_path):
        print(f"INFO: Loading and integrating head motion from {head_motion_path}")
        # Load the ARTalk tensor
        artalk_motion = torch.load(head_motion_path, map_location="cpu")
        artalk_expressions = artalk_motion[:, :100]
        artalk_head_pose = artalk_motion[:, 100:103]

        # Resample to match the target frame count `t`
        artalk_len = artalk_expressions.shape[0]
        artalk_expressions_resampled = F.interpolate(
            artalk_expressions.T.unsqueeze(0),
            size=t,
            mode='linear',
            align_corners=False
        ).squeeze(0).T.numpy()
        
        artalk_head_pose_resampled = F.interpolate(
            artalk_head_pose.T.unsqueeze(0),
            size=t,
            mode='linear',
            align_corners=False
        ).squeeze(0).T.numpy()

        # Replace the face and head motion
        face_pred = artalk_expressions_resampled * expression_scale
        
        # SMPLX V1.0 JOINT ORDER: 12: neck, 15: jaw
        # The output `motion_pred` is axis-angle (num_joints * 3)
        # So we target indices 12*3:12*3+3 and 15*3:15*3+3
        # ARTalk provides a single head rotation, which corresponds to the jaw/head joint in SMPLX
        motion_pred[:, 15*3:15*3+3] = artalk_head_pose_resampled

    # Force an upright posture by zeroing out spine and neck rotations
    print("INFO: Forcing upright posture.")
    motion_pred[:, 3*3:3*3+3] = 0  # Spine
    motion_pred[:, 6*3:6*3+3] = 0  # Spine
    motion_pred[:, 9*3:9*3+3] = 0  # Spine
    motion_pred[:, 12*3:12*3+3] = 0 # Neck

    # Create a static translation vector to position the avatar in front of the camera.
    # The render2d camera for face_only is at z=6.0, so we place the avatar at z=2.0
    trans_pred = np.zeros((t, 3), dtype=np.float32)
    trans_pred[:, 2] = 2.0 # Move the avatar forward in the z-axis

    beat_format_save(os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz"),
                     motion_pred, upsample=30//pose_fps, expressions=face_pred, trans=trans_pred)
    return t

def visualize_one(save_folder, audio_path, nopytorch3d=False):  
    npz_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz")
    motion_dict = np.load(npz_path, allow_pickle=True)
    if not nopytorch3d:
        from emage_utils.npz2pose import render2d
        v2d_face = render2d(motion_dict, (512, 512), face_only=True, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dface.mp4"), v2d_face.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dface.mp4"), audio_path, npz_path.replace(".npz", "_2dface_audio.mp4"))
        v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dbody.mp4"), audio_path, npz_path.replace(".npz", "_2dbody_audio.mp4"))
    fast_render.render_one_sequence_with_face(npz_path, os.path.dirname(npz_path), audio_path, model_folder="./emage_evaltools/smplx_models/")  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="./examples/audio")
    parser.add_argument("--save_folder", type=str, default="./examples/motion")
    parser.add_argument("--head_motion_path", type=str, default=None, help="Path to the ARTalk-generated .pt file for head motion.")
    parser.add_argument("--expression_scale", type=float, default=1.0, help="Scaling factor for ARTalk expression intensity.")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--nopytorch3d", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    face_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/face").to(device)
    upper_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/upper").to(device)
    lower_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/lower").to(device)
    hands_motion_vq = EmageVQVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/hands").to(device)
    global_motion_ae = EmageVAEConv.from_pretrained("H-Liu1997/emage_audio", subfolder="emage_vq/global").to(device)
    motion_vq = EmageVQModel(
      face_model=face_motion_vq, upper_model=upper_motion_vq,
      lower_model=lower_motion_vq, hands_model=hands_motion_vq,
      global_model=global_motion_ae).to(device)
    motion_vq.eval()

    model = EmageAudioModel.from_pretrained("H-Liu1997/emage_audio").to(device)
    model.eval()

    audio_files = [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if f.endswith(".wav")]
    sr, pose_fps = model.cfg.audio_sr, model.cfg.pose_fps
    all_t = 0
    start_time = time.time()

    for audio_path in tqdm(audio_files, desc="Inference"):
        # Construct the corresponding head motion path
        audio_fname = os.path.splitext(os.path.basename(audio_path))[0]
        head_motion_path = os.path.join(args.head_motion_path, audio_fname + ".pt") if args.head_motion_path else None

        all_t += inference(model, motion_vq, audio_path, device, args.save_folder, sr, pose_fps, head_motion_path, args.expression_scale)
        if args.visualization:
            visualize_one(args.save_folder, audio_path, args.nopytorch3d)
    print(f"generate total {all_t/pose_fps:.2f} seconds motion in {time.time()-start_time:.2f} seconds")
if __name__ == "__main__":
    main()