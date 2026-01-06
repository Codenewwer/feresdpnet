import cv2
import mediapipe as mp
import os
import math


def extract_faces_from_video(video_path, output_folder, target_fps=20, conf_threshold=0.6):
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=conf_threshold
    ) as face_detection:


        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: 无法打开视频 {video_path}")
            return

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频原始 FPS: {original_fps}")
        print(f"目标提取 FPS: {target_fps}")


        if not os.path.exists(output_folder):
            os.makedirs(output_folder)


        if original_fps <= target_fps:
            step = 1
        else:

            step = original_fps / target_fps

        current_frame_idx = 0
        next_capture_frame = 0.0
        saved_count = 0

        while True:
            success, frame = cap.read()
            if not success:
                break


            if current_frame_idx >= int(next_capture_frame):


                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)

                if results.detections:
                    for i, detection in enumerate(results.detections):

                        bboxC = detection.location_data.relative_bounding_box


                        x = int(bboxC.xmin * frame_width)
                        y = int(bboxC.ymin * frame_height)
                        w = int(bboxC.width * frame_width)
                        h = int(bboxC.height * frame_height)


                        x = max(0, x)
                        y = max(0, y)
                        w = min(w, frame_width - x)
                        h = min(h, frame_height - y)


                        if w > 0 and h > 0:

                            face_img = frame[y:y + h, x:x + w]


                            img_name = f"frame_{current_frame_idx:06d}_face_{i}.jpg"
                            save_path = os.path.join(output_folder, img_name)

                            try:
                                cv2.imwrite(save_path, face_img)
                                saved_count += 1
                            except Exception as e:
                                print(f"保存图片失败: {e}")


                next_capture_frame += step

            current_frame_idx += 1

        cap.release()
        print(f"处理完成。共保存 {saved_count} 张人脸图像至: {output_folder}")


# ---------------- 使用示例 ----------------
if __name__ == "__main__":

    my_video_path = "/public/home/"


    my_output_dir = "/output_faces"

    if os.path.exists(my_video_path):
        extract_faces_from_video(my_video_path, my_output_dir, target_fps=20, conf_threshold=0.6)
    else:
        print(f"请确保视频文件存在: {my_video_path}")
