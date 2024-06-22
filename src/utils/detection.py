# Import librarie s
import numpy as np
import os
import streamlit as st
import cv2
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from utils.labels_utils import get_labels_dics
import torch

from utils.detect import assign_team

label_mapping = {0: 'Player', 1: 'Referee', 2: 'Ball'}


def generate_file_name():
    list_video_files = os.listdir('./outputs/')
    idx = 0
    while True:
        idx += 1
        output_file_name = f'detect_{idx}'
        if output_file_name + '.mp4' not in list_video_files:
            break
    return output_file_name


def get_device_type():
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def detect(cap, stframe, team1, team2, model_players, model_keypoints,
           tac_map):

    team_colors = {team1: None, team2: None}

    hyper_params = {'players_conf': 0.4, 'keypoints_conf': 0.5, 'k_d_tol': 7}
    ball_track_hyperparams = {
        'nbr_frames_no_ball_thresh': 30,
        'ball_track_dist_thresh': 100,
        'max_track_length': 35
    }
    device = get_device_type()

    # Create progress bar
    tot_nbr_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_prog_bar = st.progress(0, text='Detection starting.')

    keypoints_map_pos, classes_names_dic, labels_dic = get_labels_dics()

    # Store the ball track history
    ball_track_history = {'src': [], 'dst': []}

    nbr_frames_no_ball = 0
    video_fps = np.round(cap.get(cv2.CAP_PROP_FPS))
    results_keypoints = None
    # Loop over input video frames
    for frame_nbr in range(1, tot_nbr_frames + 1):
        # Update progress bar
        percent_complete = int(frame_nbr / (tot_nbr_frames) * 100)
        st_prog_bar.progress(
            percent_complete,
            text=f"Detection in progress ({percent_complete}%)")

        # Read a frame from the video
        success, frame = cap.read()

        # Reset tactical map image for each new frame
        tac_map_copy = tac_map.copy()

        if nbr_frames_no_ball > ball_track_hyperparams[
                'nbr_frames_no_ball_thresh']:
            ball_track_history['dst'] = []
            ball_track_history['src'] = []

        if success:

            #################### Part 1 ####################
            # Object Detection & Coordiante Transofrmation #
            ################################################

            # Run YOLOv8 players inference on the frame
            results_players = model_players.predict(
                frame,
                conf=hyper_params['players_conf'],
                device=device,
                verbose=False)
            # Run YOLOv8 field keypoints inference each one second
            if (frame_nbr - 1 % video_fps == 0):
                results_keypoints = model_keypoints.predict(
                    frame,
                    conf=hyper_params['keypoints_conf'],
                    device=device,
                    verbose=False)

            ## Extract detections information
            bboxes_p = getattr(results_players[0].boxes.xyxy, device)().numpy(
            )  # Detected players, referees and ball (x,y,x,y) bounding boxes
            bboxes_p_c = getattr(results_players[0].boxes.xywh, device)(
            ).numpy(
            )  # Detected players, referees and ball (x,y,w,h) bounding boxes
            labels_p = list(
                getattr(results_players[0].boxes.cls, device)
                ().numpy())  # Detected players, referees and ball labels list

            # confs_p = list(
            #     getattr(results_players[0].boxes.conf, device)().numpy()
            # )

            bboxes_k_c = getattr(results_keypoints[0].boxes.xywh, device)(
            ).numpy()  # Detected field keypoints (x,y,w,h) bounding boxes
            labels_k = list(
                getattr(
                    results_keypoints[0].boxes.cls,
                    device)().numpy())  # Detected field keypoints labels list

            # Convert detected numerical labels to alphabetical labels
            detected_labels = [classes_names_dic[i] for i in labels_k]

            # Extract detected field keypoints coordiantes on the current frame
            detected_labels_src_pts = np.array([
                list(np.round(bboxes_k_c[i][:2]).astype(int))
                for i in range(bboxes_k_c.shape[0])
            ])

            # Get the detected field keypoints coordinates on the tactical map
            detected_labels_dst_pts = np.array(
                [keypoints_map_pos[i] for i in detected_labels])

            ## Calculate Homography transformation matrix when more than 4 keypoints are detected
            if len(detected_labels) > 3:
                # Always calculate homography matrix on the first frame
                if frame_nbr > 1:
                    # Determine common detected field keypoints between previous and current frames
                    common_labels = set(detected_labels_prev) & set(
                        detected_labels)
                    #
                    # When at least 4 common keypoints are detected, determine if they are displaced on average beyond a certain tolerance level
                    if len(common_labels) > 3:
                        common_label_idx_prev = [
                            detected_labels_prev.index(i)
                            for i in common_labels
                        ]  # Get labels indexes of common detected keypoints from previous frame
                        common_label_idx_curr = [
                            detected_labels.index(i) for i in common_labels
                        ]  # Get labels indexes of common detected keypoints from current frame
                        coor_common_label_prev = detected_labels_src_pts_prev[
                            common_label_idx_prev]  # Get labels coordiantes of common detected keypoints from previous frame
                        coor_common_label_curr = detected_labels_src_pts[
                            common_label_idx_curr]  # Get labels coordiantes of common detected keypoints from current frame
                        coor_error = mean_squared_error(
                            coor_common_label_prev, coor_common_label_curr
                        )  # Calculate error between previous and current common keypoints coordinates
                        update_homography = coor_error > hyper_params[
                            'k_d_tol']  # Check if error surpassed the predefined tolerance level
                    else:
                        update_homography = True
                else:
                    update_homography = True

                if update_homography:
                    homog, mask = cv2.findHomography(
                        detected_labels_src_pts,  # Calculate homography matrix
                        detected_labels_dst_pts)
            if 'homog' in locals():
                detected_labels_prev = detected_labels.copy(
                )  # Save current detected keypoint labels for next frame
                detected_labels_src_pts_prev = detected_labels_src_pts.copy(
                )  # Save current detected keypoint coordiantes for next frame

                bboxes_p_c_0 = bboxes_p_c[[
                    i == 0 for i in labels_p
                ], :]  # Get bounding boxes information (x,y,w,h) of detected players (label 0)
                bboxes_p_c_2 = bboxes_p_c[[
                    i == 2 for i in labels_p
                ], :]  # Get bounding boxes information (x,y,w,h) of detected ball(s) (label 2)

                # Get coordinates of detected players on frame (x_cencter, y_center+h/2)
                detected_ppos_src_pts = bboxes_p_c_0[:, :2] + np.array([
                    [0] * bboxes_p_c_0.shape[0], bboxes_p_c_0[:, 3] / 2
                ]).transpose()
                # Get coordinates of the first detected ball (x_center, y_center)
                detected_ball_src_pos = bboxes_p_c_2[
                    0, :2] if bboxes_p_c_2.shape[0] > 0 else None

                if detected_ball_src_pos is None:
                    nbr_frames_no_ball += 1
                else:
                    nbr_frames_no_ball = 0

                # Transform players coordinates from frame plane to tactical map plance using the calculated Homography matrix
                pred_dst_pts = [
                ]  # Initialize players tactical map coordiantes list
                for pt in detected_ppos_src_pts:  # Loop over players frame coordiantes
                    pt = np.append(np.array(pt), np.array([1]),
                                   axis=0)  # Covert to homogeneous coordiantes
                    dest_point = np.matmul(
                        homog,
                        np.transpose(pt))  # Apply homography transofrmation
                    dest_point = dest_point / dest_point[
                        2]  # Revert to 2D-coordiantes
                    pred_dst_pts.append(list(
                        np.transpose(dest_point)
                        [:2]))  # Update players tactical map coordiantes list
                pred_dst_pts = np.array(pred_dst_pts)

                # Transform ball coordinates from frame plane to tactical map plance using the calculated Homography matrix
                if detected_ball_src_pos is not None:
                    pt = np.append(np.array(detected_ball_src_pos),
                                   np.array([1]),
                                   axis=0)
                    dest_point = np.matmul(homog, np.transpose(pt))
                    dest_point = dest_point / dest_point[2]
                    detected_ball_dst_pos = np.transpose(dest_point)

                    # track ball history
                    if len(ball_track_history['src']) > 0:
                        if np.linalg.norm(
                                detected_ball_src_pos -
                                ball_track_history['src'][-1]
                        ) < ball_track_hyperparams['ball_track_dist_thresh']:
                            ball_track_history['src'].append(
                                (int(detected_ball_src_pos[0]),
                                 int(detected_ball_src_pos[1])))
                            ball_track_history['dst'].append(
                                (int(detected_ball_dst_pos[0]),
                                 int(detected_ball_dst_pos[1])))
                        else:
                            ball_track_history['src'] = [
                                (int(detected_ball_src_pos[0]),
                                 int(detected_ball_src_pos[1]))
                            ]
                            ball_track_history['dst'] = [
                                (int(detected_ball_dst_pos[0]),
                                 int(detected_ball_dst_pos[1]))
                            ]
                    else:
                        ball_track_history['src'].append(
                            (int(detected_ball_src_pos[0]),
                             int(detected_ball_src_pos[1])))
                        ball_track_history['dst'].append(
                            (int(detected_ball_dst_pos[0]),
                             int(detected_ball_dst_pos[1])))

                if len(ball_track_history
                       ) > ball_track_hyperparams['max_track_length']:
                    ball_track_history['src'].pop(0)
                    ball_track_history['dst'].pop(0)

            #################### Part 3 #####################
            # Player Team assignment & Tactical Map With Annotations #
            #################################################

            ball_color_bgr = (
                0, 0, 255)  # Color (BGR) for ball annotation on tactical map
            j = 0  # Initializing counter of detected players

            # Loop over all detected object by players detection model
            for i in range(bboxes_p.shape[0]):
                x1, y1, x2, y2 = bboxes_p[i].astype(int)
                label = label_mapping[labels_p[i]]

                roi = frame[y1:y2, x1:x2]
                roi_reshaped = roi.reshape((-1, 3))

                kmeans = KMeans(n_clusters=1)
                kmeans.fit(roi_reshaped)

                #get dom colour
                dom_color = kmeans.cluster_centers_[0]

                # convert to integer tuple
                dom_color = tuple(map(int, dom_color))

                team_name = assign_team(dom_color, team_colors, team1,
                                        team2) if label == "Player" else ""

                if labels_p[i] == 0:  # Display annotation for detected players (label 0)

                    # Add tactical map player postion color coded annotation if more than 3 field keypoints are detected
                    if 'homog' in locals():
                        tac_map_copy = cv2.circle(
                            tac_map_copy,
                            (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                            radius=15,
                            color=(255, 0, 0),
                            thickness=-1)
                        tac_map_copy = cv2.circle(
                            tac_map_copy,
                            (int(pred_dst_pts[j][0]), int(pred_dst_pts[j][1])),
                            radius=15,
                            color=dom_color,
                            thickness=-1)

                    j += 1  # Update players counter

                # Add tactical map ball postion annotation if detected
                if detected_ball_src_pos is not None and 'homog' in locals():
                    tac_map_copy = cv2.circle(tac_map_copy,
                                              (int(detected_ball_dst_pos[0]),
                                               int(detected_ball_dst_pos[1])),
                                              radius=10,
                                              color=ball_color_bgr,
                                              thickness=1)
            # Plot the tracks
            if len(ball_track_history['src']) > 0:
                points = np.hstack(ball_track_history['dst']).astype(
                    np.int32).reshape((-1, 1, 2))
                tac_map_copy = cv2.polylines(tac_map_copy, [points],
                                             isClosed=False,
                                             color=(0, 0, 100),
                                             thickness=2)

            tac_map_copy = cv2.resize(
                tac_map_copy,
                (tac_map_copy.shape[1],
                 frame.shape[0]))  # Resize tactical map
            cv2.putText(tac_map_copy, "Tactical View", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
            final_img = cv2.hconcat(
                (frame, tac_map_copy))  # Concatenate both images

            # Display the annotated frame
            stframe.image(final_img, channels="BGR")

    # Remove progress bar and return
    st_prog_bar.empty()
    return True
