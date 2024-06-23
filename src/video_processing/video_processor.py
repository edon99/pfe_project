import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import streamlit as st
from .utils import get_device_type, get_labels_dics

class VideoProcessor:

    def __init__(self, cap, stframe, model_players,
                 model_keypoints, tac_map):
        self.cap = cap
        self.stframe = stframe
        self.model_players = model_players
        self.model_keypoints = model_keypoints
        self.tac_map = tac_map

        self.team_colors = {'team1': None, 'team2': None}
        self.hyper_params = {
            'players_conf': 0.4,
            'keypoints_conf': 0.7,
            'k_d_tol': 7
        }
        self.ball_track_hyperparams = {
            'nbr_frames_no_ball_thresh': 30,
            'ball_track_dist_thresh': 100,
            'max_track_length': 35
        }
        self.device = get_device_type()
        self.keypoints_map_pos, self.classes_names_dic, self.labels_dic = get_labels_dics()
        self.ball_track_history = {'src': [], 'dst': []}
        self.video_fps = np.round(self.cap.get(cv2.CAP_PROP_FPS))

        self.results_keypoints = None
        self.results_players = None

        self.nbr_frames_no_ball = 0
        self.detected_ball_src_pos = None
        self.detected_ball_dst_pos = None

        self.detected_labels_prev = []
        self.detected_labels_src_pts_prev = []

    def process_video(self):
        tot_nbr_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st_prog_bar = st.progress(0, text='Detection starting.')

        for frame_nbr in range(1, tot_nbr_frames + 1):
            percent_complete = int(frame_nbr / tot_nbr_frames * 100)
            st_prog_bar.progress(
                percent_complete,
                text=f"Detection in progress ({percent_complete}%)")

            success, frame = self.cap.read()
            tac_map_copy = self.tac_map.copy()

            if self.nbr_frames_no_ball > self.ball_track_hyperparams[
                    'nbr_frames_no_ball_thresh']:
                self.ball_track_history['dst'] = []
                self.ball_track_history['src'] = []

            if success:
                self.detect_and_transform(frame, frame_nbr)
                tac_map_copy = self.assign_team_and_annotate(
                    frame, tac_map_copy)
                final_img = self.concatenate_images(frame, tac_map_copy)
                self.stframe.image(final_img, channels="BGR")

        st_prog_bar.empty()
        return success

    def detect_and_transform(self, frame, frame_nbr):
        if (frame_nbr - 1) % self.video_fps == 0:  # predeict each second
            self.results_players = self.model_players.predict(  #
                frame,
                conf=self.hyper_params['players_conf'],
                device=self.device,
                verbose=False)
            self.results_keypoints = self.model_keypoints.predict(
                frame,
                conf=self.hyper_params['keypoints_conf'],
                device=self.device,
                verbose=False)

        self.extract_detections(self.results_players)
        self.extract_keypoints(self.results_keypoints)

        detected_labels, detected_labels_src_pts, detected_labels_dst_pts = self.get_detected_labels_and_points(
        )

        if len(detected_labels) > 3:
            self.update_homography(detected_labels, detected_labels_src_pts,
                                   detected_labels_dst_pts, frame_nbr)

        if 'homog' in self.__dict__:
            self.transform_coordinates_and_update_tracks(
                detected_labels, detected_labels_src_pts)

    def extract_detections(self, results_players):
        self.bboxes_p = getattr(results_players[0].boxes.xyxy,
                                self.device)().numpy()
        self.bboxes_p_c = getattr(results_players[0].boxes.xywh,
                                  self.device)().numpy()
        self.labels_p = list(
            getattr(results_players[0].boxes.cls, self.device)().numpy())

    def extract_keypoints(self, results_keypoints):
        self.bboxes_k_c = getattr(results_keypoints[0].boxes.xywh,
                                  self.device)().numpy()
        self.labels_k = list(
            getattr(results_keypoints[0].boxes.cls, self.device)().numpy())

    def get_detected_labels_and_points(self):
        detected_labels = [self.classes_names_dic[i] for i in self.labels_k]
        detected_labels_src_pts = np.array([
            list(np.round(self.bboxes_k_c[i][:2]).astype(int))
            for i in range(self.bboxes_k_c.shape[0])
        ])
        detected_labels_dst_pts = np.array(
            [self.keypoints_map_pos[i] for i in detected_labels])
        return detected_labels, detected_labels_src_pts, detected_labels_dst_pts

    def update_homography(self, detected_labels, detected_labels_src_pts,
                          detected_labels_dst_pts, frame_nbr):

        common_labels = self.get_common_labels(detected_labels)
        if len(common_labels) > 3:
            coor_error = self.calculate_coordinate_error(
                detected_labels, common_labels, detected_labels_src_pts)
            update_homography = coor_error > self.hyper_params['k_d_tol']
        else:
            update_homography = True

        if update_homography or frame_nbr == 1:
            self.homog, _ = cv2.findHomography(detected_labels_src_pts,
                                               detected_labels_dst_pts)
            self.detected_labels_prev = detected_labels.copy()
            self.detected_labels_src_pts_prev = detected_labels_src_pts.copy()

    def get_common_labels(self, detected_labels):
        common_labels = set(self.detected_labels_prev) & set(detected_labels)
        return common_labels

    def calculate_coordinate_error(self, detected_labels, common_labels,
                                   detected_labels_src_pts):
        common_label_idx_prev = [
            self.detected_labels_prev.index(i) for i in common_labels
        ]
        common_label_idx_curr = [
            detected_labels.index(i) for i in common_labels
        ]
        coor_common_label_prev = self.detected_labels_src_pts_prev[
            common_label_idx_prev]
        coor_common_label_curr = detected_labels_src_pts[common_label_idx_curr]
        coor_error = mean_squared_error(coor_common_label_prev,
                                        coor_common_label_curr)
        return coor_error

    def transform_coordinates_and_update_tracks(
        self,
        detected_labels,
        detected_labels_src_pts,
    ):

        self.detected_labels_prev = detected_labels.copy()
        self.detected_labels_src_pts_prev = detected_labels_src_pts.copy()

        bboxes_p_c_0 = self.bboxes_p_c[[i == 0 for i in self.labels_p], :]
        bboxes_p_c_2 = self.bboxes_p_c[[i == 2 for i in self.labels_p], :]

        detected_ppos_src_pts = bboxes_p_c_0[:, :2] + np.array(
            [[0] * bboxes_p_c_0.shape[0], bboxes_p_c_0[:, 3] / 2]).transpose()

        detected_ball_src_pos = bboxes_p_c_2[
            0, :2] if bboxes_p_c_2.shape[0] > 0 else None

        if detected_ball_src_pos is None:
            self.nbr_frames_no_ball += 1
        else:
            detected_ball_src_pos = None
            self.nbr_frames_no_ball = 0

        self.pred_dst_pts = self.transform_players_coordinates(
            detected_ppos_src_pts)

        if detected_ball_src_pos is not None:
            self.detected_ball_dst_pos = self.transform_ball_coordinates(
                self.detected_ball_src_pos)
            self.track_ball_history(self.detected_ball_src_pos,
                                    self.detected_ball_dst_pos)

        if len(self.ball_track_history['src']
               ) > self.ball_track_hyperparams['max_track_length']:
            self.ball_track_history['src'].pop(0)
            self.ball_track_history['dst'].pop(0)

    def transform_players_coordinates(self, detected_ppos_src_pts):
        pred_dst_pts = []
        for pt in detected_ppos_src_pts:
            pt = np.append(np.array(pt), np.array([1]), axis=0)
            dest_point = np.matmul(self.homog, np.transpose(pt))
            dest_point = dest_point / dest_point[2]
            pred_dst_pts.append(list(np.transpose(dest_point)[:2]))
        return np.array(pred_dst_pts)

    def transform_ball_coordinates(self, detected_ball_src_pos):
        pt = np.append(np.array(detected_ball_src_pos), np.array([1]), axis=0)
        dest_point = np.matmul(self.homog, np.transpose(pt))
        dest_point = dest_point / dest_point[2]
        return np.transpose(dest_point)

    def track_ball_history(self, detected_ball_src_pos, detected_ball_dst_pos):
        if len(self.ball_track_history['src']) > 0:
            if np.linalg.norm(
                    detected_ball_src_pos - self.ball_track_history['src'][-1]
            ) < self.ball_track_hyperparams['ball_track_dist_thresh']:
                self.ball_track_history['src'].append(
                    (int(detected_ball_src_pos[0]),
                     int(detected_ball_src_pos[1])))
                self.ball_track_history['dst'].append(
                    (int(detected_ball_dst_pos[0]),
                     int(detected_ball_dst_pos[1])))
            else:
                self.ball_track_history['src'] = [
                    (int(detected_ball_src_pos[0]),
                     int(detected_ball_src_pos[1]))
                ]
                self.ball_track_history['dst'] = [
                    (int(detected_ball_dst_pos[0]),
                     int(detected_ball_dst_pos[1]))
                ]
        else:
            self.ball_track_history['src'].append(
                (int(detected_ball_src_pos[0]), int(detected_ball_src_pos[1])))
            self.ball_track_history['dst'].append(
                (int(detected_ball_dst_pos[0]), int(detected_ball_dst_pos[1])))

    def assign_team_and_annotate(self, frame, tac_map_copy):
        ball_color_bgr = (0, 0, 255)
        j = 0

        for i in range(self.bboxes_p.shape[0]):
            x1, y1, x2, y2 = self.bboxes_p[i].astype(int)
            label = self.labels_dic[self.labels_p[i]]

            roi = frame[y1:y2, x1:x2]
            roi_reshaped = roi.reshape((-1, 3))

            kmeans = KMeans(n_clusters=1)
            kmeans.fit(roi_reshaped)
            dom_color = tuple(map(int, kmeans.cluster_centers_[0]))

            if self.labels_p[i] == 0:
                if 'homog' in self.__dict__ and j < len(self.pred_dst_pts):
                    # tac_map_copy = cv2.circle(tac_map_copy,
                    #                           (int(self.pred_dst_pts[j][0]),
                    #                            int(self.pred_dst_pts[j][1])),
                    #                           radius=15,
                    #                           color=(255, 0, 0),
                    #                           thickness=-1)
                    tac_map_copy = cv2.circle(tac_map_copy,
                                              (int(self.pred_dst_pts[j][0]),
                                               int(self.pred_dst_pts[j][1])),
                                              radius=15,
                                              color=dom_color,
                                              thickness=-1)
                j += 1

            if self.detected_ball_src_pos is not None and 'homog' in self.__dict__:
                tac_map_copy = cv2.circle(tac_map_copy,
                                          (int(self.detected_ball_dst_pos[0]),
                                           int(self.detected_ball_dst_pos[1])),
                                          radius=10,
                                          color=ball_color_bgr,
                                          thickness=1)

        if len(self.ball_track_history['src']) > 0:
            points = np.hstack(self.ball_track_history['dst']).astype(
                np.int32).reshape((-1, 1, 2))
            tac_map_copy = cv2.polylines(tac_map_copy, [points],
                                         isClosed=False,
                                         color=(0, 0, 100),
                                         thickness=2)
        return tac_map_copy

    def isclose_int(color1, color2, tol=20):
        return all(abs(c1 - c2) <= tol for c1, c2 in zip(color1, color2))

    def assign_team(self, dom_color, team_colors, team1, team2, tol=20):
        if team_colors[team1] is None:
            team_colors[team1] = dom_color
            return team1
        elif team_colors[team2] is None:
            team_colors[team2] = dom_color
            return team2
        elif self.isclose_int(dom_color, team_colors[team1], tol):
            return team1
        elif self.isclose_int(dom_color, team_colors[team2], tol):
            return team2
        else:
            return team1 if self.isclose_int(dom_color, team_colors[team1],
                                             tol) else team2

    def concatenate_images(self, frame, tac_map_copy):
        tac_map_copy = cv2.resize(tac_map_copy,
                                  (tac_map_copy.shape[1], frame.shape[0]))
        cv2.putText(tac_map_copy, "Tactical View", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        return cv2.hconcat((frame, tac_map_copy))


def detect(cap, stframe, model_players, model_keypoints,
           tac_map):
    video_processor = VideoProcessor(cap, stframe, model_players,
                                     model_keypoints, tac_map)
    return video_processor.process_video()
