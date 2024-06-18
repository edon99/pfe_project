
import cv2
from sklearn.cluster import KMeans


def detect(stframe, cap, model, conf = 0.4):
        while cap.isOpened():
            success, frame = cap.read()
            if success:
            # results = model(frame, conf=0.4)
                results = model.track(frame, persist=True, conf=conf, verbose=False)
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame)
            else:
                print("error detecting")

def predict(model, frame, conf, format = 'xyxy'):
    """
    Predicts the bounding boxes of the objects in the image

    Returns
    -------
    pd.DataFrame
        DataFrame containing the bounding boxes
    """
    results = model(frame, conf=conf, verbose=False)
    # return results[0].boxes.xyxy.cpu().numpy()
    return results[0]


def isclose_int(color1, color2, tol=20):
    return all(abs(c1 - c2) <= tol for c1, c2 in zip(color1, color2))

label_mapping = {0: 'Player', 1: 'Referee', 2: 'Ball'}  #tunsi model
# label_mapping = {0: 'Ball', 1: 'Player', 2: 'Referee'}  #my model



def assign_team(color, team_colors, team1, team2, tol=20):
    if team_colors[team1] is None:
        team_colors[team1] = color
        return team1
    elif team_colors[team2] is None:
        team_colors[team2] = color
        return team2
    elif isclose_int(color, team_colors[team1], tol):
        return team1
    elif isclose_int(color, team_colors[team2], tol):
        return team2
    else:
        
        return team1 if isclose_int(color, team_colors[team1], tol) else team2


def detect_test(stframe, cap, model, team1, team2, conf=0.4):
    team_colors = {
    team1: None,
    team2: None
}
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model.track(frame, persist=True, conf=conf, verbose=False)
            detections_info = [] 


            for result in results:
                detections = result.boxes.xyxy.cpu().numpy()
                labels = result.boxes.cls.cpu().numpy().astype(int)     
                

                for detection, label_in in zip(detections,labels):
                    x1, y1, x2, y2 = detection[:4].astype(int)
                    label = label_mapping[label_in]
                   
                    
                    roi = frame[y1:y2, x1:x2]
                    roi_reshaped = roi.reshape((-1, 3))
                    
                   
                    # avg_color_per_row = np.average(roi, axis=0)
                    # avg_color = np.average(avg_color_per_row, axis=0)
                    # avg_color = tuple(map(int, avg_color))

                    kmeans = KMeans(n_clusters=1)
                    kmeans.fit(roi_reshaped)

                    #get dom colour
                    dom_color = kmeans.cluster_centers_[0]

                    # convert to integer tuple
                    dom_color = tuple(map(int, dom_color))

                    team_name = assign_team(dom_color, team_colors, team1, team2) if label == "Player" else ""

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    detections_info.append((center_x, center_y, dom_color))
                    
                   
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), dom_color, 2)

                    if label == 'Player':
                        frame = cv2.putText(frame, f"{team_name} {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, dom_color, 2)
                    else:
                        frame = cv2.putText(frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, dom_color, 2)                      
            # annotated_frame = results[0].plot()
            # annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            
            # for i in range(4):
            #     for j in range(i + 1, 4):
            #         center_x1, center_y1, color1 = detections_info[i]
            #         center_x2, center_y2, color2 = detections_info[j]
            #         if isclose_int(color1,color2):  # Check if the colors are the same
            #             frame = cv2.line(frame, (center_x1, center_y1), (center_x2, center_y2), (255,255,255), 2)
            
            stframe.image(frame)
        else:
            print("Error detecting")


        