from ultralytics import YOLO
import cv2
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert list to dataframe to allow interpolate
        df_ball_positions = pd.DataFrame(ball_positions, columns = ['x1', 'y1', 'x2', 'y2'])
        # interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        # ensure no missing values at start or end
        df_ball_positions = df_ball_positions.bfill()

        #convert back to list
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        
        ball_boxes = [d.get(1, [None, None, None, None]) for d in ball_positions]
        df_ball_positions = pd.DataFrame(ball_boxes, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        assert not df_ball_positions.isnull().values.any(), "Still contains NaNs!"
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window = 5, min_periods = 1, center = False).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        df_ball_positions['ball_hit'] = 0  # initialize column

        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            dy_current = df_ball_positions['delta_y'].iloc[i]
            dy_next = df_ball_positions['delta_y'].iloc[i + 1]

            negative_change = dy_current > 0 and dy_next < 0
            positive_change = dy_current < 0 and dy_next > 0

            if negative_change or positive_change:
                change_count = 0
                for offset in range(1, int(minimum_change_frames_for_hit * 1.2) + 1):
                    dy_future = df_ball_positions['delta_y'].iloc[i + offset]

                    if negative_change and dy_future < 0:
                        change_count += 1
                    elif positive_change and dy_future > 0:
                        change_count += 1

                if change_count >= minimum_change_frames_for_hit - 1:
                    df_ball_positions.at[i, 'ball_hit'] = 1
   
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stub = False, stub_path = None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            ball_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        results = self.model.predict(frame, conf = 0.15)[0]

        ball_dict = {}

        for box in results.boxes:
            result= box.xyxy.tolist()[0]
            ball_dict[1] = result
           
        return ball_dict
    
    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections): # zip enables looping over two lists in parallel
            # draw bounding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Ball ID: {track_id}", (int(bbox[0]), int(bbox[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2) # 2 means just outside borders
            output_video_frames.append(frame)

        return output_video_frames        