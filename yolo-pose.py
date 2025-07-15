import cv2
from ultralytics import YOLO

import cv2
import numpy as np
import collections
import yolo_model
import os
import argparse
from datetime import datetime

LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14

MAX_X = 320

#MODEL_NAME='yolov8n-pose.pt'
MODEL_NAME='yolo11n-pose.pt'

is_60 = True
if is_60 == False:
# 30 frmae
    FRAMECNT = 30
    SEG_W = 20
else:
    # 60 frame
    FRAMECNT = 60
    SEG_W = 10
def angle_between_vectors(A, B, C):
    # 計算向量 AB 和 AC
    AB = np.array([B[0] - A[0], B[1] - A[1]])
    AC = np.array([C[0] - A[0], C[1] - A[1]])

    # 計算內積
    dot_product = np.dot(AB, AC)

    # 計算向量長度
    norm_AB = np.linalg.norm(AB)
    norm_AC = np.linalg.norm(AC)

    # 計算夾角（弧度）
    if norm_AB == 0 or norm_AC == 0:
        return 0
    cos_theta = dot_product / (norm_AB * norm_AC)
    theta_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # 防止浮點誤差超過 [-1, 1]

    # 轉換為角度
    theta_deg = np.degrees(theta_rad)
    if np.isnan(theta_deg):
        theta_deg = 0
    return theta_deg

def get_middle_pt(a, b):
    if a == 0:
        return b
    
    if b == 0:
        return a

    return (a + b)>>1

def get_point(W, H, keypoints):
    #print(W,H)
    if len(keypoints) ==0:
        return []
    tmp = []
    for index, landmarks in enumerate(keypoints):
        #print(index, landmarks)
        if landmarks[2] >0.9:
            #print(landmarks.x, landmarks.y, landmarks.z)
            tmp.append([landmarks[0]/W,landmarks[1]/H,landmarks[2]])
        else:             
            tmp.append([0, 0, 0])
    return tmp
def get_image_key_landmarks(image, results):
    tmp = results

    if tmp[LEFT_SHOULDER][0] == tmp[RIGHT_SHOULDER][1] == 0:
        return [0, 0, 0, 0]
    if tmp[LEFT_HIP][0] == tmp[RIGHT_HIP][1] == 0:
        return [0, 0, 0, 0]
    if tmp[LEFT_KNEE][0] == tmp[RIGHT_KNEE][1] == 0:
        return [0, 0, 0, 0]
    x1 = int(image.shape[1]*tmp[LEFT_SHOULDER][0])
    y1 = int(image.shape[0]*tmp[LEFT_SHOULDER][1])
    cv2.circle(image, (x1, y1), 5, (0, 0, 255), thickness=-1)
    x2 = int(image.shape[1]*tmp[RIGHT_SHOULDER][0])
    y2 = int(image.shape[0]*tmp[RIGHT_SHOULDER][1])
    cv2.circle(image, (x2, y2), 5, (0, 0, 255), thickness=-1)

 
    x12 = get_middle_pt(x1, x2) # (x1 + x2) >> 1
    y12 = get_middle_pt(y1, y2) #(y1 + y2) >> 1
    #print(x1, y1, x2, y2, x12,y12)
    cv2.circle(image, (x12, y12), 3, (0, 0, 255), thickness=-1)

    # 2nd
    #print("hip left",tmp[23], image.shape[1]* tmp[23][0], image.shape[0]* tmp[23][1])
    #print("hip right",tmp[24], image.shape[1]* tmp[24][0], image.shape[0]* tmp[24][1])
    x3 = int(image.shape[1]* tmp[LEFT_HIP][0])
    y3 = int(image.shape[0]* tmp[LEFT_HIP][1])
    cv2.circle(image, (x3, y3), 5, (0, 0, 255), thickness=5)
    x4 = int(image.shape[1]* tmp[RIGHT_HIP][0])
    y4 = int(image.shape[0]* tmp[RIGHT_HIP][1])
    cv2.circle(image, (x4, y4), 5, (0, 0, 255), thickness=5)

    x34 = get_middle_pt(x3, x4) #(x3 + x4) >> 1
    y34 = get_middle_pt(y3, y4) #(y3 + y4) >> 1

    #print(x3,y3, x4, y4, x34,y34)
    cv2.circle(image, (x34, y34), 3, (0, 0, 255), thickness=-1)

    # 3rd
    #print("knee left",tmp[25], image.shape[1]* tmp[25][0], image.shape[0]* tmp[25][1])
    #print("knee right",tmp[26], image.shape[1]* tmp[26][0], image.shape[0]* tmp[26][1])
    x5 = int(image.shape[1]* tmp[LEFT_KNEE][0])
    y5 = int(image.shape[0]* tmp[LEFT_KNEE][1])
    cv2.circle(image, (x5, y5), 5, (0, 0, 255), thickness=-1)
    x6 = int(image.shape[1]* tmp[RIGHT_KNEE][0])
    y6 = int(image.shape[0]* tmp[RIGHT_KNEE][1])
    cv2.circle(image, (x6, y6), 5, (0, 0, 255), thickness=-1)

    x56 = get_middle_pt(x5, x6) # (x5 + x6) >> 1
    y56 = get_middle_pt(y5, y6) # (y5 + y6) >> 1
    cv2.circle(image, (x56, y56), 3, (0, 0, 255), thickness=-1)

      
    cv2.line(image, (x12, y12), (x34, y34) , (255, 250, 0), thickness=5)
    cv2.line(image, (x34, y34), (x56, y56) , (255, 0, 0), thickness=5)

    theta = angle_between_vectors((x34, y34), (x12, y12) ,  (x56, y56) )

    #print([(x12, y12), (x34, y34), (x56, y56), theta])
    
    dx = x12-x56
    dy = y12-y56

    #a = input()
    
    return [MAX_X - int( dx  / image.shape[1] * MAX_X),
            MAX_X - int( dy  / image.shape[0] * MAX_X), 
            MAX_X - int( y34 / image.shape[0] * MAX_X-40), 
            MAX_X - int(theta)]

def draw_image(width, height, data):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    n = len(data)
    x = 0
    
    yoffset = 0

    for i in range(1,n):
        #print(i)
        color = (255, 0, 0)  # Blue line in BGR
        thickness = 3
        
        start_point = (x, data[i-1][0] + yoffset)
        end_point = (x+SEG_W, data[i][0] + yoffset)
        #print(0,start_point, end_point)
        # Draw the line
        cv2.line(image, start_point,end_point, color, thickness)

        color = (0, 255, 0)  # Blue line in BGR
        start_point = (x, data[i-1][1] + yoffset)
        end_point = (x+SEG_W, data[i][1] + yoffset)
        #print(1,start_point, end_point)
        cv2.line(image, start_point,end_point, color, thickness)
        
        color = (0, 0, 255)  # Blue line in BGR
        start_point = (x, data[i-1][2] + yoffset)
        end_point = (x+SEG_W, data[i][2] + yoffset)
        #print(2,start_point, end_point)
        cv2.line(image, start_point,end_point, color, thickness)
        
        color = (255, 255, 0)  # Blue line in BGR
        start_point = (x, data[i-1][3] + yoffset)
        end_point = (x+SEG_W, data[i][3] + yoffset)
        #print(3,start_point, end_point)
        cv2.line(image, start_point,end_point, color, thickness)
        
        x+=SEG_W
    #cv2.imshow('MediaPipe Pose Train', image)
    #cv2.waitKey(0)
    return image
def get_landmarks(full_path,file, output_dir, output_filename):
 
    # yolo model
    model = YOLO(MODEL_NAME) 

    res_point = []
    cnt = 0
    data = []

    train_cnt = 0
  
    #image = cv2.imread(root_path+"\\"+file)
    
    cap = cv2.VideoCapture(full_path)
    print(full_path)
    while cap.isOpened():
        success, frame = cap.read()
        if frame is None:
            print("Ignoring empty frame.")
            break
            
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 偵測人體姿勢
        results = model(frame, verbose=False)
        keypoints_tensor = results[0].keypoints  # 是個 Keypoints 物件
        tensor_data = keypoints_tensor.data
        keypoints_list = tensor_data.tolist()

        if len(keypoints_list) > 0 and len(keypoints_list[0]) > 0:
            res = get_point(frame.shape[1], frame.shape[0], keypoints_list[0])
            #print("get_point",res)
            curv_data = get_image_key_landmarks(frame, res)

            if curv_data[0] > 0 and curv_data[1] > 0 and curv_data[2] > 0:
                # print("curv_data",curv_data)
                data.append(curv_data)
                cnt +=1

        if cnt == FRAMECNT:

            train_image = draw_image(640,640,data)
            cv2.imshow('MediaPipe Pose Org', train_image)    
            print(output_dir + "\\" + output_filename.replace(" ", "_") + "_" + file.replace(" ", "_") + "_org_" + str(train_cnt) + ".png")
         
            cv2.imwrite(output_dir+"_org" + "\\" + output_filename + "_" + file.replace(" ", "_") + "_" + str(train_cnt) + "_org_"+ ".png", frame)
            cv2.imwrite(output_dir + "\\" + output_filename + "_" + file.replace(" ", "_") + "_" + str(train_cnt) + "_train_" + ".png", train_image)
            train_cnt+=1
            cnt=0
            data=[]

        cv2.imshow('MediaPipe Pose Org', frame)    
        cv2.waitKey(1)

def train():
    data_dir = r'D:\ai\pose_detect\KNN_image_train\dataset_0610_60'  # 換成你的資料夾
    yolo_model.train("pose_model.pth",data_dir)

def data_converter():
    root_path = r"F:\ai_dataset\falling_detect\Fall_video_0607\normal"

    files = os.listdir(root_path)
    for file in files:
        full_path = root_path + "\\"+file
        get_landmarks(full_path, file, "output_normal_60", "normal")

def predict(video, conf):
    print(video)
    print(conf)
    cap = None
    if video:
        print(f"執行預測，來源影片為：{video}")
        cap = cv2.VideoCapture(video)
    else:
        # 開啟 webcam（0 是預設攝影機）
        print("執行預測（使用預設來源）")
        cap = cv2.VideoCapture(0)

    if conf == None:
        conf = 0.55
    else:
        conf = float(conf)
    pose_model, transform, device = yolo_model.load_model();
    # 載入 YOLOv8 Pose 模型（nano版本速度快，適合webcam）
    model = YOLO(MODEL_NAME)  # 可改為 yolov8s-pose.pt, yolov8m-pose.pt ...
    tracker_config = "botsort.yaml"
    q = collections.deque([])

    id_map = collections.defaultdict(collections.deque)

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
       
        #print(frame.shape)
        # 偵測人體姿勢
        #results = model(frame, verbose=False)
        results = model.track(source=frame, persist=True, tracker=tracker_config, stream=True, verbose=False)

        #print(results, dir(results))
        for r in results:
            if r.boxes.id is None:
                continue  # 沒有追蹤到的跳過
            #print(r.boxes.id.cpu().numpy().astype(int))
            ids = r.boxes.id.cpu().numpy().astype(int)
            keypoints = r.keypoints.data.tolist()        # 每個人的 keypoints
            #print(keypoints)
            for tid, kpts in zip(ids, keypoints):
                #print(tid, kpts)
                res = get_point(frame.shape[1], frame.shape[0], kpts)                
                curv_data = get_image_key_landmarks(frame, res)
                #print("get_point", tid, curv_data)
                if curv_data[0] == 0 and curv_data[1] == 0 and curv_data[2] == 0:
                    continue
                if curv_data[0] > 0 and curv_data[1] > 0 and curv_data[2] > 0:
                    # print("curv_data",curv_data)
                    id_map[tid].append(curv_data)
                if len(id_map[tid]) == FRAMECNT:
                    curv_image = draw_image(640,640,id_map[tid])
                    pred_res = yolo_model.predict(pose_model, transform,device, curv_image)
                    cv2.imshow('MediaPipe Pose Train' + str(tid), curv_image)
                    id_map[tid].popleft()

                    if len(pred_res) > 0:
                        if pred_res[0] == yolo_model.class_names[1]:
                            txt_color = (0, 255, 0)
                        else:
                            txt_color = (255, 0, 0)
                        res_display = pred_res[0]

                        if pred_res[1] != -1:
                            res_display += " " +str(pred_res[1])
                        if pred_res[1] > conf:
                            print(pred_res)
                            cv2.putText(frame, f'Prediction: {res_display}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,txt_color , 2)

                    a = cv2.waitKey(1) & 0xFF
                    #print("get " + str(a))
                    if a == ord('y'):
                        testing_filename = "live_testing" + "\\" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
                        print(testing_filename)
                        cv2.imwrite( testing_filename , curv_image)
        #for r in results:
        #    #print(r.keypoints.xy)
        #    print("xyn",r.keypoints)
        #    print("xyn",r.keypoints.xyn)

        #keypoints_tensor = results[0].keypoints  # 是個 Keypoints 物件

        #print(results)

        #tensor_data = keypoints_tensor.data
        # 轉成 list
        #keypoints_list = tensor_data.tolist()
        #keypoints_list = keypoints_tensor.xyn.tolist()

        #for i, peoples in enumerate(keypoints_list):
        #    for k in peoples:
        #        print(i,k)

        #print("keypoints_list len",len(keypoints_list))
        #if len(keypoints_list) > 0 and len(keypoints_list[0]) > 0:
        #    for i in range(len(keypoints_list)):
        #        res = get_point(frame.shape[1], frame.shape[0], keypoints_list[i])                
        #        curv_data = get_image_key_landmarks(frame, res)
        #        print("get_point", i, curv_data)
        #
        #    if curv_data[0] > 0 and curv_data[1] > 0 and curv_data[2] > 0:
        #        # print("curv_data",curv_data)
        #        q.append(curv_data)

        #pred_res = []
        #if len(q) == FRAMECNT:        
        #    curv_image = draw_image(640,640,q)
        #    pred_res = yolo_model.predict(pose_model, transform,device, curv_image)
        #    cv2.imshow('MediaPipe Pose Train', curv_image)
        #    q.popleft()
        #    a = cv2.waitKey(0) & 0xFF
        #    #print("get " + str(a))
        #    if a == ord('y'):
        #        testing_filename = "live_testing" + "\\" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".png"
        #        print(testing_filename)
        #        cv2.imwrite( testing_filename , curv_image)
        ## 在畫面上標示 keypoints
        #annotated_frame = results[0].plot()

        #if len(pred_res) > 0:
        #    if pred_res[0] == yolo_model.class_names[1]:
        #        txt_color = (0, 255, 0)
        #    else:
        #        txt_color = (255, 0, 0)
        #    res_display = pred_res[0]
        #
        #    if pred_res[1] != -1:
        #        res_display += " " +str(pred_res[1])
        #
        #    cv2.putText(annotated_frame, f'Prediction: {res_display}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,txt_color , 2)
        #
        ## 顯示畫面
        #cv2.imshow("YOLOv8 Pose", annotated_frame)
        cv2.imshow("YOLOv8 Pose", frame)

        # 印出 keypoints 資料
        #for person_id, person in enumerate(results[0].keypoints.data):
        #    print(f"Person {person_id}:")
        #    for i, (x, y, conf) in enumerate(person):
        #        print(f"  Keypoint {i}: x={x:.1f}, y={y:.1f}, conf={conf:.2f}")
        #    print("-" * 30)
        #
        # 按 'q' 鍵離開
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="AI 工具命令列介面")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # train 子命令
    parser_train = subparsers.add_parser("train", help="執行訓練程序")

    # predict 子命令
    parser_predict = subparsers.add_parser("predict", help="執行預測程序")
    parser_predict.add_argument("--video", help="指定影片路徑 (選用)")
    parser_predict.add_argument("--conf", help="conf (選用)")

    # data_converter 子命令
    parser_converter = subparsers.add_parser("data_converter", help="執行資料轉換")

    args = parser.parse_args()

    # 根據子命令執行對應函式
    if args.command == "train":
        train()
    elif args.command == "predict":
        predict(args.video, args.conf)
    elif args.command == "data_converter":
        data_converter()

if __name__ == "__main__":
    main()