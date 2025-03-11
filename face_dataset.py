import os
from mask_detection import *
import cv2
import numpy as np

def create_dataset(username):
    id = username
    if(os.path.exists('./training_dataset/{}/'.format(id))==False):
        os.makedirs('./training_dataset/{}/'.format(id))
    directory='./training_dataset/{}/'.format(id)
##############################################################################3
    #----var
    pb_path = "face_mask_detection.pb"
    node_dict = {'input':'data_1:0',
                'detection_bboxes':'loc_branch_concat_1/concat:0',
                'detection_scores':'cls_branch_concat_1/concat:0'}
    conf_thresh = 0.5
    iou_thresh = 0.4
    frame_count = 0
    FPS = "0"
    #====anchors config
    feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
    anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
    anchor_ratios = [[1, 0.62, 0.42]] * 5
    id2class = {0: 'Mask', 1: 'NoMask'}

    # Detect face
    #Loading the HOG face detector and the shape predictpr for allignment
    #====model restore from pb file
    sess,node_dict = model_restore_from_pb(pb_path, node_dict)
    tf_input = node_dict['input']
    model_shape = tf_input.shape#[N,H,W,C]
    print("model_shape = ", model_shape)
    detection_bboxes = node_dict['detection_bboxes']
    detection_scores = node_dict['detection_scores']
    is_2_write  = False
    save_path = "output.avi"
    cap, height, width, writer = video_init(is_2_write=is_2_write,save_path=save_path)

    #====generate anchors
    anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
    # for inference , the batch size is 1, the model output shape is [1, N, 4],
    # so we expand dim for anchors to [1, anchor_num, 4]
    anchors_exp = np.expand_dims(anchors, axis=0)
    idx_save=1
    while (cap.isOpened()):
        #----get image
        ret, img = cap.read()
        if ret:
            #----image processing
            img_resized = cv2.resize(img, (int(model_shape[2]), int(model_shape[1])))
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_resized = img_resized.astype('float32')
            img_resized /= 255

            #----mask detection
            y_bboxes_output, y_cls_output = sess.run([detection_bboxes, detection_scores],
                                                    feed_dict={tf_input: np.expand_dims(img_resized, axis=0)})

            #remove the batch dimension, for batch is always 1 for inference.
            y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
            y_cls = y_cls_output[0]
            # To speed up, do single class NMS, not multiple classes NMS.
            bbox_max_scores = np.max(y_cls, axis=1)
            bbox_max_score_classes = np.argmax(y_cls, axis=1)
            # keep_idx is the alive bounding box after nms.
            keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                        bbox_max_scores,
                                                        conf_thresh=conf_thresh,
                                                        iou_thresh=iou_thresh,
                                                        )
            #====draw bounding box
            for idx in keep_idxs:
                conf = float(bbox_max_scores[idx])
                class_id = bbox_max_score_classes[idx]
                bbox = y_bboxes[idx]
                # clip the coordinate, avoid the value exceed the image boundary.
                xmin = max(0, int(bbox[0] * width))
                ymin = max(0, int(bbox[1] * height))
                xmax = min(int(bbox[2] * width), width)
                ymax = min(int(bbox[3] * height), height)
                # Cắt phần khuôn mặt từ ảnh gốc
                extend = 25
                face_crop = img[ymin- extend:ymax + extend, xmin - extend : xmax + extend]

                # Kiểm tra xem có khuôn mặt hợp lệ không (tránh lỗi khi xmin/xmax hoặc ymin/ymax không đúng)
                if face_crop.size > 0:
                    face_filename = f"face_{idx_save}.jpg"  # Đặt tên file theo số thứ tự
                    idx_save += 1
                    cv2.imwrite(os.path.join(directory,face_filename), face_crop)  # Lưu ảnh
                print(f"Saved: {face_filename}")

                if class_id == 0:
                    color = (0, 255, 0)  # (B,G,R)
                else:
                    color = (0, 0, 255)  # (B,G,R)
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)

            #----FPS count
            if frame_count == 0:
                t_start = time.time()
            frame_count += 1
            if frame_count >= 10:
                FPS = "FPS=%1f" % (10 / (time.time() - t_start))
                frame_count = 0

            cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            #----image display
            cv2.imshow("demo by AI4SEC", img)

            #----image writing
            if writer is not None:
                writer.write(img)

            #----'q' key pressed?
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("get image failed")
            break
    #----release
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Press Enter to continue...")
    create_dataset(username)